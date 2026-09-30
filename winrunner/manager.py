"""Model lifecycle orchestration: planning, loading, JIT loading, eviction, recovery."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .cmdline import build_fit_args, build_server_args
from .config import LoadParams, SettingsStore
from .engine import EngineDevice, EngineInfo, EngineManager
from .events import EventBus, RequestTracker
from .faults import GPU_DEVICE_LOST, gpu_fault, tdr_hint, tdr_limited
from .hardware import HardwareMonitor
from .instance import EngineInstance
from .library import ModelEntry, ModelLibrary
from .paths import IS_WINDOWS, DataPaths
from .planner import FIT_CTX_MIN, DevicePlan, Plan, Planner, assign_layers, parse_fit_args, parse_fit_print
from .util import free_port, new_id

log = logging.getLogger("winrunner.manager")

_CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

# Automatic recovery of an engine that crashed or lost its GPU while serving.
RECOVERY_LIMIT = 3  # automatic restarts of one model ...
RECOVERY_WINDOW_S = 600.0  # ... within this period; after that the model is unloaded until requested again
RECOVERY_DELAYS_S = (3.0, 10.0, 30.0)  # pause before each reload attempt (driver reset, memory release)
DEVICE_WAIT_S = 90.0  # longest wait for the GPUs to become usable again after a reset
DEVICE_POLL_S = 2.0
DEVICE_SETTLE_MIB = 256  # free VRAM counts as settled when it grows less than this between two polls

SOURCE_LABELS = {"jit": "JIT request", "ui": "control panel", "startup": "startup", "recovery": "automatic restart"}
_RECOVERY_ABORTS = ("shutting_down", "model_replaced")  # reasons to stop a recovery at once (no retries)

# Engine projection (llama-fit-params) of a plan
MAX_PROJECTIONS = 5  # measurements per plan: the estimate is calibrated until the measured layout is the chosen one
CALIB_MIN_SPAN = 2048  # context tokens between two measurements before their difference is used as a slope
PROJECTION_ROUNDING_MIB = 3  # llama-fit-params prints whole MiB (rounded down) for 3 buffers per device
PLAN_CACHE_S = 120.0
FIT_TIMEOUT_S = 180.0


class ModelError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "model_error"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class _Recovery:
    """Replacement of a failed engine instance by a new engine process."""

    old: EngineInstance
    reason: str
    future: asyncio.Future
    task: asyncio.Task | None = None
    loading: bool = False  # the replacement engine is being started (no longer cancellable)


class ModelManager:
    def __init__(self, store: SettingsStore, paths: DataPaths, library: ModelLibrary, engines: EngineManager,
                 monitor: HardwareMonitor, bus: EventBus, tracker: RequestTracker):
        self.store = store
        self.paths = paths
        self.library = library
        self.engines = engines
        self.monitor = monitor
        self.bus = bus
        self.tracker = tracker
        self.instances: dict[str, EngineInstance] = {}
        self._load_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}
        self._engine: EngineInfo | None = None
        self._tasks: list[asyncio.Task] = []
        self._recoveries: dict[str, _Recovery] = {}  # model id -> running recovery
        self._recovery_log: dict[str, list[float]] = {}  # model id -> times of automatic restarts
        self._background: set[asyncio.Task] = set()
        self._substitutes: set[tuple[str, str]] = set()  # (requested name, model answering) already reported
        self._tdr_hint_shown = False
        self._plan_cache: dict[str, tuple[float, dict]] = {}
        self._shutting_down = False
        self.exiting: Any = lambda: False  # set by the entry point: True once the server is stopping
        self.ui_clients = 0

    # ----- engine ---------------------------------------------------------------------

    def engine(self, refresh: bool = False) -> EngineInfo | None:
        es = self.store.settings.engine
        server = self.engines.resolve_server(es.engine_path, es.active_engine, es.backend)
        if server is None:
            self._engine = None
            return None
        if refresh or self._engine is None or self._engine.path != str(server):
            self._engine = self.engines.probe(server, force=refresh)
        return self._engine

    async def engine_async(self, refresh: bool = False) -> EngineInfo | None:
        return await asyncio.to_thread(self.engine, refresh)

    async def devices(self, max_age: float = 5.0) -> tuple[list[EngineDevice], str]:
        eng = await self.engine_async()
        if eng is None:
            return [], "no engine installed"
        return await asyncio.to_thread(self.engines.list_devices, Path(eng.path), max_age)

    def device_map(self, devices: list[EngineDevice]) -> dict[str, str]:
        return self.monitor.map_engine_devices([{"name": d.name, "description": d.description} for d in devices])

    # ----- lifecycle helpers ------------------------------------------------------------

    def start_background(self) -> None:
        self._tasks.append(asyncio.create_task(self._slots_loop()))
        self._tasks.append(asyncio.create_task(self._idle_loop()))

    async def shutdown(self) -> None:
        self._shutting_down = True
        for t in self._tasks:
            t.cancel()
        for r in list(self._recoveries.values()):
            if r.task is not None and not r.loading:
                r.task.cancel()
        await asyncio.gather(*(i.stop(timeout=6) for i in list(self.instances.values())), return_exceptions=True)
        self.instances.clear()

    def ready_instances(self) -> list[EngineInstance]:
        return [i for i in self.instances.values() if i.state == "ready"]

    def instance_for_model(self, model_id: str) -> EngineInstance | None:
        for i in self.instances.values():
            if i.model_id == model_id and i.state in ("ready", "loading", "starting"):
                return i
        return None

    # ----- planning ---------------------------------------------------------------------

    def _mmproj_for(self, entry: ModelEntry, p: LoadParams) -> str | None:
        if p.mmproj == "none":
            return None
        if p.mmproj:
            return p.mmproj if Path(p.mmproj).is_file() else None
        return entry.mmproj_default or None

    def _draft_for(self, p: LoadParams) -> ModelEntry | None:
        if not p.draft_model:
            return None
        return self.library.get(p.draft_model) or self.library.resolve(p.draft_model)

    def _margins(self, devices: list[EngineDevice], p: LoadParams) -> list[int]:
        hw = self.store.settings.hardware
        sel = [d for d in devices if not p.devices or d.name in p.devices]
        return [int(hw.vram_margin_per_device.get(d.name, hw.vram_margin_mib)) for d in sel]

    def _fit_targets(self, devices: list[EngineDevice], p: LoadParams, pl: Plan, projection: bool = False) -> list[int]:
        """MiB that llama.cpp's --fit keeps free per device: the safety margin plus what its projection misses.

        The engine does not see the run-time scratch buffers of the GPU backend. ``projection``: targets for
        llama-fit-params, which in addition loads neither the vision projector nor the other server slots.
        """
        plans = {d.name: d for d in pl.devices}
        out = []
        for d, margin in zip([d for d in devices if not p.devices or d.name in p.devices], self._margins(devices, p)):
            dp = plans.get(d.name)
            extra = dp.scratch_mib if dp else 0.0
            if dp and projection:
                extra += dp.swa_extra_mib + dp.mmproj_mib
            out.append(int(margin + math.ceil(extra)))
        return out

    def _reclaim(self, evicting: list[EngineInstance]) -> dict[str, float]:
        """VRAM that loaded instances would free if they are evicted."""
        out: dict[str, float] = {}
        for inst in evicting:
            for dev, bd in inst.load_info.get("breakdown", {}).items():
                if dev != "Host":
                    out[dev] = out.get(dev, 0.0) + float(bd.get("self", 0))
            if not inst.load_info.get("breakdown"):
                for dev, b in inst.load_info.get("buffers", {}).items():
                    out[dev] = out.get(dev, 0.0) + sum(float(x) for x in b.values())
        return out

    def _evict_set(self, model_id: str) -> list[EngineInstance]:
        if not self.store.settings.server.auto_evict:
            return []
        return [i for i in self.instances.values() if i.model_id != model_id and i.state in ("ready", "loading")]

    async def plan(self, model_id: str, overrides: dict | None = None, verify: bool = False,
                   assume_evict: bool = True) -> tuple[Plan, dict[str, Any]]:
        entry = self.library.get(model_id) or self.library.resolve(model_id)
        if entry is None:
            raise ModelError(f"model '{model_id}' not found in library", 404, "model_not_found")
        eng = await self.engine_async()
        p = self.store.effective_load_params(entry.path, overrides)
        # an engine projection is compared with the free VRAM right now
        devices, dev_err = await self.devices(max_age=0.0 if verify else 3.0)
        evicting = self._evict_set(entry.id) if assume_evict else []
        mmproj = self._mmproj_for(entry, p)
        mm_size = 0
        mm_hint = None
        if mmproj:
            try:
                mm_size = os.path.getsize(mmproj)
            except OSError:
                mm_size = 0
            mm_hint = self.store.profile(entry.path).mmproj_est_mib or None
        draft = self._draft_for(p)
        hw = self.store.settings.hardware
        reclaim = self._reclaim(evicting)
        engine_fit = bool(eng and self.store.settings.engine.use_engine_fit and eng.has("--fit", "-fit"))

        def planner(calibration: dict[str, tuple[float, float]] | None = None) -> Planner:
            return Planner(
                entry.info, p, devices, self.device_map(devices),
                margin_mib=hw.vram_margin_mib,
                margin_per_device=hw.vram_margin_per_device,
                reclaim_mib=reclaim,
                mmproj_size=mm_size, mmproj_mib_hint=mm_hint,
                draft_info=draft.info if draft else None,
                engine_fit=engine_fit,
                model_id=entry.id,
                calibration=calibration,
            )

        pl = planner().plan()
        if pl.use_engine_fit and evicting:
            # the engine measures free memory itself; the evicted model is still resident during planning
            pl.notes.append("Memory projection accounts for the currently loaded model being unloaded first.")
        extra = {
            "entry": entry.to_summary(),
            "devices": [d.__dict__ for d in devices],
            "device_error": dev_err,
            "engine": eng.to_dict() if eng else None,
            "evicting": [i.model_id for i in evicting],
            "margins": self._margins(devices, p),
        }
        if verify and eng and eng.fit_params and devices and not evicting:
            try:
                pl = await self._verify_with_engine(eng, entry, p, pl, devices, planner, mmproj, draft)
            except Exception as exc:  # projection is advisory; never block a load on it
                pl.warnings.append(f"Engine memory projection failed: {exc}")
        elif verify and evicting:
            pl.notes.append("Engine projection skipped: the engine measures free VRAM directly, and "
                            f"{', '.join(i.model_id for i in evicting)} is still loaded. It runs automatically at load time, "
                            "after the current model is unloaded.")
        elif verify and eng and not eng.fit_params:
            pl.notes.append("This engine build does not include llama-fit-params; the analytic estimate is used.")
        pl.mmproj = mmproj or ""
        pl.draft = draft.path if draft else ""
        if eng:
            sel = [d.name for d in devices if not p.devices or d.name in p.devices]
            spec = build_server_args(
                eng, entry.path, p, pl, sel, 0, entry.id, "", mmproj, draft.path if draft else None,
                self._template_file(entry, p), entry.info.kind == "embedding",
                self.store.settings.engine.log_verbosity, self._fit_targets(devices, p, pl),
            )
            args = [a for a in spec.args]
            i = args.index("--port")
            args[i + 1] = "<port>"
            extra["command"] = " ".join(args)
        return pl, extra

    async def _verify_with_engine(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan,
                                  devices: list[EngineDevice], planner: Callable[..., Planner], mmproj: str | None,
                                  draft: ModelEntry | None) -> Plan:
        """Measure the plan with llama.cpp's own allocator (llama-fit-params) and refine it.

        llama.cpp's projection is exact for what it allocates at load (weights, KV cache, compute buffers). The
        estimate is calibrated with it, per device, until the measured layout is the one the planner chooses, so
        the context fills the GPUs up to the safety margins without spilling. A plan that leaves part of the model
        in system RAM is placed by the engine's own --fit at load; its layout is shown.
        """
        sel = [d.name for d in devices if not p.devices or d.name in p.devices]
        key_src = json.dumps([entry.path, entry.info.mtime, p.model_dump(mode="json"), eng.path, mmproj or "",
                              draft.path if draft else "", self.store.settings.engine.use_engine_fit,
                              [(d.name, d.total_mib, d.free_mib // 32) for d in devices],
                              [(d.name, d.free_mib, d.margin_mib, round(d.mmproj_mib)) for d in pl.devices]],
                             sort_keys=True)
        key = hashlib.sha1(key_src.encode()).hexdigest()
        cached = self._plan_cache.get(key)
        if cached and time.time() - cached[0] < PLAN_CACHE_S:
            return copy.deepcopy(cached[1])

        history: dict[str, list[tuple[int, float]]] = {}
        fitting: dict[tuple[str, str], tuple[Plan, dict]] = {}  # per KV type: the largest measured plan that fits
        last_fit: tuple[str, str] | None = None
        measured = 0
        cur = pl
        # The estimate says the model does not fit in VRAM even at the smallest context: measure it there first.
        # That measurement only calibrates the estimate, unless the planner then chooses exactly this layout.
        probing = self._worth_measuring_full(entry, pl)
        if probing:
            cur = planner().full_offload_plan(pl.kv_k, min(pl.ctx_target, FIT_CTX_MIN))
        proj: dict[str, dict[str, float]] = {}
        bump: dict[str, float] = {}  # extra MiB to keep free where a calibrated layout still measured too large
        last_measured: tuple[Plan, dict[str, dict[str, float]]] | None = None
        while measured < MAX_PROJECTIONS and not cur.use_engine_fit:
            proj = await self._project(eng, entry, p, cur, sel)
            measured += 1
            last_measured = (cur, proj)
            self._record_calibration(cur, proj, history)
            fits = self._projection_fits(cur, proj)
            nxt = planner(self._calibration(history, bump)).plan()
            same = nxt.layout_key() == cur.layout_key()
            if same and not fits:
                for d in cur.devices:
                    over = self._measured_use(d, proj.get(d.name, {})) - (d.free_mib - d.margin_mib)
                    if over > 0:
                        bump[d.name] = bump.get(d.name, 0.0) + over + 8
                nxt = planner(self._calibration(history, bump)).plan()
                same = nxt.layout_key() == cur.layout_key()
            if fits and (same or not probing):
                chosen = nxt if same else cur  # same layout: take the plan with the latest calibration
                kv = (chosen.kv_k, chosen.kv_v)
                if kv not in fitting or chosen.ctx >= fitting[kv][0].ctx:
                    fitting[kv] = (chosen, proj)
                last_fit = kv
            probing = False
            if same:
                break
            cur = nxt

        if last_fit is not None:
            # every plan in `fitting` was measured; within the KV type chosen last, the largest context wins
            final, proj = fitting[last_fit]
            self._apply_projection(final, proj)
        elif cur.use_engine_fit or last_measured is None:
            final = await self._engine_fit_projection(eng, entry, p, cur, sel, devices)
            measured += 2
        else:
            # nothing that was measured fits: the manual layout is larger than the free VRAM, or the engine needs
            # more than estimated even at the smallest context
            final, proj = last_measured
            self._apply_projection(final, proj)
            over = ", ".join(f"{d.name} by {self._measured_use(d, proj.get(d.name, {})) - (d.free_mib - d.margin_mib):,.0f} MiB"
                             for d in final.devices
                             if self._measured_use(d, proj.get(d.name, {})) > d.free_mib - d.margin_mib)
            if final.mode == "auto" and eng.has("--fit", "-fit"):
                final.use_engine_fit = True  # let llama.cpp place what does not fit
                final.full_offload = False
                final.load_mode = "mmap" if p.load_mode == "auto" else final.load_mode
                final.warnings.append(f"Engine projection: the model does not fit in VRAM ({over}); llama.cpp will "
                                      "keep the part that does not fit in system RAM.")
            else:
                final.warnings.append(f"Engine projection: this configuration exceeds the free VRAM ({over}); the "
                                      "load may fail or spill into shared system memory (much slower).")
        final.engine["measurements"] = measured
        final.engine["calibration"] = {k: [round(a, 1), round(b, 5)]
                                       for k, (a, b) in self._calibration(history, bump).items()}
        self._plan_cache[key] = (time.time(), copy.deepcopy(final))
        return final

    @staticmethod
    def _worth_measuring_full(entry: ModelEntry, pl: Plan) -> bool:
        """The estimate puts part of the model in system RAM, but the model's weights alone would fit in VRAM."""
        if pl.mode != "auto" or pl.full_offload or not pl.devices:
            return False
        weights = (entry.info.weights_bytes - entry.info.token_embd_bytes) / (1024 * 1024)
        return weights < sum(d.free_mib - d.margin_mib for d in pl.devices)

    async def _project(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan,
                       sel: list[str]) -> dict[str, dict[str, float]]:
        """llama-fit-params --fit-print for exactly the plan's layout: MiB of weights, context, compute per device."""
        args = build_fit_args(eng, entry.path, p, pl, sel, [], print_mode=True)
        if not args:
            raise RuntimeError("llama-fit-params is not available")
        rc, out = await asyncio.to_thread(_run_capture, args, FIT_TIMEOUT_S)
        proj = parse_fit_print(out) if rc == 0 else {}
        if not proj:
            raise RuntimeError(f"llama-fit-params failed ({rc}): {out.strip().splitlines()[-1] if out.strip() else ''}")
        return proj

    @staticmethod
    def _measured_use(d: DevicePlan, pr: dict[str, float]) -> float:
        """What the device will hold: llama.cpp's projection plus what the projection does not include."""
        return (pr.get("model", 0.0) + pr.get("context", 0.0) + pr.get("compute", 0.0) + PROJECTION_ROUNDING_MIB
                + d.swa_extra_mib + d.scratch_mib + d.mmproj_mib + d.draft_mib)

    def _projection_fits(self, pl: Plan, proj: dict[str, dict[str, float]]) -> bool:
        return all(d.name in proj and self._measured_use(d, proj[d.name]) <= d.free_mib - d.margin_mib
                   for d in pl.devices)

    @staticmethod
    def _record_calibration(pl: Plan, proj: dict[str, dict[str, float]], history: dict[str, list[tuple[int, float]]]) -> None:
        """Difference between llama.cpp's projection and the estimate of the same buffers, per device."""
        for d in pl.devices:
            pr = proj.get(d.name)
            if pr is None or not (d.layers or d.output_mib):
                continue
            estimate = d.weights_mib + d.output_mib + d.kv_mib - d.swa_extra_mib + d.compute_mib
            # the same rounding allowance as _measured_use: a calibrated layout fits exactly when it measures so
            engine = pr["model"] + pr["context"] + pr["compute"] + PROJECTION_ROUNDING_MIB
            history.setdefault(d.name, []).append((pl.ctx, engine - estimate))

    @staticmethod
    def _calibration(history: dict[str, list[tuple[int, float]]],
                     bump: dict[str, float] | None = None) -> dict[str, tuple[float, float]]:
        """Per device (MiB, MiB per context token), from the latest measurements."""
        out: dict[str, tuple[float, float]] = {}
        for name, pts in history.items():
            c2, e2 = pts[-1]
            prev = next(((c, e) for c, e in reversed(pts[:-1]) if abs(c - c2) >= CALIB_MIN_SPAN), None)
            if prev is None:
                a, b = e2, 0.0
            else:
                b = (e2 - prev[1]) / (c2 - prev[0])
                a = e2 - b * c2
            out[name] = (a + (bump or {}).get(name, 0.0), b)
        return out

    async def _engine_fit_projection(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan,
                                     sel: list[str], devices: list[EngineDevice]) -> Plan:
        """Layout llama.cpp's --fit chooses for a plan that does not fit in VRAM (measured for display)."""
        n_layer = entry.info.n_layer
        args = build_fit_args(eng, entry.path, p, pl, sel, self._fit_targets(devices, p, pl, projection=True),
                              print_mode=False)
        if not args:
            return pl
        rc, out = await asyncio.to_thread(_run_capture, args, FIT_TIMEOUT_S)
        fit = parse_fit_args(out)
        if rc != 0 or not fit:
            raise RuntimeError(f"llama-fit-params failed ({rc}): {out.strip().splitlines()[-1] if out.strip() else ''}")
        ngl = fit.get("ngl", -1)
        # every layer and the output layer (MTP layers are not loaded for decoding)
        full = (ngl < 0 or ngl > n_layer - entry.info.nextn_layers) and "override_tensor" not in fit
        q = copy.deepcopy(pl)
        q.gpu_layers = n_layer + 1 if ngl < 0 else min(ngl, n_layer + 1)
        q.tensor_split = fit.get("tensor_split") or pl.tensor_split
        q.n_cpu_moe = 0
        q.use_engine_fit = False
        pargs = build_fit_args(eng, entry.path, p, q, sel, [], print_mode=True)
        if fit.get("override_tensor") and pargs:
            pargs += ["-ot", fit["override_tensor"]]
        proj: dict[str, Any] = {}
        if pargs:
            rc, out = await asyncio.to_thread(_run_capture, pargs, FIT_TIMEOUT_S)
            if rc == 0:
                proj = parse_fit_print(out)
        pl.gpu_layers = q.gpu_layers
        pl.tensor_split = q.tensor_split if len(pl.devices) > 1 else None
        if "override_tensor" not in fit:
            pl.n_cpu_moe = 0  # otherwise keep the estimate's expert count for the layer map
        pl.full_offload = full
        if len(pl.devices) and pl.gpu_layers:
            assign = assign_layers(n_layer, pl.gpu_layers, pl.tensor_split or [1.0] * len(pl.devices))
            for i, d in enumerate(pl.devices):
                own = [il for il in range(n_layer) if assign[il] == i]
                d.layers = len(own)
                d.layer_first, d.layer_last = (own[0], own[-1]) if own else (-1, -1)
        self._apply_projection(pl, proj)
        pl.engine["fit"] = fit
        pl.notes = [n for n in pl.notes if not n.startswith(("Partial offload:", "MoE offload:"))]
        if full:
            pl.notes.append("Engine projection: the whole model fits in VRAM at this context.")
        else:
            ot = fit.get("override_tensor")
            pl.notes.append("Engine projection: llama.cpp keeps " + (
                "some expert or feed-forward weights in system RAM" if ot else
                f"{n_layer + 1 - q.gpu_layers} of {n_layer + 1} layers in system RAM") +
                " (prompt processing and generation will be much slower).")
        if p.load_mode == "auto":
            pl.load_mode = "none" if full else "mmap"
        return pl

    @staticmethod
    def _apply_projection(pl: Plan, proj: dict[str, dict[str, float]]) -> None:
        """Show llama.cpp's measured buffers instead of the estimate."""
        for d in pl.devices:
            pr = proj.get(d.name)
            if pr:
                d.weights_mib = pr["model"]
                d.output_mib = 0.0
                d.kv_mib = pr["context"] + d.swa_extra_mib
                d.compute_mib = pr["compute"]
                d.calib_mib = 0.0
        if "Host" in proj:
            pl.host = {"weights_mib": proj["Host"]["model"], "kv_mib": proj["Host"]["context"],
                       "compute_mib": proj["Host"]["compute"]}
        pl.totals["vram_used_mib"] = round(sum(d.used_mib for d in pl.devices), 1)
        pl.source = "engine"
        pl.engine = {**pl.engine, "projection": proj, "full_offload": pl.full_offload}

    def _template_file(self, entry: ModelEntry, p: LoadParams) -> str | None:
        if p.chat_template_mode != "custom" or not p.chat_template_custom.strip():
            return None
        h = hashlib.sha1(p.chat_template_custom.encode()).hexdigest()[:10]
        path = self.paths.templates_dir / f"{entry.id.replace('/', '_')}-{h}.jinja"
        if not path.exists():
            path.write_text(p.chat_template_custom, encoding="utf-8")
        return str(path)

    # ----- load / unload -----------------------------------------------------------------

    async def load(self, model_id: str, overrides: dict | None = None, source: str = "ui") -> EngineInstance:
        entry = self.library.get(model_id) or self.library.resolve(model_id)
        if entry is None:
            raise ModelError(f"model '{model_id}' not found in library", 404, "model_not_found")
        if entry.id in self._pending:
            return await asyncio.shield(self._pending[entry.id])
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[entry.id] = fut
        try:
            inst = await self._load_locked(entry, overrides, source)
            fut.set_result(inst)
            return inst
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()  # mark retrieved
            raise
        finally:
            self._pending.pop(entry.id, None)

    async def _load_locked(self, entry: ModelEntry, overrides: dict | None, source: str,
                           guard: Callable[[], None] | None = None) -> EngineInstance:
        async with self._load_lock:
            if guard is not None:
                guard()  # may raise: conditions can change while waiting for the lock
            eng = await self.engine_async()
            if eng is None:
                raise ModelError("No llama.cpp engine installed. Open Settings > Engine to download one.", 503,
                                 "engine_missing")
            if eng.probe_error and not eng.flags:
                raise ModelError(f"Engine is not usable: {eng.probe_error}", 503, "engine_error")
            existing = self.instance_for_model(entry.id)
            if existing and existing.state == "ready" and not overrides:
                return existing
            if existing:
                await self._unload(existing, reason="reload")
            for inst in self._evict_set(entry.id):
                await self._drain(inst)
                await self._unload(inst, reason=f"evicted for {entry.id}")
            self.bus.activity_log(f"Loading {entry.id} ({SOURCE_LABELS.get(source, source)})",
                                  category="model", model=entry.id)
            self.bus.publish("load_stage", model=entry.id, stage="planning", label="Planning memory layout")
            pl, extra = await self.plan(entry.id, overrides, verify=True, assume_evict=False)
            p = self.store.effective_load_params(entry.path, overrides)
            devices, _ = await self.devices(max_age=3.0)
            sel = [d.name for d in devices if not p.devices or d.name in p.devices]
            mmproj = pl.mmproj or None
            draft = self._draft_for(p)
            self.bus.activity_log(_placement_summary(pl), category="model", model=entry.id,
                                  level="ok" if pl.full_offload or not pl.devices else "warn")
            if self._shutting_down:  # planning can take a while; never start an engine during shutdown
                raise ModelError("WinRunner is shutting down", 503, "shutting_down")
            port = free_port()
            api_key = new_id("", 16)
            spec = build_server_args(
                eng, entry.path, p, pl, sel, port, entry.id, api_key, mmproj, draft.path if draft else None,
                self._template_file(entry, p), entry.info.kind == "embedding",
                self.store.settings.engine.log_verbosity, self._fit_targets(devices, p, pl),
            )
            prof = self.store.profile(entry.path)
            inst = EngineInstance(
                entry=entry, params=p, plan=pl, engine=eng, spec=spec, port=port, api_key=api_key, bus=self.bus,
                mmproj=mmproj, priority=self.store.settings.engine.process_priority,
                expected_load_s=prof.last_load_seconds, device_map=self.device_map(devices),
                on_exit=self._on_instance_exit, on_fault=self.report_fault,
            )
            self.instances[inst.id] = inst
            try:
                await inst.start(timeout=self.store.settings.engine.load_timeout_s)
            except Exception as exc:
                self.instances.pop(inst.id, None)
                self.bus.publish("instance_removed", iid=inst.id, model=inst.model_id, error=str(exc))
                raise ModelError(str(exc), 500, "load_failed") from exc
            if inst.pid:
                self.monitor.watch_process(inst.pid, inst.model_id)
            prof = self.store.profile(entry.path)
            prof.last_loaded = time.time()
            prof.load_count += 1
            prof.last_load_seconds = float(inst.load_info.get("load_seconds") or 0)
            est = inst.load_info.get("mmproj", {}).get("est_mib")
            if est:
                prof.mmproj_est_mib = float(est)
            self.store.set_profile(entry.path, prof)
            if self.store.settings.startup.last_model != entry.id:
                self.store.update({"startup": {"last_model": entry.id}})
            return inst

    async def unload(self, key: str) -> bool:
        rec = self._recoveries.get(key) or next((r for r in self._recoveries.values() if r.old.id == key), None)
        if rec is not None and rec.task is not None and not rec.loading:
            rec.task.cancel()  # unloaded while waiting to be restarted
            await asyncio.wait({rec.task})
            return True
        inst = self.instances.get(key) or self.instance_for_model(key)
        if not inst:
            return False
        async with self._load_lock:
            await self._unload(inst, reason="unloaded")
        return True

    async def _unload(self, inst: EngineInstance, reason: str) -> None:
        self.bus.activity_log(f"Unloading {inst.model_id} ({reason})", category="model", model=inst.model_id)
        if inst.pid:
            self.monitor.unwatch_process(inst.pid)
        await inst.stop()
        self.instances.pop(inst.id, None)
        self.bus.publish("instance_removed", iid=inst.id, model=inst.model_id)

    def _spawn(self, coro) -> asyncio.Task:
        """Start a background task and keep a reference to it until it is done."""
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _drop(self, inst: EngineInstance) -> None:
        """Forget a stopped instance."""
        if self.instances.get(inst.id) is inst:
            del self.instances[inst.id]
            self.bus.publish("instance_removed", iid=inst.id, model=inst.model_id)

    # ----- automatic recovery ------------------------------------------------------------------

    def _on_instance_exit(self, inst: EngineInstance, rc: int) -> None:
        """The engine process of a ready instance exited on its own (crash / abort)."""
        reason = inst.error or "engine process exited unexpectedly"
        # A lost GPU often ends in an abort; the engine's last words say so (debug records are skipped: they
        # can quote prompts).
        last_words = [r["text"] for r in list(inst.log)[-30:] if r["level"] != "debug"]
        what = next(filter(None, map(gpu_fault, reversed(last_words))), None)
        self._begin_recovery(inst, f"{what}: {reason}" if what else reason)

    def report_fault(self, inst: EngineInstance, reason: str) -> bool:
        """An engine reported an unrecoverable GPU error (e.g. Vulkan device lost).

        The engine process usually keeps running, but it can never use the lost GPU
        again: every further request would fail (or crash it). It is replaced by a new
        engine process. Returns True when a replacement is being started, so callers
        can wait for it and retry.
        """
        if inst.recovery is None and inst.state == "ready":
            inst.fail(f"{reason} (the engine cannot use the GPU any more)")
            self._begin_recovery(inst, reason)
        return inst.recovering

    def _begin_recovery(self, inst: EngineInstance, reason: str) -> None:
        if inst.recovery is not None or self.instances.get(inst.id) is not inst:
            return  # already handled (several requests can fail at once) or already unloaded
        model_id = inst.model_id
        if inst.pid:
            self.monitor.unwatch_process(inst.pid)
        if self._shutting_down or self.exiting():
            self._spawn(self._retire(inst))
            return
        if reason.startswith(GPU_DEVICE_LOST) and not self._tdr_hint_shown:
            tdr = self.monitor.gpu_timeout()
            if tdr_limited(tdr):
                self._tdr_hint_shown = True
                self.bus.activity_log(tdr_hint(tdr), level="warn", category="hardware")
        pending = self._pending.get(model_id)
        if pending is not None:
            # a load of this model is already queued (reload from the control panel): it replaces the engine
            inst.recovery = pending
            self._spawn(self._retire(inst))
            return
        now = time.time()
        recent = [t for t in self._recovery_log.get(model_id, []) if now - t < RECOVERY_WINDOW_S]
        if len(recent) >= RECOVERY_LIMIT:
            self._recovery_log[model_id] = recent
            self.bus.activity_log(
                f"{model_id} failed {len(recent) + 1} times within {RECOVERY_WINDOW_S / 60:.0f} minutes; automatic "
                "restart is disabled and the model was unloaded. It loads again with the next request that names it "
                "(just-in-time loading) or from the Library.", level="error", category="model", model=model_id)
            self._spawn(self._retire(inst))
            return
        recent.append(now)
        self._recovery_log[model_id] = recent
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        inst.recovery = fut
        self._pending[model_id] = fut  # requests for this model wait for the replacement instead of failing
        rec = _Recovery(old=inst, reason=reason, future=fut)
        self._recoveries[model_id] = rec
        self.bus.activity_log(f"Restarting {model_id} (automatic recovery {len(recent)} of {RECOVERY_LIMIT} "
                              f"within {RECOVERY_WINDOW_S / 60:.0f} min)", level="warn", category="model",
                              model=model_id)
        rec.task = self._spawn(self._recover(rec))

    async def _retire(self, inst: EngineInstance) -> None:
        """Stop a failed engine without replacing it."""
        try:
            await inst.stop(timeout=5)
        finally:
            self._drop(inst)

    async def _recover(self, rec: _Recovery) -> None:
        old = rec.old
        model_id = old.model_id
        overrides = old.params.model_dump(mode="json")
        used = [d.name for d in old.plan.devices]
        try:
            # Kill the failed engine first: it holds VRAM and the lost GPU context.
            await old.stop(timeout=5)
            last: Exception | None = None
            for attempt, delay in enumerate(RECOVERY_DELAYS_S, 1):
                await asyncio.sleep(delay)  # let the driver finish resetting the GPU
                self._check_recovery(model_id)
                await self._wait_for_devices(used)
                self._check_recovery(model_id)
                entry = self.library.get(model_id) or self.library.resolve(model_id)
                if entry is None:
                    raise ModelError(f"model '{model_id}' is no longer in the library", 404, "model_not_found")
                rec.loading = True
                self._drop(old)
                try:
                    new = await self._load_locked(entry, overrides, "recovery",
                                                  guard=lambda: self._check_recovery(model_id))
                except Exception as exc:
                    if isinstance(exc, ModelError) and exc.code in _RECOVERY_ABORTS:
                        raise
                    last = exc
                    rec.loading = False
                    if attempt < len(RECOVERY_DELAYS_S):
                        self.bus.activity_log(
                            f"{model_id}: restart attempt {attempt} failed ({exc}); trying again in "
                            f"{RECOVERY_DELAYS_S[attempt]:.0f} s", level="warn", category="model", model=model_id)
                    continue
                new.restarts = old.restarts + 1
                self.bus.activity_log(f"{model_id} recovered from: {rec.reason}", level="ok", category="model",
                                      model=model_id)
                rec.future.set_result(new)
                return
            assert last is not None
            raise last
        except asyncio.CancelledError:
            self._end_recovery(rec, ModelError(f"automatic restart of {model_id} was cancelled", 503,
                                               "model_unavailable"))
            self.bus.activity_log(f"Automatic restart of {model_id} cancelled", category="model", model=model_id)
            raise
        except Exception as exc:
            err = exc if isinstance(exc, ModelError) else ModelError(str(exc), 503, "model_unavailable")
            self._end_recovery(rec, err)
            if err.code in _RECOVERY_ABORTS:
                self.bus.activity_log(f"{model_id} was not restarted: {exc}", category="model", model=model_id)
            else:
                log.error("automatic restart of %s failed: %s", model_id, exc)
                self.bus.activity_log(f"{model_id}: automatic restart failed: {exc}. Load it again from the "
                                      "Library, or send a request that names it (just-in-time loading).",
                                      level="error", category="model", model=model_id)
        finally:
            if self._pending.get(model_id) is rec.future:
                del self._pending[model_id]
            if self._recoveries.get(model_id) is rec:
                del self._recoveries[model_id]

    def _end_recovery(self, rec: _Recovery, err: ModelError) -> None:
        if not rec.future.done():
            rec.future.set_exception(err)
            rec.future.exception()  # mark retrieved: nobody may be waiting
        self._drop(rec.old)

    def _check_recovery(self, model_id: str) -> None:
        """Raise when the model should no longer be restarted automatically."""
        if self._shutting_down or self.exiting():
            raise ModelError("WinRunner is shutting down", 503, "shutting_down")
        if self.store.settings.server.auto_evict:
            other = next((i.model_id for i in self.instances.values()
                          if i.model_id != model_id and i.state in ("ready", "loading", "starting")), None)
            other = other or next((m for m in self._pending if m != model_id), None)
            if other:
                raise ModelError(f"{other} has been loaded in the meantime", 409, "model_replaced")

    async def _wait_for_devices(self, names: list[str]) -> None:
        """Wait until the engine sees the GPUs again and their free memory has settled.

        After a GPU reset the driver needs a moment before the devices can be used again, and the
        memory of the killed engine can be released with a delay: loading too early fails, or
        places fewer layers on the GPUs than before.
        """
        if not names or await self.engine_async() is None:
            return
        deadline = time.monotonic() + DEVICE_WAIT_S
        prev: dict[str, int] | None = None
        announced = False
        while True:
            devs, _ = await self.devices(max_age=0)
            free = {d.name: d.free_mib for d in devs}
            missing = [n for n in names if n not in free]
            if not missing:
                if prev is not None and all(free[n] - prev.get(n, 0) < DEVICE_SETTLE_MIB for n in names):
                    return
                prev = free
            elif not announced:
                announced = True
                self.bus.activity_log(f"Waiting for {', '.join(missing)} to become available again", level="warn",
                                      category="hardware")
            if time.monotonic() >= deadline:
                if missing:
                    self.bus.activity_log(f"{', '.join(missing)} still not available; trying to load anyway",
                                          level="warn", category="hardware")
                return
            await asyncio.sleep(DEVICE_POLL_S)

    # ----- request routing ---------------------------------------------------------------

    async def instance_for_request(self, requested: str | None, jit: bool) -> EngineInstance:
        """Return a ready instance for a client-requested model name (JIT-loading if enabled).

        A name that is not in the library (clients often send fixed names such as "gpt-4o"), no
        name, or - with just-in-time loading off - a library model that is not loaded, is answered
        by the active model instead of failing (see :meth:`_substitute`).
        """
        entry = self.library.resolve(requested) if requested else None
        if entry is not None:
            inst = self.instance_for_model(entry.id)
            if inst is not None:
                return await self._await_ready(inst)
            if entry.id in self._pending:
                return await self._await_ready(await asyncio.shield(self._pending[entry.id]))
            if jit:
                return await self.load(entry.id, source="jit")
        active = self._active()
        if active is not None:
            self._note_substitute(requested, active)
            return active
        loading = [i for i in self.instances.values() if i.state in ("loading", "starting")]
        if loading:
            inst = await self._await_ready(loading[0])
        elif self._pending:  # e.g. the active model is being restarted after an engine failure
            inst = await self._await_ready(await asyncio.shield(next(iter(self._pending.values()))))
        elif jit and (default := self._default_model()):
            inst = await self.load(default, source="jit")
        else:
            raise self._no_model_error(jit)
        self._note_substitute(requested, inst)
        return inst

    def ready_for(self, requested: str | None, jit: bool) -> EngineInstance | None:
        """Non-blocking variant of :meth:`instance_for_request`.

        Returns a ready instance, ``None`` if one will become ready (load needed or
        in progress), or raises :class:`ModelError` if the request cannot be served.
        """
        entry = self.library.resolve(requested) if requested else None
        if entry is not None:
            inst = self.instance_for_model(entry.id)
            if inst is not None and inst.state == "ready":
                return inst
            if inst is not None or entry.id in self._pending or jit:
                return None
        active = self._active()
        if active is not None:
            self._note_substitute(requested, active)
            return active
        if any(i.state in ("loading", "starting") for i in self.instances.values()) or self._pending:
            return None
        if jit and self._default_model():
            return None
        raise self._no_model_error(jit)

    def substitute_model(self, jit: bool) -> str | None:
        """Id of the model that answers requests naming a model WinRunner does not know, if any."""
        active = self._active()
        if active is not None:
            return active.model_id
        loading = next((i for i in self.instances.values() if i.state in ("loading", "starting")), None)
        if loading is not None:
            return loading.model_id
        if self._pending:
            return next(iter(self._pending))
        return self._default_model() if jit else None

    def _active(self) -> EngineInstance | None:
        """The loaded model that answers requests without a (known) model name: the most recently used one."""
        ready = sorted(self.ready_instances(), key=lambda i: i.last_used, reverse=True)
        return ready[0] if ready else None

    def _default_model(self) -> str | None:
        """Model to load just in time when none is loaded: the one used last."""
        last = self.store.settings.startup.last_model
        if last and self.library.get(last):
            return last
        entries = self.library.entries()
        used = [(prof.last_loaded, e.id) for e in entries
                if (prof := self.store.settings.models.get(e.path)) and prof.last_loaded > 0]
        if used:
            return max(used)[1]
        llms = [e.id for e in entries if e.info.kind == "llm"]
        return llms[0] if len(llms) == 1 else None

    def _note_substitute(self, requested: str | None, inst: EngineInstance) -> None:
        """Tell the user once per name when requests for another model are answered by ``inst``."""
        if not requested or requested == inst.model_id:
            return
        entry = self.library.resolve(requested)
        if entry is not None and entry.id == inst.model_id:
            return  # an alias, file name or prefix of the model itself
        key = (requested, inst.model_id)
        if key in self._substitutes:
            return
        self._substitutes.add(key)
        why = ("is not loaded and just-in-time loading is off" if entry is not None
               else "is not a model in the library")
        self.bus.activity_log(f"'{requested}' {why}: requests for it are answered by {inst.model_id}",
                              category="request")

    def _no_model_error(self, jit: bool) -> ModelError:
        if not jit:
            return ModelError("No model is loaded and just-in-time loading is off. Load a model in WinRunner "
                              "(Library).", 503, "no_model_loaded")
        return ModelError("No model is loaded yet. Load a model in WinRunner (Library), or name one of its models "
                          "in the request (GET /v1/models lists them).", 503, "no_model_loaded")

    async def _drain(self, inst: EngineInstance, timeout: float = 300.0) -> None:
        """Wait for in-flight requests on an instance before it is evicted."""
        if inst.active_requests <= 0:
            return
        self.bus.activity_log(f"Waiting for {inst.active_requests} active request(s) on {inst.model_id} to finish "
                              "before switching models", level="warn", category="model")
        deadline = time.monotonic() + timeout
        while inst.active_requests > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.25)

    async def _await_ready(self, inst: EngineInstance) -> EngineInstance:
        deadline = time.monotonic() + self.store.settings.engine.load_timeout_s
        while inst.state in ("starting", "loading"):
            if time.monotonic() > deadline:
                raise ModelError("timed out waiting for model to load", 504, "load_timeout")
            await asyncio.sleep(0.2)
        if inst.state != "ready":
            raise ModelError(inst.error or f"model is {inst.state}", 503, "model_unavailable")
        return inst

    # ----- background loops ---------------------------------------------------------------

    async def _slots_loop(self) -> None:
        while True:
            busy = any(i.active_requests for i in self.instances.values())
            await asyncio.sleep(0.5 if busy else 2.0)
            if not self.ui_clients:
                continue
            for inst in self.ready_instances():
                slots = await inst.poll_slots()
                if slots:
                    self.bus.publish("slots", iid=inst.id, model=inst.model_id, slots=slots)

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            minutes = self.store.settings.server.idle_unload_minutes
            if minutes <= 0:
                continue
            cutoff = time.time() - minutes * 60
            for inst in list(self.ready_instances()):
                if inst.active_requests == 0 and inst.last_used < cutoff:
                    async with self._load_lock:
                        await self._unload(inst, reason=f"idle for {minutes} min")

    def status(self) -> dict[str, Any]:
        return {"instances": [i.status() for i in self.instances.values()],
                "pending": list(self._pending.keys())}


def _placement_summary(pl: Plan) -> str:
    """One line for the activity log: where the model goes, the context and the planned VRAM per GPU."""
    if not pl.devices:
        return f"Placement: CPU only, context {pl.ctx:,}"
    if pl.full_offload:
        where = "whole model in VRAM"
    elif pl.n_cpu_moe:
        where = f"expert weights of {pl.n_cpu_moe} layers in system RAM"
    elif pl.use_engine_fit:
        where = "part of the model in system RAM (placed by llama.cpp)"
    else:
        where = f"{pl.n_layer + 1 - pl.gpu_layers} of {pl.n_layer + 1} layers in system RAM"
    ctx = f"context {pl.ctx:,}"
    if pl.ctx_adjusted == "raised":
        ctx += f" (raised from {pl.ctx_target:,} to fill VRAM)"
    elif pl.ctx_adjusted == "reduced":
        ctx += f" (reduced from {pl.ctx_target:,} to keep the model in VRAM)"
    vram = ", ".join(f"{d.name} {(d.total_mib - d.free_mib + d.used_mib) / 1024:.1f}/{d.total_mib / 1024:.1f} GiB"
                     for d in pl.devices)
    src = "measured by the engine" if pl.source == "engine" else "estimate"
    return f"Placement ({src}): {where}, KV {pl.kv_k.upper()}, {ctx}; VRAM {vram}"


def _run_capture(args: list[str], timeout: float) -> tuple[int, str]:
    try:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                           timeout=timeout, cwd=os.path.dirname(args[0]) or None, creationflags=_CREATE_NO_WINDOW)
        out = r.stdout.decode("utf-8", errors="replace")
        if r.returncode != 0:
            out += "\n" + r.stderr.decode("utf-8", errors="replace")[-2000:]
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "timed out"
    except OSError as exc:
        return -2, str(exc)
