"""Model lifecycle orchestration: planning, loading, JIT loading, eviction, recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .cmdline import build_fit_args, build_server_args
from .config import LoadParams, SettingsStore
from .engine import EngineDevice, EngineInfo, EngineManager, engine_env
from .events import EventBus, RequestTracker
from .faults import GPU_DEVICE_LOST, gpu_fault, tdr_hint, tdr_limited
from .hardware import HardwareMonitor
from .instance import EngineInstance
from .probe import as_measurement, server_probe
from .library import ModelEntry, ModelLibrary
from .paths import IS_WINDOWS, DataPaths
from .planner import (Plan, Planner, apply_measurement, device_targets, fit_to_engine, measured_used, parse_fit_args,
                      parse_fit_print)
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
RELEASE_WAIT_S = 8.0  # longest wait for an unloaded model's VRAM to be released before planning the next load
MEASURE_TTL_S = 900.0  # engine projections (llama-fit-params) do not depend on free memory: cache them

SOURCE_LABELS = {"jit": "JIT request", "ui": "control panel", "startup": "startup", "recovery": "automatic restart"}
_RECOVERY_ABORTS = ("shutting_down", "model_replaced")  # reasons to stop a recovery at once (no retries)


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
        self._device_pci: dict[str, str] = {}  # engine device -> PCI address, from the engine's load log
        self._vram_penalty: dict[str, int] = {}  # MiB the OS did not keep resident at the last load, per device
        self._substitutes: set[tuple[str, str]] = set()  # (requested name, model answering) already reported
        self._tdr_hint_shown = False
        self._measure_cache: dict[str, tuple[float, dict]] = {}
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
        return self.monitor.map_engine_devices([{"name": d.name, "description": d.description,
                                                 "total_mib": d.total_mib, "free_mib": d.free_mib,
                                                 "pci": self._device_pci.get(d.name)} for d in devices])

    async def _physical_vram(self, devices: list[EngineDevice], fresh: bool) -> tuple[list[EngineDevice], dict[str, int]]:
        """Windows: free VRAM of each GPU as the hardware has it, not as the Vulkan driver reports it.

        On Windows the driver reports free memory = the per-process budget Windows grants (WDDM) minus usage. That
        budget keeps roughly 0.7-1.2 GB of a 16 GB card unused, so a plan based on it stops at ~14.5 GB. The GPU
        performance counters (Task Manager's dedicated memory) give the memory that no process uses. Returns the
        adjusted devices and the MiB gained per device.
        """
        if not (IS_WINDOWS and self.store.settings.hardware.use_physical_vram and devices):
            return devices, {}
        sample = await asyncio.to_thread(self.monitor.wait_sample, time.time()) if fresh else self.monitor.last
        phys = self.monitor.physical_free_mib(sample)
        if not phys:
            return devices, {}
        dmap = self.device_map(devices)
        out, gained = [], {}
        for d in devices:
            pf = phys.get(dmap.get(d.name, ""))
            if pf is not None:
                pf -= self._vram_penalty.get(d.name, 0)
            if pf is not None and d.free_mib + 64 < pf <= d.total_mib:
                gained[d.name] = int(pf) - d.free_mib
                d = replace(d, free_mib=int(pf))
            out.append(d)
        return out, gained

    def _check_residency(self, inst: EngineInstance) -> None:
        """After a load: warn if the OS keeps part of the engine's GPU buffers in system memory (very slow)."""
        async def check() -> None:
            vram: dict[str, float] = {}
            for _ in range(12):  # the GPU counters list a new process within ~30 s
                await asyncio.sleep(4.0)
                if inst.state != "ready" or not inst.pid:
                    return
                sample = await asyncio.to_thread(self.monitor.wait_sample, time.time())
                vram = (sample.get("procs", {}).get(str(inst.pid)) or {}).get("vram") or {}
                if vram:
                    break
            if not vram:
                return
            dmap = inst.device_map or {}
            for name, bufs in (inst.load_info.get("buffers") or {}).items():
                gid = dmap.get(name)
                if gid not in vram:
                    continue
                expected = sum(float(v) for v in bufs.values())
                resident = float(vram[gid]) / (1024 * 1024)
                if expected - resident > 512:
                    if IS_WINDOWS and self.store.settings.hardware.use_physical_vram:
                        missing = int(expected - resident) + 256
                        self._vram_penalty[name] = max(self._vram_penalty.get(name, 0), missing)
                        hint = (f" The next load of a model leaves {missing:,} MiB more free on {name}; reload the "
                                "model to apply it.")
                    else:
                        hint = " Raise the safety margin for this GPU (Settings > Hardware)."
                    self.bus.activity_log(
                        f"{inst.model_id}: only {resident:,.0f} of {expected:,.0f} MiB of the engine's buffers are "
                        f"in {name}'s VRAM; the rest was moved to system memory, which makes the model slow.{hint}",
                        level="warn", category="model", model=inst.model_id)
                    log.warning("%s: %s holds %.0f of %.0f MiB in VRAM", inst.model_id, name, resident, expected)

        self._spawn(check())

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
                   assume_evict: bool = True, fresh_devices: bool = False) -> tuple[Plan, dict[str, Any]]:
        """Memory plan for loading ``model_id``.

        ``verify``: measure the placement with the engine's own allocator (llama-fit-params) and correct it
        until every GPU is filled to its free memory minus the margin. Loads always verify.
        """
        entry = self.library.get(model_id) or self.library.resolve(model_id)
        if entry is None:
            raise ModelError(f"model '{model_id}' not found in library", 404, "model_not_found")
        eng = await self.engine_async()
        p = self.store.effective_load_params(entry.path, overrides)
        devices, dev_err = await self.devices(max_age=0 if fresh_devices else 3.0)
        devices, vram_gained = await self._physical_vram(devices, fresh_devices)
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
        es = self.store.settings.engine
        hw = self.store.settings.hardware
        engine_places = bool(eng and es.placement == "engine" and eng.has("--fit", "-fit"))
        planner = Planner(
            entry.info, p, devices, self.device_map(devices),
            margin_mib=hw.vram_margin_mib,
            margin_per_device=hw.vram_margin_per_device,
            reclaim_mib=self._reclaim(evicting),
            mmproj_size=mm_size, mmproj_mib_hint=mm_hint,
            draft_info=draft.info if draft else None,
            engine_fit=engine_places,
            model_id=entry.id,
            overrides_supported=bool(eng is None or eng.has("--override-tensor", "-ot")),
        )
        sel = [d.name for d in devices if not p.devices or d.name in p.devices]
        margins = self._margins(devices, p)
        can_measure = bool(eng and eng.fit_params and devices and es.use_engine_fit)
        pl: Plan | None = None
        if verify and can_measure and p.gpu_offload == "auto" and not engine_places:
            measure = self._measurer(eng, entry, p, sel, planner, mmproj, draft, margins)
            try:
                mm_before = planner.mmproj_mib
                pl = await asyncio.to_thread(fit_to_engine, planner, measure)
                if abs(planner.mmproj_mib - mm_before) > 32:
                    # the first probe measured the vision projector: fit again with its real size
                    pl = await asyncio.to_thread(fit_to_engine, planner, measure)
            except Exception as exc:  # the projection is advisory: fall back to the estimate
                log.warning("engine memory projection failed for %s: %s", entry.id, exc)
                pl = planner.plan()
                pl.warnings.append(f"Engine memory projection failed ({exc}); using the estimate.")
        if pl is None:
            pl = planner.plan()
            if verify and can_measure:
                try:
                    if engine_places:
                        await self._engine_fit_preview(eng, entry, p, pl, sel, margins)
                    else:  # manual mode: show what the engine projects for the user's settings
                        m = await asyncio.to_thread(self._measurer(eng, entry, p, sel, None, mmproj, draft, margins), pl)
                        apply_measurement(pl, m)
                        over = [d.name for d in pl.devices
                                if measured_used(m.get(d.name, {})) > device_targets(pl)[d.name] + 0.5]
                        if over:
                            pl.warnings.append(f"Engine projection: exceeds free VRAM minus the margin on {', '.join(over)}.")
                except Exception as exc:
                    pl.warnings.append(f"Engine memory projection failed: {exc}")
        pl.mmproj = mmproj or ""
        pl.draft = draft.path if draft else ""
        if evicting:
            pl.notes.append(f"Planned for the free VRAM after {', '.join(i.model_id for i in evicting)} is unloaded.")
        if vram_gained:
            pl.notes.append(
                "Planned with the physical free VRAM (as Task Manager shows it): Windows reports "
                + ", ".join(f"{n} {v:,} MiB" for n, v in sorted(vram_gained.items()))
                + " less to Vulkan (its per-process budget).")
        if verify and not can_measure and eng and not eng.fit_params:
            pl.notes.append("This engine build does not include llama-fit-params; the plan is an estimate.")
        if mmproj and p.mmproj_offload and not mm_hint:
            pl.notes.append("Vision projector memory is estimated from its file size; the exact figure is learned "
                            "at the first load and used from then on.")
        extra = {
            "entry": entry.to_summary(),
            "devices": [d.__dict__ for d in devices],
            "device_error": dev_err,
            "engine": eng.to_dict() if eng else None,
            "evicting": [i.model_id for i in evicting],
            "margins": margins,
        }
        if eng:
            spec = build_server_args(
                eng, entry.path, p, pl, sel, 0, entry.id, "", mmproj, draft.path if draft else None,
                self._template_file(entry, p), entry.info.kind == "embedding",
                self.store.settings.engine.log_verbosity, margins,
            )
            args = [a for a in spec.args]
            i = args.index("--port")
            args[i + 1] = "<port>"
            extra["command"] = " ".join(_quote(a) for a in args)
        return pl, extra

    def _measurer(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, sel: list[str],
                  planner: Planner | None = None, mmproj: str | None = None, draft: ModelEntry | None = None,
                  margins: list[int] | None = None):
        """Measure a plan's memory with the engine itself (cached: projections do not depend on free memory).

        Preferred: a llama-server probe with the exact launch arguments (see probe.py). Fallback:
        llama-fit-params --fit-print, which overstates the output device's compute buffer.
        """
        use_server = eng.has("--fit", "-fit") and eng.has("--log-verbosity", "-lv")

        def measure(pl: Plan) -> dict[str, dict[str, float]]:
            if use_server:
                port = free_port()
                spec = build_server_args(
                    eng, entry.path, p, _explicit_copy(pl), sel, port, entry.id, new_id("", 16), mmproj,
                    draft.path if draft else None, self._template_file(entry, p), entry.info.kind == "embedding",
                    max(4, self.store.settings.engine.log_verbosity), margins or [])
                args = _with_fit_on(spec.args, eng, margins or [])
                key_args = [a for a in args if a not in (str(port),)]
                key_args = [a for i, a in enumerate(key_args) if i == 0 or key_args[i - 1] != "--api-key"]
            else:
                args = build_fit_args(eng, entry.path, p, pl, sel, [], print_mode=True)
                if not args:
                    raise RuntimeError("the engine build has no llama-fit-params")
                key_args = args
            key = hashlib.sha1(json.dumps([key_args, entry.info.mtime, entry.info.file_size, eng.path,
                                           eng.build]).encode()).hexdigest()
            now = time.time()
            hit = self._measure_cache.get(key)
            if hit and now - hit[0] < MEASURE_TTL_S:
                res = hit[1]
            elif use_server:
                pr = server_probe(args, single_device=sel[0] if len(sel) == 1 else None)
                res = {"devices": as_measurement(pr, pl), "mmproj": pr.mmproj_mib}
            else:
                rc, out = _run_capture(args, 240)
                proj = parse_fit_print(out) if rc == 0 else {}
                if not proj:
                    tail = out.strip().splitlines()[-1] if out.strip() else ""
                    raise RuntimeError(f"llama-fit-params failed ({rc}): {tail[:300]}")
                res = {"devices": proj, "mmproj": None}
            if len(self._measure_cache) > 64:
                self._measure_cache.clear()
            self._measure_cache[key] = (now, res)
            learned = res.get("mmproj")
            if learned and planner is not None and p.mmproj_offload:
                planner.mmproj_mib = float(learned)
                prof = self.store.profile(entry.path)
                if abs(prof.mmproj_est_mib - float(learned)) > 1:
                    prof.mmproj_est_mib = float(learned)
                    self.store.set_profile(entry.path, prof)
            return res["devices"]

        return measure

    async def _engine_fit_preview(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan,
                                  sel: list[str], margins: list[int]) -> None:
        """Placement "engine": show what llama.cpp's --fit will choose (it decides again at load)."""
        args = build_fit_args(eng, entry.path, p, pl, sel, margins, print_mode=False)
        if not args:
            return
        rc, out = await asyncio.to_thread(_run_capture, args, 240)
        fit = parse_fit_args(out)
        if rc != 0 or not fit:
            raise RuntimeError(f"llama-fit-params failed ({rc}): {out.strip().splitlines()[-1] if out.strip() else ''}")
        ngl = fit.get("ngl", -1)
        pl.gpu_layers = entry.info.n_layer + 1 if ngl < 0 else ngl
        pl.tensor_split = fit.get("tensor_split") or pl.tensor_split
        pl.overrides = [x for x in (fit.get("override_tensor") or "").split(",") if x]
        pl.engine = {"fit": fit}
        pl.source = "engine"
        pl.full_offload = pl.gpu_layers > entry.info.n_layer and not pl.overrides
        m = await asyncio.to_thread(self._measurer(eng, entry, p, sel, None, pl.mmproj or None, None, margins),
                                    _explicit_copy(pl))
        apply_measurement(pl, m)

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
            released: list[str] = []
            if existing:
                released += [d.name for d in existing.plan.devices]
                await self._unload(existing, reason="reload")
            for inst in self._evict_set(entry.id):
                await self._drain(inst)
                released += [d.name for d in inst.plan.devices]
                await self._unload(inst, reason=f"evicted for {entry.id}")
            self.bus.activity_log(f"Loading {entry.id} ({SOURCE_LABELS.get(source, source)})",
                                  category="model", model=entry.id)
            if released:
                # the unloaded engine's VRAM is freed asynchronously by the driver: plan with the real free memory
                self.bus.publish("load_stage", model=entry.id, stage="planning", label="Waiting for VRAM release")
                await self._wait_for_release(sorted(set(released)))
            self.bus.publish("load_stage", model=entry.id, stage="planning", label="Planning memory layout")
            pl, extra = await self.plan(entry.id, overrides, verify=True, assume_evict=False, fresh_devices=True)
            p = self.store.effective_load_params(entry.path, overrides)
            devices, _ = await self.devices(max_age=60.0)
            sel = [d.name for d in devices if not p.devices or d.name in p.devices]
            mmproj = pl.mmproj or None
            draft = self._draft_for(p)
            self.bus.activity_log(f"{entry.id}: {describe_plan(pl)}", category="model", model=entry.id)
            for w in pl.warnings:
                self.bus.activity_log(f"{entry.id}: {w}", level="warn", category="model", model=entry.id)
            if self._shutting_down:  # planning can take a while; never start an engine during shutdown
                raise ModelError("WinRunner is shutting down", 503, "shutting_down")
            port = free_port()
            api_key = new_id("", 16)
            spec = build_server_args(
                eng, entry.path, p, pl, sel, port, entry.id, api_key, mmproj, draft.path if draft else None,
                self._template_file(entry, p), entry.info.kind == "embedding",
                self.store.settings.engine.log_verbosity, self._margins(devices, p),
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
                self._check_residency(inst)
            self._device_pci.update(inst.load_info.get("device_pci") or {})
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

    async def _wait_for_release(self, names: list[str]) -> None:
        """After unloading: wait until the GPUs' free memory stops growing (the driver frees VRAM with a delay)."""
        if not names or await self.engine_async() is None:
            return
        deadline = time.monotonic() + RELEASE_WAIT_S
        prev: dict[str, int] | None = None
        while True:
            devs, _ = await self.devices(max_age=0)
            free = {d.name: d.free_mib for d in devs if d.name in names}
            if prev is not None and all(free.get(n, 0) - prev.get(n, 0) < 64 for n in free):
                return
            prev = free
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(0.3)

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


def describe_plan(pl: Plan) -> str:
    """One line for the activity log: where the model goes."""
    gpus = [d for d in pl.devices if d.layers or d.output_mib]
    vram = " + ".join(f"{d.name} {d.used_mib / 1024:.1f}/{d.total_mib / 1024:.1f} GiB" for d in gpus)
    ctx = f"context {pl.ctx:,}, KV {pl.kv_k.upper()}"
    src = " (engine-measured)" if pl.source == "engine" else ""
    if pl.strategy == "cpu":
        return f"CPU only, {ctx}"
    if pl.strategy == "full":
        return f"full GPU offload{src}: {vram}, {ctx}"
    if pl.strategy == "attention_first":
        n = sum(1 for x in pl.ram_parts if x)
        return (f"GPUs first{src}: {vram}; attention + KV cache of all layers on GPU, feed-forward weights of "
                f"{n} layer(s) ({pl.totals.get('ram_parts_mib', 0) / 1024:.1f} GiB) in system RAM, {ctx}")
    if pl.strategy == "layers":
        cpu = sum(1 for x in pl.layer_home if x < 0)
        return f"GPUs first{src}: {vram}; {cpu} leading layer(s) on the CPU, {ctx}"
    if pl.strategy == "engine":
        return f"llama.cpp --fit places the layers at load, {ctx}"
    return f"manual: {pl.gpu_layers}/{pl.n_layer + 1} layers on GPU, {ctx}"


def _quote(a: str) -> str:
    import shlex

    return a if IS_WINDOWS else shlex.quote(a)


def _with_fit_on(args: list[str], eng: EngineInfo, margins: list[int]) -> list[str]:
    """Launch arguments with llama.cpp's --fit on: it then prints its projection of the placement first."""
    out = list(args)
    if "--fit" in out:
        i = out.index("--fit")
        out[i + 1] = "on"
    else:
        out += ["--fit", "on"]
    if margins and eng.has("--fit-target", "-fitt") and "--fit-target" not in out:
        out += ["--fit-target", ",".join(str(int(m)) for m in margins)]
    return out


def _explicit_copy(pl: Plan) -> Plan:
    import copy

    q = copy.copy(pl)
    q.use_engine_fit = False
    return q


def _run_capture(args: list[str], timeout: float) -> tuple[int, str]:
    try:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                           timeout=timeout, cwd=os.path.dirname(args[0]) or None, env=engine_env(),
                           creationflags=_CREATE_NO_WINDOW)
        out = r.stdout.decode("utf-8", errors="replace")
        if r.returncode != 0:
            out += "\n" + r.stderr.decode("utf-8", errors="replace")[-2000:]
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "timed out"
    except OSError as exc:
        return -2, str(exc)
