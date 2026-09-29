"""Model lifecycle orchestration: planning, loading, JIT loading, eviction, recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .cmdline import build_fit_args, build_server_args
from .config import QUANTIZED_KV, LoadParams, SettingsStore
from .engine import EngineDevice, EngineInfo, EngineManager
from .events import EventBus, RequestTracker
from .hardware import HardwareMonitor
from .instance import EngineInstance, is_host_buffer
from .library import ModelEntry, ModelLibrary
from .paths import IS_WINDOWS, DataPaths
from .planner import (AUTO_UBATCH, MIN_FIT_CTX, PLACEMENT_PREFIXES, Plan, Planner, floor_ctx, msg_ctx_reduced,
                      msg_kv_q8, msg_manual_over, msg_no_fit_min, msg_partial_experts, msg_partial_layers, msg_ubatch,
                      override_layers, parse_fit_args, parse_fit_print)
from .util import free_port, new_id

log = logging.getLogger("winrunner.manager")

_CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0
MiB = 1024 * 1024
PLAN_CACHE_TTL = 1800.0  # engine projections stay valid while free VRAM (256 MiB buckets) is unchanged


class ModelError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "model_error"):
        super().__init__(message)
        self.status = status
        self.code = code


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
        self._restart_log: list[float] = []
        self._plan_cache: dict[str, tuple[float, dict]] = {}
        self._device_pci: dict[str, dict[str, str]] = {}  # engine path -> device name -> PCI address (from load logs)
        self._spill: dict[str, dict[str, Any]] = {}  # instance id -> VRAM residency state
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
        devs, err = await asyncio.to_thread(self.engines.list_devices, Path(eng.path), max_age)
        pci = self._device_pci.get(eng.path, {})
        for d in devs:
            d.pci = pci.get(d.name, d.pci)
        return devs, err

    @staticmethod
    def _dev_dicts(devices: list[EngineDevice]) -> list[dict[str, Any]]:
        return [{"name": d.name, "description": d.description, "pci": d.pci, "free_mib": d.free_mib,
                 "total_mib": d.total_mib} for d in devices]

    def device_map(self, devices: list[EngineDevice]) -> dict[str, str]:
        return self.monitor.map_engine_devices(self._dev_dicts(devices))

    def device_roles(self, devices: list[EngineDevice]) -> dict[str, str]:
        """display | idle | busy | unknown per engine device (decides the VRAM safety margin)."""
        return self.monitor.device_roles(self._dev_dicts(devices))

    # ----- lifecycle helpers ------------------------------------------------------------

    def start_background(self) -> None:
        self._tasks.append(asyncio.create_task(self._slots_loop()))
        self._tasks.append(asyncio.create_task(self._idle_loop()))

    async def shutdown(self) -> None:
        self._shutting_down = True
        for t in self._tasks:
            t.cancel()
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

    def _margin_map(self, devices: list[EngineDevice]) -> dict[str, int]:
        """VRAM left free per device: the full margin on GPUs that drive a display or run other
        applications (their usage can grow at any time), a smaller one on GPUs only WinRunner uses."""
        hw = self.store.settings.hardware
        roles = self.device_roles(devices)
        out: dict[str, int] = {}
        for d in devices:
            if d.name in hw.vram_margin_per_device:
                out[d.name] = int(hw.vram_margin_per_device[d.name])
            elif roles.get(d.name) == "idle":
                out[d.name] = int(hw.vram_margin_idle_mib)
            else:
                out[d.name] = int(hw.vram_margin_mib)
        return out

    def _margins(self, devices: list[EngineDevice], p: LoadParams) -> list[int]:
        mm = self._margin_map(devices)
        return [mm[d.name] for d in devices if not p.devices or d.name in p.devices]

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
                   assume_evict: bool = True, projection: bool = True) -> tuple[Plan, dict[str, Any]]:
        entry = self.library.get(model_id) or self.library.resolve(model_id)
        if entry is None:
            raise ModelError(f"model '{model_id}' not found in library", 404, "model_not_found")
        eng = await self.engine_async()
        p = self.store.effective_load_params(entry.path, overrides)
        devices, dev_err = await self.devices(max_age=3.0)
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
        planner = Planner(
            entry.info, p, devices, self.device_map(devices),
            margin_mib=self.store.settings.hardware.vram_margin_mib,
            margin_per_device=self._margin_map(devices),
            reclaim_mib=self._reclaim(evicting),
            mmproj_size=mm_size, mmproj_mib_hint=mm_hint,
            draft_info=draft.info if draft else None,
            engine_fit=bool(eng and self.store.settings.engine.use_engine_fit and eng.has("--fit", "-fit")),
            model_id=entry.id,
        )
        pl = planner.plan()
        pl.mmproj = mmproj or ""
        pl.draft = draft.path if draft else ""
        if pl.use_engine_fit and evicting:
            # the engine measures free memory itself; the evicted model is still resident during planning
            pl.notes.append("Memory projection accounts for the currently loaded model being unloaded first.")
        extra = {
            "entry": entry.to_summary(),
            "devices": [d.__dict__ for d in devices],
            "device_roles": self.device_roles(devices),
            "device_error": dev_err,
            "engine": eng.to_dict() if eng else None,
            "evicting": [i.model_id for i in evicting],
            "margins": self._margins(devices, p),
        }
        if verify and eng and eng.fit_params and devices and not evicting:
            try:
                await self._verify_with_engine(eng, entry, p, pl, devices, projection=projection)
            except Exception as exc:  # projection is advisory; never block a load on it
                pl.warnings.append(f"Engine memory projection failed: {exc}")
        elif verify and evicting:
            pl.notes.append("Engine projection skipped: the engine measures free VRAM directly, and "
                            f"{', '.join(i.model_id for i in evicting)} is still loaded. It runs automatically at load time, "
                            "after the current model is unloaded.")
        elif verify and eng and not eng.fit_params:
            pl.notes.append("This engine build does not include llama-fit-params; the analytic estimate is used.")
        if eng:
            sel = [d.name for d in devices if not p.devices or d.name in p.devices]
            spec = build_server_args(
                eng, entry.path, p, pl, sel, 0, entry.id, "", mmproj, draft.path if draft else None,
                self._template_file(entry, p), entry.info.kind == "embedding",
                self.store.settings.engine.log_verbosity, self._margins(devices, p),
            )
            args = [a for a in spec.args]
            i = args.index("--port")
            args[i + 1] = "<port>"
            extra["command"] = " ".join(args)
        return pl, extra

    async def _verify_with_engine(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan,
                                  devices: list[EngineDevice], projection: bool = True) -> None:
        """Check (and correct) the plan with llama.cpp's own memory model (llama-fit-params).

        Automatic allocation: F16 KV at the requested context, then Q8_0; if the model
        still does not fit entirely in VRAM and the overflow policy is "reduce context",
        the engine is asked for the largest context at which everything fits. Manual
        allocation: the engine projects the exact configuration; with "reduce context"
        the context is shrunk until that projection fits.
        """
        sel = [d.name for d in devices if not p.devices or d.name in p.devices]
        margins = self._margins(devices, p)
        if pl.mmproj and margins and p.mmproj_offload:
            margins = [margins[0] + int(pl.totals.get("mmproj_mib", 0))] + margins[1:]
        ctx_req = pl.ctx_reduced_from or pl.ctx
        key_src = json.dumps([entry.path, entry.info.mtime, p.model_dump(mode="json"), ctx_req, pl.mode, margins,
                              [(d.name, d.free_mib // 256) for d in devices], eng.path, projection], sort_keys=True)
        key = hashlib.sha1(key_src.encode()).hexdigest()
        cached = self._plan_cache.get(key)
        if cached and time.time() - cached[0] < PLAN_CACHE_TTL:
            self._apply_projection(pl, cached[1], entry, p, devices)
            return
        n_par = (p.parallel if p.parallel > 0 else 4) if entry.info.sliding_window else 0
        if pl.mode == "manual":
            chosen = await self._project_manual(eng, entry, p, pl, sel, margins, n_par, devices, ctx_req)
        else:
            chosen = await self._fit_auto(eng, entry, p, pl, sel, margins, n_par, ctx_req)
            if chosen is not None and projection:
                chosen["projection"] = await self._fit_projection(eng, entry, p, pl, sel, n_par, chosen)
        if chosen is None:
            return
        self._plan_cache[key] = (time.time(), chosen)
        self._apply_projection(pl, chosen, entry, p, devices)

    async def _fit_auto(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan, sel: list[str],
                        margins: list[int], n_par: int, ctx_req: int) -> dict[str, Any] | None:
        info = entry.info
        if p.kv_cache_type == "auto":
            kvs = ["f16", "q8_0"] if p.flash_attn != "off" else ["f16"]
        else:
            kvs = [p.kv_cache_type]

        async def run(kv: str, ctx: int | None, ub: int) -> dict[str, Any]:
            q = Plan(**{**pl.__dict__})
            q.kv_k, q.kv_v = kv, p.kv_cache_type_v or kv
            quant = q.kv_k in QUANTIZED_KV or q.kv_v in QUANTIZED_KV
            q.flash_attn = "on" if quant and p.flash_attn == "auto" else pl.flash_attn if quant else p.flash_attn
            q.ctx = ctx or ctx_req
            q.ubatch, q.batch = ub, max(p.batch_size, ub)
            q.use_engine_fit = True
            args = build_fit_args(eng, entry.path, p, q, sel, margins, print_mode=False, auto_ctx=ctx is None,
                                  n_parallel=n_par, min_ctx=MIN_FIT_CTX)
            if not args:
                raise RuntimeError("llama-fit-params is not available")
            rc, out = await asyncio.to_thread(_run_capture, args, 180)
            fit = parse_fit_args(out)
            if rc != 0 or not fit:
                tail = out.strip().splitlines()[-1] if out.strip() else ""
                raise RuntimeError(f"llama-fit-params failed ({rc}): {tail}")
            ngl = fit.get("ngl", -1)
            full = (ngl < 0 or ngl > info.n_layer) and "override_tensor" not in fit
            got = q.ctx if ctx else min(ctx_req, int(fit.get("ctx") or ctx_req))
            return {"kv_k": q.kv_k, "kv_v": q.kv_v, "flash_attn": q.flash_attn, "fit": fit, "full": full,
                    "ctx": got, "ctx_requested": ctx_req, "ubatch": q.ubatch, "batch": q.batch}

        reduce = p.vram_overflow == "reduce_context"
        boost = p.auto_batch and p.ubatch_size < AUTO_UBATCH
        est = pl.max_ctx_full_offload or {}

        def hopeless(kv: str) -> bool:
            # each engine run costs seconds; skip configurations the estimate misses by a wide margin
            return kv in est and est[kv] < 0.85 * ctx_req

        chosen: dict[str, Any] | None = None
        for i, kv in enumerate(kvs):
            last = i == len(kvs) - 1
            if hopeless(kv) and (not last or reduce or boost):
                continue
            chosen = await run(kv, ctx_req, p.ubatch_size)
            if chosen["full"]:
                return chosen
        if reduce:
            r = await run(kvs[-1], None, p.ubatch_size)
            if r["full"] and r["ctx"] >= min(MIN_FIT_CTX, ctx_req):
                r["ctx"] = min(ctx_req, floor_ctx(r["ctx"]))
                return r
        if boost:
            # partial offload: fit with the larger micro-batch the server will be started with
            return await run(kvs[-1], ctx_req, AUTO_UBATCH)
        return chosen or await run(kvs[-1], ctx_req, p.ubatch_size)

    async def _fit_projection(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan, sel: list[str],
                              n_par: int, chosen: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Per-device memory the engine projects for the chosen configuration (llama-fit-params --fit-print)."""
        q = Plan(**{**pl.__dict__})
        q.kv_k, q.kv_v, q.flash_attn = chosen["kv_k"], chosen["kv_v"], chosen["flash_attn"]
        q.ctx, q.ubatch, q.batch = chosen["ctx"], chosen["ubatch"], chosen["batch"]
        fit = chosen.get("fit") or {}
        ngl = fit.get("ngl", -1)
        q.gpu_layers = entry.info.n_layer + 1 if ngl < 0 else ngl
        q.tensor_split = fit.get("tensor_split") or pl.tensor_split
        q.n_cpu_moe = 0
        q.use_engine_fit = False
        pargs = build_fit_args(eng, entry.path, p, q, sel, [], print_mode=True, n_parallel=n_par)
        if not pargs:
            return {}
        if fit.get("override_tensor"):
            pargs += ["-ot", fit["override_tensor"]]
        rc, out = await asyncio.to_thread(_run_capture, pargs, 120)
        return parse_fit_print(out) if rc == 0 else {}

    async def _project_manual(self, eng: EngineInfo, entry: ModelEntry, p: LoadParams, pl: Plan, sel: list[str],
                              margins: list[int], n_par: int, devices: list[EngineDevice],
                              ctx_req: int) -> dict[str, Any] | None:
        """Engine projection of an explicit (manual) layout; shrinks the context if it overflows."""
        free = {d.name: d.free_mib for d in devices}
        marg = dict(zip(sel, margins))

        async def project(ctx: int) -> dict[str, dict[str, float]]:
            q = Plan(**{**pl.__dict__})
            q.ctx = ctx
            q.use_engine_fit = False
            pargs = build_fit_args(eng, entry.path, p, q, sel, [], print_mode=True, n_parallel=n_par)
            if not pargs:
                return {}
            rc, out = await asyncio.to_thread(_run_capture, pargs, 120)
            return parse_fit_print(out) if rc == 0 else {}

        def overflow(proj: dict[str, dict[str, float]]) -> dict[str, float]:
            return {n: v["model"] + v["context"] + v["compute"] - (free.get(n, 0) - marg.get(n, 0))
                    for n, v in proj.items() if n in free}

        ctx = pl.ctx
        proj = await project(ctx)
        if not proj:
            return None
        over = overflow(proj)
        if any(x > 0 for x in over.values()) and p.vram_overflow == "reduce_context":
            # the KV cache and attention buffers scale with the context: shrink it by the worst overflow ratio
            ratios = [1 - x / proj[n]["context"] for n, x in over.items() if x > 0 and proj[n]["context"] > 0]
            if ratios and min(ratios) > 0:
                new = floor_ctx(int(ctx * min(ratios) * 0.97))
                if new >= min(MIN_FIT_CTX, ctx_req) and new < ctx:
                    proj2 = await project(new)
                    if proj2 and all(x <= 0 for x in overflow(proj2).values()):
                        ctx, proj = new, proj2
        return {"kv_k": pl.kv_k, "kv_v": pl.kv_v, "flash_attn": pl.flash_attn, "fit": {}, "full": pl.full_offload,
                "ctx": ctx, "ctx_requested": ctx_req, "ubatch": pl.ubatch, "batch": pl.batch, "projection": proj,
                "manual": True}

    @staticmethod
    def _apply_projection(pl: Plan, chosen: dict[str, Any], entry: ModelEntry, p: LoadParams,
                          devices: list[EngineDevice]) -> None:
        info = entry.info
        reduce = p.vram_overflow == "reduce_context"
        manual = bool(chosen.get("manual"))
        pl.source = "engine"
        pl.kv_k, pl.kv_v = chosen["kv_k"], chosen["kv_v"]
        pl.flash_attn = chosen["flash_attn"]
        req = int(chosen.get("ctx_requested") or pl.ctx)
        pl.ctx = int(chosen["ctx"])
        pl.ctx_reduced_from = req if pl.ctx < req else 0
        pl.ubatch, pl.batch = int(chosen["ubatch"]), int(chosen["batch"])
        fit = chosen.get("fit") or {}
        proj = chosen.get("projection") or {}
        pl.engine = {"fit": fit, "projection": proj, "full_offload": chosen["full"]}
        pl.warnings = [w for w in pl.warnings if not w.startswith(PLACEMENT_PREFIXES + ("Manual allocation",))]
        pl.notes = [n for n in pl.notes if not n.startswith(PLACEMENT_PREFIXES)]
        if pl.ctx_reduced_from:
            pl.warnings.insert(0, msg_ctx_reduced(req, pl.ctx))
        if not manual:
            pl.full_offload = bool(chosen["full"])
            ngl = int(fit.get("ngl", -1))
            pl.gpu_layers = info.n_layer + 1 if ngl < 0 else min(ngl, info.n_layer + 1)
            ts = fit.get("tensor_split")
            if ts and len(ts) == len(pl.devices):
                pl.tensor_split = [float(x) for x in ts]
                for i, (d, n) in enumerate(zip(pl.devices, ts)):
                    d.layers = int(n) - (1 if i == len(ts) - 1 and pl.gpu_layers > info.n_layer else 0)
            elif len(pl.devices) == 1:
                pl.devices[0].layers = min(pl.gpu_layers, info.n_layer)
            ot = fit.get("override_tensor", "")
            pl.cpu_expert_layers = override_layers(ot)
            pl.n_cpu_moe = len(pl.cpu_expert_layers)
            if not pl.full_offload:
                if ot:
                    pl.warnings.append(msg_partial_experts(len(pl.cpu_expert_layers), info.n_layer, reduce))
                else:
                    pl.warnings.append(msg_partial_layers(min(pl.gpu_layers, info.n_layer), info.n_layer, reduce))
                if reduce:
                    pl.notes.append(msg_no_fit_min())
                if pl.ubatch > p.ubatch_size:
                    pl.notes.append(msg_ubatch(pl.ubatch))
            if p.kv_cache_type == "auto" and pl.kv_k != "f16":
                pl.notes.append(msg_kv_q8(pl.full_offload))
            pl.load_mode = p.load_mode if p.load_mode != "auto" else ("none" if pl.full_offload else "mmap")
        for d in pl.devices:
            pr = proj.get(d.name)
            if pr:
                d.weights_mib = pr["model"]
                d.kv_mib = pr["context"]
                d.compute_mib = pr["compute"]
                d.output_mib = 0.0
                d.draft_mib = 0.0
        if "Host" in proj:
            pl.host = {"weights_mib": proj["Host"]["model"], "kv_mib": proj["Host"]["context"],
                       "compute_mib": proj["Host"]["compute"]}
        pl.totals["vram_used_mib"] = round(sum(d.used_mib for d in pl.devices), 1)
        if proj:
            pl.fits = all(d.headroom_mib >= 0 for d in pl.devices)
        elif not manual:
            pl.fits = True  # the engine fit guarantees the margins
        if manual and not pl.fits:
            over = sum(max(0.0, -d.headroom_mib) for d in pl.devices)
            pl.warnings.append(msg_manual_over(over))

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

    async def _load_locked(self, entry: ModelEntry, overrides: dict | None, source: str) -> EngineInstance:
        async with self._load_lock:
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
            self.bus.activity_log(f"Loading {entry.id} ({'JIT request' if source == 'jit' else 'control panel'})",
                                  category="model", model=entry.id)
            self.bus.publish("load_stage", model=entry.id, stage="planning", label="Planning memory layout")
            pl, extra = await self.plan(entry.id, overrides, verify=True, assume_evict=False, projection=False)
            p = self.store.effective_load_params(entry.path, overrides)
            devices, _ = await self.devices(max_age=3.0)
            sel = [d.name for d in devices if not p.devices or d.name in p.devices]
            mmproj = pl.mmproj or None
            draft = self._draft_for(p)
            where = ("CPU only" if not devices else "full GPU offload" if pl.full_offload
                     else "PARTIAL offload (part of the model in system RAM)")
            self.bus.activity_log(
                f"Plan{' (engine-verified)' if pl.source == 'engine' else ''}: {where}, KV {pl.kv_k.upper()}, "
                f"context {pl.ctx:,}", category="model", level="info" if pl.full_offload or not devices else "warn")
            if pl.ctx_reduced_from:
                self.bus.activity_log(
                    f"{entry.id}: context reduced from {pl.ctx_reduced_from:,} to {pl.ctx:,} tokens so the whole model "
                    "stays in VRAM (change 'If it does not fit' in the load settings to keep the full context)",
                    level="warn", category="model", model=entry.id)
            for w in pl.warnings:
                if w.startswith("Manual allocation"):
                    self.bus.activity_log(f"{entry.id}: {w}", level="error", category="model", model=entry.id)
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
                on_exit=self._on_instance_exit,
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
            if inst.load_info.get("device_pci"):
                self._device_pci.setdefault(eng.path, {}).update(inst.load_info["device_pci"])
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
        self._spill.pop(inst.id, None)
        await inst.stop()
        self.instances.pop(inst.id, None)
        self.bus.publish("instance_removed", iid=inst.id, model=inst.model_id)

    def _on_instance_exit(self, inst: EngineInstance, rc: int) -> None:
        if self._shutting_down or self.exiting():
            return
        now = time.time()
        self._restart_log = [t for t in self._restart_log if now - t < 600]
        if inst.pid:
            self.monitor.unwatch_process(inst.pid)
        if len(self._restart_log) >= 3:
            self.bus.activity_log(f"{inst.model_id} crashed repeatedly; automatic restart disabled", level="error",
                                  category="model")
            return
        self._restart_log.append(now)

        async def restart() -> None:
            self.instances.pop(inst.id, None)
            self.bus.publish("instance_removed", iid=inst.id, model=inst.model_id)
            self.bus.activity_log(f"Restarting {inst.model_id} after engine exit (code {rc})", level="warn",
                                  category="model")
            try:
                new = await self.load(inst.model_id, overrides=inst.params.model_dump(mode="json"), source="recovery")
                new.restarts = inst.restarts + 1
            except Exception as exc:
                log.error("restart failed: %s", exc)

        asyncio.get_running_loop().create_task(restart())

    # ----- request routing ---------------------------------------------------------------

    async def instance_for_request(self, requested: str | None, jit: bool) -> EngineInstance:
        """Return a ready instance for a client-requested model name (JIT-loading if enabled)."""
        entry = self.library.resolve(requested) if requested else None
        if entry is not None:
            inst = self.instance_for_model(entry.id)
            if inst is not None:
                return await self._await_ready(inst)
            if entry.id in self._pending:
                return await self._await_ready(await asyncio.shield(self._pending[entry.id]))
            if jit:
                return await self.load(entry.id, source="jit")
            ready = self.ready_instances()
            if ready:
                raise ModelError(
                    f"Model '{requested}' is not loaded and just-in-time loading is disabled. "
                    f"Loaded: {', '.join(i.model_id for i in ready)}", 404, "model_not_loaded")
            raise ModelError("No model is loaded and just-in-time loading is disabled.", 503, "no_model_loaded")
        # Unknown / empty model name: be lenient like most local servers and use the active model.
        loading = [i for i in self.instances.values() if i.state in ("loading", "starting")]
        ready = sorted(self.ready_instances(), key=lambda i: i.last_used, reverse=True)
        if ready:
            return ready[0]
        if loading:
            return await self._await_ready(loading[0])
        last = self.store.settings.startup.last_model
        if jit and last and self.library.get(last) and not requested:
            return await self.load(last, source="jit")
        known = ", ".join(e.id for e in self.library.entries()[:20])
        if requested:
            raise ModelError(f"Model '{requested}' not found. Available: {known}", 404, "model_not_found")
        raise ModelError("No model is loaded. Load a model in WinRunner or specify 'model'.", 503, "no_model_loaded")

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
            ready = self.ready_instances()
            if ready:
                raise ModelError(
                    f"Model '{requested}' is not loaded and just-in-time loading is disabled. "
                    f"Loaded: {', '.join(i.model_id for i in ready)}", 404, "model_not_loaded")
            raise ModelError("No model is loaded and just-in-time loading is disabled.", 503, "no_model_loaded")
        ready = sorted(self.ready_instances(), key=lambda i: i.last_used, reverse=True)
        if ready:
            return ready[0]
        if any(i.state in ("loading", "starting") for i in self.instances.values()) or self._pending:
            return None
        last = self.store.settings.startup.last_model
        if jit and last and self.library.get(last) and not requested:
            return None
        known = ", ".join(e.id for e in self.library.entries()[:20])
        if requested:
            raise ModelError(f"Model '{requested}' not found. Available: {known}", 404, "model_not_found")
        raise ModelError("No model is loaded. Load a model in WinRunner or specify 'model'.", 503, "no_model_loaded")

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

    def note_request_error(self, rec: Any, error: str) -> None:
        """Explain context overflows on models whose context was reduced to fit VRAM."""
        if "exceed" not in error or "context" not in error:
            return
        inst = self.instances.get(getattr(rec, "instance", "") or "")
        if inst is None or not inst.plan.ctx_reduced_from or inst.load_info.get("ctx_hint_shown"):
            return
        inst.load_info["ctx_hint_shown"] = True
        self.bus.activity_log(
            f"{inst.model_id}: a request needed more than the {inst.n_ctx:,}-token context this model was loaded with "
            f"(reduced from {inst.plan.ctx_reduced_from:,} so the model fits in VRAM). For longer prompts use a Q4_0 KV "
            "cache or a smaller quantization, or set 'If it does not fit' to 'Offload to CPU' (slower).",
            level="warn", category="model", model=inst.model_id)

    def check_residency(self, sample: dict[str, Any]) -> None:
        """Detect Windows paging an engine's GPU memory into system RAM.

        When VRAM is over-committed, WDDM keeps only part of a process's allocations
        resident and pages the rest over PCIe on demand: the GPUs look busy "on and
        off" and everything becomes extremely slow. The engine reports what it
        allocated on each GPU; the OS reports how much of the process is resident in
        dedicated VRAM. A persistent gap means memory was demoted.
        """
        procs = sample.get("procs") or {}
        now = time.time()
        for inst in self.ready_instances():
            if not inst.pid or inst.t_ready is None or now - inst.t_ready < 8:  # let allocations settle
                continue
            rec = procs.get(str(inst.pid)) or {}
            if "vram" not in rec:
                continue
            bufs = inst.load_info.get("buffers") or {}
            allocated = sum(float(v) for dev, b in bufs.items() if not is_host_buffer(dev)
                            for k, v in b.items() if k in ("model", "kv", "compute", "rs")) * MiB
            if allocated <= 0:
                continue
            resident = sum(float(v or 0) for v in rec["vram"].values())
            deficit = allocated - resident
            st = self._spill.setdefault(inst.id, {"count": 0, "warned": False})
            st["count"] = st["count"] + 1 if deficit > max(512 * MiB, 0.05 * allocated) else 0
            spilled = st["count"] >= 3
            mib = round(max(0.0, deficit) / MiB) if spilled else 0
            if spilled != inst.load_info.get("vram_spill_active") or (spilled and abs(mib - inst.load_info.get("vram_spill_mib", 0)) > 256):
                inst.load_info["vram_spill_active"] = spilled
                inst.load_info["vram_spill_mib"] = mib
                inst.load_info["vram_shared_mib"] = round(sum(float(v or 0) for v in (rec.get("shared") or {}).values()) / MiB)
                self.bus.publish("instance", instance=inst.status())
            if spilled and not st["warned"]:
                st["warned"] = True
                self.bus.activity_log(
                    f"{inst.model_id}: Windows moved about {mib / 1024:.1f} GiB of the model's GPU memory into system RAM "
                    "(VRAM over-committed). Prompt processing and generation will be extremely slow. Use automatic "
                    "allocation with 'Reduce context', lower the context, or close other applications using the GPU.",
                    level="error", category="model", model=inst.model_id)


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
