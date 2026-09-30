"""A running llama-server process ("instance") and its supervision."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable

import httpx
import psutil

from .cmdline import LaunchSpec
from .engine import EngineInfo, engine_env
from .events import EventBus
from .faults import describe_exit_code
from .library import ModelEntry
from .logparse import LogParser
from .paths import IS_WINDOWS
from .planner import Plan
from .config import LoadParams
from .util import new_id

log = logging.getLogger("winrunner.instance")

# Load progress checkpoints (fraction of the whole load) per engine phase.
PHASE_PROGRESS = {
    "spawn": 0.02,
    "open": 0.04,
    "fit": 0.06,
    "metadata": 0.10,
    "tensors": 0.12,
    "context": 0.84,
    "warmup": 0.90,
    "mmproj": 0.94,
    "loaded": 0.98,
    "ready": 1.0,
}
TENSOR_SPAN = (0.12, 0.84)
PHASE_LABELS = {
    "spawn": "Starting engine process",
    "open": "Opening model file",
    "fit": "Fitting parameters to device memory",
    "metadata": "Reading GGUF metadata",
    "tensors": "Loading tensors",
    "context": "Allocating KV cache and compute buffers",
    "warmup": "Warming up",
    "mmproj": "Loading vision projector",
    "loaded": "Initializing server slots",
    "ready": "Ready",
}


def _pdeathsig() -> None:  # pragma: no cover - runs in the child on Linux
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
    except Exception:
        pass


class EngineInstance:
    def __init__(
        self,
        *,
        entry: ModelEntry,
        params: LoadParams,
        plan: Plan,
        engine: EngineInfo,
        spec: LaunchSpec,
        port: int,
        api_key: str,
        bus: EventBus,
        mmproj: str | None,
        priority: str,
        expected_load_s: float,
        device_map: dict[str, str],
        on_exit: Callable[["EngineInstance", int], None] | None = None,
        on_fault: Callable[["EngineInstance", str], bool] | None = None,
    ):
        self.id = new_id("inst-", 3)
        self.entry = entry
        self.model_id = entry.id
        self.params = params
        self.plan = plan
        self.engine = engine
        self.spec = spec
        self.port = port
        self.api_key = api_key
        self.bus = bus
        self.mmproj = mmproj
        self.priority = priority
        self.expected_load_s = expected_load_s
        self.device_map = device_map
        self.on_exit = on_exit
        self.on_fault = on_fault
        self.base_url = f"http://127.0.0.1:{port}"
        self.state = "starting"
        self.phase = "spawn"
        self.progress = 0.0
        self.error = ""
        self.t_start = time.time()
        self.t_ready: float | None = None
        self.t_tensors: float | None = None
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.exit_code: int | None = None
        self._exited = asyncio.Event()
        # Set by the model manager when this engine failed while serving and a replacement is started.
        self.recovery: asyncio.Future | None = None
        self.log: deque[dict[str, Any]] = deque(maxlen=6000)
        self._parser = LogParser(jsonl="--log-jsonl" in spec.args)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: threading.Thread | None = None
        self._progress_task: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None
        self.load_info: dict[str, Any] = {
            "buffers": {}, "breakdown": {}, "offload": None, "kv": None, "ctx": {}, "flash_attn": None,
            "slots": {}, "mmproj": {}, "template_example": "", "thinking": None, "fit": [], "device_info": {},
            "backend": {}, "load_seconds": None, "errors": [],
        }
        self.props: dict[str, Any] = {}
        self.template_verified: bool | None = None
        self.requests_served = 0
        self.active_requests = 0
        self.last_used = time.time()
        self.slots: list[dict[str, Any]] = []
        self.restarts = 0

    # ----- lifecycle ------------------------------------------------------------------

    async def start(self, timeout: float) -> None:
        self._loop = asyncio.get_running_loop()
        self._set_phase("spawn")
        self._emit_state()
        args = self.spec.args
        self._log_line("info", "$ " + self.spec.display(), source="winrunner")
        kwargs: dict[str, Any] = dict(
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=os.path.dirname(self.engine.path) or None, env=engine_env(),
        )
        if IS_WINDOWS:  # pragma: no cover
            from .platform.win32 import CREATE_NO_WINDOW, PRIORITY_FLAGS

            kwargs["creationflags"] = CREATE_NO_WINDOW | PRIORITY_FLAGS.get(self.priority, 0)
        else:
            kwargs["preexec_fn"] = _pdeathsig
        try:
            self.proc = subprocess.Popen(args, **kwargs)
        except OSError as exc:
            self.fail(f"could not start engine: {exc}")
            raise RuntimeError(self.error) from exc
        self.pid = self.proc.pid
        if IS_WINDOWS:  # pragma: no cover
            from .platform.win32 import assign_to_job

            assign_to_job(int(self.proc._handle))  # type: ignore[attr-defined]
        elif self.priority != "normal":
            try:
                psutil.Process(self.pid).nice(-2 if self.priority == "high" else -1)
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
        self.state = "loading"
        self._emit_state()
        self._reader = threading.Thread(target=self._read_output, name=f"engine-log-{self.id}", daemon=True)
        self._reader.start()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                        timeout=httpx.Timeout(connect=10, read=None, write=60, pool=None),
                                        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
        self._progress_task = asyncio.create_task(self._progress_loop())
        try:
            await self._wait_ready(timeout)
        finally:
            if self._progress_task:
                self._progress_task.cancel()
        await self._after_ready()

    async def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        assert self.client is not None
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                await asyncio.sleep(0.3)  # let the reader thread flush the final log lines
                errs = self.load_info["errors"][-3:]
                detail = "; ".join(errs) if errs else "see engine log"
                self.fail(f"engine exited during load ({describe_exit_code(self.proc.returncode)}): {detail}")
                raise RuntimeError(self.error)
            if self.state == "error":
                await self.stop()
                raise RuntimeError(self.error)
            try:
                r = await self.client.get("/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                self.fail(f"engine did not become ready within {int(timeout)} s")
                await self.stop()
                raise RuntimeError(self.error)
            await asyncio.sleep(0.25)

    async def _after_ready(self) -> None:
        self.t_ready = time.time()
        self.load_info["load_seconds"] = round(self.t_ready - self.t_start, 2)
        try:
            assert self.client is not None
            r = await self.client.get("/props", timeout=10)
            if r.status_code == 200:
                self.props = r.json()
        except (httpx.HTTPError, ValueError):
            pass
        if self.state == "error":  # the engine failed while its properties were read
            await self.stop()
            raise RuntimeError(self.error)
        gguf_tmpl = self.entry.info.chat_template or ""
        eng_tmpl = self.props.get("chat_template") or ""
        if self.params.chat_template_mode == "gguf" and gguf_tmpl and eng_tmpl:
            self.template_verified = gguf_tmpl.strip() == eng_tmpl.strip()
        self._set_phase("ready")
        self.progress = 1.0
        self.state = "ready"
        self._emit_state()
        msg = f"{self.model_id} ready in {self.load_info['load_seconds']:.1f} s"
        off = self.load_info.get("offload")
        if off:
            msg += f" · {off['gpu_layers']}/{off['total_layers']} layers on GPU"
        self.bus.activity_log(msg, level="ok", category="model", model=self.model_id)
        if self.template_verified is False:
            self.bus.activity_log(
                f"{self.model_id}: engine chat template differs from the GGUF's embedded template", level="warn",
                category="model")

    async def stop(self, timeout: float = 10.0) -> None:
        if self.state in ("stopped",):
            return
        prev = self.state
        self.state = "stopping"
        self._emit_state()
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass
            try:
                await asyncio.to_thread(p.wait, timeout)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except OSError:
                    pass
                await asyncio.to_thread(p.wait, 5)
        if self.client:
            await self.client.aclose()
            self.client = None
        self.state = "stopped" if prev != "error" else "error"
        self._emit_state()

    # ----- output handling ------------------------------------------------------------

    def _read_output(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        loop = self._loop
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                if loop and not loop.is_closed():
                    loop.call_soon_threadsafe(self._on_line, line)
        except (OSError, ValueError):
            pass
        rc = self.proc.wait()
        if loop and not loop.is_closed():
            loop.call_soon_threadsafe(self._on_exit, rc)

    def _on_exit(self, rc: int) -> None:
        for ll in self._parser.flush():
            self._handle(ll.level, ll.text, ll.events)
        self.exit_code = rc
        was = self.state
        self._log_line("info" if was == "stopping" else "error",
                       f"engine process exited ({describe_exit_code(rc)})", source="winrunner")
        if was == "ready":
            self.fail(f"engine process exited unexpectedly ({describe_exit_code(rc)})")
            if self.on_exit:
                self.on_exit(self, rc)
        self._exited.set()

    async def wait_exited(self, timeout: float) -> bool:
        """Wait until the engine process has exited and its exit was handled; False on timeout."""
        if self.proc is None:
            return False
        try:
            await asyncio.wait_for(self._exited.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def report_fault(self, reason: str) -> bool:
        """The engine reported an unrecoverable GPU error. True if a replacement engine is being started."""
        if self.on_fault is None:
            return False
        return self.on_fault(self, reason)

    @property
    def recovering(self) -> bool:
        """This engine failed and a replacement is being started (or was started successfully)."""
        f = self.recovery
        return f is not None and (not f.done() or (not f.cancelled() and f.exception() is None))

    def _on_line(self, line: str) -> None:
        for ll in self._parser.feed(line):
            self._handle(ll.level, ll.text, ll.events)

    def _log_line(self, level: str, text: str, source: str = "engine") -> None:
        rec = {"t": time.time(), "level": level, "text": text, "source": source}
        self.log.append(rec)
        self.bus.publish("englog", iid=self.id, model=self.model_id, **rec)

    def _handle(self, level: str, text: str, events: list[tuple[str, dict[str, Any]]]) -> None:
        self._log_line(level, text)
        li = self.load_info
        for kind, d in events:
            if kind == "phase":
                ph = d["phase"]
                if ph == "tensors":
                    self.t_tensors = time.time()
                    if d.get("load_mode"):
                        li["load_mode"] = d["load_mode"]
                self._set_phase(ph)
            elif kind == "fit":
                li["fit"].append(d["message"])
            elif kind == "buffer":
                li["buffers"].setdefault(d["device"], {})
                li["buffers"][d["device"]][d["kind"]] = li["buffers"][d["device"]].get(d["kind"], 0.0) + d["mib"]
                if d["kind"] in ("kv", "compute"):
                    self._set_phase("context")
            elif kind == "breakdown":
                li["breakdown"][d["device"]] = d
            elif kind == "offload":
                li["offload"] = d
            elif kind == "kv":
                li["kv"] = d
            elif kind == "ctx":
                li["ctx"].update(d)
                self._set_phase("context")
            elif kind == "flash_attn":
                li["flash_attn"] = d["enabled"]
            elif kind == "slots":
                li["slots"].update(d)
            elif kind == "mmproj":
                li["mmproj"].update(d)
                if "encoder" in d or "loaded" in d:
                    self._set_phase("mmproj")
            elif kind == "template":
                if "example" in d:
                    li["template_example"] = d["example"]
                if "thinking" in d:
                    li["thinking"] = d["thinking"]
            elif kind == "device_info":
                li["device_info"][str(d["index"])] = d["info"]
            elif kind == "backend":
                li["backend"].update({k: v for k, v in d.items() if k != "path"})
            elif kind == "error":
                li["errors"].append(d["message"])
                if self.state == "loading":
                    self.bus.activity_log(f"{self.model_id}: {d['message'][:200]}", level="error", category="engine")
            elif kind == "gpu_fault":
                if self.state == "ready":
                    self.report_fault(f"{d['what']}: {d['message'][:300]}")
                elif self.state in ("starting", "loading"):
                    # the engine could come up "ready" with a GPU it can never use again
                    self.fail(f"{d['what']} while loading: {d['message'][:300]}")
            elif kind.startswith("task_"):
                self.bus.publish("task", iid=self.id, kind=kind, **d)

    def _set_phase(self, phase: str) -> None:
        if phase not in PHASE_PROGRESS:
            return
        if PHASE_PROGRESS[phase] < PHASE_PROGRESS.get(self.phase, 0) and phase != "ready":
            return
        changed = phase != self.phase
        self.phase = phase
        self.progress = max(self.progress, PHASE_PROGRESS[phase])
        if changed and self.state == "loading":
            self._emit_state()

    async def _progress_loop(self) -> None:
        """Refine progress inside the tensor loading phase.

        With full-read loading (load mode "none") bytes read by the process are an
        exact measure; with mmap the estimate is time based, using the duration
        of the previous load of this model when known.
        """
        size = max(1, self.entry.info.file_size)
        try:
            proc = psutil.Process(self.pid) if self.pid else None
        except psutil.Error:
            proc = None
        base_read = None
        while True:
            await asyncio.sleep(0.25)
            if self.phase != "tensors" or self.t_tensors is None:
                continue
            frac = None
            if proc is not None:
                try:
                    rb = proc.io_counters().read_bytes
                    if base_read is None:
                        base_read = rb
                    if rb - base_read > 16 * 1024 * 1024:
                        frac = min(0.99, (rb - base_read) / size)
                except (psutil.Error, AttributeError, NotImplementedError):
                    proc = None
            if frac is None:
                exp = self.expected_load_s * 0.8 if self.expected_load_s else max(3.0, size / (1.5 * 1024 ** 3))
                el = time.time() - self.t_tensors
                frac = min(0.95, 1 - pow(2.718, -el / max(1.0, exp)))
            lo, hi = TENSOR_SPAN
            self.progress = max(self.progress, lo + (hi - lo) * frac)
            self.bus.publish("instance_progress", iid=self.id, model=self.model_id, progress=round(self.progress, 4),
                             phase=self.phase, label=PHASE_LABELS.get(self.phase, self.phase))

    # ----- status -------------------------------------------------------------------

    @property
    def vision_enabled(self) -> bool:
        mods = self.props.get("modalities") or {}
        if mods:
            return bool(mods.get("vision"))
        return bool(self.mmproj and self.entry.has_vision)

    def fail(self, msg: str) -> None:
        """Take the instance out of service: requests are no longer routed to it."""
        self.error = msg
        self.state = "error"
        self._emit_state()
        self.bus.activity_log(f"{self.model_id}: {msg}", level="error", category="model", model=self.model_id)
        log.error("%s: %s", self.model_id, msg)

    def _emit_state(self) -> None:
        self.bus.publish("instance", instance=self.status())

    def status(self) -> dict[str, Any]:
        e = self.entry
        uptime = time.time() - self.t_ready if self.t_ready and self.state == "ready" else 0
        caps = self.props.get("chat_template_caps") or {}
        mods = self.props.get("modalities") or {}
        return {
            "id": self.id,
            "model": self.model_id,
            "name": e.info.name or e.id,
            "path": e.path,
            "state": self.state,
            "phase": self.phase,
            "phase_label": PHASE_LABELS.get(self.phase, self.phase),
            "progress": round(self.progress, 4),
            "error": self.error,
            "pid": self.pid,
            "port": self.port,
            "t_start": self.t_start,
            "t_ready": self.t_ready,
            "uptime": round(uptime, 1),
            "ctx": self.plan.ctx,
            "n_ctx_engine": (self.props.get("default_generation_settings") or {}).get("n_ctx")
            or self.load_info["ctx"].get("n_ctx"),
            "vision": self.vision_enabled,
            "audio": bool(mods.get("audio")),
            "mmproj": self.mmproj or "",
            "plan": self.plan.to_dict(),
            "params": self.params.model_dump(mode="json"),
            "engine": {"name": self.engine.name, "backend": self.engine.backend, "build": self.engine.build,
                       "version": self.engine.version, "commit": self.engine.commit},
            "command": self.spec.display(),
            "load": self.load_info,
            "template_verified": self.template_verified,
            "template_caps": caps,
            "total_slots": self.props.get("total_slots"),
            "requests_served": self.requests_served,
            "active_requests": self.active_requests,
            "last_used": self.last_used,
            "device_map": self.device_map,
            "restarts": self.restarts,
            "recovering": self.state == "error" and self.recovery is not None and not self.recovery.done(),
            "exit_code": self.exit_code,
            "generation_defaults": {
                k: v for k, v in ((self.props.get("default_generation_settings") or {}).get("params") or {}).items()
                if k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n",
                         "presence_penalty", "frequency_penalty", "dry_multiplier", "xtc_probability",
                         "typical_p", "n_predict", "reasoning_format", "chat_format", "samplers")
            },
        }

    async def poll_slots(self) -> list[dict[str, Any]]:
        if self.state != "ready" or not self.client:
            return []
        try:
            r = await self.client.get("/slots", timeout=3)
            if r.status_code == 200:
                data = r.json()
                self.slots = [
                    {
                        "id": s.get("id"),
                        "n_ctx": s.get("n_ctx"),
                        "processing": s.get("is_processing"),
                        "task": s.get("id_task"),
                        "n_prompt": s.get("n_prompt_tokens"),
                        "n_prompt_processed": s.get("n_prompt_tokens_processed"),
                        "n_decoded": ((s.get("next_token") or [{}])[0] or {}).get("n_decoded"),
                        "speculative": s.get("speculative"),
                    }
                    for s in (data if isinstance(data, list) else [])
                ]
        except (httpx.HTTPError, ValueError):
            pass
        return self.slots
