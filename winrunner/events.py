"""Event bus, history buffers and request tracking for live telemetry."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from .util import new_id

log = logging.getLogger("winrunner.events")


class EventBus:
    """Fan-out of events to websocket subscribers. ``publish`` is thread-safe."""

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.activity: deque[dict[str, Any]] = deque(maxlen=400)
        self.applog: deque[dict[str, Any]] = deque(maxlen=3000)

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, maxsize: int = 4000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, etype: str, **data: Any) -> None:
        ev = {"type": etype, "t": time.time(), **data}
        loop = self._loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._deliver(ev)
        elif not loop.is_closed():
            loop.call_soon_threadsafe(self._deliver, ev)

    def _deliver(self, ev: dict[str, Any]) -> None:
        for q in list(self._subs):
            if q.full():
                try:  # slow consumer: drop the oldest event rather than blocking producers
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass

    def activity_log(self, text: str, level: str = "info", category: str = "server", **extra: Any) -> None:
        """Short human readable entry for the activity panel."""
        rec = {"t": time.time(), "level": level, "category": category, "text": text, **extra}
        self.activity.append(rec)
        self.publish("activity", **rec)


class BusLogHandler(logging.Handler):
    """Forwards application log records to the UI."""

    def __init__(self, bus: EventBus):
        super().__init__(level=logging.INFO)
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = {logging.WARNING: "warn", logging.ERROR: "error", logging.CRITICAL: "error"}.get(
                record.levelno, "debug" if record.levelno < logging.INFO else "info")
            rec = {"t": record.created, "level": level, "source": record.name, "text": self.format(record)}
            self.bus.applog.append(rec)
            self.bus.publish("applog", **rec)
        except Exception:  # pragma: no cover - logging must never raise
            pass


@dataclass
class RequestRecord:
    id: str
    t_start: float
    client: str
    user_agent: str
    endpoint: str
    model: str
    instance: str = ""
    stream: bool = False
    images: int = 0
    phase: str = "queued"  # queued | loading_model | prompt | generating | done | error | cancelled
    prompt_total: int | None = None
    prompt_processed: int = 0
    prompt_cached: int = 0
    tokens: int = 0
    reasoning_tokens: int = 0
    t_prompt_start: float | None = None
    t_first_token: float | None = None
    t_end: float | None = None
    prompt_ms: float | None = None
    prompt_tps: float | None = None
    gen_ms: float | None = None
    gen_tps: float | None = None
    finish_reason: str = ""
    error: str = ""
    tool_calls: list[str] = field(default_factory=list)
    preview: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        d = asdict(self)
        now = time.time()
        end = self.t_end or now
        d["duration_ms"] = round((end - self.t_start) * 1000, 1)
        d["ttft_ms"] = round((self.t_first_token - self.t_start) * 1000, 1) if self.t_first_token else None
        if self.gen_tps is None and self.t_first_token and self.tokens > 1:
            dt = end - self.t_first_token
            d["live_tps"] = round((self.tokens - 1) / dt, 2) if dt > 0 else None
        return d


class RequestTracker:
    """Tracks API requests from arrival to completion and streams tokens to the UI."""

    PREVIEW_CHARS = 4000

    def __init__(self, bus: EventBus, history: int = 500):
        self.bus = bus
        self.active: dict[str, RequestRecord] = {}
        self.history: deque[RequestRecord] = deque(maxlen=history)
        self._pending: dict[str, list[list[str]]] = {}
        self._lock = threading.Lock()
        self.totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}
        self._last_progress_emit: dict[str, float] = {}
        self._flusher: asyncio.Task | None = None
        self.tps_history: deque[dict[str, Any]] = deque(maxlen=300)

    def start(self) -> None:
        if self._flusher is None:
            self._flusher = asyncio.get_running_loop().create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._flusher:
            self._flusher.cancel()
            self._flusher = None

    def begin(self, client: str, user_agent: str, endpoint: str, model: str, stream: bool, images: int,
              params: dict[str, Any] | None = None) -> RequestRecord:
        r = RequestRecord(
            id=new_id("req-", 4), t_start=time.time(), client=client, user_agent=user_agent[:120],
            endpoint=endpoint, model=model, stream=stream, images=images, params=params or {},
        )
        with self._lock:
            self.active[r.id] = r
            self.totals["requests"] += 1
        self.bus.publish("request", record=r.snapshot())
        ua = f" · {images} image(s)" if images else ""
        self.bus.activity_log(f"{endpoint} from {client} → {model or 'default model'}{ua}", category="request",
                              rid=r.id)
        return r

    def update(self, r: RequestRecord, emit: bool = True, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(r, k, v)
        if emit:
            self.bus.publish("request", record=r.snapshot())

    def progress(self, r: RequestRecord, total: int, processed: int, cache: int) -> None:
        first = r.phase != "prompt"
        if r.t_prompt_start is None:
            r.t_prompt_start = time.time()
        if r.phase != "generating":
            r.phase = "prompt"
        r.prompt_total, r.prompt_processed, r.prompt_cached = total, processed, cache
        now = time.time()
        if first or processed >= total or now - self._last_progress_emit.get(r.id, 0) > 0.1:
            self._last_progress_emit[r.id] = now
            self.bus.publish("request", record=r.snapshot())

    def tokens(self, r: RequestRecord, text: str, kind: str = "c") -> None:
        """kind: c = content, r = reasoning, t = tool call arguments."""
        if not text:
            return
        now = time.time()
        if r.t_first_token is None:
            r.t_first_token = now
            r.phase = "generating"
            self.bus.publish("request", record=r.snapshot())
        if kind == "r":
            r.reasoning_tokens += 1
        r.tokens += 1
        if len(r.preview) < self.PREVIEW_CHARS:
            r.preview += text
        with self._lock:
            self._pending.setdefault(r.id, []).append([text, kind])

    def finish(self, r: RequestRecord, finish_reason: str = "", timings: dict[str, Any] | None = None,
               usage: dict[str, Any] | None = None, error: str = "", cancelled: bool = False) -> None:
        self._flush_one(r.id)
        r.t_end = time.time()
        if timings:
            r.prompt_ms = timings.get("prompt_ms")
            r.prompt_tps = timings.get("prompt_per_second")
            r.gen_ms = timings.get("predicted_ms")
            r.gen_tps = timings.get("predicted_per_second")
            if timings.get("predicted_n") is not None:
                r.tokens = int(timings["predicted_n"])
            if timings.get("prompt_n") is not None and r.prompt_total is None:
                r.prompt_total = int(timings["prompt_n"]) + int(timings.get("cache_n") or 0)
            if timings.get("cache_n") is not None:
                r.prompt_cached = int(timings["cache_n"])
        if usage:
            if usage.get("completion_tokens") is not None:
                r.tokens = int(usage["completion_tokens"])
            if usage.get("prompt_tokens") is not None:
                r.prompt_total = int(usage["prompt_tokens"])
        if r.gen_tps is None and r.t_first_token and r.tokens > 1:
            dt = r.t_end - r.t_first_token
            r.gen_tps = round((r.tokens - 1) / dt, 2) if dt > 0 else None
        r.finish_reason = finish_reason or r.finish_reason
        r.error = error
        r.phase = "cancelled" if cancelled else "error" if error else "done"
        with self._lock:
            self.active.pop(r.id, None)
            self.history.appendleft(r)
            self._pending.pop(r.id, None)
            self.totals["prompt_tokens"] += int(r.prompt_total or 0)
            self.totals["completion_tokens"] += int(r.tokens or 0)
            if error:
                self.totals["errors"] += 1
        self._last_progress_emit.pop(r.id, None)
        snap = r.snapshot()
        self.bus.publish("request", record=snap)
        if r.gen_tps:
            self.tps_history.append({"t": r.t_end, "tg": r.gen_tps, "pp": r.prompt_tps, "n": r.tokens, "id": r.id})
        if error:
            self.bus.activity_log(f"{r.id} failed: {error[:160]}", level="error", category="request", rid=r.id)
        elif cancelled:
            self.bus.activity_log(f"{r.id} cancelled by client after {r.tokens} tokens", level="warn",
                                  category="request", rid=r.id)
        else:
            parts = [f"{r.id} complete"]
            if r.prompt_total is not None:
                pp = f" @ {r.prompt_tps:,.0f} t/s" if r.prompt_tps else ""
                cached = f" ({r.prompt_cached} cached)" if r.prompt_cached else ""
                parts.append(f"prompt {r.prompt_total} tok{cached}{pp}")
            tg = f" @ {r.gen_tps:.1f} t/s" if r.gen_tps else ""
            parts.append(f"generated {r.tokens} tok{tg}")
            if r.finish_reason:
                parts.append(f"stop: {r.finish_reason}")
            self.bus.activity_log(" · ".join(parts), level="ok", category="request", rid=r.id)

    def _flush_one(self, rid: str) -> None:
        with self._lock:
            pieces = self._pending.pop(rid, None)
        if pieces:
            r = self.active.get(rid)
            self.bus.publish("tokens", rid=rid, pieces=pieces, count=r.tokens if r else None,
                             live_tps=r.snapshot().get("live_tps") if r else None)

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(0.06)
            with self._lock:
                ids = list(self._pending.keys())
            for rid in ids:
                self._flush_one(rid)

    def recent(self, n: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            act = [r.snapshot() for r in self.active.values()]
            hist = [r.snapshot() for r in list(self.history)[:n]]
        for h in hist:
            h["preview"] = h["preview"][-600:]
        return act + hist

    def get(self, rid: str) -> dict[str, Any] | None:
        with self._lock:
            r = self.active.get(rid) or next((x for x in self.history if x.id == rid), None)
        return r.snapshot() if r else None
