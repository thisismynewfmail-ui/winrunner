"""OpenAI / LM Studio compatible API (``/v1`` and ``/api/v0``).

Requests are validated and normalised (vision inputs, LM Studio specific
fields, per-model sampling presets), routed to the engine instance serving the
requested model (loading it just in time if enabled), and streamed back while
being tapped for live telemetry (prompt processing progress, tokens, timings).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .context import AppContext
from .events import RequestRecord
from .instance import EngineInstance
from .manager import ModelError
from .util import is_loopback
from .vision import ImageError, VisionStats

log = logging.getLogger("winrunner.api")

LMS_STOP_REASONS = {"stop": "eosFound", "length": "maxPredictedTokensReached", "tool_calls": "toolCalls"}
STRIP_FIELDS = ("ttl", "draft_model")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def oai_error(status: int, message: str, code: str | None = None, etype: str | None = None) -> JSONResponse:
    if etype is None:
        etype = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": etype, "param": None, "code": code}},
                        status_code=status)


def _client(request: Request) -> str:
    return request.client.host if request.client else "?"


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class SSEParser:
    """Incremental server-sent-events splitter (works on raw bytes)."""

    def __init__(self) -> None:
        self.buf = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buf += chunk
        if b"\r" in self.buf:
            self.buf = self.buf.replace(b"\r\n", b"\n")
        out = []
        while True:
            i = self.buf.find(b"\n\n")
            if i < 0:
                break
            out.append(self.buf[:i])
            self.buf = self.buf[i + 2 :]
        return out

    def flush(self) -> list[bytes]:
        rest, self.buf = self.buf.strip(), b""
        return [rest] if rest else []


def parse_event(raw: bytes) -> tuple[str | None, str | None]:
    """(event name, data) of one SSE event; data None for comments."""
    name = None
    data_lines = []
    for line in raw.split(b"\n"):
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip(b" ").decode("utf-8", errors="replace"))
        elif line.startswith(b"event:"):
            name = line[6:].strip().decode("utf-8", errors="replace")
    return name, ("\n".join(data_lines) if data_lines else None)


class ChatTap:
    """Observes (and minimally rewrites) chat-completion stream chunks.

    * ``prompt_progress`` (injected by WinRunner) feeds the telemetry and is
      removed unless the client asked for it,
    * the usage-only chunk is removed unless the client asked for it,
    * token deltas are forwarded to the live token stream,
    * the full response is accumulated for non-streaming clients.
    """

    def __init__(self, ctx: AppContext, rec: RequestRecord, client_progress: bool, client_usage: bool,
                 lms: dict[str, Any] | None = None, kind: str = "chat"):
        self.ctx = ctx
        self.rec = rec
        self.client_progress = client_progress
        self.client_usage = client_usage
        self.lms = lms
        self.kind = kind
        self.finish_reason = ""
        self.timings: dict[str, Any] | None = None
        self.usage: dict[str, Any] | None = None
        self.error: Any = None
        self.meta: dict[str, Any] = {}
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.logprobs: list[Any] = []

    def process(self, raw: bytes) -> bytes | None:
        name, data = parse_event(raw)
        if data is None:
            return raw  # comment / ping
        if data.strip() == "[DONE]":
            return raw
        try:
            obj = json.loads(data)
        except ValueError:
            return raw
        if not isinstance(obj, dict):
            return raw
        modified = False
        if "error" in obj and not obj.get("choices"):
            self.error = obj["error"]
            return raw
        for k in ("id", "created", "model", "system_fingerprint", "object"):
            if k in obj and k not in self.meta:
                self.meta[k] = obj[k]
        pp = obj.get("prompt_progress")
        if pp is not None:
            try:
                self.ctx.tracker.progress(self.rec, int(pp.get("total", 0)), int(pp.get("processed", 0)),
                                          int(pp.get("cache", 0)))
            except (TypeError, ValueError):
                pass
            if not self.client_progress:
                del obj["prompt_progress"]
                modified = True
        choices = obj.get("choices") or []
        has_payload = False
        for ch in choices:
            delta = ch.get("delta") or {}
            if self.kind == "completion":
                text = ch.get("text")
                if text:
                    has_payload = True
                    self.ctx.tracker.tokens(self.rec, text, "c")
                    self.content.append(text)
            else:
                c = delta.get("content")
                if c:
                    has_payload = True
                    self.ctx.tracker.tokens(self.rec, c, "c")
                    self.content.append(c)
                r = delta.get("reasoning_content") or delta.get("reasoning")
                if r:
                    has_payload = True
                    self.ctx.tracker.tokens(self.rec, r, "r")
                    self.reasoning.append(r)
                for tc in delta.get("tool_calls") or []:
                    has_payload = True
                    idx = int(tc.get("index", 0))
                    slot = self.tool_calls.setdefault(idx, {"id": "", "type": "function",
                                                            "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    if tc.get("type"):
                        slot["type"] = tc["type"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                        self.rec.tool_calls.append(fn["name"])
                        self.ctx.tracker.tokens(self.rec, f"⟦{fn['name']}⟧", "t")
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                        self.ctx.tracker.tokens(self.rec, fn["arguments"], "t")
            lp = ch.get("logprobs")
            if lp and isinstance(lp, dict) and lp.get("content"):
                self.logprobs.extend(lp["content"])
            if ch.get("finish_reason"):
                has_payload = True
                self.finish_reason = ch["finish_reason"]
        if "timings" in obj:
            self.timings = obj["timings"]
            has_payload = True
        if "usage" in obj and obj["usage"]:
            self.usage = obj["usage"]
            if not choices and not self.client_usage:
                return None
        if pp is not None and not self.client_progress and not has_payload:
            return None  # progress-only chunk (the engine sends its own role chunk afterwards)
        if self.lms is not None and self.finish_reason and any(ch.get("finish_reason") for ch in choices):
            obj.update(lms_stats(self.lms, self.rec, self.finish_reason, self.timings))
            modified = True
        if modified:
            prefix = f"event: {name}\n".encode() if name else b""
            return prefix + b"data: " + _dumps(obj)
        return raw

    def aggregate(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(self.content) if self.content else None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        elif msg["content"] is None:
            msg["content"] = ""
        choice: dict[str, Any] = {"finish_reason": self.finish_reason or "stop", "index": 0, "message": msg}
        if self.logprobs:
            choice["logprobs"] = {"content": self.logprobs}
        out: dict[str, Any] = {
            "choices": [choice],
            "created": self.meta.get("created", int(time.time())),
            "model": self.meta.get("model", ""),
            "system_fingerprint": self.meta.get("system_fingerprint", ""),
            "object": "chat.completion",
            "usage": self.usage or {},
            "id": self.meta.get("id", ""),
        }
        if self.timings:
            out["timings"] = self.timings
        return out


def lms_stats(lms: dict[str, Any], rec: RequestRecord, finish: str, timings: dict | None) -> dict[str, Any]:
    t = timings or {}
    ttft = (rec.t_first_token - rec.t_start) if rec.t_first_token else None
    gen_s = (t.get("predicted_ms") or 0) / 1000 if t else None
    return {
        "stats": {
            "tokens_per_second": round(t.get("predicted_per_second") or rec.gen_tps or 0.0, 3),
            "time_to_first_token": round(ttft, 3) if ttft is not None else None,
            "generation_time": round(gen_s, 3) if gen_s is not None else None,
            "stop_reason": LMS_STOP_REASONS.get(finish, "userStopped"),
        },
        "model_info": lms["model_info"],
        "runtime": lms["runtime"],
    }


def lms_context(inst: EngineInstance) -> dict[str, Any]:
    e = inst.entry
    return {
        "model_info": {"arch": e.info.architecture, "quant": e.info.quant, "format": "gguf",
                       "context_length": inst.plan.ctx},
        "runtime": {"name": f"llama.cpp-{inst.engine.backend}", "version": str(inst.engine.build or
                                                                                  inst.engine.version),
                    "supported_formats": ["gguf"]},
    }


class OpenAIRouter:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.router = APIRouter()
        r = self.router
        for prefix, lms in (("/v1", False), ("/api/v0", True)):
            r.add_api_route(f"{prefix}/models", self._models_handler(lms), methods=["GET"])
            r.add_api_route(prefix + "/models/{model_id:path}", self._model_handler(lms), methods=["GET"])
            r.add_api_route(f"{prefix}/chat/completions", self._chat_handler(lms), methods=["POST"])
            r.add_api_route(f"{prefix}/completions", self._completion_handler(lms), methods=["POST"])
            r.add_api_route(f"{prefix}/embeddings", self._passthrough_handler("/v1/embeddings", "embeddings"),
                            methods=["POST"])
        r.add_api_route("/v1/responses", self._passthrough_handler("/v1/responses", "responses"), methods=["POST"])
        r.add_api_route("/v1/messages", self._passthrough_handler("/v1/messages", "anthropic"), methods=["POST"])
        r.add_api_route("/v1/messages/count_tokens",
                        self._passthrough_handler("/v1/messages/count_tokens", "count"), methods=["POST"])
        r.add_api_route("/v1/rerank", self._passthrough_handler("/v1/rerank", "rerank"), methods=["POST"])
        r.add_api_route("/v1/reranking", self._passthrough_handler("/v1/rerank", "rerank"), methods=["POST"])
        r.add_api_route("/v1/health", self._health, methods=["GET"])
        r.add_api_route("/v1", self._root, methods=["GET"])

    # ----- guards -----------------------------------------------------------------------

    def _guard(self, request: Request) -> JSONResponse | None:
        s = self.ctx.store.settings.server
        if not s.api_enabled:
            return oai_error(503, "The WinRunner API server is stopped. Start it from the control panel.",
                             "api_disabled")
        if s.api_key:
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
            if token != s.api_key and not (is_loopback(_client(request)) and request.headers.get("x-winrunner-ui")):
                return oai_error(401, "Invalid or missing API key.", "invalid_api_key", "authentication_error")
        return None

    async def _body(self, request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise ModelError(f"Request body is not valid JSON: {exc}", 400, "invalid_json") from exc
        if not isinstance(body, dict):
            raise ModelError("Request body must be a JSON object.", 400, "invalid_json")
        return body

    # ----- model listing -------------------------------------------------------------------

    def _model_objects(self, lms: bool) -> list[dict[str, Any]]:
        ctx = self.ctx
        loaded = {i.model_id: i for i in ctx.manager.instances.values() if i.state in ("ready", "loading")}
        entries = ctx.library.entries()
        if not ctx.store.settings.server.jit_loading:
            entries = [e for e in entries if e.id in loaded]
        out = []
        for e in entries:
            if lms:
                inst = loaded.get(e.id)
                o: dict[str, Any] = {
                    "id": e.id,
                    "object": "model",
                    "type": e.lms_type,
                    "publisher": e.publisher or "local",
                    "arch": e.info.architecture,
                    "compatibility_type": "gguf",
                    "quantization": e.info.quant,
                    "state": "loaded" if inst and inst.state == "ready" else "not-loaded",
                    "max_context_length": e.info.context_length,
                }
                if inst:
                    o["loaded_context_length"] = inst.plan.ctx
                caps = []
                if e.template.get("tools"):
                    caps.append("tool_use")
                if e.has_vision:
                    caps.append("vision")
                o["capabilities"] = caps
                out.append(o)
            else:
                out.append({"id": e.id, "object": "model", "created": int(e.info.mtime),
                            "owned_by": e.publisher or "winrunner"})
        return out

    def _models_handler(self, lms: bool):
        async def handler(request: Request) -> Response:
            if (g := self._guard(request)) is not None:
                return g
            return JSONResponse({"object": "list", "data": self._model_objects(lms)})

        return handler

    def _model_handler(self, lms: bool):
        async def handler(request: Request, model_id: str) -> Response:
            if (g := self._guard(request)) is not None:
                return g
            e = self.ctx.library.resolve(model_id)
            if e is None:
                return oai_error(404, f"Model '{model_id}' not found.", "model_not_found")
            for o in self._model_objects(lms) or []:
                if o["id"] == e.id:
                    return JSONResponse(o)
            return JSONResponse({"id": e.id, "object": "model", "created": int(e.info.mtime),
                                 "owned_by": e.publisher or "winrunner"})

        return handler

    async def _health(self, request: Request) -> Response:
        ready = self.ctx.manager.ready_instances()
        return JSONResponse({"status": "ok" if ready else "no_model", "models": [i.model_id for i in ready]})

    async def _root(self, request: Request) -> Response:
        return JSONResponse({"object": "winrunner", "api": "OpenAI / LM Studio compatible",
                             "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions",
                                           "/v1/embeddings", "/v1/responses", "/v1/messages", "/api/v0/models"]})

    # ----- instance acquisition ----------------------------------------------------------------

    async def _acquire(self, requested: str | None, rec: RequestRecord) -> AsyncIterator[bytes | EngineInstance]:
        """Yield SSE keep-alive comments while waiting for a model, then the instance."""
        ctx = self.ctx
        jit = ctx.store.settings.server.jit_loading
        inst = ctx.manager.ready_for(requested, jit)
        if inst is not None:
            yield inst
            return
        ctx.tracker.update(rec, phase="loading_model")
        task = asyncio.create_task(ctx.manager.instance_for_request(requested, jit))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=2.0)
                if done:
                    yield task.result()
                    return
                prog = ""
                for i in ctx.manager.instances.values():
                    if i.state in ("loading", "starting"):
                        prog = f" {i.model_id} {i.progress * 100:.0f}% ({i.phase})"
                yield f": winrunner loading model{prog}\n\n".encode()
        finally:
            if not task.done():
                # the client went away; let the load finish in the background
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _instance(self, requested: str | None, rec: RequestRecord) -> EngineInstance:
        inst = None
        async for item in self._acquire(requested, rec):
            if isinstance(item, EngineInstance):
                inst = item
        assert inst is not None
        return inst

    def _prepare_common(self, body: dict[str, Any], inst: EngineInstance) -> None:
        for k in STRIP_FIELDS:
            body.pop(k, None)
        mt = body.get("max_tokens")
        if mt is None and body.get("max_completion_tokens") is not None:
            body["max_tokens"] = body.pop("max_completion_tokens")
            mt = body["max_tokens"]
        if isinstance(mt, (int, float)) and mt <= 0:
            body.pop("max_tokens", None)
        prof = self.ctx.store.profile(inst.entry.path)
        for k, v in (prof.sampling or {}).items():
            if k.startswith("_") or v is None or k in body:
                continue
            if k == "max_tokens" and (not isinstance(v, (int, float)) or v <= 0):
                continue
            body[k] = v
        body["model"] = inst.model_id

    # ----- chat / completions ------------------------------------------------------------------

    def _chat_handler(self, lms: bool):
        async def handler(request: Request) -> Response:
            return await self._generate(request, "chat", lms)

        return handler

    def _completion_handler(self, lms: bool):
        async def handler(request: Request) -> Response:
            return await self._generate(request, "completion", lms)

        return handler

    async def _generate(self, request: Request, kind: str, lms: bool) -> Response:
        if (g := self._guard(request)) is not None:
            return g
        ctx = self.ctx
        path = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
        endpoint = ("/api/v0" if lms else "/v1") + path[3:]
        try:
            body = await self._body(request)
        except ModelError as exc:
            return oai_error(exc.status, str(exc), exc.code)
        requested = body.get("model")
        client_stream = bool(body.get("stream"))
        params = {k: body.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "seed")
                  if body.get(k) is not None}
        if body.get("tools"):
            params["tools"] = len(body["tools"])
        rec = ctx.tracker.begin(_client(request), request.headers.get("user-agent", ""), endpoint,
                                str(requested or ""), client_stream, 0, params)
        stats = VisionStats()
        try:
            if kind == "chat":
                if not isinstance(body.get("messages"), list) or not body["messages"]:
                    raise ModelError("'messages' must be a non-empty array.", 400, "invalid_messages")
                stats = await ctx.normalizer().normalize_chat(body["messages"])
                rec.images = stats.images
                if stats.converted:
                    ctx.bus.activity_log(f"{rec.id}: image input normalised ({'; '.join(stats.converted)})",
                                         category="request")
            elif "prompt" not in body:
                raise ModelError("'prompt' is required.", 400, "invalid_prompt")
        except ImageError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            return oai_error(400, f"Invalid image input: {exc}", "invalid_image")
        except ModelError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            return oai_error(exc.status, str(exc), exc.code)

        try:
            ready = ctx.manager.ready_for(requested, ctx.store.settings.server.jit_loading)
        except ModelError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            return oai_error(exc.status, str(exc), exc.code)

        client_progress = bool(body.get("return_progress"))
        so = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        client_usage = bool(so.get("include_usage"))
        aggregate = (not client_stream and kind == "chat" and ctx.store.settings.server.internal_stream_aggregation
                     and body.get("n", 1) in (None, 1) and not body.get("logprobs"))

        async def upstream_body(inst: EngineInstance) -> dict[str, Any]:
            if stats.images and not inst.vision_enabled:
                raise ModelError(
                    f"Model '{inst.model_id}' does not accept image input (no vision projector / mmproj loaded). "
                    "Pair an mmproj file with this model in WinRunner's library.", 400, "model_not_vision_capable")
            b = dict(body)
            self._prepare_common(b, inst)
            if client_stream or aggregate:
                b["stream"] = True
                b["return_progress"] = True
                b["stream_options"] = {**so, "include_usage": True}
            return b

        if ready is not None and stats.images and not ready.vision_enabled:
            msg = (f"Model '{ready.model_id}' does not accept image input (no vision projector / mmproj loaded). "
                   "Pair an mmproj file with this model in WinRunner's library.")
            ctx.tracker.finish(rec, error=msg)
            return oai_error(400, msg, "model_not_vision_capable")

        if client_stream:
            return StreamingResponse(
                self._stream(request, rec, path, kind, lms, requested, upstream_body, client_progress, client_usage,
                             ready),
                media_type="text/event-stream", headers=SSE_HEADERS)

        # non-streaming
        try:
            inst = ready or await self._instance(requested, rec)
            b = await upstream_body(inst)
        except ModelError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            return oai_error(exc.status, str(exc), exc.code)
        rec.instance = inst.id
        rec.model = inst.model_id
        inst.active_requests += 1
        inst.last_used = time.time()
        try:
            if aggregate:
                tap = ChatTap(ctx, rec, False, True, None, kind)
                result = await _until_disconnect(request, self._consume(inst, path, b, tap))
                if result is None:
                    ctx.tracker.finish(rec, cancelled=True)
                    return Response(status_code=499)
                status, err_body = result
                if status != 200:
                    ctx.tracker.finish(rec, error=_err_text(err_body))
                    return Response(err_body, status_code=status, media_type="application/json")
                if tap.error:
                    ctx.tracker.finish(rec, error=_err_text(tap.error))
                    return JSONResponse({"error": tap.error}, status_code=_err_status(tap.error))
                out = tap.aggregate()
                if lms:
                    out.update(lms_stats(lms_context(inst), rec, tap.finish_reason, tap.timings))
                ctx.tracker.finish(rec, tap.finish_reason, tap.timings, tap.usage)
                inst.requests_served += 1
                return JSONResponse(out)
            result = await _until_disconnect(request, inst.client.post(path, json=b))  # type: ignore[union-attr]
            if result is None:
                ctx.tracker.finish(rec, cancelled=True)
                return Response(status_code=499)
            r: httpx.Response = result
            if r.status_code != 200:
                ctx.tracker.finish(rec, error=_err_text(r.content))
                return Response(r.content, status_code=r.status_code, media_type="application/json")
            data = r.json()
            ch = (data.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content") or ch.get("text") or ""
            if text:
                ctx.tracker.tokens(rec, text, "c")
            if lms:
                data.update(lms_stats(lms_context(inst), rec, ch.get("finish_reason", ""), data.get("timings")))
            ctx.tracker.finish(rec, ch.get("finish_reason", ""), data.get("timings"), data.get("usage"))
            inst.requests_served += 1
            return JSONResponse(data)
        except httpx.HTTPError as exc:
            ctx.tracker.finish(rec, error=f"engine connection error: {exc}")
            return oai_error(502, f"Engine connection error: {exc}", "engine_unavailable")
        finally:
            inst.active_requests -= 1
            inst.last_used = time.time()

    async def _consume(self, inst: EngineInstance, path: str, body: dict[str, Any],
                       tap: ChatTap) -> tuple[int, bytes]:
        assert inst.client is not None
        async with inst.client.stream("POST", path, json=body) as resp:
            if resp.status_code != 200:
                return resp.status_code, await resp.aread()
            parser = SSEParser()
            async for chunk in resp.aiter_raw():
                for ev in parser.feed(chunk):
                    tap.process(ev)
            for ev in parser.flush():
                tap.process(ev)
        return 200, b""

    async def _stream(self, request: Request, rec: RequestRecord, path: str, kind: str, lms: bool,
                      requested: str | None, upstream_body, client_progress: bool, client_usage: bool,
                      ready: EngineInstance | None) -> AsyncIterator[bytes]:
        ctx = self.ctx
        inst: EngineInstance | None = ready
        counted = False
        tap: ChatTap | None = None
        try:
            if inst is None:
                async for item in self._acquire(requested, rec):
                    if isinstance(item, EngineInstance):
                        inst = item
                    else:
                        yield item
            assert inst is not None
            b = await upstream_body(inst)
            rec.instance, rec.model = inst.id, inst.model_id
            inst.active_requests += 1
            inst.last_used = time.time()
            counted = True
            tap = ChatTap(ctx, rec, client_progress, client_usage, lms_context(inst) if lms else None, kind)
            assert inst.client is not None
            async with inst.client.stream("POST", path, json=b) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    ctx.tracker.finish(rec, error=_err_text(err))
                    yield b"data: " + _error_payload(err, resp.status_code) + b"\n\n"
                    return
                parser = SSEParser()
                async for chunk in resp.aiter_raw():
                    for ev in parser.feed(chunk):
                        out = tap.process(ev)
                        if out is not None:
                            yield out + b"\n\n"
                for ev in parser.flush():
                    out = tap.process(ev)
                    if out is not None:
                        yield out + b"\n\n"
            if tap.error:
                ctx.tracker.finish(rec, error=_err_text(tap.error))
            else:
                ctx.tracker.finish(rec, tap.finish_reason, tap.timings, tap.usage)
                inst.requests_served += 1
        except ModelError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            yield b"data: " + _dumps({"error": {"message": str(exc), "type": "invalid_request_error",
                                                 "code": exc.code}}) + b"\n\n"
        except ImageError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            yield b"data: " + _dumps({"error": {"message": str(exc), "type": "invalid_request_error",
                                                 "code": "invalid_image"}}) + b"\n\n"
        except httpx.HTTPError as exc:
            ctx.tracker.finish(rec, error=f"engine connection error: {exc}")
            yield b"data: " + _dumps({"error": {"message": f"Engine connection error: {exc}",
                                                 "type": "server_error"}}) + b"\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            if rec.t_end is None:
                ctx.tracker.finish(rec, tap.finish_reason if tap else "", tap.timings if tap else None,
                                   cancelled=True)
            raise
        finally:
            if counted and inst is not None:
                inst.active_requests -= 1
                inst.last_used = time.time()

    # ----- other endpoints (pass-through with telemetry) ---------------------------------------

    def _passthrough_handler(self, path: str, kind: str):
        async def handler(request: Request) -> Response:
            return await self._passthrough(request, path, kind)

        return handler

    async def _passthrough(self, request: Request, path: str, kind: str) -> Response:
        if (g := self._guard(request)) is not None:
            return g
        ctx = self.ctx
        try:
            body = await self._body(request)
        except ModelError as exc:
            return oai_error(exc.status, str(exc), exc.code)
        requested = body.get("model")
        stream = bool(body.get("stream"))
        rec = ctx.tracker.begin(_client(request), request.headers.get("user-agent", ""), request.url.path,
                                str(requested or ""), stream, 0)
        try:
            if kind == "responses":
                st = await ctx.normalizer().normalize_responses(body)
                rec.images = st.images
            elif kind in ("anthropic", "count"):
                st = await ctx.normalizer().normalize_anthropic(body)
                rec.images = st.images
            else:
                st = VisionStats()
            ready = ctx.manager.ready_for(requested, ctx.store.settings.server.jit_loading)
            inst = ready or await self._instance(requested, rec)
            if st.images and not inst.vision_enabled:
                raise ModelError(f"Model '{inst.model_id}' does not accept image input (no mmproj loaded).", 400,
                                 "model_not_vision_capable")
        except ImageError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            return oai_error(400, f"Invalid image input: {exc}", "invalid_image")
        except ModelError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            return oai_error(exc.status, str(exc), exc.code)
        for k in STRIP_FIELDS:
            body.pop(k, None)
        body["model"] = inst.model_id
        rec.instance, rec.model = inst.id, inst.model_id
        assert inst.client is not None

        if stream and kind in ("responses", "anthropic"):
            async def gen() -> AsyncIterator[bytes]:
                inst.active_requests += 1
                usage: dict[str, Any] = {}
                finish = ""
                try:
                    async with inst.client.stream("POST", path, json=body) as resp:  # type: ignore[union-attr]
                        if resp.status_code != 200:
                            err = await resp.aread()
                            ctx.tracker.finish(rec, error=_err_text(err))
                            yield b"data: " + _error_payload(err, resp.status_code) + b"\n\n"
                            return
                        parser = SSEParser()
                        async for chunk in resp.aiter_raw():
                            for ev in parser.feed(chunk):
                                u, f = _tap_named_event(ctx, rec, ev, kind)
                                usage.update(u)
                                finish = f or finish
                                yield ev + b"\n\n"
                        for ev in parser.flush():
                            yield ev + b"\n\n"
                    ctx.tracker.finish(rec, finish, usage=_norm_usage(usage))
                    inst.requests_served += 1
                except (asyncio.CancelledError, GeneratorExit):
                    if rec.t_end is None:
                        ctx.tracker.finish(rec, cancelled=True)
                    raise
                except httpx.HTTPError as exc:
                    ctx.tracker.finish(rec, error=str(exc))
                finally:
                    inst.active_requests -= 1
                    inst.last_used = time.time()

            return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

        inst.active_requests += 1
        try:
            result = await _until_disconnect(request, inst.client.post(path, json=body))
            if result is None:
                ctx.tracker.finish(rec, cancelled=True)
                return Response(status_code=499)
            r: httpx.Response = result
            if r.status_code != 200:
                ctx.tracker.finish(rec, error=_err_text(r.content))
            else:
                try:
                    data = r.json()
                except ValueError:
                    data = {}
                ctx.tracker.finish(rec, "stop", usage=_norm_usage(data.get("usage") or {}) if isinstance(data, dict)
                                   else None)
                inst.requests_served += 1
            return Response(r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type", "application/json"))
        except httpx.HTTPError as exc:
            ctx.tracker.finish(rec, error=str(exc))
            return oai_error(502, f"Engine connection error: {exc}", "engine_unavailable")
        finally:
            inst.active_requests -= 1
            inst.last_used = time.time()


def _tap_named_event(ctx: AppContext, rec: RequestRecord, raw: bytes, kind: str) -> tuple[dict, str]:
    name, data = parse_event(raw)
    if not data:
        return {}, ""
    try:
        obj = json.loads(data)
    except ValueError:
        return {}, ""
    if not isinstance(obj, dict):
        return {}, ""
    t = obj.get("type") or name or ""
    usage: dict[str, Any] = {}
    finish = ""
    if kind == "responses":
        if t == "response.output_text.delta":
            ctx.tracker.tokens(rec, obj.get("delta", ""), "c")
        elif t in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            ctx.tracker.tokens(rec, obj.get("delta", ""), "r")
        elif t == "response.function_call_arguments.delta":
            ctx.tracker.tokens(rec, obj.get("delta", ""), "t")
        elif t == "response.completed":
            usage = (obj.get("response") or {}).get("usage") or {}
            finish = "stop"
    else:
        if t == "content_block_delta":
            d = obj.get("delta") or {}
            if d.get("type") == "text_delta":
                ctx.tracker.tokens(rec, d.get("text", ""), "c")
            elif d.get("type") == "thinking_delta":
                ctx.tracker.tokens(rec, d.get("thinking", ""), "r")
            elif d.get("type") == "input_json_delta":
                ctx.tracker.tokens(rec, d.get("partial_json", ""), "t")
        elif t == "message_delta":
            usage = obj.get("usage") or {}
            finish = (obj.get("delta") or {}).get("stop_reason") or ""
        elif t == "message_start":
            usage = ((obj.get("message") or {}).get("usage")) or {}
    if "prompt_progress" in obj:
        pp = obj["prompt_progress"]
        ctx.tracker.progress(rec, int(pp.get("total", 0)), int(pp.get("processed", 0)), int(pp.get("cache", 0)))
    return usage, finish


def _norm_usage(u: dict[str, Any]) -> dict[str, Any]:
    out = {}
    pt = u.get("prompt_tokens", u.get("input_tokens"))
    ct = u.get("completion_tokens", u.get("output_tokens"))
    if pt is not None:
        out["prompt_tokens"] = pt
    if ct is not None:
        out["completion_tokens"] = ct
    return out


def _err_text(err: Any) -> str:
    if isinstance(err, (bytes, bytearray)):
        try:
            err = json.loads(err)
        except ValueError:
            return err.decode("utf-8", errors="replace")[:500]
    if isinstance(err, dict):
        e = err.get("error", err)
        if isinstance(e, dict):
            return str(e.get("message") or e)[:500]
        return str(e)[:500]
    return str(err)[:500]


def _err_status(err: Any) -> int:
    if isinstance(err, dict):
        c = err.get("code")
        if isinstance(c, int) and 400 <= c < 600:
            return c
    return 500


def _error_payload(err: bytes, status: int) -> bytes:
    try:
        obj = json.loads(err)
        if isinstance(obj, dict) and "error" in obj:
            return _dumps(obj)
    except ValueError:
        pass
    return _dumps({"error": {"message": err.decode("utf-8", errors="replace")[:500], "code": status,
                             "type": "server_error"}})


async def _until_disconnect(request: Request, coro) -> Any:
    """Run ``coro`` but cancel it if the HTTP client disconnects (frees the engine slot)."""
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=1.0)
            if done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                return None
    finally:
        if not task.done():
            task.cancel()
