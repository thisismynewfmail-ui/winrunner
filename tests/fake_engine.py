"""A stand-in for llama-server used by the engine recovery tests.

It answers the probes WinRunner runs (``--version``, ``--help``, ``--list-devices``)
and serves the HTTP endpoints WinRunner uses. What each server process does is
scripted through environment variables:

``FAKE_ENGINE_PLAN``   comma separated modes, one per server start (the last one repeats)
``FAKE_ENGINE_STATE``  directory for the start counter (``starts``)
``FAKE_ENGINE_ERRORS`` ``data`` (``data: {"error": ...}``, default) or ``legacy`` (``error: {...}``)
``FAKE_ENGINE_RESET_POLLS`` after an abort, ``--list-devices`` reports no GPU this many times (a GPU reset)

Modes:
  ok                        answers every request
  device_lost               like a llama.cpp engine whose Vulkan device was lost: every generation request fails
                            with "vk::Queue::submit: ErrorDeviceLost" while /health still reports ok
  device_lost_after_tokens  streams two tokens, then fails as above (and keeps failing)
  abort_on_request          aborts (like llama.cpp on Windows, exit code 0xC0000409) when a request arrives
  crash_when_idle           aborts shortly after it became ready, without any request
  log_fault_when_idle       logs a device-lost error record shortly after it became ready and keeps running
  debug_mention             logs a debug record and an unprefixed line quoting "ErrorDeviceLost" (prompt echo)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HELP = """
-m,    --model FNAME                    model path
--host HOST                             ip address to listen, or bind to an UNIX socket
--port PORT                             port to listen (default: 8080)
-a,    --alias STRING                   set model name aliases, comma-separated (to be used by API)
--api-key KEY                           API key to use for authentication
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM, either an exact number,
                                        'auto', or 'all' (default: auto)
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: enabled)
-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold.
--log-jsonl, --no-log-jsonl             Log as JSONL
"""

DEVICE_LOST = "decode() failed: vk::Queue::submit: ErrorDeviceLost"
MODE = "ok"
JSONL = False
STATE = {"broken": False, "ready": False}
LOCK = threading.Lock()


def log(level: str, msg: str) -> None:
    with LOCK:
        if JSONL:
            print(json.dumps({"level": level, "msg": msg}), flush=True)
        else:
            print(f"0.00.100.000 {level[0].upper()} {msg}", flush=True)


def abort() -> None:
    polls = os.environ.get("FAKE_ENGINE_RESET_POLLS")
    if polls:
        with open(os.path.join(os.environ["FAKE_ENGINE_STATE"], "devices_gone"), "w", encoding="utf-8") as f:
            f.write(polls)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # no core dump
    except (ImportError, ValueError, OSError):
        pass
    os.abort()


def next_start() -> tuple[int, str]:
    path = os.path.join(os.environ["FAKE_ENGINE_STATE"], "starts")
    try:
        with open(path, encoding="utf-8") as f:
            n = int(f.read() or 0)
    except OSError:
        n = 0
    n += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(n))
    modes = [m.strip() for m in os.environ.get("FAKE_ENGINE_PLAN", "ok").split(",") if m.strip()] or ["ok"]
    return n, modes[min(n, len(modes)) - 1]


def on_ready() -> None:
    """Called when WinRunner reads /props, i.e. right before it marks the instance ready."""
    if STATE["ready"]:
        return
    STATE["ready"] = True
    if MODE == "crash_when_idle":
        threading.Timer(0.5, abort).start()
    elif MODE == "log_fault_when_idle":
        def fault() -> None:
            STATE["broken"] = True
            log("error", f"srv  update_slots: failed to decode the batch: {DEVICE_LOST}")

        threading.Timer(0.3, fault).start()
    elif MODE == "debug_mention":
        def mention() -> None:
            log("debug", 'srv  log_server_r: request:  {"messages":[{"role":"user","content":"' + DEVICE_LOST + '"}]}')
            with LOCK:
                print(f"Error Log: 15:02:39 req-34933029 failed: {DEVICE_LOST}", flush=True)
            log("info", "srv  update_slots: all slots are idle")

        threading.Timer(0.2, mention).start()


def chunk(model: str, **fields) -> dict:
    return {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 1, "model": model,
            "system_fingerprint": "fake", **fields}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _event(self, obj) -> None:
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def _stream_error(self) -> None:
        err = {"code": 500, "message": DEVICE_LOST, "type": "server_error"}
        if os.environ.get("FAKE_ENGINE_ERRORS") == "legacy":
            self.wfile.write(b"error: " + json.dumps(err).encode() + b"\n\n")
        else:
            self.wfile.write(b"data: " + json.dumps({"error": err}).encode() + b"\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/props":
            self._json(200, {"chat_template": "", "total_slots": 1, "modalities": {"vision": False},
                             "default_generation_settings": {"n_ctx": 4096, "params": {}}})
            on_ready()
        elif self.path == "/slots":
            self._json(200, [])
        else:
            self._json(404, {"error": {"code": 404, "message": "not found", "type": "not_found_error"}})

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._json(404, {"error": {"code": 404, "message": "not found", "type": "not_found_error"}})
            return
        if MODE == "abort_on_request":
            abort()
        model = body.get("model", "")
        if not body.get("stream"):
            if STATE["broken"] or MODE == "device_lost":
                STATE["broken"] = True
                self._json(500, {"error": {"code": 500, "message": DEVICE_LOST, "type": "server_error"}})
                return
            self._json(200, {"id": "cmpl-fake", "object": "text_completion", "created": 1, "model": model,
                             "choices": [{"index": 0, "text": "Hello", "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._event(chunk(model, choices=[{"index": 0, "delta": {"role": "assistant", "content": None},
                                           "finish_reason": None}],
                          prompt_progress={"total": 10, "cache": 0, "processed": 10, "time_ms": 1}))
        if STATE["broken"] or MODE == "device_lost":
            STATE["broken"] = True
            self._stream_error()
            return
        self._event(chunk(model, choices=[{"index": 0, "delta": {"role": "assistant", "content": None},
                                           "finish_reason": None}]))
        for piece in ("Hel", "lo"):
            self._event(chunk(model, choices=[{"index": 0, "delta": {"content": piece}, "finish_reason": None}]))
        if MODE == "device_lost_after_tokens":
            STATE["broken"] = True
            self._stream_error()
            return
        self._event(chunk(model, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                          timings={"prompt_n": 10, "predicted_n": 2, "prompt_ms": 20.0, "predicted_ms": 40.0,
                                   "prompt_per_second": 500.0, "predicted_per_second": 50.0}))
        self._event(chunk(model, choices=[], usage={"completion_tokens": 2, "prompt_tokens": 10, "total_tokens": 12}))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main(argv: list[str]) -> int:
    global MODE, JSONL
    if "--version" in argv:
        print("version: 9999 (abcdef1)")
        return 0
    if "--help" in argv:
        print(HELP)
        return 0
    if "--list-devices" in argv:
        gone = os.path.join(os.environ.get("FAKE_ENGINE_STATE", "."), "devices_gone")
        try:
            with open(gone, encoding="utf-8") as f:
                left = int(f.read() or 0)
        except OSError:
            left = 0
        if left > 0:  # the driver is still resetting the GPU
            with open(gone, "w", encoding="utf-8") as f:
                f.write(str(left - 1))
            print("Available devices:")
            return 0
        print("Available devices:\n  Vulkan0: Fake GPU (16368 MiB, 15000 MiB free)")
        return 0
    port = int(argv[argv.index("--port") + 1])
    JSONL = "--log-jsonl" in argv
    n, MODE = next_start()
    log("info", f"main: fake engine start {n}, mode {MODE}")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    log("info", "srv  llama_server: model loaded")
    log("info", f"main: server is listening on http://127.0.0.1:{port}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
