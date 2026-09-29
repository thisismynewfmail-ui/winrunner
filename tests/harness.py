"""Helpers for tests that run WinRunner against the scripted fake llama-server (tests/fake_engine.py).

The ``winrunner`` and ``state`` fixtures live in ``tests/conftest.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

CHAT = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}


def wait_for(cond, timeout: float = 30.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = cond()
        except Exception:
            value = None
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def starts(state: Path) -> int:
    try:
        return int((state / "starts").read_text())
    except OSError:
        return 0


def activity(ctx) -> str:
    return "\n".join(a["text"] for a in list(ctx.bus.activity))


def ready_instance(ctx):
    return next((i for i in list(ctx.manager.instances.values()) if i.state == "ready"), None)


def failed_instance(ctx):
    return next((i for i in list(ctx.manager.instances.values()) if i.state == "error" and i.recovering), None)


def model_id(ctx, name: str = "test-7b") -> str:
    return next(e.id for e in ctx.library.entries() if name in e.id)


def load(base: str, ctx, name: str = "test-7b") -> None:
    r = httpx.post(base + "/wr/api/models/load", json={"id": model_id(ctx, name)})
    assert r.status_code == 202
    wait_for(lambda: (i := ready_instance(ctx)) and i.model_id == model_id(ctx, name), what="model load")


def chat(base: str, ctx, **extra) -> httpx.Response:
    return httpx.post(base + "/v1/chat/completions", json={**CHAT, "model": model_id(ctx), **extra}, timeout=60)


def stream_lines(base: str, ctx) -> list[str]:
    with httpx.stream("POST", base + "/v1/chat/completions",
                      json={**CHAT, "model": model_id(ctx), "stream": True}, timeout=60) as s:
        return [ln for ln in s.iter_lines() if ln]
