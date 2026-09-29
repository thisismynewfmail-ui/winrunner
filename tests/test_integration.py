"""End-to-end test against a real llama-server binary.

Skipped unless these environment variables are set:
  WINRUNNER_TEST_ENGINE   path to llama-server(.exe)
  WINRUNNER_TEST_MODELS   folder containing Qwen3-0.6B (any quant) and optionally SmolVLM-256M-Instruct + mmproj
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest

ENGINE = os.environ.get("WINRUNNER_TEST_ENGINE")
MODELS = os.environ.get("WINRUNNER_TEST_MODELS")
pytestmark = pytest.mark.skipif(not (ENGINE and MODELS), reason="real engine / models not configured")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    from winrunner.app import create_app
    from winrunner.paths import DataPaths
    from winrunner.util import free_port

    data = tmp_path_factory.mktemp("data")
    (data / "settings.json").write_text(json.dumps({
        "library": {"model_dirs": [MODELS]},
        "engine": {"engine_path": ENGINE, "backend": "custom"},
        "defaults": {"context_length": 4096},
        "startup": {"open_ui": "none"},
    }))
    app, ctx = create_app(DataPaths(data))
    port = free_port()
    ctx.extras.update(bound_host="127.0.0.1", bound_port=port)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/health").status_code == 200 and ctx.library.entries():
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    yield base, ctx
    srv.should_exit = True
    t.join(timeout=30)


def _model(ctx, needle: str) -> str | None:
    return next((e.id for e in ctx.library.entries() if needle in e.id), None)


def test_models_listing(server):
    base, ctx = server
    r = httpx.get(base + "/v1/models").json()
    assert r["object"] == "list" and r["data"]
    v0 = httpx.get(base + "/api/v0/models").json()
    assert all(m["compatibility_type"] == "gguf" and m["state"] in ("loaded", "not-loaded") for m in v0["data"])


def test_jit_chat_stream_and_aggregate(server):
    base, ctx = server
    mid = _model(ctx, "qwen3")
    if not mid:
        pytest.skip("Qwen3 test model not present")
    body = {"model": mid, "messages": [{"role": "user", "content": "Say OK. /no_think"}], "max_tokens": 16, "temperature": 0}
    r = httpx.post(base + "/v1/chat/completions", json=body, timeout=300).json()
    assert r["object"] == "chat.completion" and r["model"] == mid
    assert r["choices"][0]["message"]["role"] == "assistant"
    assert r["usage"]["completion_tokens"] > 0
    with httpx.stream("POST", base + "/v1/chat/completions", json={**body, "stream": True}, timeout=120) as s:
        lines = [ln for ln in s.iter_lines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    assert not any("prompt_progress" in ln for ln in lines)
    inst = ctx.manager.instance_for_model(mid)
    assert inst.template_verified is True  # engine uses the GGUF's embedded template
    lms = httpx.post(base + "/api/v0/chat/completions", json=body, timeout=120).json()
    assert lms["stats"]["tokens_per_second"] > 0 and lms["model_info"]["format"] == "gguf"


def test_vision_webp_via_jit_swap(server):
    base, ctx = server
    mid = _model(ctx, "smolvlm")
    if not mid:
        pytest.skip("SmolVLM test model not present")
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 240), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([40, 40, 280, 200], fill=(220, 20, 20))
    b = io.BytesIO()
    img.save(b, "WEBP")
    uri = "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()
    body = {"model": mid, "max_tokens": 10, "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What color is the rectangle? One word."},
                                                      {"type": "image_url", "image_url": {"url": uri}}]}]}
    r = httpx.post(base + "/v1/chat/completions", json=body, timeout=300).json()
    assert "red" in r["choices"][0]["message"]["content"].lower()
    rec = ctx.tracker.recent(1)[0]
    assert rec["images"] == 1 and rec["phase"] == "done"


def test_image_rejected_for_text_model(server):
    base, ctx = server
    mid = _model(ctx, "qwen3")
    if not mid:
        pytest.skip("Qwen3 test model not present")
    tiny = "data:image/png;base64," + base64.b64encode(_png()).decode()
    r = httpx.post(base + "/v1/chat/completions", timeout=300, json={
        "model": mid, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": tiny}}]}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_vision_capable"


def _png() -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (8, 8)).save(b, "PNG")
    return b.getvalue()


def test_control_panel_api(server):
    base, ctx = server
    st = httpx.get(base + "/wr/api/status").json()
    assert st["product"] == "WinRunner" and st["engine"]["flag_count"] > 100
    lib = httpx.get(base + "/wr/api/library").json()
    mid = lib["models"][0]["id"]
    plan = httpx.post(base + "/wr/api/plan", json={"id": mid}).json()
    assert plan["plan"]["ctx"] > 0 and "--jinja" in plan["command"]
    assert Path(ENGINE).name in plan["command"]
