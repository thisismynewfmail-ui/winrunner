"""Shared fixtures: WinRunner served by uvicorn against the scripted fake llama-server (tests/fake_engine.py)."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from tests.gguf_writer import llama_like
from tests.harness import wait_for

FAKE_ENGINE = Path(__file__).with_name("fake_engine.py")


@pytest.fixture
def state(tmp_path: Path) -> Path:
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def winrunner(tmp_path: Path, state: Path, monkeypatch):
    """start(plan, server, errors, startup) -> (base_url, ctx) with a fake engine following ``plan``.

    ``server`` / ``startup`` are settings sections; the library holds two models (test-7b, other-1b).
    """
    import uvicorn

    from winrunner import manager as mgr
    from winrunner.app import create_app
    from winrunner.paths import DataPaths
    from winrunner.util import free_port

    monkeypatch.setattr(mgr, "RECOVERY_DELAYS_S", (0.2, 0.3, 0.3))
    monkeypatch.setattr(mgr, "DEVICE_POLL_S", 0.05)
    monkeypatch.setenv("FAKE_ENGINE_STATE", str(state))
    running = []

    def start(plan: str, server: dict | None = None, errors: str = "data", startup: dict | None = None):
        monkeypatch.setenv("FAKE_ENGINE_PLAN", plan)
        monkeypatch.setenv("FAKE_ENGINE_ERRORS", errors)
        exe = tmp_path / "engine" / "llama-server"
        exe.parent.mkdir()
        exe.write_text(f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(FAKE_ENGINE)!r}, run_name='__main__')\n")
        exe.chmod(0o755)
        models = tmp_path / "models"
        llama_like(models / "pub" / "repo" / "Test-7B-Q4_K_M.gguf")
        llama_like(models / "pub" / "other" / "Other-1B-Q8_0.gguf", n_layer=2)
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.json").write_text(json.dumps({
            "library": {"model_dirs": [str(models)]},
            "engine": {"engine_path": str(exe), "backend": "custom", "load_timeout_s": 60},
            "defaults": {"context_length": 4096},
            "server": server or {},
            "startup": {"open_ui": "none", **(startup or {})},
        }))
        app, ctx = create_app(DataPaths(data))
        port = free_port()
        ctx.extras.update(bound_host="127.0.0.1", bound_port=port)
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        t = threading.Thread(target=srv.run, daemon=True)
        t.start()
        running.append((srv, t))
        base = f"http://127.0.0.1:{port}"
        wait_for(lambda: srv.started and ctx.library.entries(), what="WinRunner start")
        return base, ctx

    yield start
    for srv, t in running:
        srv.should_exit = True
        t.join(timeout=30)
