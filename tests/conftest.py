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
FAKE_FIT_PARAMS = Path(__file__).with_name("fake_fit_params.py")


def _script(path: Path, target: Path) -> None:
    path.write_text(f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(target)!r}, run_name='__main__')\n")
    path.chmod(0o755)


@pytest.fixture
def state(tmp_path: Path) -> Path:
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def winrunner(tmp_path: Path, state: Path, monkeypatch):
    """start(plan, server, errors, startup, ...) -> (base_url, ctx) with a fake engine following ``plan``.

    ``server`` / ``startup`` are settings sections; the library holds two models (test-7b, other-1b).
    ``fit_params`` installs the fake llama-fit-params, ``devices`` sets the GPUs the engine reports
    (``name:description:total:free;...``), ``modern`` makes the engine list the options of current builds,
    ``settings`` is merged into the settings file and ``models(models_dir)`` can add model files.
    """
    import uvicorn

    from winrunner import manager as mgr
    from winrunner.app import create_app
    from winrunner.config import deep_merge
    from winrunner.paths import DataPaths
    from winrunner.util import free_port

    monkeypatch.setattr(mgr, "RECOVERY_DELAYS_S", (0.2, 0.3, 0.3))
    monkeypatch.setattr(mgr, "DEVICE_POLL_S", 0.05)
    monkeypatch.setenv("FAKE_ENGINE_STATE", str(state))
    running = []

    def start(plan: str, server: dict | None = None, errors: str = "data", startup: dict | None = None, *,
              fit_params: bool = False, devices: str | None = None, modern: bool = False,
              settings: dict | None = None, models=None):
        monkeypatch.setenv("FAKE_ENGINE_PLAN", plan)
        monkeypatch.setenv("FAKE_ENGINE_ERRORS", errors)
        if devices:
            monkeypatch.setenv("FAKE_ENGINE_DEVICES", devices)
        if modern:
            monkeypatch.setenv("FAKE_ENGINE_HELP", "modern")
        exe = tmp_path / "engine" / "llama-server"
        exe.parent.mkdir()
        _script(exe, FAKE_ENGINE)
        if fit_params:
            _script(exe.with_name("llama-fit-params"), FAKE_FIT_PARAMS)
            monkeypatch.setenv("FAKE_FIT_LOG", str(state / "fit-params.log"))
        model_dir = tmp_path / "models"
        llama_like(model_dir / "pub" / "repo" / "Test-7B-Q4_K_M.gguf")
        llama_like(model_dir / "pub" / "other" / "Other-1B-Q8_0.gguf", n_layer=2)
        if models:
            models(model_dir)
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.json").write_text(json.dumps(deep_merge({
            "library": {"model_dirs": [str(model_dir)]},
            "engine": {"engine_path": str(exe), "backend": "custom", "load_timeout_s": 60},
            "defaults": {"context_length": 4096},
            "server": server or {},
            "startup": {"open_ui": "none", **(startup or {})},
        }, settings or {})))
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
