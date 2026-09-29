"""Requests naming a model WinRunner does not know are answered by the active model instead of failing."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from tests.gguf_writer import llama_like
from tests.harness import CHAT, activity, chat, load, model_id, ready_instance

fake_engine = pytest.mark.skipif(sys.platform == "win32", reason="the fake engine is started as a POSIX script")


# ----- choosing the model (no engine needed) ----------------------------------------------------------


def make_manager(tmp_path: Path, names: list[str], **settings):
    from winrunner.config import SettingsStore
    from winrunner.engine import EngineManager
    from winrunner.events import EventBus, RequestTracker
    from winrunner.hardware import HardwareMonitor
    from winrunner.library import ModelLibrary
    from winrunner.manager import ModelManager
    from winrunner.paths import DataPaths

    models = tmp_path / "models"
    for n in names:
        llama_like(models / "pub" / n.lower() / f"{n}.gguf", n_layer=2)
    paths = DataPaths(tmp_path / "data")
    paths.ensure()
    store = SettingsStore(paths.settings_file)
    store.update({"library": {"model_dirs": [str(models)]}, **settings})
    lib = ModelLibrary(paths.gguf_index)
    lib.scan([str(models)], {})
    bus = EventBus()
    return ModelManager(store, paths, lib, EngineManager(paths.engines_dir, paths.downloads_tmp), HardwareMonitor(),
                        bus, RequestTracker(bus))


def test_default_model_is_the_last_used_one(tmp_path):
    m = make_manager(tmp_path, ["Alpha-7B-Q4_K_M", "Beta-3B-Q8_0"])
    assert m._default_model() is None  # two models, none used yet: nothing to guess
    beta = m.library.resolve("beta-3b").id
    prof = m.store.profile(m.library.get(beta).path)
    prof.last_loaded = 1000.0
    m.store.set_profile(m.library.get(beta).path, prof)
    assert m._default_model() == beta  # most recently loaded
    alpha = m.library.resolve("alpha-7b").id
    m.store.update({"startup": {"last_model": alpha}})
    assert m._default_model() == alpha  # the model loaded last takes precedence
    m.store.update({"startup": {"last_model": "deleted-model"}})
    assert m._default_model() == beta  # no longer in the library


def test_single_model_library_needs_no_history(tmp_path):
    m = make_manager(tmp_path, ["Only-7B-Q4_K_M"])
    assert m._default_model() == m.library.entries()[0].id


def test_unknown_name_without_loaded_model(tmp_path):
    from winrunner.manager import ModelError

    m = make_manager(tmp_path, ["Only-7B-Q4_K_M"])
    assert m.ready_for("gpt-4o", jit=True) is None  # the model will be loaded just in time
    assert m.substitute_model(jit=True) == m.library.entries()[0].id
    with pytest.raises(ModelError) as e:
        m.ready_for("gpt-4o", jit=False)
    assert e.value.status == 503 and e.value.code == "no_model_loaded" and "Available" not in str(e.value)
    assert m.substitute_model(jit=False) is None


# ----- end to end ---------------------------------------------------------------------------------------


@fake_engine
def test_unknown_name_uses_the_loaded_model(winrunner):
    base, ctx = winrunner("ok")
    load(base, ctx)
    for name in ("gpt-4o", "omnibrain", "gpt-4o"):
        r = chat(base, ctx, model=name)
        assert r.status_code == 200, r.text
        assert r.json()["model"] == model_id(ctx)  # the response names the model that answered
    assert activity(ctx).count("'gpt-4o' is not a model in the library") == 1  # noted once per name


@fake_engine
def test_unknown_name_loads_the_last_used_model(winrunner, tmp_path):
    # Nothing is loaded (e.g. after a restart of WinRunner): this used to fail with "Model 'gpt-4o' not found.
    # Available: ...".
    last = "test-7b-q4_k_m"
    base, ctx = winrunner("ok", startup={"last_model": last})
    assert ready_instance(ctx) is None
    r = chat(base, ctx, model="gpt-4o")
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "Hello"
    assert ready_instance(ctx).model_id == last
    with httpx.stream("POST", base + "/v1/chat/completions", json={**CHAT, "model": "my-local-model", "stream": True},
                      timeout=60) as s:
        lines = [ln for ln in s.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]"


@fake_engine
def test_without_jit_the_loaded_model_answers_for_other_library_models(winrunner):
    base, ctx = winrunner("ok", server={"jit_loading": False})
    r = chat(base, ctx, model="gpt-4o")
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_model_loaded"
    load(base, ctx, "other-1b")
    # used to fail with "Model '...' is not loaded and just-in-time loading is disabled. Loaded: ..."
    r = chat(base, ctx, model=model_id(ctx, "test-7b"))
    assert r.status_code == 200, r.text
    assert r.json()["model"] == model_id(ctx, "other-1b")
    assert ready_instance(ctx).model_id == model_id(ctx, "other-1b")  # no model switch
    assert "is not loaded and just-in-time loading is off" in activity(ctx)


@fake_engine
def test_model_lookup_by_unknown_name(winrunner):
    base, ctx = winrunner("ok")
    assert httpx.get(base + "/v1/models/gpt-4o").status_code == 404  # nothing loaded, nothing used yet
    load(base, ctx)
    r = httpx.get(base + "/v1/models/gpt-4o")
    assert r.status_code == 200
    assert r.json()["id"] == "gpt-4o" and r.json()["root"] == model_id(ctx)
    lms = httpx.get(base + "/api/v0/models/gpt-4o").json()
    assert lms["id"] == "gpt-4o" and lms["root"] == model_id(ctx) and lms["state"] == "loaded"
