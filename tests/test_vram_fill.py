"""Engine-verified memory plans end to end: WinRunner calibrates its estimate with llama-fit-params, keeps the whole
model on the GPUs and fills them up to the safety margin (fake engine and fake llama-fit-params, Linux only)."""

from __future__ import annotations

import sys

import httpx
import pytest

from tests.gguf_writer import llama_like
from tests.harness import model_id, ready_instance, wait_for

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake engine is a Python script with a shebang")

# two small GPUs: the model's KV cache, not its weights, decides how much context fits
TWO_GPUS = "Vulkan0:Fake GPU A:4096:3584;Vulkan1:Fake GPU B:4096:3968"
MARGIN = 256


def models(root):
    # 32 layers of 2048: ~0.7 GiB of weights and 64 KiB of F16 KV cache per token; trained for 131072 tokens
    llama_like(root / "pub" / "big" / "Big-3B-Q4_K_M.gguf", n_layer=32, n_embd=2048, ctx=131072)
    # 64 layers of 5120: ~9 GiB of weights, more than both GPUs together
    llama_like(root / "pub" / "huge" / "Huge-20B-Q4_K_M.gguf", n_layer=64, n_embd=5120, ctx=32768)


def plan(base: str, ctx, name: str, verify: bool, **overrides) -> dict:
    r = httpx.post(base + "/wr/api/plan", json={"id": model_id(ctx, name), "overrides": overrides, "verify": verify},
                   timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def fit_calls(state) -> list[str]:
    try:
        return (state / "fit-params.log").read_text().splitlines()
    except OSError:
        return []


@pytest.fixture
def two_gpus(winrunner, monkeypatch):
    def start(bias: str = "300,0.002"):
        monkeypatch.setenv("FAKE_FIT_BIAS", bias)
        return winrunner("ok", fit_params=True, devices=TWO_GPUS, modern=True, models=models)
    return start


def test_verified_plan_fills_both_gpus_without_overcommitting(two_gpus, state):
    base, ctx = two_gpus()  # the engine needs 300 MiB + 0.002 MiB per token more per GPU than estimated
    estimate = plan(base, ctx, "big", False)["plan"]
    r = plan(base, ctx, "big", True)
    pl = r["plan"]
    assert pl["source"] == "engine" and pl["full_offload"] and pl["ctx_adjusted"] == "raised"
    assert pl["kv_k"] == "f16" and pl["ctx"] > 4096
    assert pl["ctx"] < estimate["ctx"]  # the uncalibrated estimate would have overcommitted the GPUs
    kv_per_1k = 64 * 1024 * 1024 / (1024 * 1024)  # 64 KiB per token, both GPUs together
    for d in pl["devices"]:
        assert d["headroom_mib"] >= 0, d  # measured by the engine, plus what it does not measure
        assert d["margin_mib"] == MARGIN
    assert min(d["headroom_mib"] for d in pl["devices"]) < kv_per_1k + 32  # filled: 1K more tokens would not fit
    cmd = r["command"].split()
    assert cmd[cmd.index("-c") + 1] == str(pl["ctx"])
    assert cmd[cmd.index("-ngl") + 1] == "all" and "-ts" in cmd and cmd[cmd.index("--fit") + 1] == "off"
    calls = fit_calls(state)
    assert 2 <= len(calls) <= 5 and all("--fit-print on" in c for c in calls)
    assert not any(" -np " in f" {c} " for c in calls)


def test_calibration_uses_vram_the_estimate_would_waste(two_gpus):
    base, ctx = two_gpus(bias="-150,0")  # the engine needs less than estimated
    estimate = plan(base, ctx, "big", False)["plan"]
    pl = plan(base, ctx, "big", True)["plan"]
    assert pl["source"] == "engine" and pl["full_offload"]
    assert pl["ctx"] > estimate["ctx"]
    assert all(d["headroom_mib"] >= 0 for d in pl["devices"])
    assert min(d["headroom_mib"] for d in pl["devices"]) < 96


def test_requested_context_is_reduced_instead_of_offloading_layers(two_gpus):
    base, ctx = two_gpus()
    pl = plan(base, ctx, "big", True, context_length=131072, context_fit="fit")["plan"]
    assert pl["full_offload"] and pl["gpu_layers"] == pl["n_layer"] + 1
    assert pl["ctx_adjusted"] == "reduced" and pl["ctx"] < 131072
    assert any("Context reduced from 131,072" in w for w in pl["warnings"])
    exact = plan(base, ctx, "big", True, context_length=131072, context_fit="off")["plan"]
    assert exact["ctx"] == 131072 and not exact["full_offload"]  # only when asked to keep the exact length


def test_load_uses_the_verified_layout(two_gpus):
    base, ctx = two_gpus()
    verified = plan(base, ctx, "big", True)["plan"]
    r = httpx.post(base + "/wr/api/models/load", json={"id": model_id(ctx, "big")})
    assert r.status_code == 202
    inst = wait_for(lambda: ready_instance(ctx), what="model load")
    a = inst.spec.args
    assert a[a.index("-c") + 1] == str(verified["ctx"])
    assert a[a.index("-ngl") + 1] == "all" and a[a.index("--fit") + 1] == "off"
    assert "Placement (measured by the engine): whole model in VRAM" in "\n".join(
        x["text"] for x in list(ctx.bus.activity))


def test_model_larger_than_vram_is_placed_by_the_engine(two_gpus):
    base, ctx = two_gpus()
    r = plan(base, ctx, "huge", True)
    pl = r["plan"]
    assert not pl["full_offload"] and pl["use_engine_fit"]
    assert pl["engine"].get("fit"), pl["engine"]  # the layout llama.cpp's --fit chooses is shown
    cmd = r["command"].split()
    assert "-ngl" not in cmd and cmd[cmd.index("--fit") + 1] == "on"
    targets = [int(x) for x in cmd[cmd.index("--fit-target") + 1].split(",")]
    assert len(targets) == 2 and all(t >= MARGIN for t in targets)
