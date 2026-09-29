"""Engine-verified planning (manager._fit_auto / _apply_projection) against a fake llama-fit-params.

The fake models two 16 GiB GPUs: a full offload fits when weights + KV cache +
compute buffers stay under free VRAM minus the --fit-target margins. Without -c it
reports the largest context that fits (like llama.cpp's fit with an unset context).
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tests.fixtures import fake_model
from winrunner.config import LoadParams
from winrunner.engine import EngineDevice, EngineInfo
from winrunner.manager import ModelManager
from winrunner.planner import AUTO_UBATCH, Planner

FAKE = textwrap.dedent('''\
    #!{python}
    import os, sys
    a = sys.argv[1:]
    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default
    layer = float(os.environ.get("FAKE_LAYER_MIB", "300"))
    n_layer = 64
    weights = layer * n_layer + 400
    kv_tok = {{"f16": 0.25, "q8_0": 0.1328125}}[opt("-ctk", "f16")]
    ub = int(opt("-ub", "512"))
    compute = 400 * ub / 512
    free = [15300, 16100]
    margins = [int(x) for x in opt("--fit-target", "1024,1024").split(",")]
    cap = sum(f - m for f, m in zip(free, margins)) - 2 * compute
    with open(os.environ["FAKE_LOG"], "a") as log:
        log.write(" ".join(a) + "\\n")
    if "-c" in a:
        ctx = int(opt("-c"))
    else:
        ctx = int((cap - weights) / kv_tok) // 256 * 256
        ctx = max(min(ctx, 131072), int(opt("--fit-ctx", "4096")))
    need = weights + kv_tok * ctx
    if need <= cap:
        print(f"-c {{ctx}} -ngl 65 -ts 32,33")
    else:
        per = layer + kv_tok * ctx / n_layer
        ngl = max(0, int((cap - 400) // per))
        print(f"-c {{ctx}} -ngl {{ngl}} -ts {{ngl // 2}},{{ngl - ngl // 2}}")
''')


@pytest.fixture()
def fake_fit(tmp_path, monkeypatch):
    exe = tmp_path / "llama-fit-params"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    return exe, log


DEVS = [EngineDevice("Vulkan0", "AMD Radeon RX 6800", 16368, 15300),
        EngineDevice("Vulkan1", "AMD Radeon RX 6800", 16368, 16100)]


def _verify(exe, p: LoadParams, layer_mib: float = 300.0):
    os.environ["FAKE_LAYER_MIB"] = str(layer_mib)
    info = fake_model(ctx_train=131072, layer_mib=layer_mib)
    eng = EngineInfo(path=str(exe.parent / "llama-server"), name="e", backend="vulkan",
                     flags=["-c", "-fa", "--flash-attn", "--fit", "--fit-target", "--fit-ctx", "-fitc", "-ctk", "-ctv",
                            "-ngl", "-ts", "--tensor-split", "-ub", "-np"],
                     fa_tristate=True, ngl_all=True, fit_params=str(exe))
    entry = SimpleNamespace(path="/m/model.gguf", info=info, id="test")
    pl = Planner(info, p, DEVS, engine_fit=True).plan()
    mgr = ModelManager.__new__(ModelManager)
    sel = [d.name for d in DEVS]
    chosen = asyncio.run(mgr._fit_auto(eng, entry, p, pl, sel, [1024, 1024], 0, pl.ctx_reduced_from or pl.ctx))
    ModelManager._apply_projection(pl, chosen, entry, p, DEVS)
    return pl, chosen


def test_q8_kv_chosen_when_it_enables_full_offload(fake_fit):
    exe, log = fake_fit
    pl, chosen = _verify(exe, LoadParams(context_length=65536))
    assert chosen["full"] and chosen["kv_k"] == "q8_0" and pl.ctx == 65536
    assert pl.full_offload and pl.source == "engine" and pl.flash_attn == "on"
    calls = log.read_text().splitlines()
    # the estimate already rules out F16 at 64K by far, so only the Q8_0 configuration is run
    assert len(calls) == 1 and "-ctk q8_0" in calls[0]


def test_f16_tried_first_when_close(fake_fit):
    exe, log = fake_fit
    pl, chosen = _verify(exe, LoadParams(context_length=24576))
    calls = log.read_text().splitlines()
    assert "-ctk" not in calls[0] and chosen["kv_k"] == "f16" and pl.ctx == 24576 and len(calls) == 1


def test_context_reduced_to_engine_fit(fake_fit):
    exe, log = fake_fit
    pl, chosen = _verify(exe, LoadParams(context_length=65536), layer_mib=320.0)
    assert chosen["full"] and chosen["kv_k"] == "q8_0"
    assert 32768 < pl.ctx < 65536 and pl.ctx % 1024 == 0
    assert pl.ctx_reduced_from == 65536 and pl.full_offload
    assert pl.warnings[0].startswith("Context reduced from 65,536")
    assert not any(w.startswith("Partial offload") for w in pl.warnings)
    calls = log.read_text().splitlines()
    assert "-c " not in calls[-1] and "--fit-ctx 4096" in calls[-1]  # auto context run
    assert len(calls) <= 2


def test_cpu_offload_policy_refits_with_large_ubatch(fake_fit):
    exe, log = fake_fit
    pl, chosen = _verify(exe, LoadParams(context_length=65536, vram_overflow="cpu_offload"), layer_mib=320.0)
    assert not chosen["full"] and pl.ctx == 65536 and pl.ubatch == AUTO_UBATCH
    assert not pl.full_offload and pl.gpu_layers < 65
    assert any(w.startswith("Partial offload") for w in pl.warnings)
    calls = log.read_text().splitlines()
    assert f"-ub {AUTO_UBATCH}" in calls[-1]
