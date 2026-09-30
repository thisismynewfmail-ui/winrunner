import re

import pytest

from tests.fixtures import MiB, fake_model
from winrunner.config import LoadParams
from winrunner.engine import EngineDevice
from winrunner.planner import (LayerCost, Planner, assign_layers, device_targets, fit_to_engine, kv_layer_bytes,
                               measured_used, override_patterns, parse_fit_args, parse_fit_print, place, swa_layers)

RX6800 = [EngineDevice("Vulkan0", "AMD Radeon RX 6800 (RADV NAVI21)", 16368, 15300),
          EngineDevice("Vulkan1", "AMD Radeon RX 6800 (RADV NAVI21)", 16368, 16100)]


def test_assign_layers_matches_llamacpp():
    # 10 layers + output on 2 devices split 5/6 -> contiguous ranges, output on last device
    a = assign_layers(10, 11, [5, 6])
    assert a[:5] == [0] * 5 and a[5:] == [1] * 6
    # partial offload: last 6 slots on GPU
    b = assign_layers(10, 6, [1, 1])
    assert b[:5] == [-1] * 5 and b[5:8] == [0] * 3 and b[8:] == [1] * 3


def test_default_context_clamped_to_trained():
    pl = Planner(fake_model(ctx_train=40960), LoadParams(), RX6800).plan()
    assert pl.ctx_requested == 65536
    assert pl.ctx == 40960
    assert any("clamped" in n for n in pl.notes)


def test_default_margin_is_512_mib():
    pl = Planner(fake_model(), LoadParams(), RX6800).plan()
    assert all(d.margin_mib == 512 for d in pl.devices)


def test_small_model_full_offload_f16():
    m = fake_model(n_layer=28, n_embd=1024, n_head=16, layer_mib=14, embd_mib=84, ctx_train=40960)
    pl = Planner(m, LoadParams(), RX6800).plan()
    assert pl.full_offload and pl.kv_k == "f16" and pl.strategy == "full"
    assert pl.tensor_split and sum(pl.tensor_split) == 29
    assert all(d.headroom_mib >= 0 for d in pl.devices)
    assert pl.load_mode == "none"  # full read when fully offloaded
    assert not pl.overrides


def test_32b_uses_q8_kv_to_fit_65k():
    m = fake_model(ctx_train=131072)  # 64 layers, 290 MiB each
    pl = Planner(m, LoadParams(context_length=65536), RX6800).plan()
    assert pl.ctx == 65536
    assert pl.full_offload
    assert pl.kv_k == pl.kv_v == "q8_0"
    assert pl.flash_attn == "on"  # quantized V cache needs flash attention
    assert pl.max_ctx_full_offload["f16"] < 65536 <= pl.max_ctx_full_offload["q8_0"]


def _big_dense():
    # 64 layers of 518 MiB (87% feed-forward), like a 32B model at Q8_0: does not fit 2 x 16 GB
    return fake_model(n_layer=64, layer_mib=518, embd_mib=780, n_ff=27648, n_vocab=152064, ctx_train=131072)


@pytest.mark.parametrize("ctx", [16384, 32768, 65536, 131072])
def test_attention_first_fills_every_gpu_to_the_margin(ctx):
    m = _big_dense()
    pl = Planner(m, LoadParams(context_length=ctx), RX6800).plan()
    assert pl.ctx == ctx  # the context is never reduced to make the model fit
    assert pl.strategy == "attention_first" and not pl.full_offload
    # attention and KV cache of every layer on a GPU, only feed-forward groups in RAM
    assert pl.gpu_layers == pl.n_layer + 1 and all(h >= 0 for h in pl.layer_home)
    assert any(pl.ram_parts) and all(g in ("ffn_up", "ffn_gate", "ffn_down") for x in pl.ram_parts for g in x)
    part_mib = 518 * 0.87 / 3
    for d in pl.devices:
        assert 0 <= d.headroom_mib < part_mib, d  # filled to free - margin, less than one matrix unused
    assert pl.kv_k == "q8_0"  # auto KV: Q8_0 keeps more weights on the GPUs
    assert pl.load_mode == "none"
    assert sum(pl.tensor_split) == pl.n_layer + 1


def test_attention_first_override_patterns_match_the_ram_parts():
    pl = Planner(_big_dense(), LoadParams(context_length=65536), RX6800).plan()
    rx = [(re.compile(o.split("=")[0]), o.split("=")[1]) for o in pl.overrides]
    assert all(b == "CPU" for _, b in rx)
    for il in range(pl.n_layer):
        for g in ("ffn_up", "ffn_gate", "ffn_down"):
            name = f"blk.{il}.{g}.weight"
            hit = any(r.search(name) for r, _ in rx)
            assert hit == (g in pl.ram_parts[il]), (il, g)
        assert not any(r.search(f"blk.{il}.attn_q.weight") for r, _ in rx)
    assert not any(r.search("blk.1.ffn_up_exps.weight") for r, _ in rx)


def test_user_kv_choice_respected():
    m = fake_model(ctx_train=131072)
    pl = Planner(m, LoadParams(context_length=65536, kv_cache_type="f16"), RX6800).plan()
    assert pl.kv_k == "f16" and not pl.full_offload
    assert pl.strategy == "attention_first" and any(pl.ram_parts)


def test_whole_layers_on_cpu_only_when_attention_does_not_fit():
    # F16 KV at 128K: the KV cache alone exceeds both GPUs
    m = _big_dense()
    pl = Planner(m, LoadParams(context_length=131072, kv_cache_type="f16"), RX6800).plan()
    assert pl.ctx == 131072 and pl.strategy == "layers"
    cpu = [i for i, h in enumerate(pl.layer_home) if h < 0]
    assert cpu == list(range(len(cpu))) and 0 < len(cpu) < pl.n_layer  # leading layers only
    assert pl.gpu_layers == pl.n_layer + 1 - len(cpu)
    assert all(d.headroom_mib >= 0 for d in pl.devices)
    assert pl.load_mode == "mmap"


def test_moe_keeps_experts_in_ram_and_attention_on_gpu():
    big = fake_model(n_layer=36, n_embd=2880, n_head=64, n_head_kv=8, head_dim=64, n_ff=2880, layer_mib=1750,
                     expert_count=128, expert_frac=0.97, ctx_train=131072, sliding_window=128, arch="gpt-oss")
    pl = Planner(big, LoadParams(), RX6800).plan()
    assert not pl.full_offload and pl.strategy == "attention_first"
    assert pl.gpu_layers == pl.n_layer + 1  # every layer's attention stays on GPU
    assert 0 < sum(1 for x in pl.ram_parts if x) < pl.n_layer
    assert all(g.endswith("_exps") for x in pl.ram_parts for g in x)
    assert all(d.headroom_mib >= 0 for d in pl.devices)
    # both GPUs are actually used and filled
    assert all(d.used_mib > 14000 for d in pl.devices)


def test_cpu_only():
    pl = Planner(fake_model(), LoadParams(), []).plan()
    assert pl.gpu_layers == 0 and not pl.full_offload and pl.kv_k == "f16" and pl.strategy == "cpu"
    assert pl.load_mode == "mmap"


def test_flash_off_forces_unquantized_candidates():
    m = fake_model(ctx_train=131072)
    pl = Planner(m, LoadParams(flash_attn="off"), RX6800).plan()
    assert pl.kv_k == "f16"


def test_device_selection_single_gpu():
    pl = Planner(fake_model(), LoadParams(devices=["Vulkan1"]), RX6800).plan()
    assert [d.name for d in pl.devices] == ["Vulkan1"]
    assert pl.tensor_split is None


def test_reclaim_adds_evicted_memory():
    m = fake_model(ctx_train=131072)
    tight = [EngineDevice("Vulkan0", "RX 6800", 16368, 4000), EngineDevice("Vulkan1", "RX 6800", 16368, 4000)]
    a = Planner(m, LoadParams(context_length=8192), tight).plan()
    b = Planner(m, LoadParams(context_length=8192), tight, reclaim_mib={"Vulkan0": 11000, "Vulkan1": 12000}).plan()
    assert not a.full_offload and b.full_offload


def test_mmproj_reserved_on_first_gpu():
    m = _big_dense()
    without = Planner(m, LoadParams(context_length=32768), RX6800).plan()
    with_mm = Planner(m, LoadParams(context_length=32768), RX6800, mmproj_size=900 * MiB, mmproj_mib_hint=1200).plan()
    assert with_mm.devices[0].mmproj_mib == 1200 and with_mm.devices[1].mmproj_mib == 0
    assert 0 <= with_mm.devices[0].headroom_mib < 518 * 0.87 / 3
    assert with_mm.totals["ram_parts_mib"] > without.totals["ram_parts_mib"]


def test_swa_pattern_and_kv():
    m = fake_model(n_layer=12, sliding_window=1024, arch="gemma3", swa_pattern=None)
    swa = swa_layers(m)
    assert swa.count(False) == 2  # every 6th layer is global
    full = sum(kv_layer_bytes(m, 65536, "f16", "f16", 512, 1, True))
    windowed = sum(kv_layer_bytes(m, 65536, "f16", "f16", 512, 1, False))
    assert windowed < full / 3


def test_manual_mode():
    pl = Planner(fake_model(), LoadParams(gpu_offload="manual", n_gpu_layers=20, tensor_split=[1, 1]), RX6800).plan()
    assert pl.mode == "manual" and pl.gpu_layers == 20 and pl.tensor_split == [1.0, 1.0]
    assert pl.strategy == "manual" and not pl.overrides


def test_manual_ffn_in_ram():
    pl = Planner(_big_dense(), LoadParams(gpu_offload="manual", n_cpu_ffn=10), RX6800).plan()
    assert pl.n_cpu_ffn == 10 and pl.gpu_layers == pl.n_layer + 1
    assert [bool(x) for x in pl.ram_parts[:12]] == [True] * 10 + [False] * 2


def test_override_patterns_group_layers():
    ram = [[] for _ in range(8)]
    for i in (0, 1, 2):
        ram[i] = ["ffn_up", "ffn_gate", "ffn_down"]
    ram[3] = ["ffn_down"]
    pats = override_patterns(ram)
    assert pats == ["blk\\.(0|1|2)\\.(ffn_down|ffn_gate|ffn_up)\\.=CPU", "blk\\.3\\.ffn_down\\.=CPU"]
    assert not any("," in p for p in pats)  # entries are joined with commas on the command line


def test_place_splits_and_fills():
    costs = [LayerCost(base=10.0, kv=4.0, parts=[("ffn_up", 10.0), ("ffn_gate", 10.0), ("ffn_down", 10.0)])
             for _ in range(10)]
    lay = place(costs, caps=[150.0, 170.0], out_cost=20.0)
    assert lay is not None and lay.k_cpu == 0 and sum(lay.counts) == 11
    assert all(0 <= r < 10.0 for r in lay.rooms)  # less than one group left on each GPU
    kept = sum(len(c.parts) for c in costs) - sum(len(v) for v in lay.ram.values())
    assert kept * 10.0 == lay.gpu_parts
    # base alone does not fit: leading layers go to the CPU
    lay2 = place(costs, caps=[40.0, 50.0], out_cost=20.0)
    assert lay2 is not None and lay2.k_cpu > 0
    assert place(costs, caps=[5.0, 5.0], out_cost=20.0) is None  # not even the output layer


def _engine_like(true_compute_full, true_compute_partial):
    """A fake engine projection: exact weights/KV, compute buffers that differ from the estimate."""
    def measure(pl):
        out = {}
        for i, d in enumerate(pl.devices):
            comp = (true_compute_full if pl.regime == "full" else true_compute_partial)[i]
            out[d.name] = {"model": d.weights_mib + d.output_mib, "context": d.kv_mib, "compute": comp}
        return out
    return measure


def test_fit_to_engine_corrects_the_estimate_and_fills_to_the_margin():
    m = _big_dense()
    planner = Planner(m, LoadParams(context_length=65536), RX6800)
    calls = []
    real = _engine_like([900.0, 1300.0], [150.0, 180.0])

    def measure(pl):
        calls.append(pl)
        return real(pl)

    pl = fit_to_engine(planner, measure)
    assert pl.source == "engine" and pl.strategy == "attention_first"
    targets = device_targets(pl)
    part = 518 * 0.87 / 3
    for d in pl.devices:
        used = measured_used(pl.engine["projection"][d.name])
        assert targets[d.name] - part <= used <= targets[d.name], (d.name, used, targets[d.name])
    assert len(calls) <= 7
    assert planner.adjust == {"full": {}, "partial": {}}  # the planner is left as it was


def test_fit_to_engine_prefers_full_offload_when_it_really_fits():
    m = fake_model(ctx_train=131072)
    # the estimate thinks F16 does not fit at 32K, but the engine's buffers are small
    planner = Planner(m, LoadParams(context_length=40960), RX6800)
    est = planner.plan()
    pl = fit_to_engine(planner, _engine_like([60.0, 60.0], [40.0, 40.0]))
    assert pl.full_offload
    assert (pl.kv_k == "f16") or est.kv_k == "q8_0"


def test_parse_fit_outputs():
    f = parse_fit_args('noise\n-c 4096 -ngl 48 -ts 20,21 -ot "blk\\.14\\.ffn_(up|down|gate)_(ch|)exps=CPU"\n')
    assert f["ctx"] == 4096 and f["ngl"] == 48 and f["tensor_split"] == [20.0, 21.0] and "exps=CPU" in f["override_tensor"]
    p = parse_fit_print("Vulkan0 9000 2000 300\nVulkan1 9100 2000 500\nHost 83 0 12\n")
    assert p["Vulkan1"]["compute"] == 500 and p["Host"]["model"] == 83
