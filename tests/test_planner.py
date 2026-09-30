from tests.fixtures import MiB, fake_model
from winrunner.config import LoadParams
from winrunner.engine import EngineDevice
from winrunner.planner import (Planner, assign_layers, kv_layer_bytes, parse_fit_args, parse_fit_print,
                               swa_layers)

RX6800 = [EngineDevice("Vulkan0", "AMD Radeon RX 6800", 16368, 15300), EngineDevice("Vulkan1", "AMD Radeon RX 6800", 16368, 16100)]


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


def test_small_model_full_offload_f16():
    m = fake_model(n_layer=28, n_embd=1024, n_head=16, layer_mib=14, embd_mib=84, ctx_train=40960)
    pl = Planner(m, LoadParams(), RX6800).plan()
    assert pl.full_offload and pl.kv_k == "f16"
    assert pl.tensor_split and sum(pl.tensor_split) == 29
    assert all(d.headroom_mib >= 0 for d in pl.devices)
    assert pl.load_mode == "none"  # full read when fully offloaded


def test_32b_uses_q8_kv_to_fit_65k():
    m = fake_model(ctx_train=131072)  # 64 layers, 290 MiB each
    pl = Planner(m, LoadParams(context_length=65536, context_fit="fit"), RX6800).plan()
    assert pl.ctx == 65536
    assert pl.full_offload
    assert pl.kv_k == pl.kv_v == "q8_0"
    assert pl.flash_attn == "on"  # quantized V cache needs flash attention
    assert pl.max_ctx_full_offload["f16"] < 65536 <= pl.max_ctx_full_offload["q8_0"]


def test_user_kv_choice_respected():
    m = fake_model(ctx_train=131072)
    pl = Planner(m, LoadParams(context_length=65536, kv_cache_type="f16"), RX6800).plan()
    assert pl.kv_k == "f16" and pl.full_offload  # the context is reduced instead
    assert pl.ctx < 65536 and pl.ctx_adjusted == "reduced"
    off = Planner(m, LoadParams(context_length=65536, kv_cache_type="f16", context_fit="off"), RX6800).plan()
    assert off.kv_k == "f16" and not off.full_offload and off.ctx == 65536
    assert off.gpu_layers < off.n_layer + 1


def test_moe_keeps_experts_on_cpu_instead_of_layers():
    big = fake_model(n_layer=36, n_embd=2880, n_head=64, n_head_kv=8, head_dim=64, n_ff=2880, layer_mib=1750,
                     expert_count=128, expert_frac=0.97, ctx_train=131072, sliding_window=128, arch="gpt-oss")
    pl = Planner(big, LoadParams(), RX6800).plan()
    assert not pl.full_offload
    assert pl.gpu_layers == pl.n_layer + 1  # every layer's attention stays on GPU
    assert 0 < pl.n_cpu_moe < pl.n_layer
    assert all(d.headroom_mib >= 0 for d in pl.devices)
    # both GPUs are actually used (cost-aware split)
    assert all(d.used_mib > 8000 for d in pl.devices)


def test_cpu_only():
    pl = Planner(fake_model(), LoadParams(), []).plan()
    assert pl.gpu_layers == 0 and not pl.full_offload and pl.kv_k == "f16"
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


def test_parse_fit_outputs():
    f = parse_fit_args('noise\n-c 4096 -ngl 48 -ts 20,21 -ot "blk\\.14\\.ffn_(up|down|gate)_(ch|)exps=CPU"\n')
    assert f["ctx"] == 4096 and f["ngl"] == 48 and f["tensor_split"] == [20.0, 21.0] and "exps=CPU" in f["override_tensor"]
    p = parse_fit_print("Vulkan0 9000 2000 300\nVulkan1 9100 2000 500\nHost 83 0 12\n")
    assert p["Vulkan1"]["compute"] == 500 and p["Host"]["model"] == 83


# ---------------------------------------------------------------------------
# VRAM filling: the whole model stays on the GPUs and the context takes the free VRAM
# ---------------------------------------------------------------------------


def _dense_24b(bytes_per_param: float):
    """Mistral-Small-24B-like shape: 40 layers, 5120 wide, 8 KV heads, 131k vocabulary."""
    n_embd, n_ff, vocab = 5120, 32768, 131072
    layer_params = 4 * n_embd * n_embd * 0.625 + 3 * n_embd * n_ff
    return fake_model(n_layer=40, n_embd=n_embd, n_head=32, n_head_kv=8, head_dim=128, n_ff=n_ff, n_vocab=vocab,
                      ctx_train=131072, layer_mib=layer_params * bytes_per_param / MiB,
                      embd_mib=vocab * n_embd * bytes_per_param / MiB, arch="llama")


def _gpu_in_use(d):
    """What the operating system shows for the GPU: other programs plus this plan."""
    return d.total_mib - d.free_mib + d.used_mib


def test_q8_model_stays_in_vram_with_reduced_context():
    # Q8_0 24B with the default 65,536 context does not fit in 2 x 16 GB. It used to run 2 layers from system RAM
    # while each GPU kept ~2 GB free; now the context shrinks and the whole model runs on the GPUs.
    pl = Planner(_dense_24b(34 / 32), LoadParams(), RX6800).plan()
    assert pl.full_offload and pl.gpu_layers == pl.n_layer + 1 and pl.n_cpu_moe == 0
    assert 32768 < pl.ctx < 65536 and pl.ctx_adjusted == "reduced"
    assert any("Context reduced from 65,536" in w for w in pl.warnings)
    assert pl.tensor_split and sum(pl.tensor_split) == pl.n_layer + 1
    for d in pl.devices:
        assert d.headroom_mib >= 0
        assert _gpu_in_use(d) > d.total_mib - d.margin_mib - 600  # both GPUs filled (within one layer)
    assert min(d.headroom_mib for d in pl.devices) < 64  # the context grew until a GPU was full


def test_fill_raises_context_into_free_vram():
    pl = Planner(_dense_24b(210 / 256), LoadParams(), RX6800).plan()  # Q6_K: fits with room to spare
    assert pl.full_offload and pl.kv_k == "f16"
    assert pl.ctx > 65536 and pl.ctx_adjusted == "raised"
    assert min(d.headroom_mib for d in pl.devices) < 64
    for d in pl.devices:
        assert d.total_mib - 600 < _gpu_in_use(d) <= d.total_mib - d.margin_mib  # ~15.5-15.7 of 16 GiB in use


def test_fill_stops_at_trained_context():
    small = fake_model(n_layer=28, n_embd=1024, n_head=16, layer_mib=14, embd_mib=84, ctx_train=40960)
    pl = Planner(small, LoadParams(context_length=8192), RX6800).plan()
    assert pl.ctx == 40960 and pl.ctx_adjusted == "raised"


def test_fit_policy_keeps_requested_context_when_it_fits():
    pl = Planner(_dense_24b(210 / 256), LoadParams(context_fit="fit"), RX6800).plan()
    assert pl.ctx == 65536 and not pl.ctx_adjusted and pl.full_offload


def test_off_policy_moves_layers_instead():
    pl = Planner(_dense_24b(34 / 32), LoadParams(context_fit="off"), RX6800).plan()
    assert pl.ctx == 65536 and not pl.full_offload and pl.gpu_layers < pl.n_layer + 1


def test_manual_layout_gets_a_fitted_context():
    pl = Planner(_dense_24b(34 / 32), LoadParams(gpu_offload="manual"), RX6800).plan()
    assert pl.mode == "manual" and pl.full_offload and pl.ctx < 65536
    assert not any("exceeds free VRAM" in w for w in pl.warnings)
    over = Planner(_dense_24b(34 / 32), LoadParams(gpu_offload="manual", context_fit="off"), RX6800).plan()
    assert over.ctx == 65536 and any("exceeds free VRAM" in w for w in over.warnings)


def test_quantized_kv_reserves_flash_attention_scratch():
    m = fake_model(ctx_train=131072)
    f16 = Planner(m, LoadParams(context_length=32768, kv_cache_type="f16", context_fit="fit"), RX6800).plan()
    q8 = Planner(m, LoadParams(context_length=32768, kv_cache_type="q8_0", context_fit="fit"), RX6800).plan()
    layer_f16 = 32768 * 8 * 128 * 2 * 2 / MiB  # one layer's K and V converted to F16
    assert all(d.scratch_mib >= layer_f16 for d in q8.devices)
    assert all(d.scratch_mib < layer_f16 / 2 for d in f16.devices)  # only the flash attention index scratch


def test_calibration_changes_capacity():
    m = _dense_24b(34 / 32)
    base = Planner(m, LoadParams(), RX6800).plan()
    worse = Planner(m, LoadParams(), RX6800, calibration={"Vulkan0": (400.0, 0.0), "Vulkan1": (400.0, 0.0)}).plan()
    better = Planner(m, LoadParams(), RX6800, calibration={"Vulkan0": (-400.0, 0.0), "Vulkan1": (-400.0, 0.0)}).plan()
    assert worse.ctx < base.ctx < better.ctx
    assert all(d.headroom_mib >= 0 for d in worse.devices + better.devices)


def test_model_too_large_for_vram_still_offloads_to_cpu():
    m = fake_model(n_layer=64, layer_mib=700, ctx_train=131072)  # 44 GiB of weights
    pl = Planner(m, LoadParams(), RX6800, engine_fit=True).plan()
    assert not pl.full_offload and pl.ctx == 65536 and pl.use_engine_fit
    assert any("does not fit in VRAM even with a 4,096-token context" in n for n in pl.notes)


def test_full_offload_plans_pass_their_layout_explicitly():
    pl = Planner(_dense_24b(34 / 32), LoadParams(), RX6800, engine_fit=True).plan()
    assert pl.full_offload and not pl.use_engine_fit  # llama.cpp's --fit would move layers to meet an explicit -c
