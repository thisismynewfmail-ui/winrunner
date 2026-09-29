from tests.fixtures import fake_model
from winrunner.config import LoadParams
from winrunner.engine import EngineDevice
from winrunner.planner import (AUTO_UBATCH, MIN_FIT_CTX, Planner, assign_layers, floor_ctx, kv_layer_bytes,
                               override_layers, parse_fit_args, parse_fit_print, swa_layers)

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
    pl = Planner(m, LoadParams(context_length=65536), RX6800).plan()
    assert pl.ctx == 65536
    assert pl.full_offload
    assert pl.kv_k == pl.kv_v == "q8_0"
    assert pl.flash_attn == "on"  # quantized V cache needs flash attention
    assert pl.max_ctx_full_offload["f16"] < 65536 <= pl.max_ctx_full_offload["q8_0"]


def test_user_kv_choice_respected_context_reduced():
    # F16 KV at 65K does not fit: by default the context gives way, never the GPU placement
    m = fake_model(ctx_train=131072)
    pl = Planner(m, LoadParams(context_length=65536, kv_cache_type="f16"), RX6800).plan()
    assert pl.kv_k == "f16" and pl.full_offload and pl.fits
    assert MIN_FIT_CTX <= pl.ctx < 65536 and pl.ctx % 1024 == 0
    assert pl.ctx_reduced_from == 65536 and pl.ctx_requested == 65536
    assert pl.warnings[0].startswith("Context reduced from 65,536")
    assert pl.ubatch == 512  # full offload keeps the default micro-batch
    assert all(d.headroom_mib >= 0 for d in pl.devices)


def test_cpu_offload_policy_keeps_context():
    m = fake_model(ctx_train=131072)
    pl = Planner(m, LoadParams(context_length=65536, kv_cache_type="f16", vram_overflow="cpu_offload"), RX6800).plan()
    assert pl.kv_k == "f16" and not pl.full_offload and pl.ctx == 65536 and pl.ctx_reduced_from == 0
    assert pl.gpu_layers < pl.n_layer + 1
    assert any(w.startswith("Partial offload") and "Reduce context" in w for w in pl.warnings)
    # weights in system RAM: larger micro-batch amortises the PCIe transfer per batch
    assert pl.ubatch == AUTO_UBATCH and pl.batch >= pl.ubatch
    no_auto = Planner(m, LoadParams(context_length=65536, kv_cache_type="f16", vram_overflow="cpu_offload",
                                    auto_batch=False), RX6800).plan()
    assert no_auto.ubatch == 512


def test_q8_then_reduced_context_for_larger_model():
    m = fake_model(ctx_train=131072, layer_mib=345.0)  # ~22 GiB of weights
    pl = Planner(m, LoadParams(context_length=65536), RX6800).plan()
    assert pl.full_offload and pl.kv_k == "q8_0" and pl.flash_attn == "on"
    assert pl.ctx < 65536 and pl.ctx_reduced_from == 65536
    assert pl.max_ctx_full_offload["q8_0"] >= pl.ctx > pl.max_ctx_full_offload["f16"]
    assert pl.max_ctx_full_offload["q4_0"] > pl.max_ctx_full_offload["q8_0"]


def test_manual_full_layers_kv_auto_and_reduce():
    m = fake_model(ctx_train=131072)
    # all layers on GPU: KV auto picks Q8_0 (fits), no overflow warning
    pl = Planner(m, LoadParams(context_length=65536, gpu_offload="manual", n_gpu_layers=-1), RX6800).plan()
    assert pl.mode == "manual" and pl.kv_k == "q8_0" and pl.ctx == 65536 and pl.fits
    assert not any(w.startswith("Manual allocation") for w in pl.warnings)
    # forced F16: the context shrinks so the manual layout still fits
    pl = Planner(m, LoadParams(context_length=65536, gpu_offload="manual", n_gpu_layers=-1, kv_cache_type="f16"),
                 RX6800).plan()
    assert pl.fits and pl.ctx < 65536 and pl.gpu_layers == pl.n_layer + 1
    # ...unless the user wants the context kept: then the over-commit is reported loudly
    pl = Planner(m, LoadParams(context_length=65536, gpu_offload="manual", n_gpu_layers=-1, kv_cache_type="f16",
                               vram_overflow="cpu_offload"), RX6800).plan()
    assert not pl.fits and pl.ctx == 65536
    assert any(w.startswith("Manual allocation exceeds free VRAM") for w in pl.warnings)
    assert not pl.use_engine_fit


def test_floor_ctx_and_override_layers():
    assert floor_ctx(36999) == 36864 and floor_ctx(5000) == 4864 and floor_ctx(10) == 256
    assert override_layers(r"blk\.14\.ffn_(up|down)_exps=CPU,blk\.15\.ffn_down.*=CPU") == [14, 15]


def test_moe_keeps_experts_on_cpu_instead_of_layers():
    big = fake_model(n_layer=36, n_embd=2880, n_head=64, n_head_kv=8, head_dim=64, n_ff=2880, layer_mib=1750,
                     expert_count=128, expert_frac=0.97, ctx_train=131072, sliding_window=128, arch="gpt-oss")
    pl = Planner(big, LoadParams(), RX6800).plan()
    assert not pl.full_offload
    assert pl.gpu_layers == pl.n_layer + 1  # every layer's attention stays on GPU
    assert 0 < pl.n_cpu_moe < pl.n_layer
    assert pl.cpu_expert_layers == list(range(pl.n_cpu_moe))
    assert pl.ubatch == AUTO_UBATCH  # expert weights are streamed per micro-batch
    assert pl.ctx == 65536  # too large even at a minimal context: the requested context is kept
    assert any(n.startswith("The model does not fit in VRAM even") for n in pl.notes)
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
