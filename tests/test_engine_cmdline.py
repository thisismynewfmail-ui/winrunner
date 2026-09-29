
from tests.fixtures import fake_model
from winrunner.cmdline import build_bench_args, build_fit_args, build_server_args
from winrunner.config import LoadParams
from winrunner.engine import EngineDevice, EngineInfo, EngineManager, parse_devices, parse_help_flags, parse_version
from winrunner.planner import Planner

HELP_NEW = """
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
-lm,   --load-mode MODE                 model loading mode (default: auto)
                                        - auto: mmap, unless a device does not support it
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM, either an exact number,
                                        'auto', or 'all' (default: auto)
-ts,   --tensor-split N0,N1,N2,...      fraction of the model to offload to each GPU
-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory
-fitt, --fit-target MiB0,MiB1,MiB2,...
-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the
--mmproj-offload, --no-mmproj-offload   whether to enable GPU offloading for multimodal projector (default:
-mm,   --mmproj FILE                    path to a multimodal projector file.
--spec-type none,draft-simple,ngram-simple
                                        comma-separated list of types of speculative decoding to use (default:
-a,    --alias STRING                   set model name aliases, comma-separated (to be used by API)
--api-key KEY                           API key to use for authentication
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: enabled)
--chat-template JINJA_TEMPLATE          set custom jinja chat template (default: template taken from model's
                                        list of built-in templates:
                                        chatml, gemma, llama3,
                                        mistral-v7
                                        (env: LLAMA_ARG_CHAT_TEMPLATE)
--metrics                               enable prometheus compatible metrics endpoint (default: disabled)
--no-ui, --no-webui                     disable the Web UI
-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold.
--log-jsonl, --no-log-jsonl             Log as JSONL
--log-colors [on|off|auto]              Set colored logging
-np,   --parallel N                     number of server slots (default: -1, -1 = auto)
-kvu,  --kv-unified, -no-kvu, --no-kv-unified
"""

HELP_OLD = """
-c,    --ctx-size N                     size of the prompt context
-fa,   --flash-attn                     enable Flash Attention (default: disabled)
--no-mmap                               do not memory-map model
--mlock                                 force system to keep model in RAM
-ngl,  --gpu-layers, --n-gpu-layers N   number of layers to store in VRAM
-ts,   --tensor-split N0,N1,N2,...      fraction of the model to offload to each GPU
-mm,   --mmproj FILE                    path to a multimodal projector file
--alias STRING                          set alias for model name (to be used by REST API)
--jinja                                 use jinja template for chat (default: disabled)
"""


def engine_from_help(text: str, **kw) -> EngineInfo:
    flags, spec, builtins = parse_help_flags(text)
    return EngineInfo(path="/e/llama-server", name="e", backend="vulkan", flags=flags, spec_types=spec,
                      builtin_templates=builtins, fa_tristate="[on|off" in text, ngl_all="'all'" in text, **kw)


def test_parse_help_flags():
    flags, spec, builtins = parse_help_flags(HELP_NEW)
    for f in ("-fa", "--flash-attn", "--load-mode", "--fit", "--fit-target", "--n-cpu-moe", "--no-mmproj-offload", "--jinja",
              "--no-ui", "-lv", "--log-jsonl", "--kv-unified", "-np"):
        assert f in flags, f
    assert spec == ["none", "draft-simple", "ngram-simple"]
    assert builtins == ["chatml", "gemma", "llama3", "mistral-v7"]


def test_parse_version_formats():
    assert parse_version("version: 0.5.0-dev (build 11240, commit 680a03628)\nbuilt with GNU") == ("0.5.0-dev", 11240, "680a03628")
    assert parse_version("version: 6452 (a1b2c3d4)") == ("6452", 6452, "a1b2c3d4")


def test_parse_devices_vulkan_and_rocm():
    vk = """ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 6800 (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon RX 6800 (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: none
Available devices:
  Vulkan0: AMD Radeon RX 6800 (16368 MiB, 15300 MiB free)
  Vulkan1: AMD Radeon RX 6800 (16368 MiB, 16100 MiB free)
"""
    d = parse_devices(vk)
    assert [(x.name, x.total_mib, x.free_mib) for x in d] == [("Vulkan0", 16368, 15300), ("Vulkan1", 16368, 16100)]
    assert "fp16: 1" in d[0].details
    rocm = "  Device 0: AMD Radeon RX 6800, gfx1030 (0x1030), VMM: no, Wave Size: 32\nAvailable devices:\n  ROCm0: AMD Radeon RX 6800 (16368 MiB, 16222 MiB free)\n"
    r = parse_devices(rocm)
    assert r[0].name == "ROCm0" and "gfx1030" in r[0].details
    assert parse_devices("Available devices:\n  (none)\n") == []


def test_pick_asset_names():
    assets = [{"name": n, "url": "u", "size": 1} for n in (
        "llama-b11240-bin-win-cpu-x64.zip", "llama-b11240-bin-win-vulkan-x64.zip", "llama-b11240-bin-win-rocm-10.0-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip", "llama-b11240-bin-ubuntu-x64.tar.gz", "llama-b11240-bin-ubuntu-vulkan-x64.tar.gz")]
    import winrunner.engine as em

    orig = em._plat_key
    try:
        em._plat_key = lambda: "win"
        assert EngineManager.pick_asset(assets, "vulkan")["name"].endswith("win-vulkan-x64.zip")
        assert EngineManager.pick_asset(assets, "rocm")["name"].endswith("win-rocm-10.0-x64.zip")
        assert EngineManager.pick_asset(assets, "cpu")["name"].endswith("win-cpu-x64.zip")
        em._plat_key = lambda: "linux"
        assert EngineManager.pick_asset(assets, "vulkan")["name"].endswith("ubuntu-vulkan-x64.tar.gz")
        assert EngineManager.pick_asset(assets, "cpu")["name"] == "llama-b11240-bin-ubuntu-x64.tar.gz"
    finally:
        em._plat_key = orig


def _plan(p: LoadParams, fit: bool):
    devs = [EngineDevice("Vulkan0", "RX 6800", 16368, 15300), EngineDevice("Vulkan1", "RX 6800", 16368, 16100)]
    m = fake_model(n_layer=28, n_embd=1024, n_head=16, layer_mib=14, embd_mib=84)
    return Planner(m, p, devs, engine_fit=fit).plan(), [d.name for d in devs]


def test_server_args_new_engine_auto_fit():
    eng = engine_from_help(HELP_NEW)
    p = LoadParams(context_length=16384)
    pl, names = _plan(p, True)
    spec = build_server_args(eng, "/m/model.gguf", p, pl, names, 5100, "my-model", "secret", "/m/mmproj.gguf", None, None,
                             False, 4, [1024, 1024])
    a = spec.args
    assert a[:3] == ["/e/llama-server", "-m", "/m/model.gguf"]
    assert ["--alias", "my-model"] == a[a.index("--alias"):a.index("--alias") + 2]
    assert a[a.index("-c") + 1] == "16384"
    assert a[a.index("-fa") + 1] in ("auto", "on")
    assert "--fit" in a and a[a.index("--fit") + 1] == "on"
    assert a[a.index("--fit-target") + 1] == "1024,1024"
    assert "-ngl" not in a  # the engine fits layers itself
    assert a[a.index("--load-mode") + 1] == "none"
    assert "--mmproj" in a and "--jinja" in a and "--no-ui" in a and "--log-jsonl" in a
    assert "--chat-template" not in a  # GGUF template is used
    assert "--temp" not in a and "--top-k" not in a  # GGUF sampling defaults are not overridden


def test_server_args_old_engine_explicit_plan():
    eng = engine_from_help(HELP_OLD)
    p = LoadParams(context_length=16384)
    pl, names = _plan(p, False)
    a = build_server_args(eng, "/m/model.gguf", p, pl, names, 5100, "m", "", None, None, None, False, 4, [1024]).args
    assert a[a.index("-ngl") + 1] == "999"
    assert "-ts" in a
    assert "-fa" in a and (a.index("-fa") + 1 == len(a) or a[a.index("-fa") + 1].startswith("-"))  # boolean flag
    assert "--no-mmap" in a
    assert "--fit" not in a and "--no-ui" not in a and "--log-jsonl" not in a


def test_manual_moe_override_tensor_fallback():
    eng = engine_from_help(HELP_OLD + "-ot,   --override-tensor <pattern>=<type>\n")
    p = LoadParams(gpu_offload="manual", n_cpu_moe=2)
    pl, names = _plan(p, False)
    a = build_server_args(eng, "/m/x.gguf", p, pl, names, 1, "m", "", None, None, None, False, 4, []).args
    assert "-ot" in a and "blk\\.0\\.ffn_(up|down|gate)_exps=CPU" in a[a.index("-ot") + 1]


def test_template_override_and_extra_args():
    eng = engine_from_help(HELP_NEW)
    p = LoadParams(chat_template_mode="builtin", chat_template_builtin="chatml", extra_args="--override-kv foo=int:1")
    pl, names = _plan(p, True)
    a = build_server_args(eng, "/m/x.gguf", p, pl, names, 1, "m", "", None, None, None, False, 4, []).args
    assert a[a.index("--chat-template") + 1] == "chatml"
    assert a[-2:] == ["--override-kv", "foo=int:1"]


def test_fit_and_bench_args():
    eng = engine_from_help(HELP_NEW, fit_params="/e/llama-fit-params", bench="/e/llama-bench")
    p = LoadParams(context_length=8192)
    pl, names = _plan(p, True)
    fa = build_fit_args(eng, "/m/x.gguf", p, pl, names, [1024, 1024], print_mode=False)
    assert fa[0] == "/e/llama-fit-params" and "--fit-target" in fa and "-ngl" not in fa
    fp = build_fit_args(eng, "/m/x.gguf", p, pl, names, [1024, 1024], print_mode=True)
    assert fp[-2:] == ["--fit-print", "on"] and "-ngl" in fp
    b = build_bench_args(eng, "/m/x.gguf", p, pl, names, "512", "128", "0", 3)
    assert b[0] == "/e/llama-bench" and "-o" in b and b[b.index("-o") + 1] == "json"
    assert "/" in b[b.index("-ts") + 1]  # llama-bench uses '/' separators
