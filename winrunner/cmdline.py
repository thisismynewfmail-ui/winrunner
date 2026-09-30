"""Build llama-server / llama-fit-params / llama-bench command lines.

Arguments are only emitted when the probed engine build supports them, so the
same WinRunner release works across llama.cpp versions. The guiding rule is
"GGUF first": nothing that would override values stored in the GGUF (chat
template, sampling defaults, RoPE parameters, special tokens) is passed unless
the user explicitly configured it.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field

from .config import LoadParams
from .engine import EngineInfo
from .paths import IS_WINDOWS
from .planner import Plan


@dataclass
class LaunchSpec:
    args: list[str]
    notes: list[str] = field(default_factory=list)

    def display(self) -> str:
        return subprocess.list2cmdline(self.args) if IS_WINDOWS else shlex.join(self.args)


def _split_extra(extra: str) -> list[str]:
    extra = (extra or "").strip()
    if not extra:
        return []
    if IS_WINDOWS:
        toks = shlex.split(extra, posix=False)
        return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in toks]
    return shlex.split(extra)


def _ngl(engine: EngineInfo, n: int, n_layer: int) -> str:
    if n > n_layer:
        return "all" if engine.ngl_all else "999"
    return str(n)


def _split_str(split: list[float], sep: str) -> str:
    return sep.join(str(int(x)) if float(x).is_integer() else f"{x:g}" for x in split)


def _ffn_flag_or_overrides(engine: EngineInfo, flag: str, short: str, n: int, groups: tuple[str, ...]) -> list[str]:
    """--n-cpu-moe / --n-cpu-ffn N, or the equivalent --override-tensor entry on builds without the flag."""
    if engine.has(flag, short):
        return [flag, str(n)]
    if engine.has("--override-tensor", "-ot"):
        layers = "|".join(str(i) for i in range(n))
        return ["-ot", f"blk\\.({layers})\\.({'|'.join(groups)})\\.=CPU"]
    return []


def gpu_args(engine: EngineInfo, p: LoadParams, plan: Plan, device_names: list[str]) -> list[str]:
    """Device placement arguments (shared by the server and llama-fit-params)."""
    a: list[str] = []
    if p.devices and engine.has("--device", "-dev"):
        a += ["--device", ",".join(p.devices)]
    if not device_names:
        return a + ["-ngl", "0"]
    if plan.split_mode != "layer" and engine.has("--split-mode", "-sm"):
        a += ["-sm", plan.split_mode]
    if plan.split_mode in ("none", "row") and engine.has("--main-gpu", "-mg"):
        a += ["-mg", str(p.main_gpu)]
    if plan.use_engine_fit and engine.has("--fit", "-fit"):
        return a  # llama.cpp's --fit places the layers itself (placement "engine")
    a += ["-ngl", _ngl(engine, plan.gpu_layers, plan.n_layer)]
    if plan.tensor_split and len(device_names) > 1 and engine.has("--tensor-split", "-ts"):
        a += ["-ts", _split_str(plan.tensor_split, ",")]
    if plan.overrides and engine.has("--override-tensor", "-ot"):
        a += ["-ot", ",".join(plan.overrides)]
    if plan.n_cpu_moe > 0:
        a += _ffn_flag_or_overrides(engine, "--n-cpu-moe", "-ncmoe", plan.n_cpu_moe,
                                    ("ffn_up_exps", "ffn_down_exps", "ffn_gate_exps", "ffn_gate_up_exps"))
    if plan.n_cpu_ffn > 0:
        a += _ffn_flag_or_overrides(engine, "--n-cpu-ffn", "-ncffn", plan.n_cpu_ffn, ("ffn_up", "ffn_down", "ffn_gate"))
    return a


def fit_args(engine: EngineInfo, plan: Plan, margins: list[int]) -> list[str]:
    """llama.cpp's --fit: places the layers for placement "engine"; otherwise it only checks the projected memory.

    With an explicit -ngl the engine never changes the placement: it measures the vision projector, compares
    the projection with the free memory targets and logs the result.
    """
    a: list[str] = []
    if not engine.has("--fit", "-fit"):
        return a
    if plan.mode == "manual":
        return ["--fit", "off"]
    a += ["--fit", "on"]
    if margins and engine.has("--fit-target", "-fitt"):
        a += ["--fit-target", ",".join(str(int(m)) for m in margins)]
    return a


def context_args(engine: EngineInfo, p: LoadParams, plan: Plan, for_fit: bool = False) -> list[str]:
    a = ["-c", str(plan.ctx)]
    if p.batch_size != 2048:
        a += ["-b", str(p.batch_size)]
    if p.ubatch_size != 512:
        a += ["-ub", str(p.ubatch_size)]
    if engine.has("--flash-attn", "-fa"):
        if engine.fa_tristate:
            a += ["-fa", plan.flash_attn]
        elif plan.flash_attn in ("on", "auto"):
            a += ["-fa"]
    if plan.kv_k != "f16":
        a += ["-ctk", plan.kv_k]
    if plan.kv_v != "f16":
        a += ["-ctv", plan.kv_v]
    if not p.kv_offload:
        a += ["-nkvo"]
    if p.swa_full and engine.has("--swa-full"):
        a += ["--swa-full"]
    # llama-fit-params keeps its default single sequence: the server's slots share one unified KV cache of the
    # same size (--kv-unified), which that measures most closely
    if plan.parallel and plan.parallel > 0 and not for_fit:
        a += ["-np", str(plan.parallel)]
    if p.threads > 0:
        a += ["-t", str(p.threads)]
    if p.threads_batch > 0:
        a += ["-tb", str(p.threads_batch)]
    lm = plan.load_mode
    if engine.has("--load-mode", "-lm"):
        if lm and lm != "auto":
            a += ["--load-mode", lm]
    else:  # older builds
        if lm == "none" and engine.has("--no-mmap"):
            a += ["--no-mmap"]
        if "mlock" in lm and engine.has("--mlock"):
            a += ["--mlock"]
    if p.rope_scaling:
        a += ["--rope-scaling", p.rope_scaling]
    if p.rope_freq_base > 0:
        a += ["--rope-freq-base", f"{p.rope_freq_base:g}"]
    if p.rope_freq_scale > 0:
        a += ["--rope-freq-scale", f"{p.rope_freq_scale:g}"]
    if p.yarn_orig_ctx > 0:
        a += ["--yarn-orig-ctx", str(p.yarn_orig_ctx)]
    return a


def build_server_args(
    engine: EngineInfo,
    model_path: str,
    p: LoadParams,
    plan: Plan,
    device_names: list[str],
    port: int,
    alias: str,
    api_key: str,
    mmproj: str | None,
    draft_path: str | None,
    template_file: str | None,
    is_embedding: bool,
    verbosity: int,
    margins: list[int],
) -> LaunchSpec:
    notes: list[str] = []
    a = [engine.path, "-m", model_path, "--host", "127.0.0.1", "--port", str(port)]
    if engine.has("--alias", "-a"):
        a += ["--alias", alias]
    if api_key and engine.has("--api-key"):
        a += ["--api-key", api_key]
    a += context_args(engine, p, plan)
    a += gpu_args(engine, p, plan, device_names)
    a += fit_args(engine, plan, margins)

    if plan.parallel and plan.parallel > 1 and engine.has("--kv-unified", "-kvu"):
        a += ["--kv-unified"]  # every slot may use the whole context; memory is shared

    if mmproj:
        a += ["--mmproj", mmproj]
        if not p.mmproj_offload and engine.has("--no-mmproj-offload"):
            a += ["--no-mmproj-offload"]
        if p.image_min_tokens > 0 and engine.has("--image-min-tokens"):
            a += ["--image-min-tokens", str(p.image_min_tokens)]
        if p.image_max_tokens > 0 and engine.has("--image-max-tokens"):
            a += ["--image-max-tokens", str(p.image_max_tokens)]
    elif engine.has("--no-mmproj"):
        a += ["--no-mmproj"]

    # Chat template: the GGUF's embedded Jinja template is used unless overridden.
    if engine.has("--jinja"):
        a += ["--jinja"]
    if p.chat_template_mode == "builtin" and p.chat_template_builtin:
        a += ["--chat-template", p.chat_template_builtin]
    elif p.chat_template_mode == "custom" and template_file:
        a += ["--chat-template-file", template_file]
    if p.chat_template_kwargs.strip() and engine.has("--chat-template-kwargs"):
        a += ["--chat-template-kwargs", p.chat_template_kwargs.strip()]
    if p.reasoning_format != "auto" and engine.has("--reasoning-format"):
        a += ["--reasoning-format", p.reasoning_format]
    if p.reasoning != "auto":
        if engine.has("--reasoning", "-rea"):
            a += ["--reasoning", p.reasoning]
        elif p.reasoning == "off" and engine.has("--reasoning-budget"):
            a += ["--reasoning-budget", "0"]
    if p.reasoning_budget >= 0 and engine.has("--reasoning-budget") and p.reasoning != "off":
        a += ["--reasoning-budget", str(p.reasoning_budget)]

    if p.context_shift and engine.has("--context-shift"):
        a += ["--context-shift"]
    if p.cache_ram_mib != 8192 and engine.has("--cache-ram", "-cram"):
        a += ["--cache-ram", str(p.cache_ram_mib)]
    if p.cache_reuse > 0 and engine.has("--cache-reuse"):
        a += ["--cache-reuse", str(p.cache_reuse)]

    if draft_path:
        flag = "--spec-draft-model" if engine.has("--spec-draft-model") else "-md"
        a += [flag, draft_path]
        if p.draft_max > 0:
            if engine.has("--spec-draft-n-max"):
                a += ["--spec-draft-n-max", str(p.draft_max)]
            elif engine.has("--draft-max"):
                a += ["--draft-max", str(p.draft_max)]
    if p.spec_type and engine.has("--spec-type"):
        a += ["--spec-type", p.spec_type]

    if (is_embedding or p.embeddings) and engine.has("--embeddings", "--embedding"):
        a += ["--embeddings"]
    if engine.has("--metrics"):
        a += ["--metrics"]
    if engine.has("--slots"):
        a += ["--slots"]
    if engine.has("--no-ui"):
        a += ["--no-ui"]
    elif engine.has("--no-webui"):
        a += ["--no-webui"]
    if engine.has("--log-verbosity", "-lv"):
        a += ["-lv", str(verbosity)]
    if engine.has("--log-colors"):
        a += ["--log-colors", "off"]
    if engine.has("--log-jsonl"):
        a += ["--log-jsonl"]

    extra = _split_extra(p.extra_args)
    if extra:
        notes.append(f"{len(extra)} extra argument(s) appended from load settings")
        a += extra
    return LaunchSpec(a, notes)


def build_fit_args(engine: EngineInfo, model_path: str, p: LoadParams, plan: Plan, device_names: list[str],
                   margins: list[int], print_mode: bool) -> list[str] | None:
    """Arguments for llama-fit-params.

    ``print_mode``: measure the plan's own placement (--fit-print: model / context / compute MiB per device).
    Otherwise: let llama.cpp fit layers within ``margins`` and print the resulting arguments.
    """
    if not engine.fit_params:
        return None
    a = [engine.fit_params, "-m", model_path]
    a += context_args(engine, p, plan, for_fit=True)
    if print_mode:
        a += gpu_args(engine, p, _explicit(plan), device_names)
        a += ["--fit-print", "on"]
    else:
        a += gpu_args(engine, p, _engine_placed(plan), device_names)
        if engine.has("--fit-target", "-fitt") and margins:
            a += ["--fit-target", ",".join(str(int(m)) for m in margins)]
    return a


def _explicit(plan: Plan) -> Plan:
    import copy

    q = copy.copy(plan)
    q.use_engine_fit = False
    return q


def _engine_placed(plan: Plan) -> Plan:
    import copy

    q = copy.copy(plan)
    q.use_engine_fit = True
    return q


def build_bench_args(engine: EngineInfo, model_path: str, p: LoadParams, plan: Plan, device_names: list[str],
                     n_prompt: str, n_gen: str, depth: str, reps: int, threads: int = 0) -> list[str] | None:
    """llama-bench with the plan's placement (same layers, split and weights in RAM as the server)."""
    if not engine.bench:
        return None
    q = _explicit(plan)
    a = [engine.bench, "-m", model_path, "-o", "json", "-r", str(reps), "-p", n_prompt, "-n", n_gen]
    if depth and depth != "0":
        a += ["-d", depth]
    if p.batch_size != 2048:
        a += ["-b", str(p.batch_size)]
    if p.ubatch_size != 512:
        a += ["-ub", str(p.ubatch_size)]
    a += ["-fa", q.flash_attn if q.flash_attn != "auto" else "on"]
    a += ["-ctk", q.kv_k, "-ctv", q.kv_v]
    if device_names:
        a += ["-ngl", str(q.gpu_layers if q.gpu_layers <= q.n_layer else 999)]
        if q.split_mode != "layer":
            a += ["-sm", q.split_mode]
        if q.tensor_split and len(device_names) > 1:
            a += ["-ts", _split_str(q.tensor_split, "/")]
        if q.overrides:
            a += ["-ot", ";".join(q.overrides)]  # llama-bench: ';' within one configuration
        if q.n_cpu_moe > 0:
            a += ["-ncmoe", str(q.n_cpu_moe)]
        if q.n_cpu_ffn > 0:
            layers = "|".join(str(i) for i in range(q.n_cpu_ffn))
            a += ["-ot", f"blk\\.({layers})\\.(ffn_up|ffn_down|ffn_gate)\\.=CPU"]
        if p.devices:
            a += ["-dev", "/".join(p.devices)]
    else:
        a += ["-ngl", "0"]
    if q.load_mode and q.load_mode != "auto":
        a += ["-lm", q.load_mode]
    t = p.threads if p.threads > 0 else threads
    if t > 0:
        a += ["-t", str(t)]  # llama-bench defaults to 4 threads, the server to one per physical core
    return a
