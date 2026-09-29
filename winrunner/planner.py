"""GPU memory planning.

Given a model (parsed GGUF), the requested load parameters and the devices the
engine can see (with their current free memory), the planner decides:

* the effective context length (clamped to the trained context unless the user
  explicitly allows RoPE extension),
* the KV cache precision (``auto`` prefers F16 and falls back to Q8_0 only if
  that is what makes a full GPU offload possible),
* what gives way when the model and context do not fit in VRAM: by default the
  context is reduced so every layer stays on the GPUs (``vram_overflow =
  reduce_context``); keeping the context instead (``cpu_offload``) leaves
  layers (dense) or expert blocks (MoE) in system RAM, which the engine then
  streams over PCIe for every prompt batch - several times slower,
* how layers are distributed across GPUs (``--tensor-split``).

It produces a per-device memory breakdown (weights / KV cache / compute
buffers / vision projector) used by the UI, and the corresponding engine
arguments. The analytic estimate mirrors llama.cpp's allocation rules; when the
engine ships ``llama-fit-params`` the plan is verified against llama.cpp's own
projection before loading (see ``manager.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import QUANTIZED_KV, LoadParams
from .engine import EngineDevice
from .gguf import ModelInfo

MiB = 1024 * 1024

KV_TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}

# Default sliding-window patterns for architectures whose GGUF omits the key.
SWA_PATTERN_DEFAULTS = {
    "gemma2": 2,
    "gemma3": 6,
    "gemma3n": 5,
    "cohere2": 4,
    "gpt-oss": 2,
    "exaone4": 4,
    "llama4": 4,
}

CTX_PAD = 256
MIN_FIT_CTX = 4096  # never reduce the context below this to reach a full offload (llama.cpp uses the same floor)
AUTO_UBATCH = 2048  # micro-batch used when weights live in system RAM (amortises the PCIe transfer)


@dataclass
class DevicePlan:
    name: str
    description: str
    gpu_id: str | None
    total_mib: float
    free_mib: float
    margin_mib: float
    layer_first: int = -1
    layer_last: int = -1
    layers: int = 0
    weights_mib: float = 0.0
    kv_mib: float = 0.0
    compute_mib: float = 0.0
    mmproj_mib: float = 0.0
    draft_mib: float = 0.0
    output_mib: float = 0.0

    @property
    def used_mib(self) -> float:
        return self.weights_mib + self.kv_mib + self.compute_mib + self.mmproj_mib + self.draft_mib + self.output_mib

    @property
    def headroom_mib(self) -> float:
        return self.free_mib - self.margin_mib - self.used_mib

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["used_mib"] = round(self.used_mib, 1)
        d["headroom_mib"] = round(self.headroom_mib, 1)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 1)
        return d


@dataclass
class Plan:
    model_id: str
    mode: str
    ctx_requested: int
    ctx: int
    ctx_train: int
    kv_k: str
    kv_v: str
    flash_attn: str
    n_layer: int
    gpu_layers: int
    full_offload: bool
    n_cpu_moe: int = 0
    tensor_split: list[float] | None = None
    split_mode: str = "layer"
    devices: list[DevicePlan] = field(default_factory=list)
    host: dict[str, float] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    kv_bytes_per_token: float = 0.0
    max_ctx_full_offload: dict[str, int] = field(default_factory=dict)
    load_mode: str = "mmap"
    parallel: int = -1
    batch: int = 2048
    ubatch: int = 512
    vram_overflow: str = "reduce_context"
    ctx_reduced_from: int = 0  # requested context when it was reduced to keep the model in VRAM
    cpu_expert_layers: list[int] = field(default_factory=list)  # layers whose MoE experts stay in system RAM
    fits: bool = True  # estimated allocations stay within free VRAM minus margins
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "estimate"
    use_engine_fit: bool = False
    engine: dict[str, Any] = field(default_factory=dict)
    mmproj: str = ""
    draft: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["devices"] = [x.to_dict() for x in self.devices]
        return d


# ---------------------------------------------------------------------------
# Model memory model
# ---------------------------------------------------------------------------


def _layer_heads_kv(info: ModelInfo, il: int) -> int:
    v = info.n_head_kv
    if isinstance(v, list):
        return int(v[il]) if il < len(v) else int(v[-1]) if v else 0
    return int(v or 0)


def swa_layers(info: ModelInfo) -> list[bool]:
    n = info.n_layer
    if not info.sliding_window:
        return [False] * n
    pat = info.swa_pattern
    if isinstance(pat, list) and len(pat) >= n:
        return [bool(x) for x in pat[:n]]
    if isinstance(pat, bool):
        pat = None
    if not isinstance(pat, int) or pat <= 0:
        pat = SWA_PATTERN_DEFAULTS.get(info.architecture, 0)
    if pat <= 0:
        return [False] * n
    return [(il % pat) < (pat - 1) for il in range(n)]


def kv_layer_elems(info: ModelInfo) -> list[tuple[int, int]]:
    """(K elements, V elements) per token for each layer (0,0 = no KV)."""
    out = []
    rope_dim = int(info.rope_dim or 64)
    for il in range(info.n_layer):
        nkv = _layer_heads_kv(info, il)
        if info.full_attention_interval and (il + 1) % info.full_attention_interval != 0:
            out.append((0, 0))  # linear-attention / recurrent layer
            continue
        if nkv == 0 and (info.recurrent or isinstance(info.n_head_kv, list)):
            out.append((0, 0))
            continue
        if info.kv_lora_rank:
            out.append((info.kv_lora_rank + rope_dim, 0))  # MLA: compressed latent + rope part
            continue
        out.append((nkv * info.head_dim_k, nkv * info.head_dim_v))
    return out


def kv_cells(info: ModelInfo, ctx: int, ubatch: int, n_seq: int, swa_full: bool) -> list[int]:
    swa = swa_layers(info)
    swa_cells = ctx
    if info.sliding_window and not swa_full:
        swa_cells = min(ctx, _pad(info.sliding_window * max(1, n_seq) + ubatch, CTX_PAD))
    return [swa_cells if s else ctx for s in swa]


def kv_layer_bytes(info: ModelInfo, ctx: int, kv_k: str, kv_v: str, ubatch: int, n_seq: int,
                   swa_full: bool) -> list[float]:
    kb, vb = KV_TYPE_BYTES.get(kv_k, 2.0), KV_TYPE_BYTES.get(kv_v, 2.0)
    elems = kv_layer_elems(info)
    cells = kv_cells(info, ctx, ubatch, n_seq, swa_full)
    return [c * (k * kb + v * vb) for (k, v), c in zip(elems, cells)]


def compute_buffer_bytes(info: ModelInfo, ctx: int, ubatch: int, flash: bool, has_output: bool) -> float:
    """Approximate llama.cpp compute (graph) buffer for one device."""
    n_ff = max(info.n_ff, 1)
    if info.expert_count and info.expert_used_count:
        n_ff = max(n_ff, 1)
    act = ubatch * (info.n_embd * 8 + n_ff * 3) * 4
    if flash:
        attn = ubatch * info.n_embd * 4 * 4
    else:
        attn = float(ctx) * ubatch * max(info.n_head, 1) * 4
    logits = ubatch * info.n_vocab * 4 if has_output else 0
    return max(logits, attn) + act + 32 * MiB


def _pad(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def floor_ctx(x: int) -> int:
    """Round a fitted context down to a tidy value (multiple of 1024 above 8K, else of 256)."""
    step = 1024 if x >= 8192 else CTX_PAD
    return max(CTX_PAD, (int(x) // step) * step)


# Placement messages are generated here and by the engine verification in manager.py;
# both replace the messages that start with these prefixes.
PLACEMENT_PREFIXES = ("Context reduced from", "Partial offload:", "Expert weights of", "Micro-batch raised",
                      "The model does not fit in VRAM even", "KV cache set to Q8_0")
_SET_REDUCE = " Set 'If it does not fit' to 'Reduce context' to keep everything in VRAM."


def msg_ctx_reduced(requested: int, ctx: int) -> str:
    return (f"Context reduced from {requested:,} to {ctx:,} tokens so the whole model stays in VRAM (with part of "
            "the model in system RAM, prompt processing is several times slower). To keep the full context, set "
            "'If it does not fit' to 'Offload to CPU', use a smaller quantization, or free VRAM.")


def msg_partial_layers(on_gpu: int, n_layer: int, reduce: bool) -> str:
    return (f"Partial offload: {on_gpu} of {n_layer} layers on the GPUs, {n_layer - on_gpu} in system RAM. Prompt "
            "processing streams the CPU-side weights over PCIe for every batch and generation runs those layers on "
            "the CPU - both several times slower than a full GPU offload." + ("" if reduce else _SET_REDUCE))


def msg_partial_experts(n_moe_layers: int, n_layer: int, reduce: bool) -> str:
    return (f"Expert weights of {n_moe_layers} of {n_layer} layers stay in system RAM (attention, shared weights and "
            "the KV cache stay on the GPUs). Prompt processing streams those weights over PCIe for every batch and is "
            "much slower than a full GPU offload." + ("" if reduce else _SET_REDUCE))


def msg_ubatch(ub: int) -> str:
    return (f"Micro-batch raised to {ub} tokens: weights in system RAM are streamed to the GPU once per micro-batch, "
            "so larger batches process prompts much faster.")


def msg_no_fit_min() -> str:
    return (f"The model does not fit in VRAM even at a {MIN_FIT_CTX:,}-token context, so the requested context was "
            "kept.")


def msg_manual_over(over_mib: float) -> str:
    return (f"Manual allocation exceeds free VRAM by {over_mib / 1024:.1f} GiB. Windows will page the excess into "
            "shared system memory, which makes prompt processing and generation extremely slow. Use automatic "
            "allocation, lower the context, or put fewer layers on the GPU.")


def override_layers(pattern: str) -> list[int]:
    """Layer indices named in a tensor-override list such as 'blk\\.14\\.ffn_(up|down)_exps=CPU,...'."""
    import re

    return sorted({int(m) for m in re.findall(r"blk\\?\.(\d+)\\?\.", pattern or "")})


def msg_kv_q8(full_offload: bool) -> str:
    if full_offload:
        return "KV cache set to Q8_0 (near-lossless, half the size of F16) so more context fits in VRAM."
    return "KV cache set to Q8_0 (near-lossless) to keep as much of the model on the GPU(s) as possible."


def mmproj_estimate_mib(file_size: int) -> float:
    return file_size / MiB * 1.15 + 64


# ---------------------------------------------------------------------------
# Layer assignment (mirrors llama.cpp's layer split)
# ---------------------------------------------------------------------------


def assign_layers(n_layer: int, n_gpu: int, split: list[float]) -> list[int]:
    """Device index for each of layers 0..n_layer (index n_layer = output).

    -1 means CPU. Offloaded layers are the *last* ``n_gpu`` layers; the output
    layer is offloaded when ``n_gpu > n_layer``.
    """
    n_dev = len(split)
    res = [-1] * (n_layer + 1)
    if n_dev == 0 or n_gpu <= 0:
        return res
    i_gpu_start = max(n_layer + 1 - n_gpu, 0)
    act = min(n_gpu, n_layer + 1)
    total = sum(split) or 1.0
    cum = []
    acc = 0.0
    for s in split:
        acc += s / total
        cum.append(acc)
    for il in range(i_gpu_start, n_layer + 1):
        f = (il - i_gpu_start) / act
        d = next((i for i, c in enumerate(cum) if f < c), n_dev - 1)
        res[il] = d
    return res


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    def __init__(
        self,
        info: ModelInfo,
        params: LoadParams,
        devices: list[EngineDevice],
        device_gpu_map: dict[str, str] | None = None,
        margin_mib: int = 1024,
        margin_per_device: dict[str, int] | None = None,
        reclaim_mib: dict[str, float] | None = None,
        mmproj_size: int = 0,
        mmproj_mib_hint: float | None = None,
        draft_info: ModelInfo | None = None,
        engine_fit: bool = False,
        model_id: str = "",
    ):
        self.info = info
        self.p = params
        sel = set(params.devices or [])
        self.devices = [d for d in devices if not sel or d.name in sel]
        if params.split_mode == "none" and self.devices:
            idx = min(max(params.main_gpu, 0), len(self.devices) - 1)
            self.devices = [self.devices[idx]]
        self.map = device_gpu_map or {}
        self.margin = margin_mib
        self.margin_dev = margin_per_device or {}
        self.reclaim = reclaim_mib or {}
        self.mmproj_mib = (mmproj_mib_hint if mmproj_mib_hint else mmproj_estimate_mib(mmproj_size)) if mmproj_size else 0.0
        self.draft = draft_info
        self.engine_fit = engine_fit
        self.model_id = model_id
        self.ub = params.ubatch_size  # micro-batch in effect for the estimate
        self.b = params.batch_size

    # ----- helpers ----------------------------------------------------------------

    def _n_seq(self) -> int:
        return self.p.parallel if self.p.parallel and self.p.parallel > 0 else 4

    def _flash_for_estimate(self) -> bool:
        return self.p.flash_attn != "off"

    def _kv_pair(self, k: str) -> tuple[str, str]:
        v = self.p.kv_cache_type_v or k
        return k, v

    def resolve_ctx(self) -> tuple[int, list[str]]:
        notes = []
        req = int(self.p.context_length)
        train = int(self.info.context_length or 0)
        ctx = req
        if train and req > train and not self.p.allow_context_over_train:
            ctx = train
            notes.append(
                f"Context {req:,} exceeds the model's trained context ({train:,}); clamped to {train:,}. "
                "Enable 'Allow context above trained length' to use RoPE scaling instead."
            )
        elif train and req > train:
            notes.append(f"Context {req:,} exceeds trained context {train:,}: output quality may degrade.")
        return max(CTX_PAD, _pad(ctx, CTX_PAD) if ctx % CTX_PAD else ctx), notes

    def _layout(self, ctx: int, kv_k: str, kv_v: str, n_gpu: int, n_cpu_moe: int,
                split: list[float] | None) -> tuple[list[DevicePlan], dict[str, float], list[int]]:
        info = self.info
        n_layer = info.n_layer
        devs = [
            DevicePlan(
                name=d.name,
                description=d.description,
                gpu_id=self.map.get(d.name),
                total_mib=float(d.total_mib),
                free_mib=float(d.free_mib) + float(self.reclaim.get(d.name, 0.0)),
                margin_mib=float(self.margin_dev.get(d.name, self.margin)),
            )
            for d in self.devices
        ]
        host = {"weights_mib": 0.0, "kv_mib": 0.0, "compute_mib": 0.0}
        kvl = kv_layer_bytes(info, ctx, kv_k, kv_v, self.ub, self._n_seq(), self.p.swa_full)
        if split is None:
            split = [max(1.0, d.free_mib - d.margin_mib) for d in devs]
        assign = assign_layers(n_layer, n_gpu if devs else 0, split) if devs else [-1] * (n_layer + 1)
        flash = self._flash_for_estimate()
        host["weights_mib"] += info.token_embd_bytes / MiB  # input embeddings always stay in host memory
        for il in range(n_layer):
            d = assign[il]
            lw = info.layer_bytes[il] if il < len(info.layer_bytes) else 0
            le = info.layer_expert_bytes[il] if il < len(info.layer_expert_bytes) else 0
            kv = kvl[il] if il < len(kvl) else 0
            if d < 0:
                host["weights_mib"] += lw / MiB
                host["kv_mib"] += kv / MiB
                continue
            dp = devs[d]
            moved = le if il < n_cpu_moe else 0
            dp.weights_mib += (lw - moved) / MiB
            host["weights_mib"] += moved / MiB
            if self.p.kv_offload:
                dp.kv_mib += kv / MiB
            else:
                host["kv_mib"] += kv / MiB
            dp.layers += 1
            if dp.layer_first < 0:
                dp.layer_first = il
            dp.layer_last = il
        out_dev = assign[n_layer] if n_layer < len(assign) else -1
        extra_layers = len(info.layer_bytes) - n_layer  # e.g. MTP layers (not used for decoding)
        if extra_layers > 0:
            host["weights_mib"] += sum(info.layer_bytes[n_layer:]) / MiB
        if out_dev >= 0:
            devs[out_dev].output_mib += (info.output_bytes + info.other_bytes) / MiB
        else:
            host["weights_mib"] += (info.output_bytes + info.other_bytes) / MiB
        for i, dp in enumerate(devs):
            if dp.layers or i == out_dev:
                dp.compute_mib = compute_buffer_bytes(info, ctx, self.ub, flash, i == out_dev) / MiB
        if n_gpu < n_layer + 1 or not devs:
            host["compute_mib"] = compute_buffer_bytes(info, ctx, self.ub, flash, out_dev < 0) / MiB
        else:
            host["compute_mib"] = (self.ub * info.n_embd * 4 * 2) / MiB + 8
        if devs and self.mmproj_mib:
            if self.p.mmproj_offload:
                devs[0].mmproj_mib = self.mmproj_mib
            else:
                host["compute_mib"] += self.mmproj_mib
        if devs and self.draft is not None:
            dw = self.draft.weights_bytes / MiB
            dkv = sum(kv_layer_bytes(self.draft, ctx, "f16", "f16", self.ub, 1, False)) / MiB
            dcomp = compute_buffer_bytes(self.draft, ctx, self.ub, flash, True) / MiB
            tot = sum(split) or 1
            for dp, s in zip(devs, split):
                dp.draft_mib = (dw + dkv) * s / tot
            devs[-1].draft_mib += dcomp
        return devs, host, assign

    @staticmethod
    def _fits(devs: list[DevicePlan]) -> bool:
        return all(d.headroom_mib >= 0 for d in devs)

    def _user_split(self) -> list[float] | None:
        ts = self.p.tensor_split
        if ts and len(ts) == len(self.devices) and sum(ts) > 0:
            return [float(x) for x in ts]
        return None

    def _split_for(self, ctx: int, kv_k: str, kv_v: str, n_gpu: int, n_cpu_moe: int) -> list[float] | None:
        """Cost-aware tensor split: contiguous layer ranges sized to each GPU's capacity.

        Layers are not uniform once expert weights move to the CPU (MoE) or the
        output layer is included, so the split is computed from per-layer VRAM
        cost (weights + KV cache) rather than layer counts. The result is passed
        to llama.cpp as ``--tensor-split`` layer counts.
        """
        user = self._user_split()
        if user is not None:
            return user
        nd = len(self.devices)
        if nd <= 1:
            return None
        info = self.info
        n_layer = info.n_layer
        start = max(n_layer + 1 - n_gpu, 0)
        kvl = kv_layer_bytes(info, ctx, kv_k, kv_v, self.ub, self._n_seq(), self.p.swa_full)
        costs = []
        for il in range(start, n_layer + 1):
            if il == n_layer:
                c = info.output_bytes + info.other_bytes
            else:
                lw = info.layer_bytes[il] if il < len(info.layer_bytes) else 0
                le = info.layer_expert_bytes[il] if il < n_cpu_moe and il < len(info.layer_expert_bytes) else 0
                c = lw - le + (kvl[il] if self.p.kv_offload and il < len(kvl) else 0)
            costs.append(c / MiB)
        probe, _, _ = self._layout(ctx, kv_k, kv_v, n_gpu, n_cpu_moe, [1.0] * nd)
        flash = self._flash_for_estimate()
        caps = []
        for i, d in enumerate(probe):
            comp = compute_buffer_bytes(info, ctx, self.ub, flash, i == nd - 1) / MiB
            caps.append(max(0.0, d.free_mib - d.margin_mib - comp - d.mmproj_mib - d.draft_mib))
        total_cost, total_cap = sum(costs), sum(caps)
        if total_cap <= 0 or total_cost <= 0:
            return [max(1.0, d.free_mib) for d in probe]
        counts = [0] * nd
        cum, di = 0.0, 0
        bound = caps[0] / total_cap * total_cost
        for c in costs:
            while di < nd - 1 and cum + c / 2 > bound:
                di += 1
                bound += caps[di] / total_cap * total_cost
            counts[di] += 1
            cum += c
        return [float(x) for x in counts]

    def _try(self, ctx: int, kv: str, n_gpu: int, n_cpu_moe: int) -> tuple[bool, list[float] | None,
                                                                         list[DevicePlan], dict[str, float]]:
        """Layout for one configuration: (fits, split, devices, host)."""
        k, v = self._kv_pair(kv)
        s = self._split_for(ctx, k, v, n_gpu, n_cpu_moe)
        d, h, _ = self._layout(ctx, k, v, n_gpu, n_cpu_moe, s)
        return self._fits(d), s, d, h

    def _max_ctx(self, kv: str, n_gpu: int, n_cpu_moe: int, hi: int) -> int:
        """Largest context (multiple of CTX_PAD, <= hi) at which the configuration fits; 0 if none."""
        if not self.devices:
            return 0
        lo = CTX_PAD
        if not self._try(lo, kv, n_gpu, n_cpu_moe)[0]:
            return 0
        hi = max(lo, _pad(int(hi), CTX_PAD))
        if self._try(hi, kv, n_gpu, n_cpu_moe)[0]:
            return hi
        while hi - lo > CTX_PAD:
            mid = ((lo + hi) // 2) // CTX_PAD * CTX_PAD
            if mid <= lo or mid >= hi:
                break
            if self._try(mid, kv, n_gpu, n_cpu_moe)[0]:
                lo = mid
            else:
                hi = mid
        return lo

    def _ctx_ceiling(self) -> int:
        hi = max(CTX_PAD, int(self.info.context_length or 131072))
        if self.p.allow_context_over_train:
            hi = max(hi, int(self.p.context_length))
        return hi

    # ----- main entry ---------------------------------------------------------------

    def plan(self) -> Plan:
        info, p = self.info, self.p
        ctx, notes = self.resolve_ctx()
        ctx_req = ctx
        warnings: list[str] = []
        n_layer = info.n_layer
        full = n_layer + 1
        has_dev = bool(self.devices)
        reduce = p.vram_overflow == "reduce_context"
        self.ub, self.b = p.ubatch_size, p.batch_size

        fa = p.flash_attn
        if p.kv_cache_type == "auto":
            candidates = ["f16", "q8_0"]
        else:
            candidates = [p.kv_cache_type]
        if fa == "off":
            quant = [c for c in candidates if c in QUANTIZED_KV or (p.kv_cache_type_v or c) in QUANTIZED_KV]
            if quant and p.kv_cache_type != "auto":
                warnings.append("A quantized KV cache requires flash attention; flash attention set to 'on'.")
                fa = "on"
            candidates = [c for c in candidates if c not in QUANTIZED_KV] or ["f16"]

        manual = p.gpu_offload == "manual"
        chosen_kv = candidates[0]
        split: list[float] | None = None
        n_gpu = full if has_dev else 0
        n_cpu_moe = 0
        devs: list[DevicePlan] = []
        host: dict[str, float] = {}
        fits = True

        def reduce_to_fit(kv: str, ngl: int, ncmoe: int) -> bool:
            """Shrink the context until the configuration fits (vram_overflow = reduce_context)."""
            nonlocal ctx, split, devs, host, fits
            mc = self._max_ctx(kv, ngl, ncmoe, ctx)
            if mc < min(MIN_FIT_CTX, ctx):
                return False
            new = min(ctx, floor_ctx(mc))
            ok, s, d, h = self._try(new, kv, ngl, ncmoe)
            if not ok:
                return False
            ctx, split, devs, host, fits = new, s, d, h, True
            return True

        if not has_dev:
            n_gpu = 0
            k, v = self._kv_pair(chosen_kv)
            devs, host, _ = self._layout(ctx, k, v, 0, 0, None)
        elif manual:
            n_gpu = full if p.n_gpu_layers < 0 else min(full, p.n_gpu_layers)
            n_cpu_moe = max(0, p.n_cpu_moe)
            for kv in candidates:  # KV "auto": F16 when the manual layout fits, else Q8_0
                chosen_kv = kv
                fits, split, devs, host = self._try(ctx, kv, n_gpu, n_cpu_moe)
                if fits:
                    break
            if not fits and reduce:
                reduce_to_fit(chosen_kv, n_gpu, n_cpu_moe)
            if not fits:
                warnings.append(msg_manual_over(sum(max(0.0, -d.headroom_mib) for d in devs)))
        else:
            fitted = False
            for kv in candidates:
                ok, s, d, h = self._try(ctx, kv, full, 0)
                if ok:
                    chosen_kv, split, devs, host, n_gpu, fitted = kv, s, d, h, full, True
                    break
            if not fitted and reduce:
                chosen_kv = candidates[-1]
                fitted = reduce_to_fit(chosen_kv, full, 0)
                n_gpu = full
            if not fitted:
                chosen_kv = candidates[-1]
                k, v = self._kv_pair(chosen_kv)
                if p.auto_batch and self.ub < AUTO_UBATCH:
                    self.ub = AUTO_UBATCH
                    self.b = max(self.b, self.ub)
                    notes.append(msg_ubatch(self.ub))
                if info.expert_count and any(info.layer_expert_bytes):
                    # MoE: keep attention + KV on GPU, move expert FFNs of the first N layers to system RAM.
                    best = None
                    for n in range(0, n_layer + 1):
                        ok, s, d, h = self._try(ctx, chosen_kv, full, n)
                        if ok:
                            best = (n, s, d, h)
                            break
                    if best is None:
                        s = self._split_for(ctx, k, v, full, n_layer)
                        best = (n_layer, s) + self._layout(ctx, k, v, full, n_layer, s)[:2]
                        warnings.append("Even with all expert weights in system RAM the model does not fit in VRAM; "
                                        "reduce the context length or use a smaller quantization.")
                        fits = False
                    n_cpu_moe, split, devs, host = best
                    n_gpu = full
                    warnings.append(msg_partial_experts(n_cpu_moe, n_layer, reduce))
                else:
                    s0 = self._split_for(ctx, k, v, 0, 0)
                    best = (0, s0) + self._layout(ctx, k, v, 0, 0, s0)[:2]
                    for n in range(full, -1, -1):
                        ok, s, d, h = self._try(ctx, chosen_kv, n, 0)
                        if ok:
                            best = (n, s, d, h)
                            break
                    n_gpu, split, devs, host = best
                    on_gpu = sum(dp.layers for dp in devs)
                    warnings.append(msg_partial_layers(on_gpu, n_layer, reduce))
                if reduce:
                    notes.append(msg_no_fit_min())

        k, v = self._kv_pair(chosen_kv)
        full_offload = has_dev and n_gpu >= full and n_cpu_moe == 0
        if ctx < ctx_req:
            warnings.insert(0, msg_ctx_reduced(ctx_req, ctx))
        if p.kv_cache_type == "auto" and chosen_kv != "f16":
            notes.append(msg_kv_q8(full_offload))
        if fa == "auto" and (k in QUANTIZED_KV or v in QUANTIZED_KV):
            fa = "on"

        kv_total = sum(kv_layer_bytes(info, ctx, k, v, self.ub, self._n_seq(), p.swa_full)) / MiB
        max_ctx: dict[str, int] = {}
        if has_dev:
            saved = self.ub, self.b
            self.ub, self.b = p.ubatch_size, p.batch_size
            hi = self._ctx_ceiling()
            for kvt in ("f16", "q8_0", "q4_0"):
                if kvt != "f16" and fa == "off":
                    continue
                max_ctx[kvt] = self._max_ctx(kvt, full, 0, hi)
            self.ub, self.b = saved

        load_mode = p.load_mode
        if load_mode == "auto":
            # Full offload: read the file straight into VRAM (no page-cache copy of
            # the whole model, measurable progress). Partial: keep mmap so CPU
            # layers run from the mapped file without an extra copy.
            load_mode = "none" if full_offload else "mmap"

        pl = Plan(
            model_id=self.model_id,
            mode="manual" if manual else "auto",
            ctx_requested=int(p.context_length),
            ctx=ctx,
            ctx_train=int(info.context_length or 0),
            kv_k=k,
            kv_v=v,
            flash_attn=fa,
            n_layer=n_layer,
            gpu_layers=min(n_gpu, full),
            full_offload=full_offload,
            n_cpu_moe=n_cpu_moe,
            cpu_expert_layers=list(range(n_cpu_moe)),
            tensor_split=split if split and len(self.devices) > 1 else None,
            split_mode=p.split_mode,
            devices=devs,
            host={k2: round(v2, 1) for k2, v2 in host.items()},
            totals={
                "weights_mib": round(info.weights_bytes / MiB, 1),
                "kv_mib": round(kv_total, 1),
                "mmproj_mib": round(self.mmproj_mib, 1),
                "vram_used_mib": round(sum(d.used_mib for d in devs), 1),
                "vram_free_mib": round(sum(d.free_mib for d in devs), 1),
            },
            kv_bytes_per_token=round(sum(kv_layer_bytes(info, 1, k, v, 1, 1, True)), 1),
            max_ctx_full_offload=max_ctx,
            load_mode=load_mode,
            parallel=p.parallel,
            batch=self.b,
            ubatch=self.ub,
            vram_overflow=p.vram_overflow,
            ctx_reduced_from=ctx_req if ctx < ctx_req else 0,
            fits=bool(fits and (not devs or self._fits(devs))),
            warnings=warnings,
            notes=notes,
            use_engine_fit=self.engine_fit and not manual,
        )
        if not self.devices:
            pl.warnings.append("No GPU devices are available to the engine: the model will run on the CPU.")
        if info.sliding_window and not p.swa_full:
            pl.notes.append(f"Sliding-window attention ({info.sliding_window} tokens) reduces KV cache on SWA layers.")
        return pl


def parse_fit_args(text: str) -> dict[str, Any]:
    """Parse llama-fit-params stdout: '-c 65536 -ngl 99 -ts 20,21 -ot "..."'."""
    import shlex

    out: dict[str, Any] = {}
    line = ""
    for ln in text.splitlines():
        if ln.strip().startswith("-c "):
            line = ln.strip()
    if not line:
        return out
    toks = shlex.split(line, posix=True)
    i = 0
    while i < len(toks):
        t = toks[i]
        val = toks[i + 1] if i + 1 < len(toks) else ""
        if t == "-c":
            out["ctx"] = int(val)
        elif t == "-ngl":
            out["ngl"] = int(val)
        elif t == "-ts":
            out["tensor_split"] = [float(x) for x in val.split(",") if x]
        elif t == "-ot":
            out["override_tensor"] = val
        i += 2
    return out


def parse_fit_print(text: str) -> dict[str, dict[str, float]]:
    """Parse 'llama-fit-params --fit-print on' output: '<dev> model context compute' (MiB)."""
    res: dict[str, dict[str, float]] = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts[1:]):
            res[parts[0]] = {"model": float(parts[1]), "context": float(parts[2]), "compute": float(parts[3])}
    return res
