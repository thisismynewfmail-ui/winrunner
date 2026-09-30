"""GPU memory planning.

Given a model (parsed GGUF), the requested load parameters and the devices the
engine can see (with their current free memory), the planner decides:

* the effective context length: the requested context, never reduced to make
  the model fit (it is only clamped to the trained context, unless the user
  allows RoPE extension),
* the KV cache precision (``auto`` keeps F16 when the whole model fits in VRAM
  with it, and uses Q8_0 otherwise, which leaves more VRAM for weights),
* where every tensor lives. GPUs are filled first, up to their free memory
  minus the safety margin; only the overflow goes to system RAM:

  1. everything on the GPUs when it fits,
  2. otherwise *attention first*: the attention weights and the KV cache of
     every layer stay on the GPUs and only feed-forward weights (the dense FFN
     matrices, or the routed experts of MoE models) move to system RAM, one
     matrix at a time, until each GPU is full. Attention - the part whose cost
     grows with the context - never runs on the CPU, so long prompts stay fast,
  3. only when not even the attention part fits (very long contexts): whole
     leading layers on the CPU, the rest as in 2.

Layers are split across GPUs in contiguous ranges (``--tensor-split``) with
the output layer on the last GPU; weights kept in RAM are selected with
``--override-tensor``. The analytic estimate below mirrors llama.cpp's
allocation rules; when the engine ships ``llama-fit-params`` the plan is
measured by the engine's own allocator before loading and corrected
(:func:`fit_to_engine`).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

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
DENSE_PARTS = ("ffn_up", "ffn_gate", "ffn_down")
EXPERT_PARTS = ("ffn_up_exps", "ffn_gate_exps", "ffn_down_exps")

STRATEGIES = {
    "full": "Everything on the GPUs",
    "attention_first": "Attention and KV cache on the GPUs, part of the feed-forward weights in system RAM",
    "layers": "Leading layers on the CPU, the rest attention-first on the GPUs",
    "cpu": "CPU only",
    "manual": "Manual",
    "engine": "llama.cpp --fit decides at load",
}


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
    ram_parts_mib: float = 0.0  # feed-forward weights of this GPU's layers that are kept in system RAM
    ram_layers: int = 0  # layers of this GPU with (part of) their feed-forward weights in RAM

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
    gpu_layers: int  # -ngl: layers on GPUs, counting the output layer (n_layer + 1 = all)
    full_offload: bool
    n_cpu_moe: int = 0  # manual mode (--n-cpu-moe)
    n_cpu_ffn: int = 0  # manual mode (--n-cpu-ffn)
    tensor_split: list[float] | None = None
    split_mode: str = "layer"
    strategy: str = "full"
    devices: list[DevicePlan] = field(default_factory=list)
    host: dict[str, float] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    kv_bytes_per_token: float = 0.0
    max_ctx_full_offload: dict[str, int] = field(default_factory=dict)
    load_mode: str = "mmap"
    parallel: int = -1
    # per layer: GPU index holding the layer's attention and KV cache (-1 = CPU)
    layer_home: list[int] = field(default_factory=list)
    # per layer: tensor groups (llama.cpp names, e.g. "ffn_down") kept in system RAM
    ram_parts: list[list[str]] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)  # --override-tensor entries ("pattern=CPU")
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "estimate"
    regime: str = "full"  # "full": all weights on the GPUs, "partial": some weights in RAM (smaller GPU buffers)
    use_engine_fit: bool = False  # llama.cpp --fit places the layers itself (placement "engine")
    engine: dict[str, Any] = field(default_factory=dict)
    mmproj: str = ""
    draft: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["devices"] = [x.to_dict() for x in self.devices]
        d["strategy_label"] = STRATEGIES.get(self.strategy, self.strategy)
        d["ram_layers"] = sum(1 for x in self.ram_parts if x)
        d["cpu_layers"] = sum(1 for x in self.layer_home if x < 0)
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


def compute_buffer_bytes(info: ModelInfo, ctx: int, ubatch: int, flash: bool, has_output: bool,
                         n_outputs: int | None = None) -> float:
    """Approximate llama.cpp compute (graph) buffer for one device.

    ``n_outputs``: logits rows per micro-batch. llama-server computes logits only for the tokens it samples
    (one per slot, more with speculative decoding); None = the whole micro-batch (llama-fit-params).
    """
    n_ff = max(info.n_ff, 1)
    act = ubatch * (info.n_embd * 8 + n_ff * 3) * 4
    if flash:
        attn = ubatch * info.n_embd * 4 * 4
    else:
        attn = float(ctx) * ubatch * max(info.n_head, 1) * 4
    rows = ubatch if n_outputs is None else min(ubatch, max(1, n_outputs))
    logits = rows * info.n_vocab * 4 if has_output else 0
    return max(logits, attn) + act + 32 * MiB


def _pad(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def mmproj_estimate_mib(file_size: int) -> float:
    """Vision projector weights plus a worst-case image encoding buffer (refined after the first load)."""
    return file_size / MiB * 1.15 + 256


def layer_parts(info: ModelInfo, il: int) -> list[tuple[str, float]]:
    """Tensor groups of layer ``il`` that may be kept in system RAM (bytes), in the order they stay on the GPU."""
    parts = info.layer_parts[il] if il < len(info.layer_parts or []) else None
    if parts:
        return [(str(g), float(b)) for g, b in parts]
    # index entries without per-group sizes: split the known feed-forward bytes evenly
    ffn = info.layer_ffn_bytes[il] if il < len(info.layer_ffn_bytes) else 0
    exp = info.layer_expert_bytes[il] if il < len(info.layer_expert_bytes) else 0
    out: list[tuple[str, float]] = []
    if ffn:
        out += [(g, ffn / 3) for g in DENSE_PARTS]
    if exp:
        out += [(g, exp / 3) for g in EXPERT_PARTS]
    return out


def override_patterns(ram_parts: list[list[str]], buft: str = "CPU") -> list[str]:
    """--override-tensor entries that keep the given tensor groups of each layer in system RAM.

    Layers with the same groups share one entry: ``blk\\.(3|4|5)\\.(ffn_up|ffn_gate|ffn_down)\\.=CPU``.
    """
    by_groups: dict[tuple[str, ...], list[int]] = {}
    for il, groups in enumerate(ram_parts):
        if groups:
            by_groups.setdefault(tuple(sorted(groups)), []).append(il)
    out = []
    for groups, layers in sorted(by_groups.items(), key=lambda kv: kv[1][0]):
        lay = str(layers[0]) if len(layers) == 1 else "(" + "|".join(str(x) for x in layers) + ")"
        grp = re.escape(groups[0]) if len(groups) == 1 else "(" + "|".join(re.escape(g) for g in groups) + ")"
        out.append(f"blk\\.{lay}\\.{grp}\\.={buft}")
    return out


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
# GPU-first placement
# ---------------------------------------------------------------------------


@dataclass
class LayerCost:
    base: float  # bytes that live with the layer's device: attention/norm/router weights + KV cache
    kv: float
    parts: list[tuple[str, float]]  # tensor groups that may stay in RAM, in the order they stay on the GPU

    @property
    def full(self) -> float:
        return self.base + sum(b for _, b in self.parts)


@dataclass
class Layout:
    """Result of the placement search."""

    k_cpu: int  # leading layers entirely on the CPU (attention and KV cache included)
    counts: list[int]  # layers per GPU (the last one includes the output layer)
    ram: dict[int, list[str]]  # layer -> tensor groups kept in system RAM
    gpu_parts: float  # bytes of feed-forward weights on the GPUs
    rooms: list[float]  # capacity left per GPU (bytes)

    @property
    def n_gpu(self) -> int:
        return sum(self.counts)


def _fill_device(costs: list[LayerCost], lo: int, hi: int, room: float,
                 ram: dict[int, list[str]]) -> tuple[float, float]:
    """Keep feed-forward groups of layers [lo, hi) on one GPU, from the last layer backwards.

    ``room`` is the capacity left after the layers' base cost. Groups that do not fit go to
    ``ram``: they end up in the leading layers of the range (plus one partial layer).
    Returns (bytes kept on the GPU, room left).
    """
    kept = 0.0
    i = hi - 1
    while i >= lo:
        c = costs[i]
        tot = sum(b for _, b in c.parts)
        if tot <= room:
            room -= tot
            kept += tot
            i -= 1
            continue
        n = 0
        for _g, b in c.parts:
            if b > room:
                break
            room -= b
            kept += b
            n += 1
        ram[i] = [g for g, _ in c.parts[n:]]
        i -= 1
        while i >= lo:
            if costs[i].parts:
                ram[i] = [g for g, _ in costs[i].parts]
            i -= 1
    return kept, room


def _ranges(k: int, counts: list[int], n_layer: int) -> list[tuple[int, int]]:
    out = []
    lo = k
    for c in counts:
        hi = min(n_layer, lo + c)
        out.append((lo, hi))
        lo = hi
    return out


def _counts_for_bounds(k: int, bounds: list[int], n_layer: int) -> list[int]:
    """Layer counts per GPU for split points ``bounds`` (GPU d gets [bounds[d-1], bounds[d]))."""
    edges = [k] + list(bounds) + [n_layer]
    counts = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    counts[-1] += 1  # output layer
    return counts


def _evaluate(costs: list[LayerCost], prefix: list[float], k: int, counts: list[int], caps: list[float],
              out_cost: float) -> Layout | None:
    n_layer = len(costs)
    nd = len(caps)
    rooms = []
    for d, (lo, hi) in enumerate(_ranges(k, counts, n_layer)):
        base = prefix[hi] - prefix[lo] + (out_cost if d == nd - 1 else 0.0)
        if base > caps[d]:
            return None
        rooms.append(caps[d] - base)
    ram: dict[int, list[str]] = {}
    kept = 0.0
    for d, (lo, hi) in enumerate(_ranges(k, counts, n_layer)):
        got, rooms[d] = _fill_device(costs, lo, hi, rooms[d], ram)
        kept += got
    return Layout(k_cpu=k, counts=list(counts), ram=ram, gpu_parts=kept, rooms=rooms)


def _better(a: Layout, b: Layout | None) -> bool:
    """More feed-forward bytes on the GPUs; on a tie, the more even leftover (largest minimum room)."""
    if b is None:
        return True
    tol = 1e-9 * max(abs(a.gpu_parts), abs(b.gpu_parts), 1.0)
    if a.gpu_parts > b.gpu_parts + tol:
        return True
    if a.gpu_parts < b.gpu_parts - tol:
        return False
    ra, rb = (min(a.rooms) if a.rooms else 0.0), (min(b.rooms) if b.rooms else 0.0)
    return ra > rb + 1e-9 * max(abs(ra), abs(rb), 1.0)


def _base_feasible(prefix: list[float], k: int, caps: list[float], out_cost: float, n_layer: int) -> bool:
    """Whether the base costs of layers [k, n_layer) + output fit the GPUs in contiguous ranges (greedy from the back)."""
    nd = len(caps)
    hi = n_layer
    for d in range(nd - 1, -1, -1):
        room = caps[d] - (out_cost if d == nd - 1 else 0.0)
        if room < 0:
            return False
        lo = hi
        while lo > k and prefix[hi] - prefix[lo - 1] <= room:
            lo -= 1
        hi = lo
        if hi <= k:
            return True
    return hi <= k


def place(costs: list[LayerCost], caps: list[float], out_cost: float, fixed_split: list[float] | None = None) -> Layout | None:
    """GPU-first placement: the fewest leading CPU layers, then the split that keeps most weights on the GPUs.

    ``caps``: bytes available per GPU for layer weights + KV cache (free memory minus margin, compute
    buffers and other fixed allocations). ``fixed_split``: user-chosen proportions per GPU.
    Returns None when not even the output layer fits.
    """
    n_layer = len(costs)
    nd = len(caps)
    if nd == 0:
        return None
    prefix = [0.0]
    for c in costs:
        prefix.append(prefix[-1] + c.base)
    # fewest whole CPU layers such that attention + KV cache of the remaining layers fit (monotone)
    if not _base_feasible(prefix, n_layer, caps, out_cost, n_layer):
        return None
    lo, hi = 0, n_layer
    while lo < hi:
        mid = (lo + hi) // 2
        if _base_feasible(prefix, mid, caps, out_cost, n_layer):
            hi = mid
        else:
            lo = mid + 1
    k = lo
    n_gpu_layers = n_layer - k
    if nd == 1:
        return _evaluate(costs, prefix, k, [n_gpu_layers + 1], caps, out_cost)
    if fixed_split:
        tot = sum(fixed_split) or 1.0
        acc, bounds = 0.0, []
        for s in fixed_split[:-1]:
            acc += s / tot
            bounds.append(k + round(acc * (n_gpu_layers + 1)))
        bounds = [min(max(b, k), n_layer) for b in bounds]
        for i in range(1, len(bounds)):
            bounds[i] = max(bounds[i], bounds[i - 1])
        return _evaluate(costs, prefix, k, _counts_for_bounds(k, bounds, n_layer), caps, out_cost)
    best: Layout | None = None
    if nd == 2:
        for b in range(k, n_layer + 1):
            lay = _evaluate(costs, prefix, k, _counts_for_bounds(k, [b], n_layer), caps, out_cost)
            if lay is not None and _better(lay, best):
                best = lay
        return best
    # 3+ GPUs: start proportional to capacity, then move split points while that improves the placement
    tot = sum(max(c, 1.0) for c in caps)
    acc, bounds = 0.0, []
    for c in caps[:-1]:
        acc += max(c, 1.0) / tot
        bounds.append(min(n_layer, k + round(acc * n_gpu_layers)))
    best = _evaluate(costs, prefix, k, _counts_for_bounds(k, bounds, n_layer), caps, out_cost)
    improved = True
    steps = 0
    while improved and steps < 400:
        improved = False
        for i in range(len(bounds)):
            for delta in (-1, 1):
                nb = list(bounds)
                nb[i] += delta
                if not all(k <= x <= n_layer for x in nb) or any(nb[j] > nb[j + 1] for j in range(len(nb) - 1)):
                    continue
                steps += 1
                lay = _evaluate(costs, prefix, k, _counts_for_bounds(k, nb, n_layer), caps, out_cost)
                if lay is not None and _better(lay, best):
                    best, bounds, improved = lay, nb, True
    return best


def fits_whole(costs: list[LayerCost], caps: list[float], out_cost: float) -> bool:
    """Whether the whole model fits the GPUs (contiguous ranges, greedy from the back)."""
    nd = len(caps)
    if nd == 0:
        return False
    i = len(costs) - 1
    for d in range(nd - 1, -1, -1):
        room = caps[d] - (out_cost if d == nd - 1 else 0.0)
        if room < 0:
            return False
        while i >= 0 and costs[i].full <= room:
            room -= costs[i].full
            i -= 1
        if i < 0:
            return True
    return i < 0


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
        margin_mib: int = 512,
        margin_per_device: dict[str, int] | None = None,
        reclaim_mib: dict[str, float] | None = None,
        mmproj_size: int = 0,
        mmproj_mib_hint: float | None = None,
        draft_info: ModelInfo | None = None,
        engine_fit: bool = False,
        model_id: str = "",
        adjust_mib: dict[str, dict[str, float]] | None = None,
        overrides_supported: bool = True,
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
        self.engine_fit = engine_fit  # llama.cpp --fit places the layers at load (placement "engine")
        self.model_id = model_id
        # measured minus estimated memory per device (MiB) from the engine's own projection, per regime:
        # llama.cpp's compute buffers are much smaller when part of the weights stay in system RAM
        self.adjust: dict[str, dict[str, float]] = {"full": {}, "partial": {}}
        for regime, vals in (adjust_mib or {}).items():
            self.adjust.setdefault(regime, {}).update(vals)
        # the engine can keep individual tensors in RAM (--override-tensor); row / tensor split modes cannot
        self.overrides_ok = overrides_supported and params.split_mode in ("layer", "none")

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

    def _blank_devices(self) -> list[DevicePlan]:
        return [
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

    def layer_costs(self, ctx: int, kv_k: str, kv_v: str) -> list[LayerCost]:
        info = self.info
        kvl = kv_layer_bytes(info, ctx, kv_k, kv_v, self.p.ubatch_size, self._n_seq(), self.p.swa_full)
        out = []
        for il in range(info.n_layer):
            lw = float(info.layer_bytes[il]) if il < len(info.layer_bytes) else 0.0
            parts = layer_parts(info, il) if self.overrides_ok else []
            kv = float(kvl[il]) if il < len(kvl) else 0.0
            base = lw - sum(b for _, b in parts) + (kv if self.p.kv_offload else 0.0)
            out.append(LayerCost(base=max(0.0, base), kv=kv, parts=parts))
        return out

    def _out_bytes(self) -> float:
        return float(self.info.output_bytes + self.info.other_bytes)

    def _n_outputs(self) -> int:
        """Logits rows llama-server computes per micro-batch: one per slot (plus drafted tokens)."""
        per_seq = 1 + (max(self.p.draft_max, 16) if self.draft is not None or self.p.spec_type else 0)
        return self._n_seq() * per_seq

    def _compute(self, ctx: int, i: int, has_output: bool, regime: str) -> float:
        """Compute (graph) buffer of GPU ``i``: estimate plus the engine-measured correction."""
        name = self.devices[i].name
        comp = compute_buffer_bytes(self.info, ctx, self.p.ubatch_size, self._flash_for_estimate(), has_output,
                                    self._n_outputs())
        return comp + float(self.adjust.get(regime, {}).get(name, 0.0)) * MiB

    def _draft_share(self, ctx: int) -> list[float]:
        nd = len(self.devices)
        out = [0.0] * nd
        if self.draft is None or not nd:
            return out
        dw = self.draft.weights_bytes
        dkv = sum(kv_layer_bytes(self.draft, ctx, "f16", "f16", self.p.ubatch_size, 1, False))
        caps = [max(1.0, d.free_mib) for d in self.devices]
        tot = sum(caps)
        for i in range(nd):
            out[i] = (dw + dkv) * caps[i] / tot
        out[-1] += compute_buffer_bytes(self.draft, ctx, self.p.ubatch_size, self._flash_for_estimate(), True)
        return out

    def _fixed(self, ctx: int, regime: str) -> list[float]:
        """Bytes per GPU that do not depend on the layers placed on it (buffers, projector, draft model)."""
        nd = len(self.devices)
        draft = self._draft_share(ctx)
        fixed = []
        for i in range(nd):
            extra = self.mmproj_mib * MiB if i == 0 and self.mmproj_mib and self.p.mmproj_offload else 0.0
            fixed.append(self._compute(ctx, i, i == nd - 1, regime) + extra + draft[i])
        return fixed

    def capacities(self, ctx: int, regime: str = "full") -> list[float]:
        """Bytes available per GPU for layer weights and KV cache."""
        devs = self._blank_devices()
        fixed = self._fixed(ctx, regime)
        return [(d.free_mib - d.margin_mib) * MiB - f for d, f in zip(devs, fixed)]

    def _user_split(self) -> list[float] | None:
        ts = self.p.tensor_split
        if ts and len(ts) == len(self.devices) and sum(ts) > 0:
            return [float(x) for x in ts]
        return None

    # ----- layouts ------------------------------------------------------------------

    def _auto_layout(self, ctx: int, kv_k: str, kv_v: str) -> tuple[Layout | None, list[LayerCost], str]:
        costs = self.layer_costs(ctx, kv_k, kv_v)
        out, split = self._out_bytes(), self._user_split()
        lay = place(costs, self.capacities(ctx, "full"), out, split)
        if lay is not None and lay.k_cpu == 0 and not lay.ram:
            return lay, costs, "full"
        lay = place(costs, self.capacities(ctx, "partial"), out, split)
        if lay is not None and lay.k_cpu == 0 and not lay.ram:
            # everything fits only with the smaller buffers of a partial placement: keep one group in RAM
            first = next((il for il, c in enumerate(costs) if c.parts), None)
            if first is None:
                return None, costs, "partial"
            lay.ram[first] = [costs[first].parts[-1][0]]
        return lay, costs, "partial"

    def probe_plan(self) -> Plan | None:
        """Smallest GPU footprint: attention and KV cache on the GPUs, every feed-forward group in RAM.

        Measuring it tells the fitter the engine's real buffer sizes for partial placements.
        """
        if not self.devices or self.p.gpu_offload == "manual":
            return None
        ctx, _ = self.resolve_ctx()
        kv = self.p.kv_cache_type if self.p.kv_cache_type != "auto" else "q8_0"
        if self.p.flash_attn == "off" and kv in QUANTIZED_KV:
            kv = "f16"
        k, v = self._kv_pair(kv)
        costs = self.layer_costs(ctx, k, v)
        heavy = [LayerCost(c.base, c.kv, [(g, float("inf")) for g, _ in c.parts]) for c in costs]
        lay = place(heavy, self.capacities(ctx, "partial"), self._out_bytes(), self._user_split())
        if lay is None:
            # the estimate says not even that fits: measure the output layer alone on the last GPU
            nd = len(self.devices)
            lay = Layout(k_cpu=self.info.n_layer, counts=[0] * (nd - 1) + [1], ram={}, gpu_parts=0.0, rooms=[])
        return self._build(lay, costs, ctx, k, v, "partial", self.p.flash_attn, [], [])

    def _manual_layout(self, ctx: int, kv_k: str, kv_v: str) -> tuple[Layout, list[LayerCost]]:
        info, p = self.info, self.p
        n_layer = info.n_layer
        costs = self.layer_costs(ctx, kv_k, kv_v)
        full = n_layer + 1
        n_gpu = (full if p.n_gpu_layers < 0 else min(full, p.n_gpu_layers)) if self.devices else 0
        k = max(0, n_layer + 1 - n_gpu) if n_gpu > 0 else n_layer
        ram: dict[int, list[str]] = {}
        for il in range(n_layer):
            parts = layer_parts(info, il)
            groups = []
            if il < p.n_cpu_moe:
                groups += [g for g, _ in parts if g.endswith("exps")]
            if il < p.n_cpu_ffn:
                groups += [g for g, _ in parts if not g.endswith("exps")]
            if groups and il >= k:
                ram[il] = groups
        nd = len(self.devices)
        counts: list[int] = []
        if nd and n_gpu > 0:
            split = self._user_split() or self._balanced_split(ctx, costs, ram, k)
            assign = assign_layers(n_layer, n_gpu, split)
            counts = [sum(1 for x in assign if x == d) for d in range(nd)]
        return Layout(k_cpu=k, counts=counts, ram=ram, gpu_parts=0.0, rooms=[]), costs

    def _balanced_split(self, ctx: int, costs: list[LayerCost], ram: dict[int, list[str]], k: int) -> list[float]:
        """Layer counts proportional to each GPU's capacity, by per-layer cost (manual mode without a split)."""
        nd = len(self.devices)
        caps = [max(1.0, c) for c in self.capacities(ctx, "partial" if ram or k else "full")]
        n_layer = len(costs)
        cost = []
        for il in range(k, n_layer):
            c = costs[il]
            in_ram = set(ram.get(il, []))
            cost.append(c.base + sum(b for g, b in c.parts if g not in in_ram))
        cost.append(self._out_bytes())
        total_cost, total_cap = sum(cost) or 1.0, sum(caps)
        counts = [0] * nd
        cum, di = 0.0, 0
        bound = caps[0] / total_cap * total_cost
        for c in cost:
            while di < nd - 1 and cum + c / 2 > bound:
                di += 1
                bound += caps[di] / total_cap * total_cost
            counts[di] += 1
            cum += c
        return [float(x) for x in counts]

    # ----- building the plan -----------------------------------------------------------

    def _materialize(self, lay: Layout, costs: list[LayerCost], ctx: int,
                     regime: str) -> tuple[list[DevicePlan], dict[str, float], list[int]]:
        info = self.info
        n_layer = info.n_layer
        devs = self._blank_devices()
        nd = len(devs)
        home = [-1] * n_layer
        if nd and lay.counts:
            for d, (lo, hi) in enumerate(_ranges(lay.k_cpu, lay.counts, n_layer)):
                for il in range(lo, hi):
                    home[il] = d
        host = {"weights_mib": info.token_embd_bytes / MiB, "kv_mib": 0.0, "compute_mib": 0.0}
        for il in range(n_layer):
            c = costs[il]
            lw = float(info.layer_bytes[il]) if il < len(info.layer_bytes) else 0.0
            in_ram = set(lay.ram.get(il, []))
            ram_b = sum(b for g, b in c.parts if g in in_ram)
            d = home[il]
            if d < 0:
                host["weights_mib"] += lw / MiB
                host["kv_mib"] += c.kv / MiB
                continue
            dp = devs[d]
            dp.weights_mib += (lw - ram_b) / MiB
            host["weights_mib"] += ram_b / MiB
            if ram_b:
                dp.ram_parts_mib += ram_b / MiB
                dp.ram_layers += 1
            if self.p.kv_offload:
                dp.kv_mib += c.kv / MiB
            else:
                host["kv_mib"] += c.kv / MiB
            dp.layers += 1
            if dp.layer_first < 0:
                dp.layer_first = il
            dp.layer_last = il
        extra_layers = len(info.layer_bytes) - n_layer  # e.g. MTP layers (not used for decoding)
        if extra_layers > 0:
            host["weights_mib"] += sum(info.layer_bytes[n_layer:]) / MiB
        # the output layer is the last offloaded layer: on the last GPU that holds layers
        out_dev = max((d for d, c in enumerate(lay.counts) if c > 0), default=-1) if nd and lay.n_gpu > 0 else -1
        if out_dev >= 0:
            devs[out_dev].output_mib += self._out_bytes() / MiB
        else:
            host["weights_mib"] += self._out_bytes() / MiB
        flash = self._flash_for_estimate()
        for i, dp in enumerate(devs):
            # the first GPU also runs the batch operations of weights kept in RAM (llama.cpp op offload),
            # including the output layer when that is on the CPU
            if dp.layers or i == out_dev or i == 0:
                dp.compute_mib = self._compute(ctx, i, i == out_dev or (i == 0 and out_dev < 0), regime) / MiB
        if lay.n_gpu < n_layer + 1 or not devs:
            host["compute_mib"] = compute_buffer_bytes(info, ctx, self.p.ubatch_size, flash, out_dev < 0,
                                                       self._n_outputs()) / MiB
        else:
            host["compute_mib"] = (self.p.ubatch_size * info.n_embd * 4 * 2) / MiB + 8
        if devs and self.mmproj_mib:
            if self.p.mmproj_offload:
                devs[0].mmproj_mib = self.mmproj_mib
            else:
                host["compute_mib"] += self.mmproj_mib
        if devs and self.draft is not None:
            for dp, b in zip(devs, self._draft_share(ctx)):
                dp.draft_mib = b / MiB
        return devs, host, home

    @staticmethod
    def _fits(devs: list[DevicePlan]) -> bool:
        return all(d.headroom_mib >= -0.5 for d in devs)

    def _max_ctx_full(self, kv: str) -> int:
        if not self.devices:
            return 0
        hi = max(CTX_PAD, int(self.info.context_length or 131072))
        if self.p.allow_context_over_train:
            hi = max(hi, self.p.context_length)
        out = self._out_bytes()

        def ok(c: int) -> bool:
            k, v = kv, kv
            return fits_whole(self.layer_costs(c, k, v), self.capacities(c, "full"), out)

        lo = CTX_PAD
        if not ok(lo):
            return 0
        if ok(hi):
            return hi
        while hi - lo > CTX_PAD:
            mid = _pad((lo + hi) // 2, CTX_PAD)
            if mid >= hi:
                break
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo

    # ----- main entry ---------------------------------------------------------------

    def plan(self) -> Plan:
        info, p = self.info, self.p
        ctx, notes = self.resolve_ctx()
        warnings: list[str] = []
        n_layer = info.n_layer

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
        lay: Layout | None = None
        costs: list[LayerCost] = []
        regime = "full"

        if not self.devices:
            k, v = self._kv_pair(chosen_kv)
            costs = self.layer_costs(ctx, k, v)
            lay, regime = Layout(k_cpu=n_layer, counts=[], ram={}, gpu_parts=0.0, rooms=[]), "partial"
        elif manual:
            k, v = self._kv_pair(chosen_kv)
            lay, costs = self._manual_layout(ctx, k, v)
            regime = "partial" if lay.ram or lay.k_cpu else "full"
        else:
            for kv in candidates:
                k, v = self._kv_pair(kv)
                cand, cost, reg = self._auto_layout(ctx, k, v)
                if cand is not None and reg == "full":
                    chosen_kv, lay, costs, regime = kv, cand, cost, reg
                    break
            if lay is None:
                chosen_kv = candidates[-1]
                k, v = self._kv_pair(chosen_kv)
                lay, costs, regime = self._auto_layout(ctx, k, v)
                if lay is None:  # not even the output layer fits: CPU only
                    lay, regime = Layout(k_cpu=n_layer, counts=[], ram={}, gpu_parts=0.0, rooms=[]), "partial"

        k, v = self._kv_pair(chosen_kv)
        if p.kv_cache_type == "auto" and chosen_kv != "f16" and self.devices:
            if lay.k_cpu == 0 and not lay.ram:
                notes.append("KV cache set to Q8_0 (near-lossless) so the whole model and context fit in VRAM.")
            else:
                notes.append("KV cache set to Q8_0 (near-lossless): half the size of F16, which keeps more weights on the GPUs.")
        return self._build(lay, costs, ctx, k, v, regime, fa, warnings, notes)

    def _build(self, lay: Layout, costs: list[LayerCost], ctx: int, k: str, v: str, regime: str, fa: str,
               warnings: list[str], notes: list[str]) -> Plan:
        info, p = self.info, self.p
        n_layer = info.n_layer
        full = n_layer + 1
        manual = p.gpu_offload == "manual"
        engine_places = self.engine_fit and not manual
        devs, host, home = self._materialize(lay, costs, ctx, regime)
        n_gpu = lay.n_gpu if devs else 0
        ram_parts = [list(lay.ram.get(il, [])) for il in range(n_layer)]
        ram_mib = sum(dp.ram_parts_mib for dp in devs)
        full_offload = bool(devs) and n_gpu >= full and not any(ram_parts)

        if not devs or n_gpu == 0:
            strategy = "cpu"
        elif manual:
            strategy = "manual"
        elif engine_places:
            strategy = "engine"
        elif lay.k_cpu > 0:
            strategy = "layers"
        elif any(ram_parts):
            strategy = "attention_first"
        else:
            strategy = "full"

        n_ram = sum(1 for x in ram_parts if x)
        if strategy == "attention_first":
            what = "expert" if info.expert_count else "feed-forward"
            notes.append(
                f"Attention and the KV cache of all {n_layer} layers stay on the GPU(s); {what} weights of "
                f"{n_ram} layer(s) ({_size(ram_mib)}) are kept in system RAM. Each GPU is filled to its free "
                "memory minus the safety margin.")
        elif strategy == "layers":
            more = f", and feed-forward weights of {n_ram} more layer(s) are in system RAM" if n_ram else ""
            hint = "A shorter context" if k in QUANTIZED_KV else "A Q8_0 KV cache or a shorter context"
            notes.append(
                f"The attention part of all layers does not fit at this context: the first {lay.k_cpu} of {n_layer} "
                f"layers run on the CPU (with their KV cache){more}. {hint} would keep more on the GPUs.")
        if fa == "auto" and (k in QUANTIZED_KV or v in QUANTIZED_KV):
            fa = "on"
        if manual and devs and not self._fits(devs):
            warnings.append("Manual configuration exceeds free VRAM on at least one GPU; the load may fail "
                            "or be much slower.")

        kv_total = sum(kv_layer_bytes(info, ctx, k, v, p.ubatch_size, self._n_seq(), p.swa_full)) / MiB
        max_ctx = {}
        if self.devices:
            for kvt in ("f16", "q8_0"):
                max_ctx[kvt] = self._max_ctx_full(kvt)

        load_mode = p.load_mode
        if load_mode == "auto":
            # Full offload: read the file straight into VRAM. Weights kept in RAM by --override-tensor are
            # also best loaded without mmap (llama.cpp can then repack them / use pinned memory).
            # Whole layers on the CPU: mmap avoids a second copy of those layers.
            load_mode = "none" if strategy in ("full", "attention_first") or (strategy == "engine" and full_offload) \
                else "mmap"

        split = [float(c) for c in lay.counts] if len(self.devices) > 1 and lay.counts and n_gpu else None
        if manual and split and self._user_split():
            split = self._user_split()  # the user's proportions, as entered
        overrides = override_patterns(ram_parts) if strategy in ("attention_first", "layers") else []

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
            n_cpu_moe=max(0, p.n_cpu_moe) if manual else 0,
            n_cpu_ffn=max(0, p.n_cpu_ffn) if manual else 0,
            tensor_split=split,
            split_mode=p.split_mode,
            strategy=strategy,
            devices=devs,
            host={k2: round(v2, 1) for k2, v2 in host.items()},
            totals={
                "weights_mib": round(info.weights_bytes / MiB, 1),
                "kv_mib": round(kv_total, 1),
                "mmproj_mib": round(self.mmproj_mib, 1),
                "vram_used_mib": round(sum(d.used_mib for d in devs), 1),
                "vram_free_mib": round(sum(d.free_mib for d in devs), 1),
                "ram_parts_mib": round(ram_mib, 1),
            },
            kv_bytes_per_token=round(sum(kv_layer_bytes(info, 1, k, v, 1, 1, True)), 1),
            max_ctx_full_offload=max_ctx,
            load_mode=load_mode,
            parallel=p.parallel,
            layer_home=home,
            ram_parts=ram_parts,
            overrides=overrides,
            warnings=warnings,
            notes=notes,
            regime=regime,
            use_engine_fit=engine_places,
        )
        if not self.devices:
            pl.warnings.append("No GPU devices are available to the engine: the model will run on the CPU.")
        if info.sliding_window and not p.swa_full:
            pl.notes.append(f"Sliding-window attention ({info.sliding_window} tokens) reduces KV cache on SWA layers.")
        return pl

    def calibrate(self, pl: Plan, measured: dict[str, dict[str, float]]) -> float:
        """Fold the engine's projection of ``pl`` into the per-device corrections. Returns the largest change (MiB)."""
        adj = self.adjust.setdefault(pl.regime, {})
        worst = 0.0
        for d in pl.devices:
            md = measured.get(d.name)
            if md is None:
                continue
            delta = measured_used(md) - (d.weights_mib + d.output_mib + d.kv_mib + d.compute_mib)
            adj[d.name] = adj.get(d.name, 0.0) + delta
            worst = max(worst, abs(delta))
        return worst


def _size(mib: float) -> str:
    return f"{mib / 1024:.1f} GiB" if mib >= 1024 else f"{mib:.0f} MiB"


# ---------------------------------------------------------------------------
# Engine measurement
# ---------------------------------------------------------------------------

Measure = Callable[[Plan], dict[str, dict[str, float]]]
MAX_FIT_ROUNDS = 5
SETTLE_MIB = 16.0


def device_targets(plan: Plan) -> dict[str, float]:
    """MiB each GPU may hold for model, KV cache and compute buffers (free - margin - projector - draft)."""
    return {d.name: d.free_mib - d.margin_mib - d.mmproj_mib - d.draft_mib for d in plan.devices}


def measured_used(m: dict[str, float]) -> float:
    return float(m.get("model", 0)) + float(m.get("context", 0)) + float(m.get("compute", 0))


def fits_measured(pl: Plan, m: dict[str, dict[str, float]]) -> bool:
    targets = device_targets(pl)
    return all(measured_used(m.get(d.name, {})) <= targets[d.name] + 0.5 for d in pl.devices)


def _placement_key(pl: Plan) -> tuple:
    return (pl.kv_k, pl.kv_v, pl.flash_attn, pl.gpu_layers, tuple(pl.tensor_split or ()), tuple(pl.overrides))


def _rank(pl: Plan, m: dict[str, dict[str, float]]) -> tuple[int, float]:
    """Full offload first, then the most model and KV cache bytes on the GPUs."""
    on_gpu = sum(float(m.get(d.name, {}).get("model", 0)) + float(m.get(d.name, {}).get("context", 0))
                 for d in pl.devices)
    return (1 if pl.full_offload else 0, on_gpu)


def fit_to_engine(planner: Planner, measure: Measure, rounds: int = MAX_FIT_ROUNDS) -> Plan:
    """Correct the analytic plan with the engine's own memory projection.

    Each round measures the proposed placement and re-plans with the per-device difference between the
    measured and the estimated memory (kept separately for full and partial placements, whose compute
    buffers differ). Partial placements are calibrated first with a minimum-footprint probe; when only a
    partial placement was found, full offload is measured once as well. The best placement the engine
    confirms to fit - full offload first, then the most bytes on the GPUs - is returned with the measured
    numbers. Raises whatever ``measure`` raises.
    """
    import copy

    saved = copy.deepcopy(planner.adjust)
    tried: dict[tuple, tuple[Plan, dict[str, dict[str, float]]]] = {}
    fitting: list[tuple[tuple[int, float], int, Plan, dict[str, dict[str, float]]]] = []
    state: dict[str, Any] = {"last": None, "full_measured": False, "n": 0}

    def run(pl: Plan) -> tuple[bool, float] | None:
        key = _placement_key(pl)
        if key in tried:
            return None
        m = measure(pl)
        state["n"] += 1
        tried[key] = (pl, m)
        state["last"] = (pl, m)
        if pl.regime == "full":
            state["full_measured"] = True
        ok = fits_measured(pl, m)
        if ok:
            fitting.append((_rank(pl, m), state["n"], pl, m))
        return ok, planner.calibrate(pl, m)

    try:
        probed = False
        for _ in range(rounds):
            pl = planner.plan()
            if pl.regime == "partial" and not probed:
                probed = True
                probe = planner.probe_plan()
                if probe is not None:
                    run(probe)
                    pl = planner.plan()
            res = run(pl)
            if res is None:
                break  # converged on a placement that was already measured
            ok, worst = res
            if ok and worst <= SETTLE_MIB:
                break
        if planner.devices and not state["full_measured"] and not any(f[2].full_offload for f in fitting):
            # Full offload was ruled out by an estimate only. Its buffers are larger than those of partial
            # placements; start from the partial correction and measure it (once corrected, if needed).
            full = planner.adjust.setdefault("full", {})
            for name, v in planner.adjust.get("partial", {}).items():
                full.setdefault(name, v)
            for _ in range(2):
                pl = planner.plan()
                if pl.regime != "full":
                    break
                res = run(pl)
                if res is None or res[0]:
                    break
    finally:
        planner.adjust = saved
    if fitting:
        fitting.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, pl, m = fitting[0]
    elif state["last"] is not None:
        pl, m = state["last"]
        targets = device_targets(pl)
        over = [f"{d.name} by {measured_used(m.get(d.name, {})) - targets[d.name]:.0f} MiB"
                for d in pl.devices if measured_used(m.get(d.name, {})) > targets[d.name] + 0.5]
        if over:
            pl.warnings.append("Engine projection: this configuration exceeds the free VRAM target on "
                               + ", ".join(over) + ".")
    else:
        return planner.plan()
    apply_measurement(pl, m)
    pl.engine["rounds"] = state["n"]
    return pl


def apply_measurement(pl: Plan, m: dict[str, dict[str, float]]) -> None:
    """Replace the estimated per-device numbers with the engine's projection."""
    for d in pl.devices:
        md = m.get(d.name)
        if not md:
            continue
        d.weights_mib = float(md.get("model", 0))
        d.kv_mib = float(md.get("context", 0))
        d.compute_mib = float(md.get("compute", 0))
        d.output_mib = 0.0
    if "Host" in m:
        h = m["Host"]
        pl.host = {"weights_mib": float(h.get("model", 0)), "kv_mib": float(h.get("context", 0)),
                   "compute_mib": float(h.get("compute", 0))}
    pl.totals["vram_used_mib"] = round(sum(d.used_mib for d in pl.devices), 1)
    pl.source = "engine"
    pl.engine = {"projection": m}


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
