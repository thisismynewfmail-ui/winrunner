"""GPU memory planning.

Given a model (parsed GGUF), the requested load parameters and the devices the
engine can see (with their current free memory), the planner decides:

* whether the whole model stays in VRAM. It does whenever it fits with at least
  a 4K context: running any part of a model from system RAM is very slow in
  llama.cpp (the attention of those layers runs on the CPU, their weights are
  copied to the first GPU for every prompt batch, and the GPUs stop working as
  a pipeline), so the context is fitted before anything moves to the CPU,
* the effective context length, fitted to the VRAM according to
  ``LoadParams.context_fit``: ``fill`` (default) takes the largest context that
  fits, so the KV cache fills the GPUs up to the safety margin (at most the
  trained context unless RoPE extension is allowed); ``fit`` takes the requested
  context, reduced when needed; ``off`` takes exactly the requested context,
* the KV cache precision (``auto`` prefers F16 and uses Q8_0 only when F16
  cannot reach the requested context in VRAM),
* how layers are distributed across GPUs (``--tensor-split``), and - only when
  the model does not fit even at the minimum context - how many layers (dense)
  or expert blocks (MoE) stay on the CPU.

It produces a per-device memory breakdown (weights / KV cache / compute
buffers / run-time scratch / vision projector) used by the UI, and the
corresponding engine arguments. The analytic estimate mirrors llama.cpp's
allocation rules; when the engine ships ``llama-fit-params`` it is calibrated
with llama.cpp's own projection before loading (see ``manager.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import DEFAULT_VRAM_MARGIN_MIB, QUANTIZED_KV, LoadParams
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
# Smallest context the planner fits to before any part of the model leaves VRAM (llama.cpp's --fit-ctx default).
FIT_CTX_MIN = 4096
# Graph inputs are kept this many times per GPU when llama.cpp runs the GPUs as a pipeline (GGML_SCHED_MAX_COPIES).
SCHED_COPIES = 4


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
    # Allocated by the GPU backend while it runs (e.g. flash attention's F16 copy of a quantized KV cache); not part
    # of the buffers llama.cpp measures, so it is reserved here.
    scratch_mib: float = 0.0
    # Correction from llama.cpp's own projection of this layout (see manager.py); negative when the estimate is high.
    calib_mib: float = 0.0
    # Part of kv_mib that exists only with several server slots (their sliding windows); llama-fit-params measures a
    # single sequence.
    swa_extra_mib: float = 0.0

    @property
    def used_mib(self) -> float:
        return (self.weights_mib + self.kv_mib + self.compute_mib + self.mmproj_mib + self.draft_mib + self.output_mib
                + self.scratch_mib + self.calib_mib)

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
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "estimate"
    use_engine_fit: bool = False
    engine: dict[str, Any] = field(default_factory=dict)
    mmproj: str = ""
    draft: str = ""
    ctx_target: int = 0  # requested context after clamping, before it was fitted to the VRAM
    context_fit: str = "off"
    ctx_adjusted: str = ""  # "raised" (fills free VRAM) | "reduced" (keeps the model in VRAM) | ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["devices"] = [x.to_dict() for x in self.devices]
        return d

    def layout_key(self) -> tuple:
        """What the engine is told to allocate (two plans with equal keys load identically)."""
        return (self.ctx, self.kv_k, self.kv_v, self.flash_attn, self.gpu_layers, self.n_cpu_moe,
                tuple(self.tensor_split or ()), self.use_engine_fit)


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


def kv_dequant_bytes(info: ModelInfo, ctx: int, kv_k: str, kv_v: str, ubatch: int, n_seq: int,
                     swa_full: bool) -> list[float]:
    """Per layer: the F16 copy of the layer's K and V that flash attention makes of a quantized KV cache.

    The GPU backends convert a quantized cache for prompt batches (Vulkan scratch buffer, CUDA/HIP memory pool).
    The copy covers the filled part of the cache, so it grows with the conversation up to the full context.
    """
    if kv_k not in QUANTIZED_KV and kv_v not in QUANTIZED_KV:
        return [0.0] * info.n_layer
    elems = kv_layer_elems(info)
    cells = kv_cells(info, ctx, ubatch, n_seq, swa_full)
    return [c * (k + v) * 2.0 for (k, v), c in zip(elems, cells)]


def compute_buffer_bytes(info: ModelInfo, ctx: int, ubatch: int, flash: bool, has_output: bool,
                         copies: int = 1) -> float:
    """Approximate llama.cpp compute (graph) buffer for one device.

    ``copies``: with pipeline parallelism the graph inputs (attention mask, hidden state) are kept this many times.
    """
    n_ff = max(info.n_ff, 1)
    act = ubatch * (info.n_embd * 8 + n_ff * 3) * 4
    if flash:
        attn = ubatch * info.n_embd * 4 * 4
    else:
        attn = float(ctx) * ubatch * max(info.n_head, 1) * 4
    logits = ubatch * info.n_vocab * 4 if has_output else 0
    inputs = (_pad(ctx, CTX_PAD) * ubatch * (2 if flash else 4) + ubatch * info.n_embd * 4) * max(1, copies)
    return max(logits, attn) + act + inputs + 32 * MiB


def _pad(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


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
        margin_mib: int = DEFAULT_VRAM_MARGIN_MIB,
        margin_per_device: dict[str, int] | None = None,
        reclaim_mib: dict[str, float] | None = None,
        mmproj_size: int = 0,
        mmproj_mib_hint: float | None = None,
        draft_info: ModelInfo | None = None,
        engine_fit: bool = False,
        model_id: str = "",
        calibration: dict[str, tuple[float, float]] | None = None,
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
        # per device (MiB, MiB per context token): difference between llama.cpp's projection and this estimate
        self.calibration = calibration or {}
        self.fa, self.candidates, self.fa_warning = self._resolve_fa_kv()

    # ----- helpers ----------------------------------------------------------------

    def _n_seq(self) -> int:
        return self.p.parallel if self.p.parallel and self.p.parallel > 0 else 4

    def _resolve_fa_kv(self) -> tuple[str, list[str], str]:
        """Flash attention mode, KV cache type candidates (preferred first) and a warning, if any."""
        p = self.p
        fa = p.flash_attn
        candidates = ["f16", "q8_0"] if p.kv_cache_type == "auto" else [p.kv_cache_type]
        if fa != "off":
            return fa, candidates, ""
        explicit_quant = p.kv_cache_type in QUANTIZED_KV or p.kv_cache_type_v in QUANTIZED_KV
        if explicit_quant:
            return "on", candidates, "A quantized KV cache requires flash attention; flash attention set to 'on'."
        return fa, [c for c in candidates if c not in QUANTIZED_KV] or ["f16"], ""

    def _flash_for_estimate(self) -> bool:
        return self.fa != "off"

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

    def _ctx_ceiling(self, ctx_target: int) -> int:
        """Largest context 'fill' may raise to: the trained context (or the requested one when it is larger)."""
        train = int(self.info.context_length or 0)
        return max(ctx_target, (train // CTX_PAD) * CTX_PAD)

    def _layout(self, ctx: int, kv_k: str, kv_v: str, n_gpu: int, n_cpu_moe: int,
                split: list[float] | None) -> tuple[list[DevicePlan], dict[str, float], list[int]]:
        info, p = self.info, self.p
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
        n_seq = self._n_seq()
        flash = self._flash_for_estimate()
        kvl = kv_layer_bytes(info, ctx, kv_k, kv_v, p.ubatch_size, n_seq, p.swa_full)
        kv1 = kv_layer_bytes(info, ctx, kv_k, kv_v, p.ubatch_size, 1, p.swa_full) if n_seq > 1 else kvl
        dq = kv_dequant_bytes(info, ctx, kv_k, kv_v, p.ubatch_size, n_seq, p.swa_full) if flash else [0.0] * n_layer
        if split is None:
            split = [max(1.0, d.free_mib - d.margin_mib) for d in devs]
        assign = assign_layers(n_layer, n_gpu if devs else 0, split) if devs else [-1] * (n_layer + 1)
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
            if p.kv_offload:
                dp.kv_mib += kv / MiB
                dp.swa_extra_mib += (kv - kv1[il]) / MiB
                dp.scratch_mib = max(dp.scratch_mib, dq[il] / MiB)  # attention runs where the layer's KV cache is
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
        # llama.cpp runs several GPUs as a pipeline when every layer is on them (and keeps copies of the inputs)
        pipeline = len(devs) > 1 and n_gpu > n_layer and p.split_mode == "layer" and p.kv_offload and n_cpu_moe <= 0
        copies = SCHED_COPIES if pipeline else 1
        # Vulkan flash attention: index scratch when the unified KV cache holds several sequences
        sparse = 2.0 * _pad(ctx, CTX_PAD) * p.ubatch_size / MiB if n_seq > 1 and flash and p.kv_offload else 0.0
        for i, dp in enumerate(devs):
            if dp.layers or i == out_dev:
                dp.compute_mib = compute_buffer_bytes(info, ctx, p.ubatch_size, flash, i == out_dev, copies) / MiB
                if dp.layers:
                    dp.scratch_mib += sparse
                cal = self.calibration.get(dp.name)
                if cal:
                    dp.calib_mib = cal[0] + cal[1] * ctx
        if n_gpu < n_layer + 1 or not devs:
            host["compute_mib"] = compute_buffer_bytes(info, ctx, p.ubatch_size, flash, out_dev < 0) / MiB
        else:
            host["compute_mib"] = (p.ubatch_size * info.n_embd * 4 * 2) / MiB + 8
        if devs and self.mmproj_mib:
            if p.mmproj_offload:
                devs[0].mmproj_mib = self.mmproj_mib
            else:
                host["compute_mib"] += self.mmproj_mib
        if devs and self.draft is not None:
            dw = self.draft.weights_bytes / MiB
            dkv = sum(kv_layer_bytes(self.draft, ctx, "f16", "f16", p.ubatch_size, 1, False)) / MiB
            dcomp = compute_buffer_bytes(self.draft, ctx, p.ubatch_size, flash, True) / MiB
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
        cost (weights + KV cache) rather than layer counts. Every GPU gets the
        same share of its capacity, so they all fill up together. The result is
        passed to llama.cpp as ``--tensor-split`` layer counts.
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
        kvl = kv_layer_bytes(info, ctx, kv_k, kv_v, self.p.ubatch_size, self._n_seq(), self.p.swa_full)
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
        caps = [max(0.0, d.free_mib - d.margin_mib - d.compute_mib - d.mmproj_mib - d.draft_mib - d.scratch_mib
                    - d.calib_mib) for d in probe]
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

    def _candidate_splits(self, ctx: int, kv_k: str, kv_v: str, n_gpu: int, n_cpu_moe: int) -> list[list[float] | None]:
        """The cost-aware split, and the same split with one layer moved to a neighbouring GPU.

        Layers come in whole units; the neighbours let a GPU with more room left take the layer that no longer
        fits on the other one. A split set by the user is used as it is.
        """
        base = self._split_for(ctx, kv_k, kv_v, n_gpu, n_cpu_moe)
        if base is None or self._user_split() is not None:
            return [base]
        out: list[list[float] | None] = [base]
        for i in range(len(base) - 1):
            for src, dst in ((i, i + 1), (i + 1, i)):
                if base[src] >= 1:
                    s = list(base)
                    s[src] -= 1
                    s[dst] += 1
                    out.append(s)
        return out

    def _placement(self, ctx: int, kv_k: str, kv_v: str, n_gpu: int,
                   n_cpu_moe: int) -> tuple[bool, list[float] | None, list[DevicePlan], dict[str, float]]:
        """(fits, split, devices, host) at ``ctx``: the fitting split with the most headroom left, if any."""
        best = None
        first = None
        for s in self._candidate_splits(ctx, kv_k, kv_v, n_gpu, n_cpu_moe):
            devs, host, _ = self._layout(ctx, kv_k, kv_v, n_gpu, n_cpu_moe, s)
            if first is None:
                first = (s, devs, host)
            if self._fits(devs):
                score = min((d.headroom_mib for d in devs), default=0.0)
                if best is None or score > best[0]:
                    best = (score, s, devs, host)
        if best is not None:
            return True, best[1], best[2], best[3]
        assert first is not None
        return False, first[0], first[1], first[2]

    def _fits_at(self, ctx: int, kv_k: str, kv_v: str, n_gpu: int, n_cpu_moe: int) -> bool:
        return self._placement(ctx, kv_k, kv_v, n_gpu, n_cpu_moe)[0]

    def _max_ctx(self, kv_k: str, kv_v: str, n_gpu: int, n_cpu_moe: int, lo: int, hi: int) -> int:
        """Largest context in [lo, hi] (a multiple of CTX_PAD) at which the layout fits in VRAM; 0 if not even lo."""
        lo = max(CTX_PAD, _pad(lo, CTX_PAD))
        if not self.devices or hi < lo or not self._fits_at(lo, kv_k, kv_v, n_gpu, n_cpu_moe):
            return 0
        if self._fits_at(hi, kv_k, kv_v, n_gpu, n_cpu_moe):
            return hi
        while hi - lo > CTX_PAD:
            mid = (lo + hi) // 2 // CTX_PAD * CTX_PAD
            if mid <= lo:
                mid = lo + CTX_PAD
            if self._fits_at(mid, kv_k, kv_v, n_gpu, n_cpu_moe):
                lo = mid
            else:
                hi = mid
        return lo

    def _max_ctx_full(self, kv: str) -> int:
        """Largest context with the whole model in VRAM (shown in the plan)."""
        hi = max(CTX_PAD, int(self.info.context_length or 131072))
        if self.p.allow_context_over_train:
            hi = max(hi, self.p.context_length)
        return self._max_ctx(kv, kv, self.info.n_layer + 1, 0, CTX_PAD, _pad(hi, CTX_PAD))

    # ----- main entry ---------------------------------------------------------------

    def plan(self) -> Plan:
        info, p = self.info, self.p
        ctx_target, notes = self.resolve_ctx()
        warnings: list[str] = [self.fa_warning] if self.fa_warning else []
        n_layer = info.n_layer
        full = n_layer + 1
        manual = p.gpu_offload == "manual"
        policy = p.context_fit if self.devices else "off"
        candidates = self.candidates

        if manual:
            n_gpu = (full if p.n_gpu_layers < 0 else min(full, p.n_gpu_layers)) if self.devices else 0
            n_cpu_moe = max(0, p.n_cpu_moe)
        else:
            n_gpu, n_cpu_moe = (full if self.devices else 0), 0

        # The layout to keep in VRAM: the whole model (automatic) or the configured layers (manual). The requested
        # context decides the KV precision (F16 unless only a smaller cache reaches it); the context is then fitted.
        chosen: str | None = None
        ctx = ctx_target
        if self.devices:
            chosen = next((kv for kv in candidates if self._fits_at(ctx_target, *self._kv_pair(kv), n_gpu, n_cpu_moe)),
                          None)
            if chosen is not None:
                if policy == "fill":
                    ctx = self._max_ctx(*self._kv_pair(chosen), n_gpu, n_cpu_moe, ctx_target,
                                        self._ctx_ceiling(ctx_target)) or ctx_target
            elif policy != "off" and ctx_target > FIT_CTX_MIN:
                best, best_kv = 0, None
                for kv in candidates:
                    c = self._max_ctx(*self._kv_pair(kv), n_gpu, n_cpu_moe, FIT_CTX_MIN, ctx_target)
                    if c > best:
                        best, best_kv = c, kv
                if best_kv is not None:
                    chosen, ctx = best_kv, best

        adjusted = "raised" if ctx > ctx_target else "reduced" if ctx < ctx_target else ""
        if adjusted == "raised":
            notes.append(f"Context raised from {ctx_target:,} to {ctx:,} tokens: the KV cache fills the free VRAM "
                         "(Context in VRAM: Fill). Choose 'Up to requested' to keep the requested length.")
        elif adjusted == "reduced":
            what = "the configured layers stay" if manual else "the whole model stays"
            warnings.append(
                f"Context reduced from {ctx_target:,} to {ctx:,} tokens so that {what} in VRAM. Running part of the "
                "model from system RAM instead would make prompt processing many times slower. Choose 'Exact' under "
                "Context in VRAM to keep the requested length.")

        split: list[float] | None
        if manual or chosen is not None:
            kv = chosen or (candidates[-1] if self.devices else candidates[0])
            k, v = self._kv_pair(kv)
            _, split, devs, host = self._placement(ctx, k, v, n_gpu, n_cpu_moe)
            if manual and self.devices and not self._fits(devs):
                warnings.append("Manual configuration exceeds free VRAM on at least one GPU; the load may fail "
                                "or spill into shared system memory (much slower).")
        elif not self.devices:
            kv = candidates[0]
            k, v = self._kv_pair(kv)
            n_gpu, split = 0, None
            devs, host, _ = self._layout(ctx, k, v, 0, 0, None)
        else:
            # Not even the minimum context fits with the whole model in VRAM.
            kv = candidates[-1]
            k, v = self._kv_pair(kv)
            if info.expert_count and any(info.layer_expert_bytes):
                # MoE: keep attention + KV on GPU, move expert FFNs of the first N layers to system RAM.
                best = None
                for n in range(0, n_layer + 1):
                    ok, s, d, h = self._placement(ctx, k, v, full, n)
                    if ok:
                        best = (n, s, d, h)
                        break
                if best is None:
                    best = (n_layer,) + self._placement(ctx, k, v, full, n_layer)[1:]
                    warnings.append("Even with all expert weights in system RAM the model does not fit in VRAM; "
                                    "reduce the context length or use a smaller quantization.")
                n_cpu_moe, split, devs, host = best
                n_gpu = full
                notes.append(
                    f"MoE offload: expert weights of the first {n_cpu_moe} of {n_layer} layers stay in system "
                    "RAM; attention, shared weights and the KV cache stay on the GPU(s)."
                )
            else:
                best = (0,) + self._placement(ctx, k, v, 0, 0)[1:]
                for n in range(full, -1, -1):
                    ok, s, d, h = self._placement(ctx, k, v, n, 0)
                    if ok:
                        best = (n, s, d, h)
                        break
                n_gpu, split, devs, host = best
                on_gpu = sum(dp.layers for dp in devs)
                notes.append(f"Partial offload: {on_gpu} of {n_layer} layers on GPU, {n_layer - on_gpu} on the "
                             "CPU (prompt processing and generation will be much slower).")
            if policy != "off" and ctx_target > FIT_CTX_MIN:
                notes.append(f"The model does not fit in VRAM even with a {FIT_CTX_MIN:,}-token context.")

        if p.kv_cache_type == "auto" and kv != "f16":
            if chosen is not None:
                notes.append("KV cache set to Q8_0 (near-lossless): with F16 the requested context does not fit in VRAM.")
            elif manual:
                notes.append("KV cache set to Q8_0 (near-lossless) to use less VRAM.")
            else:
                notes.append("KV cache set to Q8_0 (near-lossless) to keep as much of the model on the GPU(s) as possible.")
        return self._build(ctx_target=ctx_target, ctx=ctx, kv=kv, n_gpu=n_gpu, n_cpu_moe=n_cpu_moe, split=split,
                           devs=devs, host=host, notes=notes, warnings=warnings, adjusted=adjusted)

    def full_offload_plan(self, kv: str, ctx: int) -> Plan:
        """The whole model in VRAM at ``ctx`` - whether or not it fits (used to measure it with the engine)."""
        k, v = self._kv_pair(kv)
        full = self.info.n_layer + 1
        _, split, devs, host = self._placement(ctx, k, v, full, 0)
        ctx_target, notes = self.resolve_ctx()
        return self._build(ctx_target=ctx_target, ctx=ctx, kv=kv, n_gpu=full, n_cpu_moe=0, split=split, devs=devs,
                           host=host, notes=notes, warnings=[], adjusted="reduced" if ctx < ctx_target else "")

    def _build(self, *, ctx_target: int, ctx: int, kv: str, n_gpu: int, n_cpu_moe: int, split: list[float] | None,
               devs: list[DevicePlan], host: dict[str, float], notes: list[str], warnings: list[str],
               adjusted: str) -> Plan:
        info, p = self.info, self.p
        k, v = self._kv_pair(kv)
        full = info.n_layer + 1
        manual = p.gpu_offload == "manual"
        full_offload = bool(self.devices) and n_gpu >= full and n_cpu_moe == 0
        fa = self.fa
        if fa == "auto" and (k in QUANTIZED_KV or v in QUANTIZED_KV):
            fa = "on"
        kv_total = sum(kv_layer_bytes(info, ctx, k, v, p.ubatch_size, self._n_seq(), p.swa_full)) / MiB
        max_ctx = {kvt: self._max_ctx_full(kvt) for kvt in ("f16", "q8_0")} if self.devices else {}

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
            n_layer=info.n_layer,
            gpu_layers=min(n_gpu, full),
            full_offload=full_offload,
            n_cpu_moe=n_cpu_moe,
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
            warnings=warnings,
            notes=notes,
            # The engine's own --fit only places what does not fit: with the whole model in VRAM the plan's
            # layout is passed explicitly, because llama.cpp can only fit an explicit context by moving layers
            # into system RAM.
            use_engine_fit=self.engine_fit and not manual and not full_offload,
            ctx_target=ctx_target,
            context_fit=p.context_fit if self.devices else "off",
            ctx_adjusted=adjusted,
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
