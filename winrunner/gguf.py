"""GGUF file reader.

A dependency-free reader for the GGUF container used by llama.cpp. Only the
header is parsed (metadata key/values and tensor descriptors); tensor data is
never touched, so even 100+ GB models are indexed in milliseconds. Large
arrays (tokenizer vocab, merges) are skipped but their size is recorded; the
BOS/EOS token strings are resolved directly from the vocab array.

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import mmap
import os
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL = range(8)
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    T_UINT8: "B",
    T_INT8: "b",
    T_UINT16: "H",
    T_INT16: "h",
    T_UINT32: "I",
    T_INT32: "i",
    T_FLOAT32: "f",
    T_BOOL: "?",
    T_UINT64: "Q",
    T_INT64: "q",
    T_FLOAT64: "d",
}
_TYPE_NAMES = {
    T_UINT8: "u8", T_INT8: "i8", T_UINT16: "u16", T_INT16: "i16", T_UINT32: "u32",
    T_INT32: "i32", T_FLOAT32: "f32", T_BOOL: "bool", T_STRING: "str", T_ARRAY: "arr",
    T_UINT64: "u64", T_INT64: "i64", T_FLOAT64: "f64",
}

# ggml tensor types: id -> (name, block size, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
}

# llama_ftype (general.file_type) -> label
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}

KEEP_ARRAY_NUMERIC = 8192  # numeric arrays up to this length are kept (per-layer params)
KEEP_ARRAY_STRING = 64  # string arrays up to this length are kept

SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# Architectures whose GGUFs are embedding (pooling) models.
EMBEDDING_ARCHS = {
    "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "t5encoder",
    "modern-bert", "neo-bert", "eurobert",
}


class GGUFError(Exception):
    pass


@dataclass
class ArrayInfo:
    """Placeholder for an array that was too large to keep in memory."""

    elem_type: str
    count: int

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<array {self.elem_type}[{self.count}]>"


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    type_id: int
    offset: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, (f"T{self.type_id}", 1, 0))[0]

    @property
    def n_bytes(self) -> int:
        _, bs, ts = GGML_TYPES.get(self.type_id, ("?", 1, 2))
        return (self.n_elements // bs) * ts


class _Reader:
    def __init__(self, buf: mmap.mmap | bytes, endian: str):
        self.buf = buf
        self.pos = 0
        self.e = endian
        self.str_len_fmt = endian + "Q"

    def unpack(self, fmt: str) -> Any:
        s = struct.Struct(self.e + fmt)
        v = s.unpack_from(self.buf, self.pos)
        self.pos += s.size
        return v[0] if len(v) == 1 else v

    def string(self) -> str:
        (n,) = struct.unpack_from(self.str_len_fmt, self.buf, self.pos)
        self.pos += struct.calcsize(self.str_len_fmt)
        if n > 64 * 1024 * 1024:
            raise GGUFError(f"implausible string length {n}")
        raw = self.buf[self.pos : self.pos + n]
        self.pos += n
        return bytes(raw).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        (n,) = struct.unpack_from(self.str_len_fmt, self.buf, self.pos)
        self.pos += struct.calcsize(self.str_len_fmt) + n

    def value(self, vtype: int, key: str = "") -> Any:
        if vtype in _SCALAR_FMT:
            return self.unpack(_SCALAR_FMT[vtype])
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            etype = self.unpack("I")
            count = self.unpack("Q") if self.str_len_fmt.endswith("Q") else self.unpack("I")
            return self.array(etype, count, key)
        raise GGUFError(f"unknown metadata value type {vtype} for key {key!r}")

    def array(self, etype: int, count: int, key: str) -> Any:
        if etype in _SCALAR_FMT:
            fmt = _SCALAR_FMT[etype]
            size = struct.calcsize(fmt)
            if count <= KEEP_ARRAY_NUMERIC:
                vals = list(struct.unpack_from(f"{self.e}{count}{fmt}", self.buf, self.pos))
                self.pos += size * count
                return vals
            self.pos += size * count
            return ArrayInfo(_TYPE_NAMES[etype], count)
        if etype == T_STRING:
            if count <= KEEP_ARRAY_STRING:
                return [self.string() for _ in range(count)]
            start = self.pos
            for _ in range(count):
                self.skip_string()
            info = ArrayInfo("str", count)
            info.start = start  # type: ignore[attr-defined]
            return info
        if etype == T_ARRAY:
            return [self.value(T_ARRAY, key) for _ in range(count)]
        raise GGUFError(f"unknown array element type {etype} for key {key!r}")

    def string_at_index(self, start: int, index: int) -> str:
        pos = self.pos
        self.pos = start
        try:
            for _ in range(index):
                self.skip_string()
            return self.string()
        finally:
            self.pos = pos


@dataclass
class GGUFFile:
    path: str
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    token_strings: dict[int, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)


def read_gguf(path: str | os.PathLike, resolve_token_ids: tuple[str, ...] = (
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.padding_token_id",
)) -> GGUFFile:
    """Parse the GGUF header of ``path``."""
    path = str(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8 or head[:4] != GGUF_MAGIC:
            raise GGUFError(f"not a GGUF file: {path}")
        size = os.fstat(f.fileno()).st_size
        (ver_le,) = struct.unpack("<I", head[4:8])
        endian = "<"
        version = ver_le
        if ver_le & 0xFFFF == 0 and ver_le > 0xFFFF:  # big-endian file
            endian = ">"
            (version,) = struct.unpack(">I", head[4:8])
        if version not in (1, 2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            r = _Reader(mm, endian)
            if version == 1:
                r.str_len_fmt = endian + "I"
            r.pos = 8
            count_fmt = "I" if version == 1 else "Q"
            n_tensors = r.unpack(count_fmt)
            n_kv = r.unpack(count_fmt)
            if n_tensors > 10_000_000 or n_kv > 1_000_000:
                raise GGUFError("implausible header counts (corrupt file?)")
            md: dict[str, Any] = {}
            for _ in range(n_kv):
                key = r.string()
                vtype = r.unpack("I")
                md[key] = r.value(vtype, key)

            tensors: list[TensorInfo] = []
            for _ in range(n_tensors):
                name = r.string()
                n_dims = r.unpack("I")
                dims = [r.unpack(count_fmt) for _ in range(n_dims)]
                ttype = r.unpack("I")
                off = r.unpack("Q")
                tensors.append(TensorInfo(name, dims, ttype, off))
            if r.pos > size:
                raise GGUFError("truncated GGUF header")

            tokens: dict[int, str] = {}
            vocab = md.get("tokenizer.ggml.tokens")
            for k in resolve_token_ids:
                tid = md.get(k)
                if not isinstance(tid, int) or tid < 0:
                    continue
                if isinstance(vocab, list) and tid < len(vocab):
                    tokens[tid] = vocab[tid]
                elif isinstance(vocab, ArrayInfo) and hasattr(vocab, "start") and tid < vocab.count:
                    tokens[tid] = r.string_at_index(vocab.start, tid)  # type: ignore[attr-defined]
    return GGUFFile(path=path, version=version, metadata=md, tensors=tensors, token_strings=tokens)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Everything WinRunner needs to know about a GGUF, JSON serialisable."""

    path: str
    file_name: str
    file_size: int
    mtime: float
    split_files: list[str]
    gguf_version: int
    kind: str  # "llm" | "embedding" | "mmproj" | "other"
    architecture: str
    name: str
    basename: str
    size_label: str
    organization: str
    quant: str
    file_type: int | None
    n_params: int
    n_layer: int
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: Any  # int or per-layer list
    head_dim_k: int
    head_dim_v: int
    n_vocab: int
    context_length: int
    rope_freq_base: float | None
    rope_scaling: str
    rope_scaling_factor: float | None
    rope_orig_ctx: int | None
    rope_dim: int | None
    expert_count: int
    expert_used_count: int
    sliding_window: int
    swa_pattern: Any
    full_attention_interval: int
    kv_lora_rank: int
    key_length_mla: int
    value_length_mla: int
    recurrent: bool
    pooling_type: int | None
    nextn_layers: int
    tokenizer_model: str
    bos_token: str | None
    eos_token: str | None
    eot_token: str | None
    add_bos: bool | None
    chat_template: str
    named_templates: dict[str, str]
    sampling: dict[str, Any]
    mmproj: dict[str, Any]
    weights_bytes: int
    layer_bytes: list[int]
    layer_expert_bytes: list[int]
    layer_ffn_bytes: list[int]
    token_embd_bytes: int
    output_bytes: int
    output_tied: bool
    other_bytes: int
    tensor_types: dict[str, int]
    metadata_brief: dict[str, Any]
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInfo":
        names = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in names})


def split_parts(path: Path) -> list[Path]:
    """All files of a split GGUF given its first part (or [path])."""
    m = SPLIT_RE.search(path.name)
    if not m:
        return [path]
    total = int(m.group(2))
    stem = path.name[: m.start()]
    parts = [path.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]
    return parts


def _brief(v: Any) -> Any:
    if isinstance(v, ArrayInfo):
        return f"[{v.elem_type} x {v.count}]"
    if isinstance(v, str) and len(v) > 400:
        return v[:400] + f"... ({len(v)} chars)"
    if isinstance(v, list) and len(v) > 64:
        return f"[{len(v)} values]"
    if isinstance(v, float):
        return round(v, 8)
    return v


def _int(v: Any, default: int = 0) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return int(max(v))
    return default


_LAYER_RE = re.compile(r"^blk\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps\.")
_FFN_RE = re.compile(r"\.ffn_(up|down|gate|gate_up)\.")


def summarize(path: str | os.PathLike) -> ModelInfo:
    p = Path(path)
    parts = split_parts(p)
    g = read_gguf(p)
    md = g.metadata
    tensors = list(g.tensors)
    file_size = 0
    for part in parts:
        try:
            file_size += part.stat().st_size
        except OSError:
            pass
    for part in parts[1:]:
        try:
            tensors.extend(read_gguf(part, resolve_token_ids=()).tensors)
        except (OSError, GGUFError):
            pass

    arch = str(md.get("general.architecture", "") or "")
    gtype = str(md.get("general.type", "") or "")
    a = lambda k, d=None: md.get(f"{arch}.{k}", d)  # noqa: E731

    is_mmproj = gtype == "mmproj" or arch == "clip" or any(k.startswith("clip.") for k in md)
    pooling = md.get(f"{arch}.pooling_type")
    kind = "llm"
    if is_mmproj:
        kind = "mmproj"
    elif arch in EMBEDDING_ARCHS or (pooling not in (None, 0) and not md.get("tokenizer.chat_template")):
        kind = "embedding"
    elif not arch:
        kind = "other"

    # --- tensor accounting -------------------------------------------------
    n_layer = _int(a("block_count"), 0)
    layer_bytes = [0] * n_layer
    layer_expert = [0] * n_layer
    layer_ffn = [0] * n_layer
    token_embd = output = other = 0
    has_output = False
    n_params = 0
    type_hist: dict[str, int] = {}
    for t in tensors:
        nb = t.n_bytes
        n_params += t.n_elements
        type_hist[t.type_name] = type_hist.get(t.type_name, 0) + nb
        m = _LAYER_RE.match(t.name)
        if m:
            il = int(m.group(1))
            if il >= len(layer_bytes):  # e.g. MTP / nextn layers beyond block_count
                grow = il + 1 - len(layer_bytes)
                layer_bytes += [0] * grow
                layer_expert += [0] * grow
                layer_ffn += [0] * grow
            layer_bytes[il] += nb
            if _EXPERT_RE.search(t.name):
                layer_expert[il] += nb
            elif _FFN_RE.search(t.name):
                layer_ffn[il] += nb
        elif t.name.startswith("token_embd."):
            token_embd += nb
        elif t.name.startswith("output.") and not t.name.startswith("output_norm"):
            output += nb
            has_output = True
        else:
            other += nb
    weights = sum(layer_bytes) + token_embd + output + other

    # --- hyperparameters ---------------------------------------------------
    n_embd = _int(a("embedding_length"), 0)
    n_head_raw = a("attention.head_count", 0)
    n_head = _int(n_head_raw, 0)
    n_head_kv = a("attention.head_count_kv", n_head_raw)
    head_dim_k = _int(a("attention.key_length"), 0) or (n_embd // n_head if n_head else 0)
    head_dim_v = _int(a("attention.value_length"), 0) or head_dim_k
    vocab = md.get("tokenizer.ggml.tokens")
    n_vocab = _int(a("vocab_size"), 0) or (
        vocab.count if isinstance(vocab, ArrayInfo) else len(vocab) if isinstance(vocab, list) else 0
    )

    ftype = md.get("general.file_type")
    quant = FILE_TYPES.get(ftype, "") if isinstance(ftype, int) else ""
    if not quant and type_hist:
        quant = max(type_hist.items(), key=lambda kv: kv[1])[0]

    rope_scaling = str(a("rope.scaling.type", "") or "")
    templates: dict[str, str] = {}
    for k, v in md.items():
        if k.startswith("tokenizer.chat_template.") and isinstance(v, str):
            templates[k.split(".", 2)[2]] = v

    sampling = {
        k.split(".", 2)[2]: (round(v, 6) if isinstance(v, float) else v)
        for k, v in md.items()
        if k.startswith("general.sampling.") and not isinstance(v, ArrayInfo)
    }

    mm: dict[str, Any] = {}
    if is_mmproj:
        mm = {
            "projector_type": md.get("clip.projector_type") or md.get("clip.vision.projector_type") or "",
            "has_vision": bool(md.get("clip.has_vision_encoder", False)),
            "has_audio": bool(md.get("clip.has_audio_encoder", False)),
            "image_size": md.get("clip.vision.image_size"),
            "patch_size": md.get("clip.vision.patch_size"),
            "projection_dim": md.get("clip.vision.projection_dim") or md.get("clip.audio.projection_dim"),
            "vision_layers": md.get("clip.vision.block_count"),
            "vision_embd": md.get("clip.vision.embedding_length"),
            "audio_projector_type": md.get("clip.audio.projector_type") or "",
        }
        if not mm["has_vision"] and not mm["has_audio"]:
            mm["has_vision"] = any(k.startswith("clip.vision.") for k in md)

    tok = g.token_strings
    bos_id = md.get("tokenizer.ggml.bos_token_id")
    eos_id = md.get("tokenizer.ggml.eos_token_id")
    eot_id = md.get("tokenizer.ggml.eot_token_id")

    brief = {k: _brief(v) for k, v in md.items() if k != "tokenizer.chat_template" and not k.startswith("tokenizer.chat_template.")}

    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0

    swa_pattern = a("attention.sliding_window_pattern")
    if isinstance(swa_pattern, ArrayInfo):
        swa_pattern = None

    return ModelInfo(
        path=str(p),
        file_name=p.name,
        file_size=file_size,
        mtime=mtime,
        split_files=[str(x) for x in parts] if len(parts) > 1 else [],
        gguf_version=g.version,
        kind=kind,
        architecture=arch,
        name=str(md.get("general.name", "") or ""),
        basename=str(md.get("general.basename", "") or ""),
        size_label=str(md.get("general.size_label", "") or ""),
        organization=str(md.get("general.organization", "") or md.get("general.author", "") or ""),
        quant=quant,
        file_type=ftype if isinstance(ftype, int) else None,
        n_params=n_params,
        n_layer=n_layer,
        n_embd=n_embd,
        n_ff=_int(a("feed_forward_length"), 0),
        n_head=n_head,
        n_head_kv=n_head_kv if not isinstance(n_head_kv, ArrayInfo) else n_head,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        n_vocab=n_vocab,
        context_length=_int(a("context_length"), 0),
        rope_freq_base=a("rope.freq_base"),
        rope_scaling=rope_scaling,
        rope_scaling_factor=a("rope.scaling.factor"),
        rope_orig_ctx=a("rope.scaling.original_context_length"),
        rope_dim=a("rope.dimension_count"),
        expert_count=_int(a("expert_count"), 0),
        expert_used_count=_int(a("expert_used_count"), 0),
        sliding_window=_int(a("attention.sliding_window"), 0),
        swa_pattern=swa_pattern,
        full_attention_interval=_int(a("full_attention_interval"), 0),
        kv_lora_rank=_int(a("attention.kv_lora_rank"), 0),
        key_length_mla=_int(a("attention.key_length_mla"), 0),
        value_length_mla=_int(a("attention.value_length_mla"), 0),
        recurrent=any(k.startswith(f"{arch}.ssm.") or k.startswith(f"{arch}.wkv.") for k in md),
        pooling_type=pooling if isinstance(pooling, int) else None,
        nextn_layers=_int(a("nextn_predict_layers"), 0),
        tokenizer_model=str(md.get("tokenizer.ggml.model", "") or ""),
        bos_token=tok.get(bos_id) if isinstance(bos_id, int) else None,
        eos_token=tok.get(eos_id) if isinstance(eos_id, int) else None,
        eot_token=tok.get(eot_id) if isinstance(eot_id, int) else None,
        add_bos=md.get("tokenizer.ggml.add_bos_token"),
        chat_template=str(md.get("tokenizer.chat_template", "") or ""),
        named_templates=templates,
        sampling=sampling,
        mmproj=mm,
        weights_bytes=weights,
        layer_bytes=layer_bytes,
        layer_expert_bytes=layer_expert,
        layer_ffn_bytes=layer_ffn,
        token_embd_bytes=token_embd,
        output_bytes=output if has_output else token_embd,
        output_tied=not has_output,
        other_bytes=other,
        tensor_types=type_hist,
        metadata_brief=brief,
    )
