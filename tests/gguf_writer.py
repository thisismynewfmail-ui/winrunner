"""Tiny GGUF writer used to create synthetic test models (header + zeroed tensor data)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

T = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "str": 8, "arr": 9, "u64": 10,
     "i64": 11, "f64": 12}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
TYPE_SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 12: (256, 144), 14: (256, 210)}


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _vtype(v: Any) -> int:
    if isinstance(v, bool):
        return T["bool"]
    if isinstance(v, int):
        return T["u32"] if 0 <= v < 2**32 else T["i64"]
    if isinstance(v, float):
        return T["f32"]
    if isinstance(v, str):
        return T["str"]
    if isinstance(v, list):
        return T["arr"]
    raise TypeError(type(v))


def _value(v: Any) -> bytes:
    t = _vtype(v)
    if t == T["str"]:
        return _s(v)
    if t == T["arr"]:
        et = _vtype(v[0]) if v else T["u32"]
        out = struct.pack("<IQ", et, len(v))
        for x in v:
            out += _s(x) if et == T["str"] else struct.pack("<" + FMT[et], x)
        return out
    return struct.pack("<" + FMT[t], v)


def write_gguf(path: Path, metadata: dict[str, Any], tensors: list[tuple[str, list[int], int]]) -> Path:
    """tensors: (name, shape, ggml type id)."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        out += _s(k) + struct.pack("<I", _vtype(v)) + _value(v)
    offset = 0
    sizes = []
    for name, shape, ttype in tensors:
        n = 1
        for d in shape:
            n *= d
        bs, ts = TYPE_SIZES[ttype]
        nbytes = n // bs * ts
        sizes.append(nbytes)
        out += _s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<IQ", ttype, offset)
        offset += (nbytes + 31) // 32 * 32
    pad = (-len(out)) % 32
    out += b"\0" * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)
        f.truncate(len(out) + offset)  # sparse zeroed tensor data
    return path


def llama_like(path: Path, n_layer: int = 4, n_embd: int = 256, n_vocab: int = 1000, n_head: int = 8, n_head_kv: int = 2,
               ctx: int = 8192, template: str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
               arch: str = "llama", extra: dict | None = None, experts: int = 0) -> Path:
    tokens = [f"tok{i}" for i in range(n_vocab)]
    tokens[1] = "<s>"
    tokens[2] = "<|im_end|>"
    md = {
        "general.architecture": arch,
        "general.name": "Test Model",
        "general.file_type": 15,
        f"{arch}.block_count": n_layer,
        f"{arch}.context_length": ctx,
        f"{arch}.embedding_length": n_embd,
        f"{arch}.feed_forward_length": n_embd * 4,
        f"{arch}.attention.head_count": n_head,
        f"{arch}.attention.head_count_kv": n_head_kv,
        f"{arch}.rope.freq_base": 10000.0,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.chat_template": template,
        "general.sampling.temp": 0.6,
        "general.sampling.top_p": 0.95,
    }
    if experts:
        md[f"{arch}.expert_count"] = experts
        md[f"{arch}.expert_used_count"] = 2
    if extra:
        md.update(extra)
    tensors = [("token_embd.weight", [n_embd, n_vocab], 12), ("output_norm.weight", [n_embd], 0),
               ("output.weight", [n_embd, n_vocab], 14)]
    for i in range(n_layer):
        tensors += [(f"blk.{i}.attn_q.weight", [n_embd, n_embd], 12), (f"blk.{i}.attn_k.weight", [n_embd, n_embd // 4], 12),
                    (f"blk.{i}.attn_v.weight", [n_embd, n_embd // 4], 12), (f"blk.{i}.attn_norm.weight", [n_embd], 0)]
        if experts:
            tensors += [(f"blk.{i}.ffn_up_exps.weight", [n_embd, n_embd * 2, experts], 12),
                        (f"blk.{i}.ffn_down_exps.weight", [n_embd * 2, n_embd, experts], 12)]
        else:
            tensors += [(f"blk.{i}.ffn_up.weight", [n_embd, n_embd * 4], 12), (f"blk.{i}.ffn_down.weight", [n_embd * 4, n_embd], 12)]
    return write_gguf(path, md, tensors)


def mmproj(path: Path, projection_dim: int = 256) -> Path:
    md = {
        "general.architecture": "clip",
        "general.type": "mmproj",
        "clip.projector_type": "idefics3",
        "clip.has_vision_encoder": True,
        "clip.vision.image_size": 512,
        "clip.vision.patch_size": 16,
        "clip.vision.projection_dim": projection_dim,
        "clip.vision.embedding_length": 768,
    }
    return write_gguf(path, md, [("v.patch_embd.weight", [16, 16, 3, 768], 1), ("mm.model.fc.weight", [768, projection_dim], 1)])
