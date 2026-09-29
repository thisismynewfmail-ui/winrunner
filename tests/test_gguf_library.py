from pathlib import Path

import pytest

from tests.gguf_writer import llama_like, mmproj, write_gguf
from winrunner.gguf import GGUFError, read_gguf, split_parts, summarize
from winrunner.library import ModelLibrary


def test_summarize_llama_like(tmp_path: Path):
    p = llama_like(tmp_path / "pub" / "repo" / "Test-7B-Q4_K_M.gguf", n_layer=4, n_embd=256, n_vocab=1000)
    info = summarize(p)
    assert info.kind == "llm"
    assert info.architecture == "llama"
    assert info.n_layer == 4 and info.n_embd == 256 and info.n_head == 8 and info.n_head_kv == 2
    assert info.head_dim_k == 32
    assert info.context_length == 8192
    assert info.quant == "Q4_K_M"
    assert info.bos_token == "<s>" and info.eos_token == "<|im_end|>"
    assert info.n_vocab == 1000
    assert "<|im_start|>" in info.chat_template
    assert info.sampling == {"temp": 0.6, "top_p": 0.95}
    assert len(info.layer_bytes) == 4 and all(b > 0 for b in info.layer_bytes)
    assert info.token_embd_bytes == 256 * 1000 // 256 * 144
    assert info.weights_bytes == sum(info.layer_bytes) + info.token_embd_bytes + info.output_bytes + info.other_bytes
    assert not info.output_tied


def test_large_vocab_is_skipped_but_tokens_resolved(tmp_path: Path):
    p = llama_like(tmp_path / "big.gguf", n_vocab=5000)
    g = read_gguf(p)
    vocab = g.metadata["tokenizer.ggml.tokens"]
    assert getattr(vocab, "count", None) == 5000  # not materialised
    assert g.token_strings == {1: "<s>", 2: "<|im_end|>"}


def test_mmproj_detection(tmp_path: Path):
    info = summarize(mmproj(tmp_path / "mmproj-model-f16.gguf", projection_dim=256))
    assert info.kind == "mmproj"
    assert info.mmproj["has_vision"] is True
    assert info.mmproj["projection_dim"] == 256
    assert info.mmproj["projector_type"] == "idefics3"


def test_not_gguf(tmp_path: Path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 100)
    with pytest.raises(GGUFError):
        read_gguf(p)


def test_truncated_header(tmp_path: Path):
    p = llama_like(tmp_path / "t.gguf")
    data = p.read_bytes()[:200]
    p.write_bytes(data)
    with pytest.raises(Exception):
        read_gguf(p)


def test_split_parts(tmp_path: Path):
    first = tmp_path / "m-00001-of-00003.gguf"
    parts = split_parts(first)
    assert [x.name for x in parts] == ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"]
    assert split_parts(tmp_path / "single.gguf") == [tmp_path / "single.gguf"]


def test_library_pairs_mmproj_and_resolves(tmp_path: Path):
    llama_like(tmp_path / "ggml-org" / "SmolVLM-GGUF" / "SmolVLM-256M-Q8_0.gguf", n_embd=256)
    mmproj(tmp_path / "ggml-org" / "SmolVLM-GGUF" / "mmproj-SmolVLM-256M-Q8_0.gguf", projection_dim=256)
    mmproj(tmp_path / "ggml-org" / "SmolVLM-GGUF" / "mmproj-SmolVLM-256M-f16.gguf", projection_dim=256)
    llama_like(tmp_path / "other" / "Qwen" / "qwen3-8b-q4_k_m.gguf", experts=4)
    lib = ModelLibrary(tmp_path / "index.json")
    res = lib.scan([str(tmp_path)])
    assert res["models"] == 2
    vlm = lib.get("smolvlm-256m-q8_0")
    assert vlm is not None and vlm.has_vision and vlm.lms_type == "vlm"
    assert vlm.mmproj_default.endswith("mmproj-SmolVLM-256M-f16.gguf")  # f16 preferred over q8_0
    assert vlm.mmproj_compatible() is True
    assert vlm.publisher == "ggml-org"
    q = lib.get("qwen3-8b-q4_k_m")
    assert q is not None and not q.has_vision and q.to_summary()["is_moe"]
    # resolution by various client-supplied names
    assert lib.resolve("SmolVLM-256M-Q8_0.gguf") is vlm
    assert lib.resolve("SMOLVLM-256M-Q8_0") is vlm
    assert lib.resolve("other/qwen") is q
    assert lib.resolve("qwen3-8b") is q
    assert lib.resolve("does-not-exist") is None
    # cached rescan
    res2 = lib.scan([str(tmp_path)])
    assert res2["models"] == 2


def test_library_id_collisions(tmp_path: Path):
    llama_like(tmp_path / "a" / "r1" / "model.gguf")
    llama_like(tmp_path / "b" / "r2" / "model.gguf")
    lib = ModelLibrary(tmp_path / "index.json")
    lib.scan([str(tmp_path)])
    ids = sorted(e.id for e in lib.entries())
    assert ids == ["a/model", "b/model"]


def test_write_gguf_roundtrip_types(tmp_path: Path):
    p = write_gguf(tmp_path / "t.gguf", {"general.architecture": "x", "a.bool": True, "a.float": 1.5, "a.list": [1, 2, 3],
                                          "a.strs": ["x", "y"]}, [("t", [32], 0)])
    g = read_gguf(p)
    assert g.metadata["a.bool"] is True
    assert g.metadata["a.float"] == 1.5
    assert g.metadata["a.list"] == [1, 2, 3]
    assert g.metadata["a.strs"] == ["x", "y"]
    assert g.tensors[0].n_bytes == 128
