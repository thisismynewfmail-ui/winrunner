"""Synthetic model descriptions for planner tests (shapes of real models)."""

from winrunner.gguf import ModelInfo

MiB = 1024 * 1024


def fake_model(
    n_layer=64, n_embd=5120, n_head=64, n_head_kv=8, head_dim=128, n_ff=25600, n_vocab=151936,
    ctx_train=40960, layer_mib=290.0, embd_mib=417.0, expert_count=0, expert_frac=0.0, ffn_frac=0.87,
    sliding_window=0, swa_pattern=None, arch="qwen3", name="Test 32B",
) -> ModelInfo:
    """A model whose layers are ``layer_mib`` each: ``ffn_frac`` of it dense feed-forward weights (or
    ``expert_frac`` routed experts for MoE models), in three tensor groups like real GGUFs."""
    lb = int(layer_mib * MiB)
    exp = int(lb * expert_frac) if expert_count else 0
    ffn = 0 if expert_count else int(lb * ffn_frac)
    groups = ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"] if expert_count else ["ffn_up", "ffn_gate", "ffn_down"]
    part = (exp or ffn) // 3
    return ModelInfo(
        path=f"/models/{name}.gguf", file_name=f"{name}.gguf", file_size=int((lb * n_layer) + 2 * embd_mib * MiB),
        mtime=0.0, split_files=[], gguf_version=3, kind="llm", architecture=arch, name=name, basename=name,
        size_label="", organization="", quant="Q4_K_M", file_type=15, n_params=0, n_layer=n_layer,
        n_embd=n_embd, n_ff=n_ff, n_head=n_head, n_head_kv=n_head_kv, head_dim_k=head_dim, head_dim_v=head_dim,
        n_vocab=n_vocab, context_length=ctx_train, rope_freq_base=1e6, rope_scaling="", rope_scaling_factor=None,
        rope_orig_ctx=None, rope_dim=head_dim, expert_count=expert_count, expert_used_count=8 if expert_count else 0,
        sliding_window=sliding_window, swa_pattern=swa_pattern, full_attention_interval=0, kv_lora_rank=0,
        key_length_mla=0, value_length_mla=0, recurrent=False, pooling_type=None, nextn_layers=0,
        tokenizer_model="gpt2", bos_token=None, eos_token="<|im_end|>", eot_token=None, add_bos=False,
        chat_template="", named_templates={}, sampling={}, mmproj={},
        weights_bytes=int(lb * n_layer + 2 * embd_mib * MiB), layer_bytes=[lb] * n_layer,
        layer_expert_bytes=[exp] * n_layer, layer_ffn_bytes=[ffn] * n_layer,
        token_embd_bytes=int(embd_mib * MiB), output_bytes=int(embd_mib * MiB), output_tied=False,
        other_bytes=0, tensor_types={}, metadata_brief={},
        layer_parts=[[[g, part] for g in groups] if part else [] for _ in range(n_layer)],
    )
