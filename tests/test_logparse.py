import json

from winrunner.logparse import LogParser


def events(lines, jsonl=False):
    p = LogParser(jsonl=jsonl)
    out = []
    for ln in lines:
        for ll in p.feed(ln):
            out.extend(ll.events)
    for ll in p.flush():
        out.extend(ll.events)
    return out


def test_load_sequence_text_mode():
    lines = [
        "0.00.019.406 I srv    load_model: loading model '/m/q.gguf'",
        "0.00.036.595 I srv    load_model: [mtmd] estimated worst-case memory usage of mmproj is 119.95 MiB (took 17.07 ms)",
        "0.00.193.536 I common_memory_breakdown_print: |   - Vulkan0 (RX 6800)  | 16368 = 1003 + (14120 = 11000 +    2800 +     320) +        1245 |",
        "0.00.340.571 I load_tensors: loading model tensors, this can take a while... (load_mode = none)",
        "0.00.363.369 I load_tensors: offloaded 65/65 layers to GPU",
        "0.00.363.377 I load_tensors:      Vulkan0 model buffer size =  9570.24 MiB",
        "0.00.363.377 I load_tensors:  CPU_Mapped model buffer size =   417.00 MiB",
        "0.00.366.540 I llama_context: n_ctx                 = 65536",
        "0.00.367.071 I llama_kv_cache:    Vulkan0 KV buffer size =  4352.00 MiB",
        "0.00.465.426 I llama_kv_cache: size = 8704.00 MiB ( 65536 cells,  64 layers,  4/4 seqs), K (q8_0): 4352.00 MiB, V (q8_0): 4352.00 MiB",
        "0.00.467.104 I resolve_fused_ops: Flash Attention enabled",
        "0.00.471.348 I sched_reserve:    Vulkan0 compute buffer size =   302.00 MiB",
        "0.01.364.764 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 65536, kv_unified = 'true'",
        "0.01.369.837 I srv          init: init: chat template, example_format: '<|im_start|>system",
        "You are a helpful assistant<|im_end|>",
        "<|im_start|>assistant",
        "'",
        "0.01.372.028 I srv          init: init: chat template, thinking = 1",
        "0.01.372.148 I srv  llama_server: model loaded",
    ]
    ev = events(lines)
    kinds = [k for k, _ in ev]
    assert ("phase", {"phase": "open", "path": "/m/q.gguf"}) in ev
    assert ("mmproj", {"est_mib": 119.95}) in ev
    bd = [d for k, d in ev if k == "breakdown"][0]
    assert bd["device"] == "Vulkan0" and bd["model"] == 11000 and bd["context"] == 2800 and bd["compute"] == 320
    assert ("offload", {"gpu_layers": 65, "total_layers": 65}) in ev
    assert ("buffer", {"kind": "model", "device": "Vulkan0", "mib": 9570.24}) in ev
    assert ("buffer", {"kind": "kv", "device": "Vulkan0", "mib": 4352.0}) in ev
    assert ("buffer", {"kind": "compute", "device": "Vulkan0", "mib": 302.0}) in ev
    kv = [d for k, d in ev if k == "kv"][0]
    assert kv["k_type"] == "q8_0" and kv["cells"] == 65536
    assert ("flash_attn", {"enabled": True}) in ev
    assert ("slots", {"n_slots": 4, "n_ctx_slot": 65536}) in ev
    tmpl = [d for k, d in ev if k == "template" and "example" in d][0]
    assert tmpl["example"].startswith("<|im_start|>system\nYou are")
    assert ("template", {"thinking": True}) in ev
    assert kinds[-1] == "phase"


def test_request_timings():
    lines = [
        "1.03.872.755 I slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 148",
        "1.07.669.688 I slot print_timing: id  0 | task 7 | prompt eval time =    3701.03 ms /   148 tokens (   25.01 ms per token,    39.99 tokens per second)",
        "1.07.669.736 I slot print_timing: id  0 | task 7 |        eval time =      95.75 ms /     6 tokens (   19.15 ms per token,    52.22 tokens per second)",
        "1.07.670.190 I slot      release: id  0 | task 7 | stop processing: n_tokens = 153, truncated = 0",
        "1.05.868.586 I image decoded (batch 1/1) in 149 ms",
    ]
    ev = dict((k, d) for k, d in events(lines))
    assert ev["task_prompt"]["n_prompt"] == 148 and ev["task_prompt"]["task"] == 7
    assert ev["task_prompt_timing"]["tps"] == 39.99
    assert ev["task_eval_timing"]["n"] == 6
    assert ev["task_end"]["n_tokens"] == 153
    assert ev["image"]["ms"] == 149


def test_errors_and_jsonl():
    lines = [json.dumps({"type": "log", "time": 1, "level": "error", "msg": "ggml_vulkan: Device memory allocation of size 123 failed.\n"}),
             json.dumps({"type": "log", "time": 2, "level": "info", "msg": "load_tensors: offloaded 10/65 layers to GPU\n"}),
             json.dumps({"type": "log", "time": 3, "level": "warn", "msg": "some warning\n"})]
    ev = events(lines, jsonl=True)
    assert ev[0][0] == "error"
    assert ("offload", {"gpu_layers": 10, "total_layers": 65}) in ev
    fatal = events(["0.00.1.0 I llama_model_load: error loading model: vk::DeviceLostError"])
    assert any(k == "error" for k, _ in fatal)
