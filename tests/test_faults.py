"""Recognising unrecoverable engine failures (GPU lost, crashes) in engine output."""

import json

import pytest

from winrunner.api_openai import ChatTap, _tap_named_event, parse_event
from winrunner.app import recommendations
from winrunner.events import EventBus, RequestTracker
from winrunner.faults import GPU_DEVICE_LOST, GPU_KERNEL_FAILURE, describe_exit_code, gpu_fault, tdr_hint, tdr_limited
from winrunner.logparse import LogParser

DEVICE_LOST = "decode() failed: vk::Queue::submit: ErrorDeviceLost"


@pytest.mark.parametrize("text", [
    DEVICE_LOST,
    "terminate called after throwing an instance of 'vk::DeviceLostError'",
    "ggml_vulkan: waitForFences failed: VK_ERROR_DEVICE_LOST",
    "ggml_vulkan: device lost",
])
def test_device_lost_is_recognised(text):
    assert gpu_fault(text) == GPU_DEVICE_LOST


@pytest.mark.parametrize("text", [
    "ROCm error: an illegal memory access was encountered",
    "CUDA error: unspecified launch failure",
    "hipErrorIllegalAddress",
])
def test_gpu_kernel_failures_are_recognised(text):
    assert gpu_fault(text) == GPU_KERNEL_FAILURE


@pytest.mark.parametrize("text", [
    None, "", "Context size has been exceeded.", "vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
    "the request exceeds the available context size", "Invalid input batch.",
])
def test_other_errors_are_not_gpu_faults(text):
    assert gpu_fault(text) is None


def test_describe_exit_code():
    assert describe_exit_code(3221226505) == "code 3221226505 = 0xC0000409, the engine aborted after a fatal error"
    assert describe_exit_code(3221225477) == "code 3221225477 = 0xC0000005, access violation"
    assert describe_exit_code(3221225725) == "code 3221225725 = 0xC00000FD, stack overflow"
    assert describe_exit_code(0xC0001234) == "code 3221230132 = 0xC0001234"
    assert describe_exit_code(-6) == "code -6 = SIGABRT, the engine aborted after a fatal error"
    assert describe_exit_code(-15) == "code -15 = SIGTERM"
    assert describe_exit_code(1) == "code 1"
    assert describe_exit_code(None) == "unknown exit code"


def test_tdr_limited():
    assert tdr_limited({"level": 3, "delay_s": 2}) is True  # Windows default
    assert tdr_limited({"level": 3, "delay_s": 60}) is False
    assert tdr_limited({"level": 0, "delay_s": 2}) is False  # detection disabled
    assert tdr_limited(None) is False
    assert "2 s" in tdr_hint({"delay_s": 2}) and "gpu-timeout.bat" in tdr_hint({"delay_s": 2})


def test_tdr_recommendation():
    base = {"cores_physical": 6, "cores_logical": 12, "cpu": "Ryzen", "ram_total": 64 << 30,
            "gpus": [{"name": "AMD Radeon RX 6800", "vendor": "AMD", "vram_total": 16 << 30}]}
    titles = [r["title"] for r in recommendations({**base, "gpu_timeout": {"level": 3, "delay_s": 2}})]
    assert "GPU timeout (TDR)" in titles
    assert "GPU timeout (TDR)" not in [r["title"] for r in recommendations({**base, "gpu_timeout": {"level": 3,
                                                                                                   "delay_s": 60}})]
    assert "GPU timeout (TDR)" not in [r["title"] for r in recommendations({**base, "gpu_timeout": None})]


def _events(lines, jsonl):
    p = LogParser(jsonl=jsonl)
    out = []
    for ln in lines:
        for ll in p.feed(ln):
            out.extend(ll.events)
    return out


def test_engine_error_records_report_gpu_faults():
    faults = [d for k, d in _events([
        json.dumps({"level": "error", "msg": f"srv  update_slots: failed to decode the batch: {DEVICE_LOST}"}),
        json.dumps({"level": "error", "msg": f"slot update_slots: id  0 | task 12 | {DEVICE_LOST}"}),
        "terminate called after throwing an instance of 'vk::DeviceLostError'",
    ], jsonl=True) if k == "gpu_fault"]
    assert len(faults) == 3 and all(f["what"] == GPU_DEVICE_LOST for f in faults)
    text = _events([f"0.05.123.456 E srv  update_slots: failed to decode the batch: {DEVICE_LOST}"], jsonl=False)
    assert [k for k, _ in text].count("gpu_fault") == 1


def test_quoted_prompts_are_not_gpu_faults():
    # Prompts only reach the log in debug records, or as unprefixed continuation lines of multi-line messages.
    ev = _events([
        json.dumps({"level": "debug", "msg": 'srv  log_server_r: request:  {"content":"' + DEVICE_LOST + '"}'}),
        json.dumps({"level": "info", "msg": "srv  params_from_: Chat format: Content-only"}),
        f"Error Log: 15:02:39 req-34933029 failed: {DEVICE_LOST}",
    ], jsonl=True)
    assert "gpu_fault" not in [k for k, _ in ev]
    ev = _events([
        "0.05.123.456 D srv  log_server_r: request: {...}",
        f"error: {DEVICE_LOST}",  # continuation line (no timestamp prefix)
    ], jsonl=False)
    assert "gpu_fault" not in [k for k, _ in ev]


class _Ctx:
    def __init__(self):
        self.bus = EventBus()
        self.tracker = RequestTracker(self.bus)


def test_stream_errors_are_captured_in_both_formats():
    err = {"code": 500, "message": DEVICE_LOST, "type": "server_error"}
    assert parse_event(b"error: " + json.dumps(err).encode()) == ("error", json.dumps(err))
    for raw in (b"error: " + json.dumps(err).encode(), b"data: " + json.dumps({"error": err}).encode()):
        ctx = _Ctx()
        rec = ctx.tracker.begin("127.0.0.1", "ua", "/v1/chat/completions", "m", True, 0)
        tap = ChatTap(ctx, rec, False, False)
        assert tap.process(raw) == raw  # forwarded unchanged
        assert tap.error == err


def test_passthrough_stream_errors():
    ctx = _Ctx()
    rec = ctx.tracker.begin("127.0.0.1", "ua", "/v1/messages", "m", True, 0)
    anthropic = b'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"' + DEVICE_LOST.encode() + b'"}}'
    assert _tap_named_event(ctx, rec, anthropic, "anthropic")[2] == {"type": "api_error", "message": DEVICE_LOST}
    legacy = b'error: {"code":500,"message":"' + DEVICE_LOST.encode() + b'","type":"server_error"}'
    assert _tap_named_event(ctx, rec, legacy, "responses")[2]["message"] == DEVICE_LOST
    ok = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hi"}'
    assert _tap_named_event(ctx, rec, ok, "responses") == ({}, "", None)


def test_tracker_restart_discards_partial_output():
    ctx = _Ctx()
    rec = ctx.tracker.begin("127.0.0.1", "ua", "/v1/chat/completions", "m", False, 0)
    ctx.tracker.progress(rec, 100, 50, 10)
    ctx.tracker.tokens(rec, "Hel", "c")
    rec.tool_calls.append("f")
    ctx.tracker.restart(rec, DEVICE_LOST)
    assert rec.retries == 1 and rec.phase == "loading_model"
    assert rec.tokens == 0 and rec.preview == "" and rec.prompt_total is None and rec.t_first_token is None
    assert rec.tool_calls == []
    assert DEVICE_LOST in ctx.bus.activity[-1]["text"]
    ctx.tracker.tokens(rec, "Hello", "c")
    ctx.tracker.finish(rec, "stop")
    assert rec.phase == "done" and rec.tokens == 1 and rec.retries == 1
