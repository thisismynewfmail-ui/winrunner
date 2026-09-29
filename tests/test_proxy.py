"""Stream tap / aggregation behaviour against real llama-server chunk shapes."""

import asyncio
import json

from winrunner.api_openai import ChatTap, SSEParser, parse_event
from winrunner.events import EventBus, RequestTracker


class Ctx:
    def __init__(self):
        self.bus = EventBus()
        self.tracker = RequestTracker(self.bus)


def chunk(obj):
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode()


BASE = {"created": 1, "id": "chatcmpl-x", "model": "m", "system_fingerprint": "b1", "object": "chat.completion.chunk"}
STREAM = [
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"role": "assistant", "content": None}}],
           "prompt_progress": {"total": 20, "cache": 0, "processed": 10, "time_ms": 5}}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"role": "assistant", "content": None}}],
           "prompt_progress": {"total": 20, "cache": 0, "processed": 20, "time_ms": 9}}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"role": "assistant", "content": None}}]}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"reasoning_content": "Think"}}]}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"content": "Hel"}}]}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"content": "lo"}}]}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}}]}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "{\"city\": "}}]}}]}),
    chunk({**BASE, "choices": [{"finish_reason": None, "index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "\"Paris\"}"}}]}}]}),
    chunk({**BASE, "choices": [{"finish_reason": "tool_calls", "index": 0, "delta": {}}],
           "timings": {"prompt_n": 20, "predicted_n": 5, "predicted_per_second": 40.0, "prompt_per_second": 300.0}}),
    chunk({**BASE, "choices": [], "usage": {"completion_tokens": 5, "prompt_tokens": 20, "total_tokens": 25}}),
    b"data: [DONE]",
]


def run_tap(client_progress=False, client_usage=False, lms=None):
    ctx = Ctx()
    rec = ctx.tracker.begin("127.0.0.1", "ua", "/v1/chat/completions", "m", True, 0)
    tap = ChatTap(ctx, rec, client_progress, client_usage, lms)
    out = [tap.process(c) for c in STREAM]
    return tap, [o for o in out if o is not None], rec


def test_progress_and_usage_stripped_by_default():
    tap, out, rec = run_tap()
    texts = b"\n".join(out).decode()
    assert "prompt_progress" not in texts
    assert '"choices":[]' not in texts
    assert out[-1] == b"data: [DONE]"
    # the native role chunk passes through unchanged (byte identical)
    assert out[0] == STREAM[2]
    assert rec.prompt_total == 20 and rec.prompt_processed == 20
    assert tap.usage["completion_tokens"] == 5


def test_progress_and_usage_kept_when_requested():
    tap, out, _ = run_tap(client_progress=True, client_usage=True)
    texts = b"\n".join(out).decode()
    assert texts.count("prompt_progress") == 2 and '"choices":[]' in texts


def test_aggregate_matches_openai_shape():
    tap, _, _ = run_tap()
    agg = tap.aggregate()
    msg = agg["choices"][0]["message"]
    assert msg["content"] == "Hello" and msg["reasoning_content"] == "Think"
    assert msg["tool_calls"] == [{"id": "call1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]
    assert agg["choices"][0]["finish_reason"] == "tool_calls"
    assert agg["object"] == "chat.completion" and agg["id"] == "chatcmpl-x"
    assert agg["usage"]["total_tokens"] == 25 and agg["timings"]["predicted_n"] == 5


def test_lmstudio_stats_attached_to_final_chunk():
    lms = {"model_info": {"arch": "qwen3"}, "runtime": {"name": "llama.cpp-vulkan"}}
    _, out, _ = run_tap(lms=lms)
    final = [json.loads(o[6:]) for o in out if o.startswith(b"data: {") and b"finish_reason\":\"tool_calls" in o][0]
    assert final["stats"]["stop_reason"] == "toolCalls"
    assert final["stats"]["tokens_per_second"] == 40.0
    assert final["model_info"]["arch"] == "qwen3"


def test_sse_parser_splits_across_chunks():
    p = SSEParser()
    raw = b"data: {\"a\":1}\n\ndata: {\"b\":2}\r\n\r\n: ping\n\nevent: response.created\ndata: {\"type\":\"x\"}\n\n"
    got = []
    for i in range(0, len(raw), 7):
        got += p.feed(raw[i:i + 7])
    got += p.flush()
    assert len(got) == 4
    assert parse_event(got[2]) == (None, None)
    assert parse_event(got[3]) == ("response.created", '{"type":"x"}')


def test_tracker_finish_summary():
    ctx = Ctx()
    loop = asyncio.new_event_loop()
    ctx.bus.bind(loop)
    rec = ctx.tracker.begin("1.2.3.4", "ua", "/v1/chat/completions", "m", False, 1)
    ctx.tracker.tokens(rec, "a", "c")
    ctx.tracker.finish(rec, "stop", {"prompt_n": 10, "cache_n": 5, "predicted_n": 3, "predicted_per_second": 12.5,
                                     "prompt_per_second": 100.0, "prompt_ms": 50, "predicted_ms": 240})
    assert rec.phase == "done" and rec.tokens == 3 and rec.prompt_total == 15 and rec.prompt_cached == 5
    assert ctx.tracker.totals["completion_tokens"] == 3
    assert ctx.tracker.recent()[0]["id"] == rec.id
    loop.close()
