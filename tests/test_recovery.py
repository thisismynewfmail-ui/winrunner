"""Engine failure recovery, end to end against a scripted fake llama-server (tests/fake_engine.py).

Covers the two field failures this recovery exists for:
* the GPU is lost while the engine runs ("decode() failed: vk::Queue::submit: ErrorDeviceLost"): the engine
  stays alive but can never compute again, so every later request used to fail the same way;
* the engine process dies (abort 0xC0000409 on the next request, or while idle): the model used to disappear.
"""

from __future__ import annotations

import json
import sys
import time

import httpx
import pytest

from tests.harness import (CHAT, activity, chat, failed_instance, load, model_id, ready_instance, starts,
                           stream_lines, wait_for)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake engine is started as a POSIX script")


def test_device_lost_request_runs_again_on_restarted_engine(winrunner, state):
    base, ctx = winrunner("device_lost,ok")
    r = chat(base, ctx)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "Hello"
    assert starts(state) == 2
    rec = ctx.tracker.recent(1)[0]
    assert rec["retries"] == 1 and rec["phase"] == "done" and not rec["error"]
    assert ready_instance(ctx).restarts == 1
    log = activity(ctx)
    assert "GPU device lost" in log and "Restarting" in log and "recovered" in log
    # Before the fix every following request failed with ErrorDeviceLost until the model was reloaded by hand.
    for _ in range(3):
        assert chat(base, ctx).status_code == 200
    assert starts(state) == 2


def test_streaming_request_runs_again_when_nothing_was_sent(winrunner, state):
    base, ctx = winrunner("device_lost,ok")
    lines = stream_lines(base, ctx)
    assert lines[-1] == "data: [DONE]"
    assert not any("error" in ln for ln in lines)
    assert "".join(json.loads(ln[6:])["choices"][0]["delta"].get("content") or ""
                   for ln in lines if ln.startswith("data: {") and json.loads(ln[6:])["choices"]) == "Hello"
    assert starts(state) == 2 and ctx.tracker.recent(1)[0]["retries"] == 1


@pytest.mark.parametrize("errors", ["data", "legacy"])
def test_device_lost_after_output_restarts_engine_for_next_request(winrunner, state, errors):
    base, ctx = winrunner("device_lost_after_tokens,ok", errors=errors)
    lines = stream_lines(base, ctx)
    text = "\n".join(lines)
    # part of the answer already reached the client: it gets the error instead of a silently truncated answer
    assert '"Hel"' in text and "ErrorDeviceLost" in text and "[DONE]" not in text
    rec = ctx.tracker.recent(1)[0]
    assert "ErrorDeviceLost" in rec["error"] and rec["retries"] == 0
    wait_for(lambda: (i := ready_instance(ctx)) and i.restarts == 1, what="engine restart")
    lines = stream_lines(base, ctx)
    assert lines[-1] == "data: [DONE]" and not any("error" in ln for ln in lines)
    assert starts(state) == 2


def test_engine_abort_on_request_runs_request_again(winrunner, state):
    base, ctx = winrunner("abort_on_request,ok")
    r = chat(base, ctx)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "Hello"
    assert starts(state) == 2
    log = activity(ctx)
    assert "exited unexpectedly (code -6 = SIGABRT" in log and "recovered" in log


def test_engine_crash_while_idle_restarts_automatically(winrunner, state):
    base, ctx = winrunner("crash_when_idle,ok")
    load(base, ctx)
    wait_for(lambda: starts(state) == 2 and (i := ready_instance(ctx)) and i.restarts == 1,
             what="automatic restart")
    assert "exited unexpectedly" in activity(ctx)
    assert chat(base, ctx).status_code == 200


def test_logged_device_loss_while_idle_restarts_and_requests_wait(winrunner, state, monkeypatch):
    from winrunner import manager as mgr

    monkeypatch.setattr(mgr, "RECOVERY_DELAYS_S", (1.5, 0.3, 0.3))
    base, ctx = winrunner("log_fault_when_idle,ok", server={"jit_loading": False})
    load(base, ctx)
    wait_for(lambda: failed_instance(ctx), what="device loss reported in the engine log")
    st = httpx.get(base + "/wr/api/status").json()
    assert any(i["recovering"] for i in st["instances"])
    # Requests during the restart wait for it, even with just-in-time loading off and a model name that is not in
    # the library (served by the active model).
    r = httpx.post(base + "/v1/chat/completions", json={**CHAT, "model": "gpt-4o"}, timeout=60)
    assert r.status_code == 200, r.text
    assert starts(state) == 2 and ready_instance(ctx).restarts == 1


def test_prompt_text_in_engine_log_is_not_a_fault(winrunner, state):
    base, ctx = winrunner("debug_mention")
    load(base, ctx)
    inst = ready_instance(ctx)
    wait_for(lambda: any("Error Log:" in x["text"] for x in list(inst.log)), what="log lines")
    time.sleep(0.3)
    assert inst.state == "ready" and starts(state) == 1
    assert chat(base, ctx).status_code == 200


def test_repeated_failures_stop_automatic_restarts(winrunner, state):
    from winrunner.manager import RECOVERY_LIMIT

    base, ctx = winrunner("crash_when_idle")
    load(base, ctx)
    wait_for(lambda: "automatic restart is disabled" in activity(ctx), timeout=60, what="restart limit")
    wait_for(lambda: not ctx.manager.instances and not ctx.manager._pending, what="broken engine removed")
    time.sleep(0.5)
    assert starts(state) == 1 + RECOVERY_LIMIT


def test_unload_cancels_pending_restart(winrunner, state, monkeypatch):
    from winrunner import manager as mgr

    monkeypatch.setattr(mgr, "RECOVERY_DELAYS_S", (5.0,))
    base, ctx = winrunner("crash_when_idle,ok")
    load(base, ctx)
    inst = wait_for(lambda: failed_instance(ctx), what="engine crash")
    r = httpx.post(base + "/wr/api/models/unload", json={"id": inst.id})
    assert r.json() == {"unloaded": True}
    wait_for(lambda: not ctx.manager.instances and not ctx.manager._pending, what="restart cancelled")
    assert "cancelled" in activity(ctx)
    time.sleep(0.3)
    assert starts(state) == 1


def test_restart_waits_for_the_gpus_after_a_reset(winrunner, state, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_RESET_POLLS", "3")  # the driver reports no GPU for a moment
    base, ctx = winrunner("abort_on_request,ok")
    r = chat(base, ctx)
    assert r.status_code == 200, r.text
    assert "Waiting for Vulkan0 to become available again" in activity(ctx)
    assert starts(state) == 2
    inst = ready_instance(ctx)
    assert [d.name for d in inst.plan.devices] == ["Vulkan0"]  # loaded on the GPU again, not on the CPU


def test_loading_another_model_stops_a_pending_restart(winrunner, state, monkeypatch):
    from winrunner import manager as mgr

    monkeypatch.setattr(mgr, "RECOVERY_DELAYS_S", (2.0, 0.3, 0.3))
    base, ctx = winrunner("crash_when_idle,ok")
    load(base, ctx)
    wait_for(lambda: failed_instance(ctx), what="engine crash")
    load(base, ctx, "other-1b")  # the user moves on while the crashed model waits to be restarted
    wait_for(lambda: "was not restarted: " in activity(ctx), what="restart abandoned")
    time.sleep(0.3)
    assert [i.model_id for i in ctx.manager.instances.values()] == [model_id(ctx, "other-1b")]
    assert starts(state) == 2 and not ctx.manager._pending
