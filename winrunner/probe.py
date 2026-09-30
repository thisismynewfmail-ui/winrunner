"""Exact memory projection of a launch configuration, measured by llama-server itself.

With ``--fit on`` llama-server first projects the memory that the given placement
will use on every device (a dry run of the real allocation: model, KV cache and
compute buffers, with the server's own batch and output limits), prints it, and
only then loads the weights. A *probe* starts the server with the exact launch
arguments, reads that projection - plus the vision projector's worst-case memory
and the free memory the engine sees - and stops the process before any weights
are read. ``llama-fit-params --fit-print`` is the fallback for builds whose server
prints no projection; it reserves logits for a whole micro-batch, so it
overstates the output device by up to ``ubatch x vocabulary x 4`` bytes.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .engine import engine_env
from .logparse import LogParser
from .paths import IS_WINDOWS

log = logging.getLogger("winrunner.probe")

_CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0
_FIT_PREFIX = re.compile(r"(?:common_params_fit_impl|llama_params_fit_impl):\s*(.*)$")
_DEV_LINE = re.compile(r"^-\s+(\S+)\s+\((.*)\):\s+(\d+)\s+total,\s+(\d+)\s+used,\s+(-?\d+)\s+free\s+vs\.\s+target\s+of\s+(-?\d+)")
_TOTAL_LINE = re.compile(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory")
_MMPROJ = re.compile(r"estimated worst-case memory usage of mmproj is ([\d.]+) MiB")
_STOP = ("load_tensors:", "llama_model_load:", "model loaded", "server is listening")


@dataclass
class Projection:
    # name -> used / free (before the load) / target MiB, and model / context / compute MiB when printed
    devices: dict[str, dict[str, float]] = field(default_factory=dict)
    total_used: float = 0.0
    total_free: float = 0.0
    mmproj_mib: float | None = None
    lines: list[str] = field(default_factory=list)


def parse_projection(messages: list[str], single_device: str | None = None,
                     breakdowns: dict[str, dict[str, Any]] | None = None) -> Projection | None:
    """Projection from llama-server / llama.cpp fit log messages (``None`` if it is not there).

    ``breakdowns``: memory breakdown table rows of the same dry run (logparse "breakdown" events).
    """
    pr = Projection()
    have_total = False
    for msg in messages:
        m = _MMPROJ.search(msg)
        if m:
            pr.mmproj_mib = float(m.group(1))
        fm = _FIT_PREFIX.search(msg)
        if not fm:
            continue
        body = fm.group(1).strip()
        pr.lines.append(body)
        m = _DEV_LINE.match(body)
        if m and not have_total:
            used, after = float(m.group(4)), float(m.group(5))
            pr.devices[m.group(1)] = {"used": used, "free": after + used, "target": float(m.group(6))}
            continue
        m = _TOTAL_LINE.search(body)
        if m and not have_total:
            pr.total_used, pr.total_free = float(m.group(1)), float(m.group(2))
            have_total = True
    if not have_total:
        return None
    if not pr.devices and single_device:
        pr.devices[single_device] = {"used": pr.total_used, "free": pr.total_free, "target": 0.0}
    for name, bd in (breakdowns or {}).items():
        if name in pr.devices and "model" in bd:
            pr.devices[name].update(model=float(bd["model"]), context=float(bd["context"]),
                                    compute=float(bd["compute"]))
    return pr


def server_probe(args: list[str], single_device: str | None = None, timeout: float = 180.0) -> Projection:
    """Start llama-server with ``args`` (which must include ``--fit on``), read the projection, stop it."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=os.path.dirname(args[0]) or None, env=engine_env(),
                            creationflags=_CREATE_NO_WINDOW)
    parser = LogParser(jsonl="--log-jsonl" in args)
    messages: list[str] = []
    breakdowns: dict[str, dict[str, Any]] = {}
    done = threading.Event()

    def reader() -> None:
        assert proc.stdout is not None
        try:
            for raw in iter(proc.stdout.readline, b""):
                for ll in parser.feed(raw.decode("utf-8", errors="replace")):
                    messages.append(ll.text)
                    for kind, d in ll.events:
                        if kind == "breakdown" and d.get("device") and d["device"] not in breakdowns:
                            breakdowns[d["device"]] = d  # the first table is the projection of the given placement
                    if _TOTAL_LINE.search(ll.text) or any(s in ll.text for s in _STOP):
                        done.set()
                        return
        except (OSError, ValueError):
            pass
        done.set()

    t = threading.Thread(target=reader, name="engine-probe", daemon=True)
    t.start()
    t0 = time.monotonic()
    try:
        done.wait(timeout)
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            pass
        t.join(timeout=2)
    pr = parse_projection(messages, single_device, breakdowns)
    if pr is None:
        errs = [m for m in messages if "error" in m.lower() or "failed" in m.lower()][-3:]
        tail = "; ".join(errs) or (messages[-1] if messages else f"no output (exit code {proc.returncode})")
        if time.monotonic() - t0 >= timeout:
            tail = f"timed out after {timeout:.0f} s"
        raise RuntimeError(f"llama-server printed no memory projection: {tail[:400]}")
    log.debug("probe: %s", pr.devices)
    return pr


def as_measurement(pr: Projection, plan: Any) -> dict[str, dict[str, float]]:
    """Per-device dict in the planner's measurement format (model / context / compute MiB).

    The server reports totals per device; weights and KV cache come from the plan (exact tensor sizes),
    the remainder is the compute buffer.
    """
    out: dict[str, dict[str, float]] = {}
    for d in plan.devices:
        dev = pr.devices.get(d.name)
        if dev is None:
            continue
        if "model" in dev and abs(dev["model"] + dev["context"] + dev["compute"] - dev["used"]) <= 2:
            out[d.name] = {"model": dev["model"], "context": dev["context"], "compute": dev["compute"]}
            continue
        model = d.weights_mib + d.output_mib
        ctx = d.kv_mib
        rest = dev["used"] - model - ctx
        if rest < 0:  # the estimate of the weights was a little high: the total is what counts
            shrink = min(model, -rest)
            model -= shrink
            rest += shrink
            if rest < 0:
                ctx = max(0.0, ctx + rest)
                rest = 0.0
        out[d.name] = {"model": model, "context": ctx, "compute": rest}
    return out
