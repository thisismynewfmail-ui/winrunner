"""Structured parsing of llama-server log output.

The engine log is the most detailed source of truth about what happened during
a load (devices found, layers offloaded, buffer sizes per device, KV cache
size, flash attention state, vision projector) and during requests (slot
activity, prompt/generation timings). This module turns raw lines into typed
events. It accepts both plain text (``0.00.123.456 I message``) and JSONL
(``--log-jsonl``) output, and handles multi-line messages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .util import strip_ansi

_TEXT_PREFIX = re.compile(r"^(\d+\.\d+\.\d+\.\d+)\s+([IWEDTN])\s(.*)$")
_LEVELS = {"I": "info", "W": "warn", "E": "error", "D": "debug", "T": "trace", "N": "info"}
_JSON_LEVELS = {"info": "info", "warn": "warn", "warning": "warn", "error": "error", "debug": "debug",
                "trace": "trace", "none": "info", "cont": "info"}

FATAL_HINTS = (
    "ErrorOutOfDeviceMemory",
    "out of memory",
    "failed to allocate",
    "failed to load model",
    "unable to load model",
    "error loading model",
    "hipErrorOutOfMemory",
    "vk::DeviceLostError",
    "exiting due to model loading error",
)


@dataclass
class LogLine:
    level: str
    text: str
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


F = r"([\d.]+)"
Handler = Callable[[re.Match], tuple[str, dict[str, Any]]]


def _f(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return 0.0


RULES: list[tuple[re.Pattern, Handler]] = [
    (re.compile(r"loading model '(.*?)'"), lambda m: ("phase", {"phase": "open", "path": m.group(1)})),
    (re.compile(r"estimated worst-case memory usage of mmproj is " + F + " MiB"),
     lambda m: ("mmproj", {"est_mib": _f(m.group(1))})),
    (re.compile(r"fitting params to device memory"), lambda m: ("phase", {"phase": "fit"})),
    (re.compile(r"(?:common_params_fit_impl|llama_params_fit_impl|common_fit_params|llama_params_fit):\s*(.*)"),
     lambda m: ("fit", {"message": m.group(1).strip()})),
    (re.compile(r"memory_breakdown_print:\s*\|\s+-\s+(\S+)\s+\((.*?)\)\s+\|\s+(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*"
                r"(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)\s*\+\s*(-?\d+)"),
     lambda m: ("breakdown", {"device": m.group(1), "description": m.group(2), "total": int(m.group(3)),
                              "free": int(m.group(4)), "self": int(m.group(5)), "model": int(m.group(6)),
                              "context": int(m.group(7)), "compute": int(m.group(8)),
                              "unaccounted": int(m.group(9))})),
    (re.compile(r"memory_breakdown_print:\s*\|\s+-\s+Host\s+\|\s+(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)"),
     lambda m: ("breakdown", {"device": "Host", "self": int(m.group(1)), "model": int(m.group(2)),
                              "context": int(m.group(3)), "compute": int(m.group(4))})),
    (re.compile(r"llama_model_loader: loaded meta data with (\d+) key-value pairs and (\d+) tensors"),
     lambda m: ("phase", {"phase": "metadata", "kv": int(m.group(1)), "tensors": int(m.group(2))})),
    (re.compile(r"load_tensors: loading model tensors.*?(?:\(load_mode = (\S+?)\))?$"),
     lambda m: ("phase", {"phase": "tensors", "load_mode": m.group(1) or ""})),
    (re.compile(r"load_tensors: offloaded (\d+)/(\d+) layers to GPU"),
     lambda m: ("offload", {"gpu_layers": int(m.group(1)), "total_layers": int(m.group(2))})),
    (re.compile(r"load_tensors:\s+(\S+) model buffer size =\s+" + F + " MiB"),
     lambda m: ("buffer", {"kind": "model", "device": m.group(1), "mib": _f(m.group(2))})),
    (re.compile(r"(?:llama_kv_cache\w*|llama_memory\w*):\s+(\S+) KV buffer size =\s+" + F + " MiB"),
     lambda m: ("buffer", {"kind": "kv", "device": m.group(1), "mib": _f(m.group(2))})),
    (re.compile(r"(?:llama_memory_recurrent\w*|llama_memory\w*):\s+(\S+) RS buffer size =\s+" + F + " MiB"),
     lambda m: ("buffer", {"kind": "rs", "device": m.group(1), "mib": _f(m.group(2))})),
    (re.compile(r"llama_kv_cache\w*: size =\s+" + F + r" MiB \(\s*(\d+) cells,\s+(\d+) layers,\s*(\d+)/(\d+) seqs\), "
                r"K \((\w+)\):\s+" + F + r" MiB, V \((\w+)\):\s+" + F + " MiB"),
     lambda m: ("kv", {"mib": _f(m.group(1)), "cells": int(m.group(2)), "layers": int(m.group(3)),
                       "seqs": int(m.group(5)), "k_type": m.group(6), "k_mib": _f(m.group(7)),
                       "v_type": m.group(8), "v_mib": _f(m.group(9))})),
    (re.compile(r"(?:sched_reserve|llama_context|graph_reserve):\s+(\S+) compute buffer size =\s+" + F + " MiB"),
     lambda m: ("buffer", {"kind": "compute", "device": m.group(1), "mib": _f(m.group(2))})),
    (re.compile(r"llama_context:\s+(\S+)\s+output buffer size =\s+" + F + " MiB"),
     lambda m: ("buffer", {"kind": "output", "device": m.group(1), "mib": _f(m.group(2))})),
    (re.compile(r"llama_context: (n_ctx|n_ctx_seq|n_batch|n_ubatch|n_seq_max)\s+=\s+(\d+)"),
     lambda m: ("ctx", {m.group(1): int(m.group(2))})),
    (re.compile(r"llama_context: flash_attn\s+=\s+(\S+)"), lambda m: ("ctx", {"flash_attn_param": m.group(1)})),
    (re.compile(r"llama_context: kv_unified\s+=\s+(\S+)"), lambda m: ("ctx", {"kv_unified": m.group(1) == "true"})),
    (re.compile(r"llama_context: freq_base\s+=\s+" + F), lambda m: ("ctx", {"freq_base": _f(m.group(1))})),
    (re.compile(r"pipeline parallelism enabled"), lambda m: ("ctx", {"pipeline_parallel": True})),
    (re.compile(r"Flash Attention (enabled|disabled)", re.IGNORECASE),
     lambda m: ("flash_attn", {"enabled": m.group(1).lower() == "enabled"})),
    (re.compile(r"flash_attn = auto -> (enabled|disabled)"),
     lambda m: ("flash_attn", {"enabled": m.group(1) == "enabled"})),
    (re.compile(r"n_parallel is set to auto, using n_parallel = (\d+) and kv_unified = (\w+)"),
     lambda m: ("slots", {"n_parallel": int(m.group(1)), "kv_unified": m.group(2) == "true"})),
    (re.compile(r"initializing, n_slots = (\d+), n_ctx_slot = (\d+)"),
     lambda m: ("slots", {"n_slots": int(m.group(1)), "n_ctx_slot": int(m.group(2))})),
    (re.compile(r"warming up the model"), lambda m: ("phase", {"phase": "warmup"})),
    (re.compile(r"clip_model_loader: has (vision|audio) encoder"),
     lambda m: ("mmproj", {"encoder": m.group(1)})),
    (re.compile(r"load_hparams: projector:\s+(\S+)"), lambda m: ("mmproj", {"projector": m.group(1)})),
    (re.compile(r"clip_ctx: CLIP using (\S+) backend"), lambda m: ("mmproj", {"backend": m.group(1)})),
    (re.compile(r"loaded multimodal model, '(.*?)'"), lambda m: ("mmproj", {"loaded": m.group(1)})),
    (re.compile(r"chat template, thinking = (\d)"), lambda m: ("template", {"thinking": m.group(1) == "1"})),
    (re.compile(r"(?:llama_server|main): model loaded"), lambda m: ("phase", {"phase": "loaded"})),
    (re.compile(r"(?:server is listening on|listening on) (http\S+)"),
     lambda m: ("phase", {"phase": "ready", "url": m.group(1)})),
    (re.compile(r"ggml_vulkan: Found (\d+) Vulkan devices"), lambda m: ("backend", {"vulkan_devices": int(m.group(1))})),
    (re.compile(r"ggml_vulkan: (\d+) = (.*)"), lambda m: ("device_info", {"index": int(m.group(1)), "info": m.group(2)})),
    (re.compile(r"ggml_cuda_init: found (\d+) (ROCm|CUDA) devices"),
     lambda m: ("backend", {"devices": int(m.group(1)), "api": m.group(2)})),
    (re.compile(r"^\s*Device (\d+): (.*)"), lambda m: ("device_info", {"index": int(m.group(1)), "info": m.group(2)})),
    (re.compile(r"load_backend: loaded (\S+) backend from (.*)"),
     lambda m: ("backend", {"loaded": m.group(1), "path": m.group(2)})),
    (re.compile(r"image decoded \(batch (\d+)/(\d+)\) in (\d+) ms"),
     lambda m: ("image", {"batch": int(m.group(1)), "batches": int(m.group(2)), "ms": int(m.group(3))})),
    (re.compile(r"(?:audio|image) slice encoded in (\d+) ms"), lambda m: ("image", {"ms": int(m.group(1))})),
]

_SLOT = re.compile(r"slot\s+\S+:\s+id\s+(\d+)\s+\|\s+task\s+(-?\d+)\s+\|\s+(.*)$")
_SLOT_RULES: list[tuple[re.Pattern, Callable[[re.Match], tuple[str, dict[str, Any]]]]] = [
    (re.compile(r"processing task"), lambda m: ("task_start", {})),
    (re.compile(r"new prompt, n_ctx_slot = (\d+), n_keep = (\d+), task\.n_tokens = (\d+)"),
     lambda m: ("task_prompt", {"n_ctx_slot": int(m.group(1)), "n_keep": int(m.group(2)),
                                "n_prompt": int(m.group(3))})),
    (re.compile(r"prompt processing progress, n_tokens = (\d+), batch\.n_tokens = (\d+), progress = " + F),
     lambda m: ("task_progress", {"n_tokens": int(m.group(1)), "progress": _f(m.group(3))})),
    (re.compile(r"prompt eval time =\s+" + F + r" ms /\s+(\d+) tokens \(\s*" + F + r" ms per token,\s+" + F +
                " tokens per second\\)"),
     lambda m: ("task_prompt_timing", {"ms": _f(m.group(1)), "n": int(m.group(2)), "tps": _f(m.group(4))})),
    (re.compile(r"^\s*eval time =\s+" + F + r" ms /\s+(\d+) tokens \(\s*" + F + r" ms per token,\s+" + F +
                " tokens per second\\)"),
     lambda m: ("task_eval_timing", {"ms": _f(m.group(1)), "n": int(m.group(2)), "tps": _f(m.group(4))})),
    (re.compile(r"draft acceptance rate = " + F + r" \(\s*(\d+) accepted /\s*(\d+) generated\)"),
     lambda m: ("task_draft", {"rate": _f(m.group(1)), "accepted": int(m.group(2)), "generated": int(m.group(3))})),
    (re.compile(r"stop processing: n_tokens = (\d+), truncated = (\d)"),
     lambda m: ("task_end", {"n_tokens": int(m.group(1)), "truncated": m.group(2) == "1"})),
    (re.compile(r"cached n_tokens = (\d+)"), lambda m: ("task_cache", {"cached": int(m.group(1))})),
]


class LogParser:
    """Stateful line parser (handles multi-line template dumps in text mode)."""

    def __init__(self, jsonl: bool = False):
        self.jsonl = jsonl
        self._multiline: list[str] | None = None
        self._multiline_level = "info"

    def feed(self, raw: str) -> list[LogLine]:
        raw = strip_ansi(raw.rstrip("\r\n"))
        if not raw:
            return []
        if self.jsonl and raw.startswith("{"):
            try:
                obj = json.loads(raw)
            except ValueError:
                obj = None
            if isinstance(obj, dict) and "msg" in obj:
                level = _JSON_LEVELS.get(str(obj.get("level", "info")).lower(), "info")
                msg = str(obj.get("msg", "")).rstrip("\n")
                return [self._parse_message(level, msg)]
        m = _TEXT_PREFIX.match(raw)
        if m:
            out = []
            if self._multiline is not None:
                out.append(self._finish_multiline())
            level = _LEVELS.get(m.group(2), "info")
            msg = m.group(3)
            if "example_format: '" in msg and not msg.rstrip().endswith("'"):
                self._multiline = [msg]
                self._multiline_level = level
                return out
            out.append(self._parse_message(level, msg))
            return out
        if self._multiline is not None:
            self._multiline.append(raw)
            if raw.rstrip().endswith("'"):
                return [self._finish_multiline()]
            return []
        level = "error" if raw.lower().startswith(("error", "terminate called")) else "info"
        return [self._parse_message(level, raw)]

    def flush(self) -> list[LogLine]:
        return [self._finish_multiline()] if self._multiline is not None else []

    def _finish_multiline(self) -> LogLine:
        text = "\n".join(self._multiline or [])
        self._multiline = None
        return self._parse_message(self._multiline_level, text)

    def _parse_message(self, level: str, msg: str) -> LogLine:
        ll = LogLine(level=level, text=msg)
        if "example_format: '" in msg:
            ex = msg.split("example_format: '", 1)[1]
            if ex.endswith("'"):
                ex = ex[:-1]
            ll.events.append(("template", {"example": ex}))
            return ll
        sm = _SLOT.search(msg)
        if sm:
            slot, task, rest = int(sm.group(1)), int(sm.group(2)), sm.group(3)
            for rx, fn in _SLOT_RULES:
                mm = rx.search(rest)
                if mm:
                    kind, data = fn(mm)
                    data.update({"slot": slot, "task": task})
                    ll.events.append((kind, data))
                    break
            return ll
        first = msg.split("\n", 1)[0]
        for rx, fn in RULES:
            mm = rx.search(first)
            if mm:
                ll.events.append(fn(mm))
                break
        if level == "error" or any(h.lower() in first.lower() for h in FATAL_HINTS):
            ll.events.append(("error", {"message": first.strip()}))
        return ll
