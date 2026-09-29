"""Chat template analysis and local rendering.

The GGUF's embedded Jinja template (``tokenizer.chat_template``) is what the
engine uses (``--jinja``). This module identifies the prompt format family and
capabilities for display, and renders previews with a sandboxed Jinja
environment. When a model is loaded, the engine's own ``/apply-template``
endpoint is preferred for an exact rendering.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("Harmony (gpt-oss)", ("<|channel|>", "<|start|>")),
    ("Llama 4", ("<|header_start|>",)),
    ("Llama 3", ("<|start_header_id|>",)),
    ("Gemma", ("<start_of_turn>",)),
    ("DeepSeek", ("<｜User｜>",)),
    ("Mistral (Tekken)", ("[SYSTEM_PROMPT]",)),
    ("Mistral / Llama 2", ("[INST]",)),
    ("GLM", ("[gMASK]",)),
    ("Phi-4", ("<|im_sep|>",)),
    ("Phi-3", ("<|assistant|>", "<|end|>")),
    ("Command-R", ("<|START_OF_TURN_TOKEN|>",)),
    ("Granite", ("<|start_of_role|>",)),
    ("Kimi", ("<|im_middle|>",)),
    ("Seed-OSS", ("<seed:bos>",)),
    ("ChatML", ("<|im_start|>",)),
    ("Zephyr", ("<|user|>",)),
    ("Vicuna", ("USER:", "ASSISTANT:")),
]


def detect_family(template: str) -> str:
    if not template:
        return "None (engine fallback)"
    for name, markers in FAMILIES:
        if all(m in template for m in markers):
            return name
    return "Custom"


def analyze(template: str) -> dict[str, Any]:
    t = template or ""
    low = t.lower()
    return {
        "family": detect_family(t),
        "length": len(t),
        "tools": "tools" in t and ("tool_call" in low or "function" in low),
        "reasoning": any(
            m in t for m in ("<think>", "enable_thinking", "reasoning_content", "thinking", "<|channel|>analysis")
        ),
        "thinking_toggle": "enable_thinking" in t,
        "reasoning_effort": "reasoning_effort" in t,
        "system_role": "system" in t,
        "multimodal_parts": "image" in low or "content is not string" in low or "['type']" in t,
        "raises_errors": "raise_exception" in t,
    }


SAMPLE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "And of Italy?"},
]


class TemplateError(Exception):
    pass


def render(
    template: str,
    messages: list[dict] | None = None,
    bos_token: str | None = "",
    eos_token: str | None = "",
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render ``template`` with HF-style globals in a sandbox."""
    try:
        from jinja2 import TemplateError as JinjaError
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:  # pragma: no cover
        raise TemplateError("jinja2 is not installed") from exc

    def raise_exception(msg: str) -> None:
        raise TemplateError(msg)

    def tojson(x: Any, indent: int | None = None, ensure_ascii: bool = False, sort_keys: bool = False) -> str:
        return json.dumps(x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)

    def strftime_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    try:
        tmpl = env.from_string(template)
        ctx = {
            "messages": messages if messages is not None else SAMPLE_CONVERSATION,
            "bos_token": bos_token or "",
            "eos_token": eos_token or "",
            "add_generation_prompt": add_generation_prompt,
            "tools": tools,
        }
        if extra:
            ctx.update(extra)
        out = tmpl.render(**ctx)
        # Templates written for typed content ([{"type": "text", ...}]) render plain
        # strings as nothing; the engine converts content automatically, so do the same.
        msgs = ctx["messages"]
        probe = next((m.get("content") for m in msgs if isinstance(m.get("content"), str) and m.get("content")), None)
        if probe and probe not in out:
            ctx["messages"] = [
                {**m, "content": [{"type": "text", "text": m["content"]}]} if isinstance(m.get("content"), str) else m
                for m in msgs
            ]
            typed = tmpl.render(**ctx)
            if probe in typed:
                return typed
        return out
    except TemplateError:
        raise
    except JinjaError as exc:
        raise TemplateError(f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # templates can fail in many ways; report, don't crash
        raise TemplateError(f"{type(exc).__name__}: {exc}") from exc
