"""Recognising engine failures that only a new engine process can recover from.

When a GPU is reset while llama.cpp is using it (on Windows usually by the
display driver's Timeout Detection and Recovery after a GPU job ran longer than
``TdrDelay``; also driver crashes or unstable clocks), the engine's GPU context
is *lost*: Vulkan reports ``VK_ERROR_DEVICE_LOST`` ("vk::Queue::submit:
ErrorDeviceLost"). llama-server survives this and keeps answering ``/health``,
but a lost device can never be used again: every later request fails the same
way, or the process aborts on the next one (Windows exit code 0xC0000409).
"""

from __future__ import annotations

import re
import signal
from typing import Any

GPU_DEVICE_LOST = "GPU device lost"
GPU_KERNEL_FAILURE = "GPU kernel failure"

_GPU_FAULTS: list[tuple[re.Pattern[str], str]] = [
    # Vulkan: "vk::Queue::submit: ErrorDeviceLost", "vk::DeviceLostError", "VK_ERROR_DEVICE_LOST", "device lost"
    (re.compile(r"device[ _-]?lost", re.IGNORECASE), GPU_DEVICE_LOST),
    # CUDA / HIP (ROCm): the GPU context is unusable after these
    (re.compile(r"(?:cuda|hip)Error(?:IllegalAddress|IllegalInstruction|LaunchFailure|HardwareStackError)"
                r"|illegal memory access|unspecified launch failure", re.IGNORECASE), GPU_KERNEL_FAILURE),
]


def gpu_fault(text: Any) -> str | None:
    """What went wrong when ``text`` (an engine error message) reports an unrecoverable GPU failure.

    Only apply this to text produced by the engine itself (error responses,
    error-level log records), never to prompts or generated text.
    """
    if not text:
        return None
    s = str(text)
    for rx, what in _GPU_FAULTS:
        if rx.search(s):
            return what
    return None


# NTSTATUS values seen as process exit codes on Windows.
_WINDOWS_STATUS = {
    0xC0000005: "access violation",
    0xC0000017: "out of memory",
    0xC000001D: "illegal instruction (the engine build needs CPU features this processor lacks)",
    0xC00000FD: "stack overflow",
    0xC0000135: "a required DLL was not found",
    0xC0000142: "a DLL failed to initialise",
    0xC000013A: "interrupted (Ctrl+C)",
    0xC0000374: "heap corruption",
    # abort() / __fastfail: llama.cpp aborts like this on fatal errors, e.g. a GPU call failing after a device loss
    0xC0000409: "the engine aborted after a fatal error",
    0xC0000602: "the engine aborted after a fatal error",
}
_SIGNAL_TEXT = {  # (SIGKILL does not exist on Windows)
    getattr(signal, name): text
    for name, text in (("SIGABRT", "the engine aborted after a fatal error"), ("SIGSEGV", "access violation"),
                       ("SIGKILL", "killed (possibly by the out-of-memory killer)"))
    if hasattr(signal, name)
}


def describe_exit_code(rc: int | None) -> str:
    """Human readable process exit code, e.g. ``code 3221226505 = 0xC0000409, the engine aborted ...``."""
    if rc is None:
        return "unknown exit code"
    if rc < 0:  # POSIX: terminated by a signal
        try:
            sig = signal.Signals(-rc)
        except ValueError:
            return f"code {rc}"
        text = _SIGNAL_TEXT.get(sig)
        return f"code {rc} = {sig.name}" + (f", {text}" if text else "")
    status = rc & 0xFFFFFFFF
    if status >= 0xC0000000:  # NTSTATUS error value (Windows)
        text = _WINDOWS_STATUS.get(status)
        return f"code {rc} = 0x{status:08X}" + (f", {text}" if text else "")
    return f"code {rc}"


# ----- Windows GPU timeout (TDR) ------------------------------------------------------------------

TDR_MIN_DELAY_S = 10  # below this, long GPU jobs (big prompts, big models, models partly in RAM) can be reset
TDR_SCRIPT = r"scripts\gpu-timeout.bat"


def tdr_limited(tdr: dict[str, Any] | None) -> bool:
    """True when Windows resets GPUs after a short timeout (the default is 2 s)."""
    if not tdr:
        return False
    return tdr.get("level", 3) != 0 and tdr.get("delay_s", 2) < TDR_MIN_DELAY_S


def tdr_hint(tdr: dict[str, Any]) -> str:
    return (f"Windows resets a GPU when one GPU job runs longer than {tdr.get('delay_s', 2)} s (TdrDelay); llama.cpp "
            "then loses the GPU ('ErrorDeviceLost'). Long prompts on large models, or models partly in system RAM, "
            f"can exceed this limit. To raise it, run {TDR_SCRIPT} as administrator and restart Windows.")
