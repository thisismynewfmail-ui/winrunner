"""A stand-in for llama-fit-params used by the memory planning tests.

It measures a layout the way llama.cpp's projection reports it (weights, KV cache of a single sequence and compute
buffers per device) from WinRunner's own estimate plus a deliberate, known error, so the tests can check that
WinRunner calibrates its plan with the engine's measurement and still fills the GPUs without overcommitting them.

Environment:

``FAKE_ENGINE_DEVICES``  GPUs: ``name:description:total MiB:free MiB;...`` (the same as the fake engine's)
``FAKE_FIT_BIAS``        ``a,b``: every GPU holding layers needs ``a`` MiB + ``b`` MiB per context token more compute
                         buffer than WinRunner estimates (default ``300,0.002``)
``FAKE_FIT_LOG``         file that receives the arguments of every call, one line each
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from winrunner.config import LoadParams  # noqa: E402
from winrunner.engine import EngineDevice  # noqa: E402
from winrunner.gguf import summarize  # noqa: E402
from winrunner.planner import Planner, assign_layers  # noqa: E402

FLAGS = {"--swa-full", "-nkvo", "--no-kv-offload"}


def devices() -> list[EngineDevice]:
    spec = os.environ.get("FAKE_ENGINE_DEVICES") or "Vulkan0:Fake GPU:16368:15000"
    out = []
    for item in spec.split(";"):
        name, desc, total, free = item.split(":")
        out.append(EngineDevice(name, desc, int(total), int(free)))
    return out


def parse(argv: list[str]) -> dict[str, str]:
    a: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i] in FLAGS:
            a[argv[i]] = "1"
            i += 1
        else:
            a[argv[i]] = argv[i + 1] if i + 1 < len(argv) else ""
            i += 2
    return a


def measure(a: dict[str, str], n_gpu: int, split: list[float] | None, n_cpu_moe: int) -> dict[str, tuple[float, ...]]:
    info = summarize(a["-m"])
    ctx = int(a["-c"])
    p = LoadParams(context_length=max(256, ctx), ubatch_size=int(a.get("-ub", 512)), batch_size=int(a.get("-b", 2048)),
                   flash_attn=a.get("-fa", "auto"), swa_full="--swa-full" in a, kv_offload="-nkvo" not in a,
                   parallel=1)  # no -np: one sequence
    devs = devices()
    bias = [float(x) for x in (os.environ.get("FAKE_FIT_BIAS") or "300,0.002").split(",")]
    planner = Planner(info, p, devs, margin_mib=0)
    kv_k, kv_v = a.get("-ctk", "f16"), a.get("-ctv", "f16")
    layout, host, _ = planner._layout(ctx, kv_k, kv_v, n_gpu, n_cpu_moe, split or [float(d.free_mib) for d in devs])
    rows: dict[str, tuple[float, ...]] = {}
    for d in layout:
        extra = bias[0] + bias[1] * ctx if (d.layers or d.output_mib) else 0.0
        rows[d.name] = (d.weights_mib + d.output_mib, d.kv_mib, d.compute_mib + extra)
    rows["Host"] = (host["weights_mib"], host["kv_mib"], host["compute_mib"])
    return rows


def main(argv: list[str]) -> int:
    log = os.environ.get("FAKE_FIT_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(" ".join(argv) + "\n")
    a = parse(argv)
    info = summarize(a["-m"])
    full = info.n_layer + 1
    ngl = a.get("-ngl", "auto")
    n_gpu = full if ngl in ("auto", "all", "-1") else min(full, int(ngl))
    split = [float(x) for x in a["-ts"].replace("/", ",").split(",")] if "-ts" in a else None
    n_cpu_moe = int(a.get("--n-cpu-moe", 0))
    if a.get("--fit-print") == "on":
        for name, vals in measure(a, n_gpu, split, n_cpu_moe).items():
            print(name, *(int(v) for v in vals))
        return 0
    # fit mode: the most layers (counted from the output layer) that stay within the targets
    devs = devices()
    targets = [float(x) for x in a.get("--fit-target", "1024").split(",")]
    targets += [targets[-1]] * (len(devs) - len(targets))
    caps = {d.name: d.free_mib - t for d, t in zip(devs, targets)}
    for n in range(full, -1, -1):
        s = [max(1.0, c) for c in caps.values()]
        rows = measure(a, n, s, 0)
        if all(sum(rows[name]) <= cap for name, cap in caps.items()):
            if n == full:
                print(f"-c {a['-c']} -ngl -1")
            else:
                owner = assign_layers(info.n_layer, n, s)
                counts = [sum(1 for x in owner if x == i) for i in range(len(devs))]
                print(f"-c {a['-c']} -ngl {n}" + (f" -ts {','.join(map(str, counts))}" if len(devs) > 1 else ""))
            return 0
    print("failed to fit", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
