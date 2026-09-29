"""Developer preview: run WinRunner with two simulated Radeon RX 6800 GPUs.

For UI development on machines without the target GPUs. It patches device
discovery and telemetry *in this process only* (the product itself never
reports simulated hardware). The real llama.cpp engine still runs on the CPU.

    python tests/preview_server.py --data-dir ./data-preview
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from winrunner import engine as engine_mod  # noqa: E402
from winrunner import hardware as hw_mod  # noqa: E402

GIB = 1024 ** 3
FAKE_DEVICES = [
    engine_mod.EngineDevice("Vulkan0", "AMD Radeon RX 6800", 16368, 15180,
                            "AMD Radeon RX 6800 (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: none"),
    engine_mod.EngineDevice("Vulkan1", "AMD Radeon RX 6800", 16368, 16044,
                            "AMD Radeon RX 6800 (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: none"),
]


def fake_list_devices(self, server, max_age=0.0):
    return list(FAKE_DEVICES), ""


def fake_detect(self):
    self.cpu_name = "AMD Ryzen 5 3600 6-Core Processor"
    self.gpus = [
        hw_mod.GpuInfo(id="gpu0", name="AMD Radeon RX 6800", vendor="AMD", vram_total=16368 * 1024 * 1024, bus=3,
                       driver="24.9.1 (32.0.12011.1036)"),
        hw_mod.GpuInfo(id="gpu1", name="AMD Radeon RX 6800", vendor="AMD", vram_total=16368 * 1024 * 1024, bus=8,
                       driver="24.9.1 (32.0.12011.1036)"),
    ]


_t0 = time.time()


def fake_sample_gpus(self, watch):
    t = time.time() - _t0
    out = []
    for i, g in enumerate(self.gpus):
        busy = 0.5 + 0.5 * math.sin(t / 7 + i)
        util = max(0.0, min(100.0, 18 + 75 * busy + random.uniform(-4, 4)))
        rec = {"id": g.id, "name": g.name, "vram_total": g.vram_total,
               "vram_used": (1.2 + i * 0.1 + 11.8) * GIB, "shared_used": 0.3 * GIB, "util": round(util, 1),
               "temp_edge": 52 + 20 * busy, "temp_hotspot": 60 + 28 * busy, "temp_mem": 58 + 16 * busy,
               "power": 40 + 160 * busy, "clk_gfx": 500 + 1800 * busy, "clk_mem": 1990.0, "fan_rpm": 900 + 900 * busy,
               "_proc_vram": {pid: 11.8 * GIB for pid in watch}, "_proc_util": {pid: util * 0.95 for pid in watch}}
        out.append(rec)
    return out


def main() -> int:
    engine_mod.EngineManager.list_devices = fake_list_devices
    hw_mod.HardwareMonitor._detect = fake_detect
    hw_mod.HardwareMonitor._sample_gpus = fake_sample_gpus
    from winrunner.__main__ import main as wr_main

    return wr_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
