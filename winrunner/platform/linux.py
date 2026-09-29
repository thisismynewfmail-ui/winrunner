"""Linux GPU telemetry: amdgpu sysfs and nvidia-smi."""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("winrunner.linux")


def _read(path: str | Path) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str | Path) -> int | None:
    v = _read(path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def amdgpu_cards() -> list[dict[str, Any]]:
    """Static info for amdgpu devices from /sys/class/drm."""
    out = []
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(card):
            continue  # connectors (card0-DP-1)
        dev = Path(card) / "device"
        vendor = _read(dev / "vendor")
        if vendor != "0x1002":
            continue
        total = _read_int(dev / "mem_info_vram_total")
        if not total:
            continue
        name = _read(dev / "product_name") or ""
        pci = os.path.basename(os.path.realpath(dev))
        hwmons = glob.glob(str(dev / "hwmon" / "hwmon*"))
        out.append(
            {
                "sysfs": str(dev),
                "hwmon": hwmons[0] if hwmons else "",
                "name": name or f"AMD GPU {_read(dev / 'device') or ''}".strip(),
                "vendor_id": 0x1002,
                "device_id": int(_read(dev / "device") or "0", 16),
                "vram_total": total,
                "pci": pci,
                "bus": int(pci.split(":")[1], 16) if pci.count(":") >= 2 else None,
            }
        )
    return out


def amdgpu_sample(card: dict[str, Any]) -> dict[str, float]:
    dev = Path(card["sysfs"])
    hw = Path(card["hwmon"]) if card.get("hwmon") else None
    out: dict[str, float] = {}
    used = _read_int(dev / "mem_info_vram_used")
    if used is not None:
        out["vram_used"] = float(used)
    gtt = _read_int(dev / "mem_info_gtt_used")
    if gtt is not None:
        out["shared_used"] = float(gtt)
    busy = _read_int(dev / "gpu_busy_percent")
    if busy is not None:
        out["util"] = float(busy)
    mbusy = _read_int(dev / "mem_busy_percent")
    if mbusy is not None:
        out["activity_mem"] = float(mbusy)
    if hw:
        for i in range(1, 6):
            label = _read(hw / f"temp{i}_label")
            val = _read_int(hw / f"temp{i}_input")
            if val is None:
                continue
            key = {"edge": "temp_edge", "junction": "temp_hotspot", "mem": "temp_mem"}.get((label or "").lower())
            if key:
                out[key] = val / 1000.0
        p = _read_int(hw / "power1_average") or _read_int(hw / "power1_input")
        if p is not None:
            out["power"] = p / 1e6
        cap = _read_int(hw / "power1_cap")
        if cap:
            out["power_limit"] = cap / 1e6
        fan = _read_int(hw / "fan1_input")
        if fan is not None:
            out["fan_rpm"] = float(fan)
        f1 = _read_int(hw / "freq1_input")
        if f1:
            out["clk_gfx"] = f1 / 1e6
        f2 = _read_int(hw / "freq2_input")
        if f2:
            out["clk_mem"] = f2 / 1e6
    return out


def nvidia_sample() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.gr,clocks.mem,fan.speed,pci.bus"
    try:
        r = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in r.stdout.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 11:
            continue

        def num(s: str) -> float | None:
            try:
                return float(s)
            except ValueError:
                return None

        out.append(
            {
                "index": int(cols[0]),
                "name": cols[1],
                "vram_total": (num(cols[2]) or 0) * 1024 * 1024,
                "vram_used": (num(cols[3]) or 0) * 1024 * 1024,
                "util": num(cols[4]),
                "temp_edge": num(cols[5]),
                "power": num(cols[6]),
                "clk_gfx": num(cols[7]),
                "clk_mem": num(cols[8]),
                "fan_pct": num(cols[9]),
                "bus": int(cols[10], 16) if cols[10].startswith("0x") else None,
            }
        )
    return out


def cpu_brand() -> str:
    txt = _read("/proc/cpuinfo") or ""
    for line in txt.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""
