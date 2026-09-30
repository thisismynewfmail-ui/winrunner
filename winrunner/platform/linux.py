"""Linux GPU telemetry: amdgpu sysfs, DRM fdinfo (per-process VRAM) and nvidia-smi."""

from __future__ import annotations

import glob
import grp
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("winrunner.linux")

# Roots are module level so tests can point them at a fake tree.
SYS_DRM = "/sys/class/drm"
PROC = "/proc"
DEV_DRI = "/dev/dri"
PCI_IDS = ("/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids", "/usr/share/pci.ids")

_PCIE_GEN = {"2.5": "1.0", "5.0": "2.0", "8.0": "3.0", "16.0": "4.0", "32.0": "5.0", "64.0": "6.0"}


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


def _pci_name(vendor: int, device: int, sub_vendor: int | None, sub_device: int | None) -> str:
    """Marketing name from the pci.ids database (subsystem entry first), '' when unknown."""
    path = next((p for p in PCI_IDS if os.path.isfile(p)), None)
    if not path:
        return ""
    ven, dev = f"{vendor:04x}", f"{device:04x}"
    sub = f"{sub_vendor:04x} {sub_device:04x}" if sub_vendor is not None and sub_device is not None else None
    in_vendor = in_device = False
    dev_name = ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                if not line.startswith("\t"):
                    if in_vendor:
                        break
                    in_vendor = line[:4].lower() == ven
                    continue
                if not in_vendor:
                    continue
                if line.startswith("\t\t"):
                    if in_device and sub and line[2:11].lower() == sub:
                        return line[11:].strip()
                    continue
                if in_device:
                    break
                if line[1:5].lower() == dev:
                    in_device = True
                    dev_name = line[5:].strip()
    except OSError:
        return ""
    return dev_name


def _pcie_link(dev: Path) -> str:
    speed = _read(dev / "current_link_speed") or ""
    width = _read(dev / "current_link_width") or ""
    max_w = _read(dev / "max_link_width") or ""
    m = re.match(r"([\d.]+)\s*GT/s", speed)
    if not m or not width:
        return ""
    gen = _PCIE_GEN.get(m.group(1), m.group(1) + " GT/s")
    txt = f"PCIe {gen} x{width}"
    if max_w and max_w != width:
        txt += f" (card supports x{max_w})"
    return txt


def amdgpu_cards() -> list[dict[str, Any]]:
    """Static info for amdgpu devices from /sys/class/drm."""
    out = []
    for card in sorted(glob.glob(os.path.join(SYS_DRM, "card[0-9]*"))):
        if "-" in os.path.basename(card):
            continue  # connectors (card0-DP-1)
        dev = Path(card) / "device"
        vendor = _read(dev / "vendor")
        if vendor != "0x1002":
            continue
        total = _read_int(dev / "mem_info_vram_total")
        if not total:
            continue
        pci = os.path.basename(os.path.realpath(dev))
        dev_id = int(_read(dev / "device") or "0", 16)
        sub_v, sub_d = _read(dev / "subsystem_vendor"), _read(dev / "subsystem_device")
        name = _read(dev / "product_name") or _pci_name(
            0x1002, dev_id, int(sub_v, 16) if sub_v else None, int(sub_d, 16) if sub_d else None)
        hwmons = sorted(glob.glob(str(dev / "hwmon" / "hwmon*")))
        vis = _read_int(dev / "mem_info_vis_vram_total")
        out.append(
            {
                "sysfs": str(dev),
                "hwmon": hwmons[0] if hwmons else "",
                "name": name or f"AMD GPU 0x{dev_id:04x}",
                "vendor_id": 0x1002,
                "device_id": dev_id,
                "vram_total": total,
                "pci": pci,
                "bus": int(pci.split(":")[1], 16) if pci.count(":") >= 2 else None,
                "boot_vga": _read(dev / "boot_vga") == "1",
                # Resizable BAR: the CPU can map all of VRAM (otherwise only a 256 MiB window)
                "rebar": (vis >= total - (64 << 20)) if vis else None,
                "pcie": _pcie_link(dev),
                "driver": _read("/sys/module/amdgpu/version") or f"amdgpu (Linux {os.uname().release})",
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


_UNITS = {"": 1, "B": 1, "KiB": 1024, "kB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}


def _fdinfo_bytes(v: str) -> int | None:
    m = re.match(r"\s*(\d+)\s*(\w*)", v)
    if not m:
        return None
    return int(m.group(1)) * _UNITS.get(m.group(2), 1)


def proc_vram(pids: set[int]) -> dict[str, dict[int, int]]:
    """VRAM used by each process on each GPU (PCI address -> pid -> bytes), from DRM fdinfo.

    amdgpu reports per DRM client: ``drm-pdev``, ``drm-client-id`` and ``drm-memory-vram`` (older kernels) or
    ``drm-resident-vram`` / ``drm-total-vram``. File descriptors of the same client are counted once.
    """
    out: dict[str, dict[int, int]] = {}
    for pid in pids:
        seen: set[tuple[str, str]] = set()
        base = os.path.join(PROC, str(pid), "fdinfo")
        try:
            fds = os.listdir(base)
        except OSError:
            continue
        for fd in fds:
            txt = _read(os.path.join(base, fd))
            if not txt or "drm-pdev" not in txt:
                continue
            kv: dict[str, str] = {}
            for line in txt.splitlines():
                k, _, v = line.partition(":")
                kv[k.strip()] = v.strip()
            pdev, client = kv.get("drm-pdev", ""), kv.get("drm-client-id", fd)
            if not pdev or (pdev, client) in seen:
                continue
            seen.add((pdev, client))
            raw = kv.get("drm-resident-vram") or kv.get("drm-memory-vram") or kv.get("drm-total-vram")
            b = _fdinfo_bytes(raw) if raw else None
            if b:
                out.setdefault(pdev, {})
                out[pdev][pid] = out[pdev].get(pid, 0) + b
    return out


def match_engine_devices(devices: list[dict[str, Any]], cards: list[dict[str, Any]],
                         used_now: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Map engine devices (Vulkan0, ...) to amdgpu cards.

    Identical GPUs cannot be told apart by name. The memory in use can: the engine reports each device's
    free memory, sysfs each card's used VRAM, and the GPU driving the desktop uses hundreds of MiB more.
    Without a clear difference the order is Mesa's device order: the boot display GPU first, then PCI order.
    """
    pool = sorted(cards, key=lambda c: (not c.get("boot_vga"), c.get("bus") if c.get("bus") is not None else 999))
    mapping: dict[str, dict[str, Any]] = {}
    for d in devices:
        total = float(d.get("total_mib") or 0) * 1024 * 1024
        cand = [c for c in pool if not total or abs(c["vram_total"] - total) < 512 * 1024 * 1024] or list(pool)
        if not cand:
            break
        best = cand[0]
        if len(cand) > 1 and d.get("free_mib") is not None and d.get("total_mib"):
            eng_used = (float(d["total_mib"]) - float(d["free_mib"])) * 1024 * 1024
            scored = sorted(cand, key=lambda c: abs(used_now.get(c["pci"], 0.0) - eng_used))
            gap = abs(abs(used_now.get(scored[1]["pci"], 0.0) - eng_used) - abs(used_now.get(scored[0]["pci"], 0.0) - eng_used))
            if gap > 96 * 1024 * 1024:
                best = scored[0]
        mapping[d["name"]] = best
        pool.remove(best)
    return mapping


def mesa_version() -> str:
    """Version of the Mesa Vulkan driver package (RADV), '' when unknown."""
    exe = shutil.which("dpkg-query")
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, "-W", "-f=${Version}", "mesa-vulkan-drivers"], capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def render_access() -> dict[str, Any]:
    """Whether this user may use the GPUs for compute (/dev/dri/renderD*)."""
    nodes = sorted(glob.glob(os.path.join(DEV_DRI, "renderD*")))
    ok = [n for n in nodes if os.access(n, os.R_OK | os.W_OK)]
    try:
        groups = sorted({grp.getgrgid(g).gr_name for g in os.getgroups()})
    except (KeyError, OSError):
        groups = []
    return {"nodes": nodes, "accessible": ok, "groups": groups}


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
    txt = _read(os.path.join(PROC, "cpuinfo")) or ""
    for line in txt.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""
