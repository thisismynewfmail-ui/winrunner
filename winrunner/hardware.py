"""Hardware detection and live telemetry.

Windows: DXGI (adapters), PDH (VRAM / utilisation, as in Task Manager), ADL
(AMD temperatures, clocks, power, fan). Linux: amdgpu sysfs / nvidia-smi.
CPU and memory via psutil everywhere.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import psutil

log = logging.getLogger("winrunner.hardware")

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:  # pragma: no cover - exercised on Windows only
    from .platform import win32 as _plat
else:
    from .platform import linux as _plat  # type: ignore[no-redef]


@dataclass
class GpuInfo:
    id: str  # gpu0, gpu1 ... in PCI bus order
    name: str
    vendor: str
    vram_total: int
    bus: int | None = None
    driver: str = ""
    luid: str = ""
    adl_index: int | None = None
    sysfs: dict[str, Any] = field(default_factory=dict)
    nvidia_index: int | None = None


_VENDORS = {0x1002: "AMD", 0x10DE: "NVIDIA", 0x8086: "Intel", 0x1414: "Microsoft"}


def _norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(r\)|\(tm\)|graphics|series|\s+", " ", s)
    return " ".join(s.split())


class HardwareMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = max(0.25, float(interval))
        self.gpus: list[GpuInfo] = []
        self.cpu_name = ""
        self._pdh = None
        self._adl = None
        self._watch: dict[int, str] = {}  # pid -> label
        self._proc_cache: dict[int, psutil.Process] = {}
        self._prev_io: dict[int, tuple[float, int]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._callbacks: list[Callable[[dict], None]] = []
        self.last: dict[str, Any] = {}
        self._detect()
        psutil.cpu_percent(percpu=True)  # prime

    # ----- detection -------------------------------------------------------------

    def _detect(self) -> None:
        self.cpu_name = _plat.cpu_brand() or platform.processor() or "Unknown CPU"
        gpus: list[GpuInfo] = []
        try:
            if IS_WINDOWS:
                gpus = self._detect_windows()
            else:
                gpus = self._detect_linux()
        except Exception:
            log.exception("GPU detection failed")
        gpus.sort(key=lambda g: (g.bus if g.bus is not None else 999, g.id))
        for i, g in enumerate(gpus):
            g.id = f"gpu{i}"
        self.gpus = gpus

    def _detect_windows(self) -> list[GpuInfo]:  # pragma: no cover - Windows only
        adapters = _plat.dxgi_adapters()
        drivers = _plat.gpu_driver_versions()
        self._pdh = _plat.PdhGpuCounters()
        self._adl = _plat.AdlSensors()
        out: list[GpuInfo] = []
        for a in adapters:
            if a["vendor_id"] == 0x1414:  # Microsoft Basic Render / remote adapters
                continue
            bus = None
            addr = _plat.adapter_pci_address(a["luid_low"], a["luid_high"])
            if addr:
                bus = addr[0]
            g = GpuInfo(
                id="",
                name=a["name"],
                vendor=_VENDORS.get(a["vendor_id"], f"0x{a['vendor_id']:04x}"),
                vram_total=a["vram_total"],
                bus=bus,
                driver=drivers.get(a["name"], ""),
                luid=a["luid"],
            )
            out.append(g)
        if self._adl and self._adl.ok:
            by_bus = {x["bus"]: x for x in self._adl.adapters}
            unmatched = [x for x in self._adl.adapters]
            amd = [g for g in out if g.vendor == "AMD"]
            for g in amd:
                if g.bus is not None and g.bus in by_bus:
                    g.adl_index = by_bus[g.bus]["adl_index"]
                    unmatched = [u for u in unmatched if u["bus"] != g.bus]
            # D3DKMT unavailable: fall back to enumeration order (both are PCI ordered)
            rest = [g for g in amd if g.adl_index is None]
            for g, u in zip(rest, sorted(unmatched, key=lambda u: u["bus"])):
                g.adl_index = u["adl_index"]
                if g.bus is None:
                    g.bus = u["bus"]
        return out

    def _detect_linux(self) -> list[GpuInfo]:
        out: list[GpuInfo] = []
        for c in _plat.amdgpu_cards():
            out.append(
                GpuInfo(id="", name=c["name"], vendor="AMD", vram_total=c["vram_total"], bus=c.get("bus"), sysfs=c)
            )
        for n in _plat.nvidia_sample():
            out.append(
                GpuInfo(
                    id="", name=n["name"], vendor="NVIDIA", vram_total=int(n["vram_total"]), bus=n.get("bus"),
                    nvidia_index=n["index"],
                )
            )
        return out

    def system_info(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        return {
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "machine": platform.machine(),
            "hostname": platform.node(),
            "cpu": self.cpu_name,
            "cores_physical": psutil.cpu_count(logical=False) or 0,
            "cores_logical": psutil.cpu_count(logical=True) or 0,
            "ram_total": vm.total,
            "python": platform.python_version(),
            "gpus": [
                {k: v for k, v in asdict(g).items() if k not in ("sysfs",)} for g in self.gpus
            ],
            "gpu_timeout": self.gpu_timeout(),
            "telemetry": {
                "pdh": bool(self._pdh and getattr(self._pdh, "ok", False)),
                "adl": bool(self._adl and getattr(self._adl, "ok", False)),
                "sysfs": any(g.sysfs for g in self.gpus),
            },
        }

    @staticmethod
    def gpu_timeout() -> dict[str, int] | None:
        """Windows GPU timeout (TDR) settings; None on platforms without one."""
        fn = getattr(_plat, "tdr_settings", None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception:
            log.debug("TDR settings unavailable", exc_info=True)
            return None

    # ----- engine device mapping ---------------------------------------------

    def map_engine_devices(self, devices: list[dict[str, Any]]) -> dict[str, str]:
        """Map engine device names (Vulkan0, ROCm1 ...) to gpu ids.

        Devices are matched by name, and in order among identically named
        devices (engines enumerate in PCI order, matching our bus sort).
        """
        mapping: dict[str, str] = {}
        pool = list(self.gpus)
        for d in devices:
            desc = _norm_name(d.get("description", ""))
            best = None
            for g in pool:
                gn = _norm_name(g.name)
                if gn and (gn in desc or desc in gn):
                    best = g
                    break
            if best is None and pool:
                best = pool[0]
            if best is not None:
                mapping[d["name"]] = best.id
                pool.remove(best)
        return mapping

    # ----- process watch -------------------------------------------------------

    def watch_process(self, pid: int, label: str) -> None:
        self._watch[pid] = label

    def unwatch_process(self, pid: int) -> None:
        self._watch.pop(pid, None)
        self._proc_cache.pop(pid, None)
        self._prev_io.pop(pid, None)

    # ----- sampling ---------------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        t = time.time()
        per_core = psutil.cpu_percent(percpu=True)
        try:
            freq = psutil.cpu_freq()
            freq_mhz = freq.current if freq else None
        except Exception:
            freq_mhz = None
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        data: dict[str, Any] = {
            "t": t,
            "cpu": {
                "util": round(sum(per_core) / max(1, len(per_core)), 1),
                "per_core": [round(x, 1) for x in per_core],
                "freq_mhz": round(freq_mhz) if freq_mhz else None,
            },
            "mem": {
                "total": vm.total,
                "used": vm.total - vm.available,
                "available": vm.available,
                "swap_total": sw.total,
                "swap_used": sw.used,
            },
            "gpus": [],
            "procs": {},
        }
        watch = set(self._watch)
        try:
            data["gpus"] = self._sample_gpus(watch)
        except Exception:
            log.debug("gpu sample failed", exc_info=True)
        for pid in list(watch):
            p = self._proc_cache.get(pid)
            try:
                if p is None:
                    p = psutil.Process(pid)
                    p.cpu_percent()
                    self._proc_cache[pid] = p
                with p.oneshot():
                    mi = p.memory_info()
                    rec: dict[str, Any] = {
                        "label": self._watch.get(pid, ""),
                        "cpu": round(p.cpu_percent() / max(1, psutil.cpu_count() or 1), 1),
                        "rss": mi.rss,
                        "private": getattr(mi, "private", None) or getattr(mi, "vms", 0),
                        "threads": p.num_threads(),
                    }
                    try:
                        io = p.io_counters()
                        rb = int(io.read_bytes)
                        prev = self._prev_io.get(pid)
                        rec["read_bytes"] = rb
                        rec["read_rate"] = (rb - prev[1]) / max(1e-3, t - prev[0]) if prev else 0.0
                        self._prev_io[pid] = (t, rb)
                    except (psutil.AccessDenied, AttributeError, NotImplementedError):
                        pass
                data["procs"][str(pid)] = rec
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.unwatch_process(pid)
        for g in data["gpus"]:
            pv = g.pop("_proc_vram", {})
            pu = g.pop("_proc_util", {})
            for pid, v in pv.items():
                if str(pid) in data["procs"]:
                    data["procs"][str(pid)].setdefault("vram", {})[g["id"]] = v
            for pid, v in pu.items():
                if str(pid) in data["procs"]:
                    data["procs"][str(pid)].setdefault("gpu_util", {})[g["id"]] = v
        self.last = data
        return data

    def _sample_gpus(self, watch: set[int]) -> list[dict[str, Any]]:
        out = []
        if IS_WINDOWS:  # pragma: no cover
            pdh = self._pdh.sample(watch) if self._pdh else {}
            for g in self.gpus:
                rec: dict[str, Any] = {"id": g.id, "name": g.name, "vram_total": g.vram_total}
                c = pdh.get(g.luid.lower()) if g.luid else None
                if c:
                    rec["vram_used"] = c["dedicated"]
                    rec["shared_used"] = c["shared"]
                    rec["util"] = round(c["util"], 1)
                    rec["engines"] = {k: round(v, 1) for k, v in c["engines"].items() if v > 0.05}
                    rec["_proc_vram"] = c.get("proc_dedicated", {})
                    rec["_proc_util"] = c.get("proc_util", {})
                if self._adl and g.adl_index is not None:
                    s = self._adl.read(g.adl_index)
                    for k in ("temp_edge", "temp_hotspot", "temp_mem", "clk_gfx", "clk_mem", "fan_rpm", "fan_pct",
                              "activity_mem", "gfx_voltage", "bus_lanes", "bus_speed"):
                        if k in s:
                            rec[k] = s[k]
                    if "asic_power" in s:
                        rec["power"] = s["asic_power"]
                    elif "gfx_power" in s:
                        rec["power"] = s["gfx_power"]
                    if "util" not in rec and "activity_gfx" in s:
                        rec["util"] = s["activity_gfx"]
                    if "vram_used" not in rec and "vram_used_mb" in s:
                        rec["vram_used"] = s["vram_used_mb"] * 1024 * 1024
                out.append(rec)
            return out
        nv = None
        for g in self.gpus:
            rec = {"id": g.id, "name": g.name, "vram_total": g.vram_total}
            if g.sysfs:
                rec.update(_plat.amdgpu_sample(g.sysfs))
            elif g.nvidia_index is not None:
                if nv is None:
                    nv = {n["index"]: n for n in _plat.nvidia_sample()}
                n = nv.get(g.nvidia_index)
                if n:
                    rec.update({k: v for k, v in n.items() if k not in ("index", "name", "bus") and v is not None})
            out.append(rec)
        return out

    # ----- background thread -----------------------------------------------------

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._callbacks.append(fn)

    def start(self) -> None:
        if self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if IS_WINDOWS and self._pdh:  # pragma: no cover
            self._pdh.close()

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                s = self.sample()
                for fn in list(self._callbacks):
                    try:
                        fn(s)
                    except Exception:
                        log.debug("telemetry callback failed", exc_info=True)
            except Exception:
                log.exception("telemetry sample failed")
            next_t += self.interval
            delay = next_t - time.monotonic()
            if delay < 0:
                next_t = time.monotonic()
                delay = self.interval
            self._stop.wait(delay)


def recommended_threads() -> tuple[int, int]:
    """(generation threads, batch threads) for this CPU."""
    phys = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    logical = psutil.cpu_count(logical=True) or phys
    return phys, logical
