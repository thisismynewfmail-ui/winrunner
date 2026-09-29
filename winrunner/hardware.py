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
    display: bool | None = None  # drives a monitor (None = unknown)


_VENDORS = {0x1002: "AMD", 0x10DE: "NVIDIA", 0x8086: "Intel", 0x1414: "Microsoft"}


def _pci_bus(addr: str) -> int | None:
    """Bus number from a PCI address such as '0000:0b:00.0'."""
    parts = (addr or "").split(":")
    if len(parts) < 3:
        return None
    try:
        return int(parts[-2], 16)
    except ValueError:
        return None


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
                display=bool(a.get("outputs", 0)) if "outputs" in a else None,
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
            "telemetry": {
                "pdh": bool(self._pdh and getattr(self._pdh, "ok", False)),
                "adl": bool(self._adl and getattr(self._adl, "ok", False)),
                "sysfs": any(g.sysfs for g in self.gpus),
            },
        }

    # ----- engine device mapping ---------------------------------------------

    def map_engine_devices(self, devices: list[dict[str, Any]]) -> dict[str, str]:
        """Map engine device names (Vulkan0, ROCm1 ...) to gpu ids."""
        return self._match_devices(devices)[0]

    def _match_devices(self, devices: list[dict[str, Any]]) -> tuple[dict[str, str], set[str]]:
        """(engine device -> gpu id, names whose match is certain).

        1. PCI address (learned from the engine's load log) against the adapter's bus number.
        2. Unique model name.
        3. Identical cards: the engine device with the most free memory is paired with the
           card that has no display and the least VRAM in use. The pairing is only
           trusted when the free-memory difference is clear (a display costs >=192 MiB)
           and no WinRunner engine is loaded (its own allocations would dominate).
        """
        mapping: dict[str, str] = {}
        certain: set[str] = set()
        pool = list(self.gpus)
        for d in devices:
            bus = _pci_bus(d.get("pci", ""))
            g = next((g for g in pool if bus is not None and g.bus == bus), None)
            if g is not None:
                mapping[d["name"]] = g.id
                certain.add(d["name"])
                pool.remove(g)
        rest = [d for d in devices if d["name"] not in mapping]
        groups: dict[str, list[dict[str, Any]]] = {}
        for d in rest:
            desc = _norm_name(d.get("description", ""))
            exact = next((g.id for g in pool if _norm_name(g.name) == desc), "")
            key = exact or next((g.id for g in pool if _norm_name(g.name) and
                                 (_norm_name(g.name) in desc or desc in _norm_name(g.name))), "")
            groups.setdefault(key, []).append(d)
        used = {r.get("id"): r.get("vram_used") or 0 for r in (self.last.get("gpus") or [])}
        for key, devs in groups.items():
            first = next((g for g in pool if g.id == key), None)
            if first is None:
                continue
            name = _norm_name(first.name)
            cands = [g for g in pool if _norm_name(g.name) == name]
            if not cands:
                continue
            if len(devs) == 1 and len(cands) == 1:
                mapping[devs[0]["name"]] = cands[0].id
                certain.add(devs[0]["name"])
                pool.remove(cands[0])
                continue
            by_free = sorted(devs, key=lambda d: -(d.get("free_mib") or 0))
            by_idle = sorted(cands, key=lambda g: (bool(g.display), used.get(g.id, 0), g.bus if g.bus is not None else 999))
            frees = [d.get("free_mib") or 0 for d in by_free]
            gap = min((a - b for a, b in zip(frees, frees[1:])), default=0)
            shows = [g.display for g in cands]
            # free memory only identifies the display card while no WinRunner engine holds VRAM
            idle_engines = not getattr(self, "_watch", None)
            sure = (None not in shows and sum(1 for x in shows if x) == 1 and gap >= 192 and len(devs) == len(cands)
                    and idle_engines)
            for d, g in zip(by_free, by_idle):
                mapping[d["name"]] = g.id
                if sure:
                    certain.add(d["name"])
                pool.remove(g)
        for d in devices:  # anything left: enumeration order
            if d["name"] not in mapping and pool:
                mapping[d["name"]] = pool.pop(0).id
        return mapping, certain

    def device_roles(self, devices: list[dict[str, Any]]) -> dict[str, str]:
        """Per engine device: "display" (drives a monitor), "busy" (other applications use
        >400 MiB of it), "idle" (only WinRunner uses it) or "unknown"."""
        mapping, certain = self._match_devices(devices)
        by_id = {g.id: g for g in self.gpus}
        used = {r.get("id"): r.get("vram_used") for r in (self.last.get("gpus") or [])}
        own: dict[str, float] = {}
        for rec in (self.last.get("procs") or {}).values():
            for gid, v in (rec.get("vram") or {}).items():
                own[gid] = own.get(gid, 0.0) + float(v or 0)
        roles: dict[str, str] = {}
        for d in devices:
            g = by_id.get(mapping.get(d["name"], ""))
            if g is None:
                roles[d["name"]] = "unknown"
                continue
            # an uncertain pairing among identical cards still has a known role if none of them drives a display
            group = [g] if d["name"] in certain else [x for x in self.gpus if _norm_name(x.name) == _norm_name(g.name)]
            if any(x.display is None for x in group) or (len(group) > 1 and any(x.display for x in group)):
                roles[d["name"]] = "unknown"
            elif g.display:
                roles[d["name"]] = "display"
            elif any(used.get(x.id) is None for x in group):
                roles[d["name"]] = "unknown"
            else:
                other = max(float(used[x.id] or 0) - own.get(x.id, 0.0) for x in group)
                roles[d["name"]] = "idle" if other < 400 * 1024 * 1024 else "busy"
        return roles

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
            ps = g.pop("_proc_shared", {})
            pu = g.pop("_proc_util", {})
            for pid, v in pv.items():
                if str(pid) in data["procs"]:
                    data["procs"][str(pid)].setdefault("vram", {})[g["id"]] = v
            for pid, v in ps.items():
                if str(pid) in data["procs"]:
                    data["procs"][str(pid)].setdefault("shared", {})[g["id"]] = v
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
                    rec["_proc_shared"] = c.get("proc_shared", {})
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
