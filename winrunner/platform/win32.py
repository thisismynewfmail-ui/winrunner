"""Windows hardware access through ctypes (no extra dependencies).

* DXGI            - adapter names, dedicated VRAM size, adapter LUIDs
* D3DKMT (gdi32)  - PCI bus number for each LUID (to correlate with ADL)
* PDH             - the same GPU performance counters Task Manager uses:
                    per-adapter dedicated/shared memory usage, per-engine
                    utilisation, per-process VRAM usage
* ADL (atiadlxx)  - AMD sensor data: temperatures (edge / junction / memory),
                    clocks, board power, fan speed, activity
* Job objects     - child engine processes are terminated automatically if
                    WinRunner exits unexpectedly
* Registry        - CPU brand string

Every entry point is defensive: failures return ``None``/empty results and are
logged once, never raised into the caller.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from ctypes import wintypes
from typing import Any

log = logging.getLogger("winrunner.win32")

if sys.platform != "win32":  # pragma: no cover - imported only on Windows
    raise ImportError("win32 module is Windows-only")

HRESULT = ctypes.c_long
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


# ---------------------------------------------------------------------------
# Common structures
# ---------------------------------------------------------------------------


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    def key(self) -> str:
        """Format used in PDH instance names: luid_0xHHHHHHHH_0xLLLLLLLL."""
        return f"luid_0x{self.HighPart & 0xFFFFFFFF:08x}_0x{self.LowPart & 0xFFFFFFFF:08x}"


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, s: str) -> "GUID":
        s = s.strip("{}")
        p = s.split("-")
        g = cls()
        g.Data1 = int(p[0], 16)
        g.Data2 = int(p[1], 16)
        g.Data3 = int(p[2], 16)
        rest = bytes.fromhex(p[3] + p[4])
        for i in range(8):
            g.Data4[i] = rest[i]
        return g


# ---------------------------------------------------------------------------
# DXGI adapter enumeration
# ---------------------------------------------------------------------------


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", ctypes.c_uint),
    ]


DXGI_ADAPTER_FLAG_SOFTWARE = 2
DXGI_ERROR_NOT_FOUND = ctypes.c_long(0x887A0002).value
IID_IDXGIFactory1 = "{770aae78-f26f-4dba-a829-253c83d1b387}"


def _vcall(obj: ctypes.c_void_p, index: int, restype: Any, argtypes: list, *args: Any) -> Any:
    """Call a COM vtable method by index."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(vtbl[index])(obj, *args)


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _vcall(obj, 2, ctypes.c_ulong, [])


def dxgi_adapters() -> list[dict[str, Any]]:
    """Hardware adapters as reported by DXGI (software adapters excluded)."""
    out: list[dict[str, Any]] = []
    try:
        dxgi = ctypes.WinDLL("dxgi")
        create = dxgi.CreateDXGIFactory1
        create.restype = HRESULT
        create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        factory = ctypes.c_void_p()
        iid = GUID.from_string(IID_IDXGIFactory1)
        hr = create(ctypes.byref(iid), ctypes.byref(factory))
        if hr != 0 or not factory:
            _warn_once("dxgi", "CreateDXGIFactory1 failed: 0x%08x", hr & 0xFFFFFFFF)
            return out
        try:
            i = 0
            while i < 64:
                adapter = ctypes.c_void_p()
                hr = _vcall(factory, 12, HRESULT, [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)], i, ctypes.byref(adapter))
                if hr == DXGI_ERROR_NOT_FOUND or hr != 0:
                    break
                try:
                    desc = DXGI_ADAPTER_DESC1()
                    hr = _vcall(adapter, 10, HRESULT, [ctypes.POINTER(DXGI_ADAPTER_DESC1)], ctypes.byref(desc))
                    if hr == 0 and not (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE):
                        outputs = 0  # monitors attached to this adapter (IDXGIAdapter::EnumOutputs)
                        while outputs < 16:
                            output = ctypes.c_void_p()
                            if _vcall(adapter, 7, HRESULT, [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)], outputs,
                                      ctypes.byref(output)) != 0:
                                break
                            _release(output)
                            outputs += 1
                        out.append(
                            {
                                "index": i,
                                "name": desc.Description.strip(),
                                "vendor_id": desc.VendorId,
                                "device_id": desc.DeviceId,
                                "subsys_id": desc.SubSysId,
                                "revision": desc.Revision,
                                "vram_total": int(desc.DedicatedVideoMemory),
                                "shared_total": int(desc.SharedSystemMemory),
                                "luid": desc.AdapterLuid.key(),
                                "luid_low": desc.AdapterLuid.LowPart,
                                "luid_high": desc.AdapterLuid.HighPart,
                                "outputs": outputs,
                            }
                        )
                finally:
                    _release(adapter)
                i += 1
        finally:
            _release(factory)
    except OSError as exc:
        _warn_once("dxgi-load", "DXGI unavailable: %s", exc)
    except Exception:
        _warn_once("dxgi-exc", "DXGI enumeration failed")
        log.debug("dxgi", exc_info=True)
    # Deduplicate (the same adapter can be listed once per output in rare setups).
    uniq: dict[str, dict] = {}
    for a in out:
        uniq.setdefault(a["luid"], a)
    return list(uniq.values())


# ---------------------------------------------------------------------------
# D3DKMT: LUID -> PCI location
# ---------------------------------------------------------------------------


class D3DKMT_OPENADAPTERFROMLUID(ctypes.Structure):
    _fields_ = [("AdapterLuid", LUID), ("hAdapter", ctypes.c_uint)]


class D3DKMT_QUERYADAPTERINFO(ctypes.Structure):
    _fields_ = [
        ("hAdapter", ctypes.c_uint),
        ("Type", ctypes.c_int),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint),
    ]


class D3DKMT_ADAPTERADDRESS(ctypes.Structure):
    _fields_ = [("BusNumber", ctypes.c_uint), ("DeviceNumber", ctypes.c_uint), ("FunctionNumber", ctypes.c_uint)]


class D3DKMT_CLOSEADAPTER(ctypes.Structure):
    _fields_ = [("hAdapter", ctypes.c_uint)]


KMTQAITYPE_ADAPTERADDRESS = 6


def adapter_pci_address(luid_low: int, luid_high: int) -> tuple[int, int, int] | None:
    try:
        gdi = ctypes.WinDLL("gdi32")
        open_ = D3DKMT_OPENADAPTERFROMLUID()
        open_.AdapterLuid.LowPart = luid_low
        open_.AdapterLuid.HighPart = luid_high
        if gdi.D3DKMTOpenAdapterFromLuid(ctypes.byref(open_)) != 0:
            return None
        try:
            addr = D3DKMT_ADAPTERADDRESS()
            q = D3DKMT_QUERYADAPTERINFO()
            q.hAdapter = open_.hAdapter
            q.Type = KMTQAITYPE_ADAPTERADDRESS
            q.pPrivateDriverData = ctypes.cast(ctypes.byref(addr), ctypes.c_void_p)
            q.PrivateDriverDataSize = ctypes.sizeof(addr)
            if gdi.D3DKMTQueryAdapterInfo(ctypes.byref(q)) != 0:
                return None
            if addr.BusNumber > 255 or addr.DeviceNumber > 31 or addr.FunctionNumber > 7:
                return None
            return (addr.BusNumber, addr.DeviceNumber, addr.FunctionNumber)
        finally:
            close = D3DKMT_CLOSEADAPTER()
            close.hAdapter = open_.hAdapter
            gdi.D3DKMTCloseAdapter(ctypes.byref(close))
    except Exception:
        _warn_once("d3dkmt", "D3DKMT adapter address query failed")
        return None


# ---------------------------------------------------------------------------
# PDH performance counters
# ---------------------------------------------------------------------------


class _PDH_VALUE_UNION(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("AnsiStringValue", ctypes.c_char_p),
        ("WideStringValue", ctypes.c_wchar_p),
    ]


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("u", _PDH_VALUE_UNION)]


class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", PDH_FMT_COUNTERVALUE)]


PDH_FMT_DOUBLE = 0x00000200
PDH_FMT_LARGE = 0x00000400
PDH_FMT_NOCAP100 = 0x00008000
PDH_MORE_DATA = ctypes.c_long(0x800007D2).value
PDH_CSTATUS_VALID_DATA = 0x0
PDH_CSTATUS_NEW_DATA = 0x1


class PdhGpuCounters:
    """Collects GPU counters for all adapters (wildcard instances)."""

    COUNTERS = {
        "dedicated": (r"\GPU Adapter Memory(*)\Dedicated Usage", PDH_FMT_LARGE),
        "shared": (r"\GPU Adapter Memory(*)\Shared Usage", PDH_FMT_LARGE),
        "engine": (r"\GPU Engine(*)\Utilization Percentage", PDH_FMT_DOUBLE | PDH_FMT_NOCAP100),
        "proc_dedicated": (r"\GPU Process Memory(*)\Dedicated Usage", PDH_FMT_LARGE),
        "proc_shared": (r"\GPU Process Memory(*)\Shared Usage", PDH_FMT_LARGE),
    }
    REBUILD_INTERVAL = 30.0  # re-expand wildcards so new processes/engines appear

    def __init__(self) -> None:
        self.ok = False
        self._query = wintypes.HANDLE()
        self._counters: dict[str, wintypes.HANDLE] = {}
        self._built_at = 0.0
        try:
            self._pdh = ctypes.WinDLL("pdh")
            self._pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
            self._pdh.PdhAddEnglishCounterW.argtypes = [
                wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)
            ]
            self._pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
            self._pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
            self._pdh.PdhGetFormattedCounterArrayW.argtypes = [
                wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
            ]
            for fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhCloseQuery",
                       "PdhGetFormattedCounterArrayW"):
                getattr(self._pdh, fn).restype = ctypes.c_long
            self._build()
        except OSError as exc:
            _warn_once("pdh", "PDH unavailable: %s", exc)

    def _build(self) -> None:
        if self._query:
            self._pdh.PdhCloseQuery(self._query)
            self._query = wintypes.HANDLE()
        self._counters = {}
        if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            self.ok = False
            return
        for name, (path, _fmt) in self.COUNTERS.items():
            h = wintypes.HANDLE()
            if self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(h)) == 0:
                self._counters[name] = h
        self.ok = bool(self._counters)
        self._built_at = time.monotonic()
        if self.ok:
            self._pdh.PdhCollectQueryData(self._query)  # prime rate counters

    def _array(self, counter: wintypes.HANDLE, fmt: int) -> list[tuple[str, float]]:
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        st = self._pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if st != PDH_MORE_DATA or size.value == 0:
            return []
        buf = (ctypes.c_byte * size.value)()
        st = self._pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size), ctypes.byref(count), buf)
        if st != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM_W))
        out = []
        for i in range(count.value):
            it = items[i]
            if it.FmtValue.CStatus not in (PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA):
                continue
            v = it.FmtValue.u.doubleValue if fmt & PDH_FMT_DOUBLE else float(it.FmtValue.u.largeValue)
            out.append((it.szName or "", v))
        return out

    def sample(self, watch_pids: set[int] | None = None) -> dict[str, dict[str, Any]]:
        """Per-LUID: dedicated, shared, util, engines{type:%}, proc_dedicated / proc_shared {pid: bytes}."""
        if not self.ok:
            return {}
        try:
            if time.monotonic() - self._built_at > self.REBUILD_INTERVAL:
                self._build()
                time.sleep(0.05)
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                return {}
            res: dict[str, dict[str, Any]] = {}

            def slot(luid: str) -> dict[str, Any]:
                return res.setdefault(
                    luid, {"dedicated": 0.0, "shared": 0.0, "engines": {}, "util": 0.0, "proc_dedicated": {},
                           "proc_shared": {}, "proc_util": {}}
                )

            def luid_of(inst: str) -> str | None:
                i = inst.find("luid_")
                if i < 0:
                    return None
                parts = inst[i:].split("_")
                if len(parts) < 3:
                    return None
                return "_".join(parts[:3]).lower()

            for name in ("dedicated", "shared"):
                if name in self._counters:
                    for inst, v in self._array(self._counters[name], self.COUNTERS[name][1]):
                        lu = luid_of(inst)
                        if lu:
                            slot(lu)[name] += v
            if "engine" in self._counters:
                per_engine: dict[tuple[str, str], float] = {}
                eng_type: dict[tuple[str, str], str] = {}
                for inst, v in self._array(self._counters["engine"], self.COUNTERS["engine"][1]):
                    lu = luid_of(inst)
                    if not lu:
                        continue
                    low = inst.lower()
                    e_idx = low.split("_eng_")[1].split("_")[0] if "_eng_" in low else "?"
                    etype = inst.split("engtype_")[1] if "engtype_" in inst else "?"
                    key = (lu, e_idx)
                    per_engine[key] = per_engine.get(key, 0.0) + v
                    eng_type[key] = etype
                    if watch_pids and low.startswith("pid_"):
                        try:
                            pid = int(low.split("_")[1])
                        except (IndexError, ValueError):
                            pid = -1
                        if pid in watch_pids:
                            pu = slot(lu)["proc_util"]
                            pu[pid] = max(pu.get(pid, 0.0), v)
                for (lu, _e), v in per_engine.items():
                    s = slot(lu)
                    et = eng_type[(lu, _e)]
                    s["engines"][et] = max(s["engines"].get(et, 0.0), min(v, 100.0))
                for lu, s in res.items():
                    s["util"] = max(s["engines"].values()) if s["engines"] else 0.0
            for cname in ("proc_dedicated", "proc_shared"):
                if not watch_pids or cname not in self._counters:
                    continue
                for inst, v in self._array(self._counters[cname], self.COUNTERS[cname][1]):
                    low = inst.lower()
                    if not low.startswith("pid_"):
                        continue
                    try:
                        pid = int(low.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    lu = luid_of(inst)
                    if lu and pid in watch_pids:
                        pd = slot(lu)[cname]
                        pd[pid] = pd.get(pid, 0.0) + v
            return res
        except Exception:
            _warn_once("pdh-sample", "PDH sampling failed")
            log.debug("pdh", exc_info=True)
            return {}

    def close(self) -> None:
        if self._query:
            try:
                self._pdh.PdhCloseQuery(self._query)
            except Exception:
                pass
            self._query = wintypes.HANDLE()


# ---------------------------------------------------------------------------
# AMD Display Library (ADL) sensors
# ---------------------------------------------------------------------------

ADL_MAX_PATH = 256
ADL_PMLOG_MAX_SENSORS = 256


class ADL_AdapterInfo(ctypes.Structure):
    _fields_ = [
        ("iSize", ctypes.c_int),
        ("iAdapterIndex", ctypes.c_int),
        ("strUDID", ctypes.c_char * ADL_MAX_PATH),
        ("iBusNumber", ctypes.c_int),
        ("iDeviceNumber", ctypes.c_int),
        ("iFunctionNumber", ctypes.c_int),
        ("iVendorID", ctypes.c_int),
        ("strAdapterName", ctypes.c_char * ADL_MAX_PATH),
        ("strDisplayName", ctypes.c_char * ADL_MAX_PATH),
        ("iPresent", ctypes.c_int),
        ("iExist", ctypes.c_int),
        ("strDriverPath", ctypes.c_char * ADL_MAX_PATH),
        ("strDriverPathExt", ctypes.c_char * ADL_MAX_PATH),
        ("strPNPString", ctypes.c_char * ADL_MAX_PATH),
        ("iOSDisplayIndex", ctypes.c_int),
    ]


class ADLSingleSensorData(ctypes.Structure):
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class ADLPMLogDataOutput(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("sensors", ADLSingleSensorData * ADL_PMLOG_MAX_SENSORS)]


# ADL_PMLOG_SENSORS (adl_defines.h)
PMLOG = {
    "clk_gfx": 1,
    "clk_mem": 2,
    "clk_soc": 3,
    "temp_edge": 8,
    "temp_mem": 9,
    "temp_vrvddc": 10,
    "temp_vrmvdd": 11,
    "fan_rpm": 14,
    "fan_pct": 15,
    "soc_power": 17,
    "activity_gfx": 19,
    "activity_mem": 20,
    "gfx_voltage": 21,
    "asic_power": 23,
    "temp_hotspot": 27,
    "temp_gfx": 28,
    "temp_soc": 29,
    "gfx_power": 30,
    "bus_speed": 40,
    "bus_lanes": 41,
}
# Plausible value ranges used to reject garbage from mismatched ADL versions.
_PMLOG_RANGES = {
    "clk_gfx": (0, 5000), "clk_mem": (0, 5000), "clk_soc": (0, 5000),
    "temp_edge": (1, 150), "temp_mem": (1, 150), "temp_vrvddc": (1, 150), "temp_vrmvdd": (1, 150),
    "temp_hotspot": (1, 150), "temp_gfx": (1, 150), "temp_soc": (1, 150),
    "fan_rpm": (0, 10000), "fan_pct": (0, 100), "soc_power": (0, 1000), "activity_gfx": (0, 100),
    "activity_mem": (0, 100), "gfx_voltage": (0, 2000), "asic_power": (0, 1000), "gfx_power": (0, 1000),
    "bus_speed": (0, 100000), "bus_lanes": (0, 32),
}

_ADL_MALLOC = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_int)


class AdlSensors:
    """AMD GPU sensor access through atiadlxx.dll (installed with the Adrenalin driver)."""

    def __init__(self) -> None:
        self.ok = False
        self.adapters: list[dict[str, Any]] = []
        self._ctx = ctypes.c_void_p()
        try:
            self._dll = ctypes.WinDLL("atiadlxx")
        except OSError:
            return  # not an AMD system (or driver missing)
        try:
            libc = ctypes.CDLL("msvcrt")
            libc.malloc.restype = ctypes.c_void_p
            libc.malloc.argtypes = [ctypes.c_size_t]
            self._malloc_cb = _ADL_MALLOC(lambda size: libc.malloc(max(1, size)))
            d = self._dll
            d.ADL2_Main_Control_Create.argtypes = [_ADL_MALLOC, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            d.ADL2_Main_Control_Create.restype = ctypes.c_int
            if d.ADL2_Main_Control_Create(self._malloc_cb, 1, ctypes.byref(self._ctx)) != 0:
                _warn_once("adl-init", "ADL2_Main_Control_Create failed")
                return
            n = ctypes.c_int(0)
            d.ADL2_Adapter_NumberOfAdapters_Get.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            if d.ADL2_Adapter_NumberOfAdapters_Get(self._ctx, ctypes.byref(n)) != 0 or n.value <= 0:
                return
            infos = (ADL_AdapterInfo * n.value)()
            d.ADL2_Adapter_AdapterInfo_Get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            if d.ADL2_Adapter_AdapterInfo_Get(self._ctx, infos, ctypes.sizeof(infos)) != 0:
                return
            seen: set[int] = set()
            for info in infos:
                if info.iBusNumber in seen or info.iVendorID not in (1002, 0x1002):
                    continue
                active = ctypes.c_int(0)
                try:
                    d.ADL2_Adapter_Active_Get.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
                    d.ADL2_Adapter_Active_Get(self._ctx, info.iAdapterIndex, ctypes.byref(active))
                except AttributeError:
                    pass
                seen.add(info.iBusNumber)
                self.adapters.append(
                    {
                        "adl_index": info.iAdapterIndex,
                        "bus": info.iBusNumber,
                        "device": info.iDeviceNumber,
                        "function": info.iFunctionNumber,
                        "name": info.strAdapterName.decode(errors="replace").strip(),
                        "udid": info.strUDID.decode(errors="replace"),
                    }
                )
            self._pmlog = getattr(d, "ADL2_New_QueryPMLogData_Get", None)
            if self._pmlog is not None:
                self._pmlog.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ADLPMLogDataOutput)]
                self._pmlog.restype = ctypes.c_int
            self._vram = getattr(d, "ADL2_Adapter_VRAMUsage_Get", None)
            if self._vram is not None:
                self._vram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
                self._vram.restype = ctypes.c_int
            self.ok = bool(self.adapters)
        except Exception:
            _warn_once("adl-exc", "ADL initialisation failed")
            log.debug("adl", exc_info=True)

    def read(self, adl_index: int) -> dict[str, float]:
        out: dict[str, float] = {}
        if not self.ok:
            return out
        try:
            if self._pmlog is not None:
                data = ADLPMLogDataOutput()
                data.size = ctypes.sizeof(data)
                if self._pmlog(self._ctx, adl_index, ctypes.byref(data)) == 0:
                    for key, idx in PMLOG.items():
                        s = data.sensors[idx]
                        if s.supported:
                            lo, hi = _PMLOG_RANGES[key]
                            if lo <= s.value <= hi:
                                out[key] = float(s.value)
            if self._vram is not None:
                mb = ctypes.c_int(0)
                if self._vram(self._ctx, adl_index, ctypes.byref(mb)) == 0 and mb.value > 0:
                    out["vram_used_mb"] = float(mb.value)
        except Exception:
            _warn_once("adl-read", "ADL sensor read failed")
        return out


# ---------------------------------------------------------------------------
# Process management helpers
# ---------------------------------------------------------------------------


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9
_job_handle: int | None = None


def kill_on_exit_job() -> int | None:
    """A job object whose processes die when WinRunner's handle closes."""
    global _job_handle
    if _job_handle:
        return _job_handle
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        _job_handle = job
        return job
    except Exception:
        _warn_once("job", "could not create job object")
        return None


def assign_to_job(process_handle: int) -> bool:
    job = kill_on_exit_job()
    if not job:
        return False
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        return bool(k32.AssignProcessToJobObject(job, process_handle))
    except Exception:
        return False


PRIORITY_FLAGS = {
    "normal": 0x00000020,
    "above_normal": 0x00008000,
    "high": 0x00000080,
}
CREATE_NO_WINDOW = 0x08000000


def cpu_brand() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except OSError:
        return ""


def gpu_driver_versions() -> dict[str, str]:
    """Display adapter name -> driver version from the device class registry."""
    out: dict[str, str] = {}
    try:
        import winreg

        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as k:
                        desc = str(winreg.QueryValueEx(k, "DriverDesc")[0])
                        ver = str(winreg.QueryValueEx(k, "DriverVersion")[0])
                        try:
                            rs = str(winreg.QueryValueEx(k, "RadeonSoftwareVersion")[0])
                            ver = f"{rs} ({ver})"
                        except OSError:
                            pass
                        out.setdefault(desc, ver)
                except OSError:
                    continue
    except Exception:
        pass
    return out


_WEBVIEW2_CLIENT = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def webview2_version() -> str | None:
    """Installed Microsoft Edge WebView2 Runtime version, or ``None`` when it is missing.

    Uses the registry keys Microsoft documents for detecting the Evergreen runtime
    (machine-wide and per-user) and also checks that the runtime files are still on
    disk, because an uninstalled or damaged runtime can leave the ``pv`` value behind.
    """
    try:
        import os
        import winreg
    except ImportError:
        return None
    keys = [
        (winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\WOW6432Node\\" + _WEBVIEW2_CLIENT),
        (winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\" + _WEBVIEW2_CLIENT),
        (winreg.HKEY_CURRENT_USER, "Software\\" + _WEBVIEW2_CLIENT),
    ]
    roots = [os.path.join(os.environ.get(v, ""), "Microsoft", "EdgeWebView", "Application")
             for v in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA") if os.environ.get(v)]
    for hive, path in keys:
        try:
            with winreg.OpenKey(hive, path) as k:
                pv = str(winreg.QueryValueEx(k, "pv")[0]).strip()
        except OSError:
            continue
        if not pv or pv == "0.0.0.0":
            continue
        if any(os.path.isfile(os.path.join(r, pv, "msedgewebview2.exe")) for r in roots):
            return pv
    return None
