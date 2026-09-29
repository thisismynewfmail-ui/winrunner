"""VRAM residency (Windows paging) detection and GPU role detection."""

from __future__ import annotations

from types import SimpleNamespace

from winrunner.hardware import GpuInfo, HardwareMonitor
from winrunner.manager import ModelManager

MiB = 1024 * 1024


class Bus:
    def __init__(self):
        self.events, self.activity = [], []

    def publish(self, etype, **d):
        self.events.append((etype, d))

    def activity_log(self, text, level="info", **kw):
        self.activity.append((level, text))


def _mgr_with_instance(bufs):
    inst = SimpleNamespace(
        id="inst-1", model_id="m", pid=4242, state="ready", t_ready=0.0,
        load_info={"buffers": bufs, "vram_spill_active": False, "vram_spill_mib": 0},
    )
    inst.status = lambda: {"id": inst.id}
    mgr = ModelManager.__new__(ModelManager)
    mgr.instances = {inst.id: inst}
    mgr._spill = {}
    mgr.bus = Bus()
    return mgr, inst


BUFS = {"Vulkan0": {"model": 9000.0, "kv": 3000.0, "compute": 400.0},
        "Vulkan1": {"model": 9500.0, "kv": 3000.0, "compute": 400.0},
        "CPU_Mapped": {"model": 600.0}, "CPU": {"output": 2.0}}


def _sample(resident_mib, shared_mib=0.0):
    half = resident_mib / 2 * MiB
    return {"procs": {"4242": {"vram": {"gpu0": half, "gpu1": half},
                               "shared": {"gpu0": shared_mib * MiB, "gpu1": 0.0}}}}


def test_resident_model_is_not_flagged():
    mgr, inst = _mgr_with_instance(BUFS)
    for _ in range(5):
        mgr.check_residency(_sample(25600))  # allocations + driver overhead all resident
    assert not inst.load_info["vram_spill_active"] and not mgr.bus.activity


def test_paged_out_memory_is_flagged_once_after_three_samples():
    mgr, inst = _mgr_with_instance(BUFS)
    mgr.check_residency(_sample(22000, 3500))
    mgr.check_residency(_sample(22000, 3500))
    assert not inst.load_info["vram_spill_active"]  # needs a persistent gap
    for _ in range(4):
        mgr.check_residency(_sample(22000, 3500))
    assert inst.load_info["vram_spill_active"] and 3000 < inst.load_info["vram_spill_mib"] < 3400
    assert inst.load_info["vram_shared_mib"] == 3500
    errors = [t for lvl, t in mgr.bus.activity if lvl == "error"]
    assert len(errors) == 1 and "into system RAM" in errors[0]
    # recovers when the memory is resident again
    mgr.check_residency(_sample(25600))
    assert not inst.load_info["vram_spill_active"]


def _monitor(gpus, used):
    mon = HardwareMonitor.__new__(HardwareMonitor)
    mon._watch = {}
    mon.gpus = gpus
    mon.last = {"gpus": [{"id": g.id, "vram_used": used.get(g.id)} for g in gpus], "procs": {}}
    return mon


def _two_6800s(display0=True):
    return [GpuInfo(id="gpu0", name="AMD Radeon RX 6800", vendor="AMD", vram_total=16 << 30, bus=11, display=display0),
            GpuInfo(id="gpu1", name="AMD Radeon RX 6800", vendor="AMD", vram_total=16 << 30, bus=12, display=not display0)]


def test_identical_cards_matched_by_free_memory_and_display():
    # the display card is the second PCI slot; the engine lists it first (less free VRAM)
    mon = _monitor(_two_6800s(display0=False), {"gpu0": 60 * MiB, "gpu1": 900 * MiB})
    devs = [{"name": "Vulkan0", "description": "AMD Radeon RX 6800", "free_mib": 15100},
            {"name": "Vulkan1", "description": "AMD Radeon RX 6800", "free_mib": 16000}]
    assert mon.map_engine_devices(devs) == {"Vulkan0": "gpu1", "Vulkan1": "gpu0"}
    assert mon.device_roles(devs) == {"Vulkan0": "display", "Vulkan1": "idle"}
    # with a WinRunner engine holding VRAM, free memory no longer identifies the display card
    mon._watch = {1234: "model"}
    assert set(mon.device_roles(devs).values()) == {"unknown"}


def test_pci_address_wins_and_ambiguity_is_conservative():
    mon = _monitor(_two_6800s(display0=True), {"gpu0": 700 * MiB, "gpu1": 50 * MiB})
    devs = [{"name": "Vulkan0", "description": "AMD Radeon RX 6800", "free_mib": 15900, "pci": "0000:0c:00.0"},
            {"name": "Vulkan1", "description": "AMD Radeon RX 6800", "free_mib": 15900, "pci": "0000:0b:00.0"}]
    assert mon.map_engine_devices(devs) == {"Vulkan0": "gpu1", "Vulkan1": "gpu0"}
    assert mon.device_roles(devs) == {"Vulkan0": "idle", "Vulkan1": "display"}
    # same free memory and no PCI information: cannot tell which one drives the display
    for d in devs:
        d.pop("pci")
    assert set(mon.device_roles(devs).values()) == {"unknown"}


def test_similar_names_match_exactly():
    gpus = [GpuInfo(id="gpu0", name="AMD Radeon RX 6800", vendor="AMD", vram_total=16 << 30, bus=3, display=True),
            GpuInfo(id="gpu1", name="AMD Radeon RX 6800 XT", vendor="AMD", vram_total=16 << 30, bus=8, display=False)]
    mon = _monitor(gpus, {"gpu0": 900 * MiB, "gpu1": 50 * MiB})
    devs = [{"name": "Vulkan0", "description": "AMD Radeon RX 6800 XT", "free_mib": 16000},
            {"name": "Vulkan1", "description": "AMD Radeon RX 6800", "free_mib": 15000}]
    assert mon.map_engine_devices(devs) == {"Vulkan0": "gpu1", "Vulkan1": "gpu0"}
    assert mon.device_roles(devs) == {"Vulkan0": "idle", "Vulkan1": "display"}
