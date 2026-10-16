"""Windows: plan with physical VRAM instead of the smaller WDDM budget the Vulkan driver reports."""

import asyncio
from types import SimpleNamespace

from winrunner import hardware, manager
from winrunner.config import Settings
from winrunner.engine import EngineDevice
from winrunner.hardware import GpuInfo, HardwareMonitor

MiB = 1024 * 1024


def _monitor(sample_used_mib: dict[str, int]) -> HardwareMonitor:
    mon = HardwareMonitor.__new__(HardwareMonitor)
    mon.gpus = [GpuInfo(id="gpu0", name="AMD Radeon RX 6800", vendor="AMD", vram_total=16368 * MiB, bus=0x0b),
                GpuInfo(id="gpu1", name="AMD Radeon RX 6800", vendor="AMD", vram_total=16368 * MiB, bus=0x0e)]
    mon._thread = None
    mon.last = {"t": 1.0, "gpus": [{"id": g, "vram_total": 16368 * MiB, "vram_used": u * MiB}
                                   for g, u in sample_used_mib.items()], "procs": {}}
    return mon


def test_physical_free_is_total_minus_all_processes():
    mon = _monitor({"gpu0": 900, "gpu1": 150})
    assert mon.physical_free_mib() == {"gpu0": 15468.0, "gpu1": 16218.0}


def test_engine_devices_matched_by_pci_address_from_the_engine_log():
    mon = _monitor({"gpu0": 900, "gpu1": 150})
    devs = [{"name": "Vulkan0", "description": "AMD Radeon RX 6800", "pci": "0000:0e:00.0"},
            {"name": "Vulkan1", "description": "AMD Radeon RX 6800", "pci": "0000:0b:00.0"}]
    assert mon.map_engine_devices(devs) == {"Vulkan0": "gpu1", "Vulkan1": "gpu0"}


def test_identical_gpus_paired_by_memory_in_use(monkeypatch):
    monkeypatch.setattr(hardware, "IS_WINDOWS", True)
    mon = _monitor({"gpu0": 150, "gpu1": 1100})  # gpu1 drives the desktop
    # the budget Windows gives Vulkan is smaller on the busy GPU: Vulkan0 is gpu1 here
    devs = [{"name": "Vulkan0", "description": "AMD Radeon RX 6800", "total_mib": 16368, "free_mib": 14300},
            {"name": "Vulkan1", "description": "AMD Radeon RX 6800", "total_mib": 16368, "free_mib": 15250}]
    assert mon.map_engine_devices(devs) == {"Vulkan0": "gpu1", "Vulkan1": "gpu0"}
    # no clear difference: name order
    mon2 = _monitor({"gpu0": 150, "gpu1": 170})
    assert mon2.map_engine_devices(devs) == {"Vulkan0": "gpu0", "Vulkan1": "gpu1"}


def _fake_manager(mon, use_physical=True):
    s = Settings()
    s.hardware.use_physical_vram = use_physical
    fake = SimpleNamespace(store=SimpleNamespace(settings=s), monitor=mon, _vram_penalty={}, _device_pci={})
    fake.device_map = lambda devices: manager.ModelManager.device_map(fake, devices)
    return fake


def test_windows_plans_with_physical_free_vram(monkeypatch):
    monkeypatch.setattr(manager, "IS_WINDOWS", True)
    monkeypatch.setattr(hardware, "IS_WINDOWS", True)
    mon = _monitor({"gpu0": 1100, "gpu1": 150})
    # what the AMD driver reports on Windows: budget - usage, ~1 GB below the physical free memory
    devs = [EngineDevice("Vulkan0", "AMD Radeon RX 6800", 16368, 14300),
            EngineDevice("Vulkan1", "AMD Radeon RX 6800", 16368, 15250)]
    fake = _fake_manager(mon)
    out, gained = asyncio.run(manager.ModelManager._physical_vram(fake, devs, False))
    assert [d.free_mib for d in out] == [15268, 16218]
    assert gained == {"Vulkan0": 968, "Vulkan1": 968}
    # with the 512 MiB margin a 16 GB card now ends at ~15.5 GiB in use (Task Manager), not ~14.5
    assert 16368 - (out[1].free_mib - 512) - 150 <= 512 + 1
    # a GPU that lost memory to system RAM at the last load gets that much more room
    fake._vram_penalty["Vulkan1"] = 700
    out2, _ = asyncio.run(manager.ModelManager._physical_vram(fake, devs, False))
    assert out2[1].free_mib == 15518
    # setting off, or not Windows: the engine's figures
    assert asyncio.run(manager.ModelManager._physical_vram(_fake_manager(mon, False), devs, False))[0] == devs
    monkeypatch.setattr(manager, "IS_WINDOWS", False)
    assert asyncio.run(manager.ModelManager._physical_vram(fake, devs, False))[0] == devs


def test_residency_check_warns_and_leaves_room_next_time(monkeypatch):
    monkeypatch.setattr(manager, "IS_WINDOWS", True)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(manager.asyncio, "sleep", no_sleep)
    mon = _monitor({"gpu0": 150, "gpu1": 150})
    mon.last["procs"] = {"77": {"vram": {"gpu0": 14000 * MiB, "gpu1": 15500 * MiB}}}
    logged = []
    fake = _fake_manager(mon)
    fake.bus = SimpleNamespace(activity_log=lambda msg, **kw: logged.append((msg, kw.get("level"))))
    inst = SimpleNamespace(state="ready", pid=77, model_id="m", device_map={"Vulkan0": "gpu0", "Vulkan1": "gpu1"},
                           load_info={"buffers": {"Vulkan0": {"model": 14000.0, "kv": 1200.0, "compute": 300.0},
                                                  "Vulkan1": {"model": 14800.0, "kv": 400.0, "compute": 250.0}}})

    async def run():
        tasks = []
        fake._spawn = lambda coro: tasks.append(asyncio.ensure_future(coro))
        manager.ModelManager._check_residency(fake, inst)
        await asyncio.gather(*tasks)

    asyncio.run(run())
    assert len(logged) == 1 and "Vulkan0" in logged[0][0] and logged[0][1] == "warn"
    assert fake._vram_penalty == {"Vulkan0": 1500 + 256}
