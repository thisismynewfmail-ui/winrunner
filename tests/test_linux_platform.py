import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Linux platform module")

MiB = 1024 * 1024


def _card(root: Path, n: int, pci: str, used_mib: int, boot_vga: bool, width: str = "16", vis_mib: int = 16368):
    dev = root / "devices" / "pci0000:00" / pci
    dev.mkdir(parents=True)
    files = {
        "vendor": "0x1002", "device": "0x73bf", "subsystem_vendor": "0x1002", "subsystem_device": "0x0e3a",
        "mem_info_vram_total": str(16368 * MiB), "mem_info_vram_used": str(used_mib * MiB),
        "mem_info_vis_vram_total": str(vis_mib * MiB), "boot_vga": "1" if boot_vga else "0",
        "current_link_speed": "16.0 GT/s PCIe", "current_link_width": width, "max_link_width": "16",
        "gpu_busy_percent": "37",
    }
    for k, v in files.items():
        (dev / k).write_text(v + "\n")
    hw = dev / "hwmon" / f"hwmon{n}"
    hw.mkdir(parents=True)
    (hw / "temp1_label").write_text("edge\n")
    (hw / "temp1_input").write_text("51000\n")
    (hw / "power1_average").write_text("45000000\n")
    drm = root / "class" / "drm"
    drm.mkdir(parents=True, exist_ok=True)
    (drm / f"card{n}").mkdir()
    os.symlink(dev, drm / f"card{n}" / "device")
    (drm / f"card{n}-DP-1").mkdir()  # connectors are ignored
    return dev


@pytest.fixture
def fake_sys(tmp_path, monkeypatch):
    from winrunner.platform import linux

    _card(tmp_path, 0, "0000:0b:00.0", 900, True)
    _card(tmp_path, 1, "0000:0e:00.0", 20, False, width="4", vis_mib=256)
    ids = tmp_path / "pci.ids"
    ids.write_text("1002  Advanced Micro Devices, Inc. [AMD/ATI]\n"
                   "\t73bf  Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]\n"
                   "\t\t1002 0e3a  Radeon RX 6800\n"
                   "10de  NVIDIA Corporation\n")
    monkeypatch.setattr(linux, "SYS_DRM", str(tmp_path / "class" / "drm"))
    monkeypatch.setattr(linux, "PCI_IDS", (str(ids),))
    monkeypatch.setattr(linux, "PROC", str(tmp_path / "proc"))
    return tmp_path


def test_amdgpu_cards_details(fake_sys):
    from winrunner.platform import linux

    cards = linux.amdgpu_cards()
    assert [c["pci"] for c in cards] == ["0000:0b:00.0", "0000:0e:00.0"]
    a, b = cards
    assert a["name"] == "Radeon RX 6800" and a["bus"] == 0x0b and a["vram_total"] == 16368 * MiB
    assert a["boot_vga"] and not b["boot_vga"]
    assert a["rebar"] is True and b["rebar"] is False
    assert a["pcie"] == "PCIe 4.0 x16" and b["pcie"] == "PCIe 4.0 x4 (card supports x16)"
    s = linux.amdgpu_sample(a)
    assert s["vram_used"] == 900 * MiB and s["util"] == 37 and s["temp_edge"] == 51 and s["power"] == 45


def test_proc_vram_from_fdinfo(fake_sys):
    from winrunner.platform import linux

    fd = fake_sys / "proc" / "4242" / "fdinfo"
    fd.mkdir(parents=True)
    client = "drm-driver:\tamdgpu\ndrm-client-id:\t{cid}\ndrm-pdev:\t{pdev}\ndrm-memory-vram:\t{kib} KiB\n"
    (fd / "7").write_text(client.format(cid=11, pdev="0000:0b:00.0", kib=1024 * 1024))
    (fd / "8").write_text(client.format(cid=11, pdev="0000:0b:00.0", kib=1024 * 1024))  # same client (dup fd)
    (fd / "9").write_text("drm-driver:\tamdgpu\ndrm-client-id:\t12\ndrm-pdev:\t0000:0e:00.0\n"
                          "drm-resident-vram:\t2048 MiB\ndrm-memory-vram:\t1 KiB\n")
    (fd / "0").write_text("pos:\t0\nflags:\t02\n")
    got = linux.proc_vram({4242, 99999})
    assert got == {"0000:0b:00.0": {4242: 1024 * MiB}, "0000:0e:00.0": {4242: 2048 * MiB}}


def test_engine_devices_matched_by_memory_in_use(fake_sys):
    from winrunner.platform import linux

    cards = linux.amdgpu_cards()
    used = {"0000:0b:00.0": 900 * MiB, "0000:0e:00.0": 20 * MiB}
    # Vulkan0 has almost everything free: the headless card, although the display card comes first on the bus
    devs = [{"name": "Vulkan0", "total_mib": 16368, "free_mib": 16340},
            {"name": "Vulkan1", "total_mib": 16368, "free_mib": 15460}]
    m = linux.match_engine_devices(devs, cards, used)
    assert m["Vulkan0"]["pci"] == "0000:0e:00.0" and m["Vulkan1"]["pci"] == "0000:0b:00.0"
    # no usable difference: Mesa's order, boot display GPU first
    same = {"0000:0b:00.0": 30 * MiB, "0000:0e:00.0": 30 * MiB}
    devs2 = [{"name": "Vulkan0", "total_mib": 16368, "free_mib": 16338},
             {"name": "Vulkan1", "total_mib": 16368, "free_mib": 16338}]
    m2 = linux.match_engine_devices(devs2, cards, same)
    assert m2["Vulkan0"]["boot_vga"] and not m2["Vulkan1"]["boot_vga"]


def test_hardware_monitor_uses_sysfs(fake_sys, monkeypatch):
    from winrunner import hardware

    monkeypatch.setattr(hardware.HardwareMonitor, "_detect", hardware.HardwareMonitor._detect)
    mon = hardware.HardwareMonitor()
    assert [g.pci for g in mon.gpus] == ["0000:0b:00.0", "0000:0e:00.0"]
    assert mon.gpus[1].pcie.startswith("PCIe 4.0 x4")
    m = mon.map_engine_devices([{"name": "Vulkan0", "description": "AMD Radeon RX 6800 (RADV NAVI21)",
                                 "total_mib": 16368, "free_mib": 15460},
                                {"name": "Vulkan1", "description": "AMD Radeon RX 6800 (RADV NAVI21)",
                                 "total_mib": 16368, "free_mib": 16340}])
    assert m == {"Vulkan0": "gpu0", "Vulkan1": "gpu1"}
    info = mon.system_info()
    assert info["gpus"][0]["rebar"] is True and "sysfs" not in info["gpus"][0]
