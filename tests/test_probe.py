import json
import sys
import time
from pathlib import Path

import pytest

from winrunner.config import Settings, SettingsStore, migrate
from winrunner.logparse import LogParser
from winrunner.probe import as_measurement, parse_projection, server_probe

# llama-server b11269 with --fit on and an explicit placement (-ngl all -ts 17,14 -ot ...), -lv 4
LOG = """0.00.094.478 I srv  llama_server: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true
0.00.096.576 I srv    load_model: loading model '/models/SmolLM2-135M-Instruct-Q8_0.gguf'
0.00.096.600 I srv    load_model: [mtmd] estimated worst-case memory usage of mmproj is 812.25 MiB (took 3.10 ms)
0.00.096.657 I common_params_fit_impl: getting device memory data for initial parameters:
0.00.206.830 I common_memory_breakdown_print: | memory breakdown [MiB]                         | total    free    self   model   context   compute    unaccounted |
0.00.206.851 I common_memory_breakdown_print: |   - Vulkan0 (AMD Radeon RX 6800 (RADV NAVI21)) | 16368 = 15507 + ( 133 =    60 +      54 +      19) +         728 |
0.00.206.851 I common_memory_breakdown_print: |   - Vulkan1 (AMD Radeon RX 6800 (RADV NAVI21)) | 16368 = 15507 + ( 136 =    75 +      41 +      20) +         725 |
0.00.206.852 I common_memory_breakdown_print: |   - Host                                       |                    39 =    29 +       0 +      10                |
0.00.215.359 I common_params_fit_impl: projected memory use with initial parameters [MiB]:
0.00.215.376 I common_params_fit_impl:   - Vulkan0 (AMD Radeon RX 6800 (RADV NAVI21)):  16368 total,    133 used,  15374 free vs. target of    512
0.00.215.377 I common_params_fit_impl:   - Vulkan1 (AMD Radeon RX 6800 (RADV NAVI21)):  16368 total,    136 used,  15371 free vs. target of    512
0.00.215.378 I common_params_fit_impl: projected to use 269 MiB of device memory vs. 31014 MiB of free device memory
0.00.215.378 I common_params_fit_impl: targets for free memory can be met on all devices, no changes needed
0.00.298.588 I load_tensors: loading model tensors, this can take a while... (load_mode = none)
"""


def _parse(text: str, jsonl: bool = False):
    lp = LogParser(jsonl=jsonl)
    msgs, bd = [], {}
    for line in text.splitlines():
        for ll in lp.feed(line):
            msgs.append(ll.text)
            for kind, d in ll.events:
                if kind == "breakdown" and d["device"] not in bd:
                    bd[d["device"]] = d
    return msgs, bd


def test_parse_projection_multi_gpu():
    pr = parse_projection(*_parse(LOG)[:1], breakdowns=_parse(LOG)[1])
    assert pr is not None and pr.total_used == 269 and pr.mmproj_mib == 812.25
    v0 = pr.devices["Vulkan0"]
    assert v0["used"] == 133 and v0["free"] == 15507 and v0["target"] == 512
    assert (v0["model"], v0["context"], v0["compute"]) == (60, 54, 19)
    assert pr.devices["Vulkan1"]["compute"] == 20


def test_parse_projection_jsonl_and_single_gpu():
    lines = [
        {"level": "info", "msg": "common_params_fit_impl: projected to use 9000 MiB of device memory vs. 15800 MiB of free device memory\n"},
        {"level": "info", "msg": "common_params_fit_impl: will leave 6800 >= 512 MiB of free device memory, no changes needed\n"},
    ]
    msgs, bd = _parse("\n".join(json.dumps(x) for x in lines), jsonl=True)
    pr = parse_projection(msgs, "Vulkan0", bd)
    assert pr.devices == {"Vulkan0": {"used": 9000.0, "free": 15800.0, "target": 0.0}}
    assert parse_projection(["srv  load_model: loading model 'x'"]) is None


def test_as_measurement_uses_the_engine_breakdown():
    class D:
        def __init__(self, name):
            self.name, self.weights_mib, self.output_mib, self.kv_mib = name, 70.0, 0.0, 50.0

    class P:
        devices = [D("Vulkan0"), D("Vulkan1")]

    msgs, bd = _parse(LOG)
    m = as_measurement(parse_projection(msgs, None, bd), P())
    assert m["Vulkan0"] == {"model": 60, "context": 54, "compute": 19}
    # without a breakdown: the estimate's weights and KV, the rest is compute, the total is the engine's
    pr = parse_projection(msgs)
    m2 = as_measurement(pr, P())
    assert sum(m2["Vulkan0"].values()) == 133 and m2["Vulkan0"]["context"] == 50


@pytest.mark.skipif(sys.platform == "win32", reason="uses a script as the engine")
def test_server_probe_stops_the_engine_after_the_projection(tmp_path: Path):
    log = tmp_path / "log.txt"
    log.write_text(LOG)
    marker = tmp_path / "loaded"
    exe = tmp_path / "llama-server"
    exe.write_text(f"#!{sys.executable}\nimport sys, time\nprint(open({str(log)!r}).read(), flush=True)\n"
                   f"time.sleep(30)\nopen({str(marker)!r}, 'w').write('x')\n")
    exe.chmod(0o755)
    t0 = time.monotonic()
    pr = server_probe([str(exe), "--fit", "on"])
    assert time.monotonic() - t0 < 20 and not marker.exists()
    assert pr.devices["Vulkan1"]["used"] == 136 and pr.mmproj_mib == 812.25


@pytest.mark.skipif(sys.platform == "win32", reason="uses a script as the engine")
def test_server_probe_reports_engine_errors(tmp_path: Path):
    exe = tmp_path / "llama-server"
    exe.write_text(f"#!{sys.executable}\nprint('0.00.001.000 E main: error: failed to load model', flush=True)\n")
    exe.chmod(0o755)
    with pytest.raises(RuntimeError, match="failed to load model"):
        server_probe([str(exe)])


def test_settings_migration_from_1x(tmp_path: Path):
    old = {"version": 1, "hardware": {"vram_margin_mib": 1024, "vram_margin_per_device": {"Vulkan0": 1500}}}
    f = tmp_path / "settings.json"
    f.write_text(json.dumps(old))
    s = SettingsStore(f).settings
    assert s.hardware.vram_margin_mib == 512
    assert s.hardware.vram_margin_per_device == {"Vulkan0": 1500}  # explicit per-GPU values are kept
    assert json.loads(f.read_text())["version"] == Settings().version
    custom = {"version": 1, "hardware": {"vram_margin_mib": 800}}
    assert migrate(custom) and custom["hardware"]["vram_margin_mib"] == 800
    assert not migrate({"version": Settings().version})
