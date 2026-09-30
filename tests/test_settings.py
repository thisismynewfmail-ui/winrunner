import json

from winrunner.config import DEFAULT_VRAM_MARGIN_MIB, SETTINGS_VERSION, SettingsStore


def test_old_default_margin_is_migrated(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"version": 1, "hardware": {"vram_margin_mib": 1024},
                             "defaults": {"context_length": 32768}}))
    s = SettingsStore(f).settings
    assert s.version == SETTINGS_VERSION
    assert s.hardware.vram_margin_mib == DEFAULT_VRAM_MARGIN_MIB == 256
    assert s.defaults.context_length == 32768 and s.defaults.context_fit == "fill"
    saved = json.loads(f.read_text())
    assert saved["version"] == SETTINGS_VERSION and saved["hardware"]["vram_margin_mib"] == 256


def test_custom_margins_are_kept(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"version": 1, "hardware": {"vram_margin_mib": 1536,
                                                        "vram_margin_per_device": {"Vulkan0": 1024}}}))
    s = SettingsStore(f).settings
    assert s.hardware.vram_margin_mib == 1536 and s.hardware.vram_margin_per_device == {"Vulkan0": 1024}
    assert s.version == SETTINGS_VERSION


def test_current_file_is_not_rewritten(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"version": SETTINGS_VERSION, "hardware": {"vram_margin_mib": 1024}}))
    before = f.read_text()
    assert SettingsStore(f).settings.hardware.vram_margin_mib == 1024  # set by the user in the current version
    assert f.read_text() == before


def test_new_settings_use_new_defaults(tmp_path):
    s = SettingsStore(tmp_path / "settings.json").settings
    assert s.hardware.vram_margin_mib == 256 and s.version == SETTINGS_VERSION
    assert s.defaults.context_fit == "fill" and s.defaults.gpu_offload == "auto"
