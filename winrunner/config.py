"""Persistent settings.

Settings are stored as JSON in ``<data>/settings.json``. Unknown keys are ignored
and missing keys take their defaults, so files written by older or newer
versions load cleanly. Writes are atomic (write to a temp file, then replace).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .paths import default_model_dirs

log = logging.getLogger("winrunner.config")

KV_TYPES = ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"]
QUANTIZED_KV = {"q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class ServerSettings(_Model):
    host: str = "0.0.0.0"
    port: int = 5070
    api_enabled: bool = True
    cors: bool = True
    api_key: str = ""
    lan_control_panel: bool = False
    jit_loading: bool = True
    auto_evict: bool = True
    idle_unload_minutes: int = 0
    internal_stream_aggregation: bool = True
    max_image_edge: int = 0
    fetch_remote_images: bool = True
    request_history: int = 500

    @field_validator("port")
    @classmethod
    def _port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be 1-65535")
        return v


class EngineSettings(_Model):
    backend: Literal["vulkan", "rocm", "cpu", "custom"] = "vulkan"
    engine_path: str = ""  # explicit llama-server path; empty = managed engine
    active_engine: str = ""  # name of the managed engine directory in use
    process_priority: Literal["normal", "above_normal", "high"] = "above_normal"
    log_verbosity: int = 4
    use_engine_fit: bool = True
    load_timeout_s: int = 900


class HardwareSettings(_Model):
    vram_margin_mib: int = 1024
    vram_margin_per_device: dict[str, int] = Field(default_factory=dict)
    telemetry_interval_s: float = 1.0


class LoadParams(_Model):
    """Model load configuration (global defaults and per-model overrides)."""

    context_length: int = 65536
    allow_context_over_train: bool = False
    gpu_offload: Literal["auto", "manual"] = "auto"
    n_gpu_layers: int = -1  # manual mode: -1 = all layers
    n_cpu_moe: int = 0  # manual mode
    devices: list[str] = Field(default_factory=list)  # empty = all GPUs
    split_mode: Literal["layer", "row", "none", "tensor"] = "layer"
    tensor_split: list[float] = Field(default_factory=list)  # empty = automatic
    main_gpu: int = 0
    flash_attn: Literal["auto", "on", "off"] = "auto"
    kv_cache_type: str = "auto"  # auto | one of KV_TYPES
    kv_cache_type_v: str = ""  # empty = same as K
    kv_offload: bool = True
    batch_size: int = 2048
    ubatch_size: int = 512
    parallel: int = -1  # -1 = engine auto
    threads: int = 0  # 0 = engine auto (physical cores)
    threads_batch: int = 0
    load_mode: str = "auto"  # auto | mmap | none | mlock | mmap+mlock | dio
    mmproj: str = ""  # "" = auto-pair from folder, "none" = disabled, else path
    mmproj_offload: bool = True
    image_min_tokens: int = 0
    image_max_tokens: int = 0
    reasoning_format: Literal["auto", "deepseek", "deepseek-legacy", "none"] = "auto"
    reasoning: Literal["auto", "on", "off"] = "auto"
    reasoning_budget: int = -1
    chat_template_mode: Literal["gguf", "builtin", "custom"] = "gguf"
    chat_template_builtin: str = ""
    chat_template_custom: str = ""
    chat_template_kwargs: str = ""
    context_shift: bool = False
    swa_full: bool = False
    cache_ram_mib: int = 8192
    cache_reuse: int = 0
    draft_model: str = ""  # library model id
    draft_max: int = 0
    spec_type: str = ""
    rope_scaling: str = ""  # "" = from model
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    yarn_orig_ctx: int = 0
    embeddings: bool = False
    extra_args: str = ""

    @field_validator("context_length")
    @classmethod
    def _ctx(cls, v: int) -> int:
        if v < 256:
            raise ValueError("context_length must be >= 256")
        return v

    @field_validator("kv_cache_type")
    @classmethod
    def _kv(cls, v: str) -> str:
        v = v.lower()
        if v != "auto" and v not in KV_TYPES:
            raise ValueError(f"unsupported KV cache type {v}")
        return v

    @field_validator("kv_cache_type_v")
    @classmethod
    def _kvv(cls, v: str) -> str:
        v = v.lower()
        if v and v not in KV_TYPES:
            raise ValueError(f"unsupported KV cache type {v}")
        return v


SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "seed",
)


class ModelProfile(_Model):
    alias: str = ""
    load: dict[str, Any] = Field(default_factory=dict)
    sampling: dict[str, Any] = Field(default_factory=dict)
    last_loaded: float = 0.0
    load_count: int = 0
    last_load_seconds: float = 0.0
    mmproj_est_mib: float = 0.0


class LibrarySettings(_Model):
    model_dirs: list[str] = Field(default_factory=default_model_dirs)
    download_dir: str = ""
    hf_token: str = ""


class UISettings(_Model):
    theme: str = "classic"
    custom_themes: dict[str, dict[str, str]] = Field(default_factory=dict)
    crt_effect: bool = True
    glow: bool = True
    animations: Literal["full", "reduced", "off"] = "full"
    ui_scale: float = 1.0
    boot_sequence: bool = True
    token_boundaries: bool = True
    sidebar_position: Literal["right", "bottom"] = "right"
    clock_24h: bool = True


class StartupSettings(_Model):
    open_ui: Literal["window", "browser", "none"] = "window"
    autoload_last_model: bool = False
    last_model: str = ""


class Settings(_Model):
    version: int = 1
    server: ServerSettings = Field(default_factory=ServerSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    defaults: LoadParams = Field(default_factory=LoadParams)
    library: LibrarySettings = Field(default_factory=LibrarySettings)
    ui: UISettings = Field(default_factory=UISettings)
    startup: StartupSettings = Field(default_factory=StartupSettings)
    models: dict[str, ModelProfile] = Field(default_factory=dict)


# Maps that are replaced as a whole when present in a settings patch (so keys can be removed).
REPLACE_KEYS = {"custom_themes", "vram_margin_per_device"}


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` into a copy of ``base``.

    Dicts are merged, everything else (including lists) is replaced; the maps in
    ``REPLACE_KEYS`` are replaced too.
    """
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE_KEYS:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    # os.replace can transiently fail on Windows if another process (antivirus,
    # indexer) holds the target open; retry briefly.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


class SettingsStore:
    """Thread-safe owner of the persisted :class:`Settings`."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._listeners: list[Callable[[Settings, dict], None]] = []
        self._settings = self._load()

    def _load(self) -> Settings:
        if not self.path.exists():
            s = Settings()
            _atomic_write_json(self.path, s.model_dump(mode="json"))
            return s
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return Settings.model_validate(raw)
        except Exception as exc:  # corrupt or invalid file: keep a backup, start clean
            backup = self.path.with_suffix(f".invalid-{int(time.time())}.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            log.error("settings file invalid (%s); backed up to %s and reset", exc, backup)
            s = Settings()
            _atomic_write_json(self.path, s.model_dump(mode="json"))
            return s

    @property
    def settings(self) -> Settings:
        return self._settings

    def snapshot(self) -> dict:
        with self._lock:
            return self._settings.model_dump(mode="json")

    def on_change(self, fn: Callable[[Settings, dict], None]) -> None:
        self._listeners.append(fn)

    def update(self, patch: dict) -> Settings:
        """Apply a partial update (validated) and persist it."""
        with self._lock:
            merged = deep_merge(self._settings.model_dump(mode="json"), patch)
            new = Settings.model_validate(merged)
            self._settings = new
            _atomic_write_json(self.path, new.model_dump(mode="json"))
        for fn in list(self._listeners):
            try:
                fn(new, patch)
            except Exception:  # listener errors must not break saving
                log.exception("settings listener failed")
        return new

    def save(self) -> None:
        with self._lock:
            _atomic_write_json(self.path, self._settings.model_dump(mode="json"))

    # ----- per-model profiles -------------------------------------------------

    def profile(self, key: str) -> ModelProfile:
        with self._lock:
            p = self._settings.models.get(key)
            return p.model_copy(deep=True) if p else ModelProfile()

    def set_profile(self, key: str, profile: ModelProfile) -> None:
        with self._lock:
            models = dict(self._settings.models)
            models[key] = profile
            self._settings.models = models
            self.save()

    def effective_load_params(self, key: str, overrides: dict | None = None) -> LoadParams:
        """Global defaults <- saved per-model profile <- one-off overrides."""
        with self._lock:
            base = self._settings.defaults.model_dump(mode="json")
            prof = self._settings.models.get(key)
            if prof:
                base.update({k: v for k, v in prof.load.items() if k in LoadParams.model_fields})
        if overrides:
            base.update({k: v for k, v in overrides.items() if k in LoadParams.model_fields})
        return LoadParams.model_validate(base)
