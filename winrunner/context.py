"""Shared application state passed to the HTTP layers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import SettingsStore
from .engine import EngineManager
from .events import EventBus, RequestTracker
from .hardware import HardwareMonitor
from .library import ModelLibrary
from .manager import ModelManager
from .paths import DataPaths
from .vision import ImageNormalizer


@dataclass
class AppContext:
    paths: DataPaths
    store: SettingsStore
    library: ModelLibrary
    engines: EngineManager
    monitor: HardwareMonitor
    bus: EventBus
    tracker: RequestTracker
    manager: ModelManager
    http: httpx.AsyncClient
    started_at: float = field(default_factory=time.time)
    metrics_history: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def normalizer(self) -> ImageNormalizer:
        s = self.store.settings.server
        return ImageNormalizer(self.http, max_edge=s.max_image_edge, fetch_remote=s.fetch_remote_images)
