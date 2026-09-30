"""Filesystem locations used by WinRunner.

WinRunner is a portable application: everything it writes lives below one data
directory (``<install dir>/data`` by default, overridable with ``--data-dir`` or
the ``WINRUNNER_DATA`` environment variable).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "static"

IS_WINDOWS = sys.platform == "win32"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


class DataPaths:
    """Resolved directories inside the data directory."""

    def __init__(self, root: Path | str | None = None):
        if root is None:
            root = os.environ.get("WINRUNNER_DATA") or (APP_ROOT / "data")
        self.root = Path(root).expanduser().resolve()
        self.settings_file = self.root / "settings.json"
        self.cache_dir = self.root / "cache"
        self.gguf_index = self.cache_dir / "gguf_index.json"
        self.plan_cache = self.cache_dir / "plan_cache.json"
        self.engines_dir = self.root / "engines"
        self.logs_dir = self.root / "logs"
        self.chats_dir = self.root / "chats"
        self.templates_dir = self.root / "templates"
        self.bench_file = self.root / "benchmarks.json"
        self.downloads_tmp = self.root / "downloads"
        self.default_models_dir = APP_ROOT / "models"

    def ensure(self) -> None:
        for d in (
            self.root,
            self.cache_dir,
            self.engines_dir,
            self.logs_dir,
            self.chats_dir,
            self.templates_dir,
            self.downloads_tmp,
        ):
            d.mkdir(parents=True, exist_ok=True)


def default_model_dirs() -> list[str]:
    """Model folders scanned out of the box.

    Includes LM Studio's model folders so existing downloads are reused.
    """
    home = Path.home()
    candidates = [
        APP_ROOT / "models",
        home / ".lmstudio" / "models",
        home / ".cache" / "lm-studio" / "models",
    ]
    if not IS_WINDOWS:
        candidates.append(home / ".cache" / "llama.cpp")  # llama-server -hf downloads
    return [str(p) for p in candidates]
