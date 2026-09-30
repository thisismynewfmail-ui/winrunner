"""llama.cpp engine management.

* Discovers ``llama-server`` builds (managed ones in ``<data>/engines`` or a
  user supplied path).
* Probes a build: version, supported command line flags (parsed from
  ``--help`` so the argument builder adapts to the exact build), and the GPU
  devices it sees with their free memory (``--list-devices``).
* Downloads official release builds from GitHub (Vulkan, ROCm/HIP, CPU).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from .paths import EXE_SUFFIX, IS_WINDOWS
from .util import strip_ansi

log = logging.getLogger("winrunner.engine")

GITHUB_REPO = "ggml-org/llama.cpp"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}"
GITHUB_WEB = f"https://github.com/{GITHUB_REPO}"

_CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

# Release asset patterns per (platform, backend)
ASSET_PATTERNS: dict[tuple[str, str], list[str]] = {
    ("win", "vulkan"): [r"-bin-win-vulkan-x64\.zip$"],
    ("win", "rocm"): [r"-bin-win-rocm-[\d.]+-x64\.zip$", r"-bin-win-hip[-\w.]*-x64\.zip$"],
    ("win", "cpu"): [r"-bin-win-cpu-x64\.zip$", r"-bin-win-avx2-x64\.zip$"],
    ("linux", "vulkan"): [r"-bin-ubuntu-vulkan-x64\.(tar\.gz|zip)$"],
    ("linux", "rocm"): [r"-bin-ubuntu-rocm-[\d.]+-x64\.(tar\.gz|zip)$"],
    ("linux", "cpu"): [r"-bin-ubuntu-x64\.(tar\.gz|zip)$"],
}
BACKEND_LABELS = {"vulkan": "Vulkan", "rocm": "ROCm (HIP)", "cpu": "CPU", "cuda": "CUDA", "custom": "Custom"}


def _plat_key() -> str:
    return "win" if IS_WINDOWS else "linux"


def engine_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for llama.cpp processes."""
    env = dict(os.environ)
    if not IS_WINDOWS:
        # Vulkan on Linux (RADV): give llama.cpp's buffers the highest residency priority, so that when the
        # desktop or a browser needs VRAM the driver moves their memory to system RAM, not the model weights
        # (weights in system RAM would be read over PCIe for every token).
        env.setdefault("GGML_VK_ENABLE_MEMORY_PRIORITY", "1")
        # AMD's own Vulkan driver (AMDVLK), if installed, hands over to Mesa's RADV: faster for llama.cpp
        env.setdefault("AMD_VULKAN_ICD", "RADV")
    if extra:
        env.update(extra)
    return env


def _run(args: list[str], timeout: float = 30.0, cwd: str | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            cwd=cwd,
            env=engine_env(),
            creationflags=_CREATE_NO_WINDOW,
        )
        return r.returncode, strip_ansi(r.stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        return -1, strip_ansi(out) + "\n[timed out]"
    except OSError as exc:
        return -2, str(exc)


@dataclass
class EngineDevice:
    name: str  # Vulkan0, ROCm1, CUDA0 ...
    description: str
    total_mib: int
    free_mib: int
    details: str = ""


@dataclass
class EngineInfo:
    path: str
    name: str
    backend: str
    version: str = ""
    build: int = 0
    commit: str = ""
    flags: list[str] = field(default_factory=list)
    spec_types: list[str] = field(default_factory=list)
    builtin_templates: list[str] = field(default_factory=list)
    fa_tristate: bool = True
    ngl_all: bool = True
    fit_params: str = ""
    bench: str = ""
    probe_error: str = ""
    probed_at: float = 0.0

    def has(self, *flags: str) -> bool:
        s = set(self.flags)
        return any(f in s for f in flags)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backend_label"] = BACKEND_LABELS.get(self.backend, self.backend)
        d["flag_count"] = len(self.flags)
        return d


def detect_backend(engine_dir: Path) -> str:
    names = {p.name.lower() for p in engine_dir.iterdir()} if engine_dir.is_dir() else set()
    joined = " ".join(names)
    if "ggml-hip" in joined or "amdhip64" in joined:
        return "rocm"
    if "ggml-vulkan" in joined:
        return "vulkan"
    if "ggml-cuda" in joined:
        return "cuda"
    return "cpu"


_FLAG_RE = re.compile(r"(?:^|[\s,])(--?[a-zA-Z][a-zA-Z0-9-]*)")
_VERSION_RE = re.compile(r"version:\s*(\S+)(?:\s*\((?:build\s*)?(\d+)?[,\s]*(?:commit\s*)?([0-9a-f]{6,})?\))?")
_DEVICE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z_]*\d+):\s*(.*?)\s*\((\d+) MiB,\s*(\d+) MiB free\)\s*$")


def parse_help_flags(text: str) -> tuple[list[str], list[str], list[str]]:
    flags: set[str] = set()
    spec: list[str] = []
    builtins: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("-") and not stripped.startswith("- ") and not stripped.startswith("---"):
            # The description starts after a run of 2+ spaces that is not
            # followed by another flag ("-fa,   --flash-attn [on|off]   set ...").
            cut = re.search(r"\s{2,}(?=[^\s-])", stripped)
            head = stripped[: cut.start()] if cut else stripped
            for m in _FLAG_RE.finditer(" " + head):
                flags.add(m.group(1))
            if head.startswith("--spec-type"):
                parts = head.split(None, 1)
                if len(parts) == 2:
                    spec = [x for x in parts[1].split(",") if x]
        if "list of built-in templates:" in line and not builtins:
            chunk = []
            for nxt in lines[i + 1 : i + 12]:
                if "(env:" in nxt or nxt.strip().startswith("-"):
                    break
                chunk.append(nxt.strip())
            builtins = [x.strip() for x in " ".join(chunk).split(",") if x.strip()]
    return sorted(flags), spec, builtins


def parse_version(text: str) -> tuple[str, int, str]:
    for line in text.splitlines():
        m = _VERSION_RE.search(line)
        if m:
            ver = m.group(1)
            build = int(m.group(2)) if m.group(2) else 0
            if not build and ver.isdigit():
                build = int(ver)
            return ver, build, m.group(3) or ""
    return "", 0, ""


def parse_devices(text: str) -> list[EngineDevice]:
    devs: list[EngineDevice] = []
    details: dict[int, str] = {}
    for line in text.splitlines():
        # ggml_vulkan: 0 = AMD Radeon RX 6800 (AMD proprietary driver) | uma: 0 | fp16: 1 | ...
        m = re.match(r"^\s*ggml_vulkan:\s*(\d+)\s*=\s*(.*)$", line)
        if m:
            details[int(m.group(1))] = m.group(2).strip()
            continue
        # Device 0: AMD Radeon RX 6800, gfx1030 (0x1030), VMM: no, Wave Size: 32
        m = re.match(r"^\s*Device\s+(\d+):\s*(.*)$", line)
        if m:
            details[int(m.group(1))] = m.group(2).strip()
            continue
        m = _DEVICE_RE.match(line)
        if m:
            name = m.group(1)
            if name.upper().startswith("CPU"):
                continue
            devs.append(EngineDevice(name, m.group(2), int(m.group(3)), int(m.group(4))))
    for d in devs:
        idx = re.search(r"(\d+)$", d.name)
        if idx and int(idx.group(1)) in details:
            d.details = details[int(idx.group(1))]
    return devs


class EngineManager:
    def __init__(self, engines_dir: Path, downloads_dir: Path):
        self.engines_dir = engines_dir
        self.downloads_dir = downloads_dir
        self._probe_cache: dict[tuple[str, float], EngineInfo] = {}
        self._lock = threading.Lock()
        self._devices_cache: tuple[float, str, list[EngineDevice]] | None = None
        self.install_state: dict[str, Any] = {"active": False}

    # ----- discovery --------------------------------------------------------------

    def installed(self) -> list[dict[str, Any]]:
        out = []
        if not self.engines_dir.is_dir():
            return out
        for d in sorted(self.engines_dir.iterdir()):
            if not d.is_dir():
                continue
            exe = self._find_server(d)
            if not exe:
                continue
            manifest = {}
            mf = d / "engine.json"
            if mf.exists():
                try:
                    manifest = json.loads(mf.read_text(encoding="utf-8"))
                except ValueError:
                    manifest = {}
            out.append(
                {
                    "name": d.name,
                    "dir": str(d),
                    "server": str(exe),
                    "backend": manifest.get("backend") or detect_backend(exe.parent),
                    "tag": manifest.get("tag", ""),
                    "installed_at": manifest.get("installed_at", 0),
                    "asset": manifest.get("asset", ""),
                }
            )
        return out

    @staticmethod
    def _find_server(root: Path) -> Path | None:
        target = "llama-server" + EXE_SUFFIX
        direct = root / target
        if direct.is_file():
            return direct
        for p in sorted(root.rglob(target)):
            if p.is_file():
                return p
        return None

    def resolve_server(self, engine_path: str, active_engine: str, backend: str) -> Path | None:
        if engine_path:
            p = Path(os.path.expandvars(engine_path)).expanduser()
            if p.is_dir():
                p = self._find_server(p) or p
            return p if p.is_file() else None
        inst = self.installed()
        if active_engine:
            for e in inst:
                if e["name"] == active_engine:
                    return Path(e["server"])
        same = [e for e in inst if e["backend"] == backend] or inst
        if same:
            same.sort(key=lambda e: (_tag_num(e["tag"]) or _tag_num(e["name"]), e["installed_at"]), reverse=True)
            return Path(same[0]["server"])
        w = shutil.which("llama-server")
        return Path(w) if w else None

    # ----- probing ----------------------------------------------------------------

    def probe(self, server: Path, force: bool = False) -> EngineInfo:
        try:
            mtime = server.stat().st_mtime
        except OSError as exc:
            return EngineInfo(path=str(server), name=server.parent.name, backend="?", probe_error=str(exc))
        key = (str(server), mtime)
        with self._lock:
            if not force and key in self._probe_cache:
                return self._probe_cache[key]
        cwd = str(server.parent)
        info = EngineInfo(path=str(server), name=server.parent.name, backend=detect_backend(server.parent))
        rc, out = _run([str(server), "--version"], timeout=30, cwd=cwd)
        info.version, info.build, info.commit = parse_version(out)
        if rc not in (0, 1) and not info.version:
            info.probe_error = f"--version failed ({rc}): {out.strip()[-400:]}"
        rc, out = _run([str(server), "--help"], timeout=30, cwd=cwd)
        info.flags, info.spec_types, info.builtin_templates = parse_help_flags(out)
        info.fa_tristate = bool(re.search(r"--flash-attn\s+\[?on\|off", out))
        info.ngl_all = "'all'" in out
        if not info.flags:
            info.probe_error = info.probe_error or f"--help produced no flags ({rc}): {out.strip()[-400:]}"
        for tool, attr in (("llama-fit-params", "fit_params"), ("llama-bench", "bench")):
            p = server.parent / (tool + EXE_SUFFIX)
            if p.is_file():
                setattr(info, attr, str(p))
        info.probed_at = time.time()
        with self._lock:
            self._probe_cache[key] = info
        return info

    def list_devices(self, server: Path, max_age: float = 0.0) -> tuple[list[EngineDevice], str]:
        """GPU devices visible to the engine, with current free memory."""
        now = time.time()
        with self._lock:
            c = self._devices_cache
            if max_age and c and c[1] == str(server) and now - c[0] <= max_age:
                return c[2], ""
        rc, out = _run([str(server), "--list-devices"], timeout=60, cwd=str(server.parent))
        devs = parse_devices(out)
        err = "" if devs or rc == 0 else out.strip()[-600:]
        with self._lock:
            self._devices_cache = (now, str(server), devs)
        return devs, err

    # ----- installation -----------------------------------------------------------

    def _release_entry(self, rel: dict[str, Any]) -> dict[str, Any]:
        assets = [
            {"name": a["name"], "size": a.get("size", 0), "url": a["browser_download_url"]}
            for a in rel.get("assets", [])
        ]
        return {
            "tag": rel.get("tag_name", ""),
            "published": rel.get("published_at", ""),
            "prerelease": rel.get("prerelease", False),
            "assets": assets,
            "backends": {b: self.pick_asset(assets, b) for b in ("vulkan", "rocm", "cpu")},
        }

    async def _guessed_release(self, c: httpx.AsyncClient, tag: str) -> dict[str, Any]:
        """Release entry built from the official asset names (when the GitHub API is unavailable)."""
        ext = ".zip" if IS_WINDOWS else ".tar.gz"
        plat = "win" if IS_WINDOWS else "ubuntu"
        guesses = {
            "vulkan": f"llama-{tag}-bin-{plat}-vulkan-x64{ext}",
            "cpu": f"llama-{tag}-bin-{plat}-{'cpu-' if IS_WINDOWS else ''}x64{ext}",
        }
        assets = []
        for name in guesses.values():
            url = f"{GITHUB_WEB}/releases/download/{tag}/{name}"
            h = await c.head(url)
            if h.status_code < 400:
                assets.append({"name": name, "size": int(h.headers.get("content-length", 0)), "url": url})
        return {"tag": tag, "published": "", "prerelease": False, "assets": assets,
                "backends": {b: self.pick_asset(assets, b) for b in ("vulkan", "rocm", "cpu")}}

    async def releases(self, limit: int = 8) -> list[dict[str, Any]]:
        """Recent releases with assets matching this platform."""
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "WinRunner"}
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as c:
            try:
                r = await c.get(f"{GITHUB_API}/releases", params={"per_page": limit})
                r.raise_for_status()
                return [self._release_entry(rel) for rel in r.json()]
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("GitHub API unavailable (%s); falling back to release redirect", exc)
            # Fallback: find the latest tag from the web redirect and guess asset names.
            r = await c.get(f"{GITHUB_WEB}/releases/latest", follow_redirects=False)
            loc = r.headers.get("location", "")
            m = re.search(r"/tag/([^/?#]+)", loc)
            if not m:
                raise RuntimeError("could not determine the latest llama.cpp release")
            return [await self._guessed_release(c, m.group(1))]

    async def release(self, tag: str) -> dict[str, Any]:
        """One release by tag (e.g. b11269), with assets matching this platform."""
        if not re.match(r"^[\w.+-]{1,64}$", tag):
            raise ValueError(f"invalid release tag {tag!r}")
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "WinRunner"}
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as c:
            try:
                r = await c.get(f"{GITHUB_API}/releases/tags/{tag}")
                if r.status_code == 404:
                    raise RuntimeError(f"llama.cpp release {tag} does not exist")
                r.raise_for_status()
                return self._release_entry(r.json())
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("GitHub API unavailable (%s); using the official asset names for %s", exc, tag)
            rel = await self._guessed_release(c, tag)
            if not rel["assets"]:
                raise RuntimeError(f"no downloads found for llama.cpp release {tag}")
            return rel

    @staticmethod
    def pick_asset(assets: list[dict[str, Any]], backend: str) -> dict[str, Any] | None:
        for pat in ASSET_PATTERNS.get((_plat_key(), backend), []):
            rx = re.compile(pat, re.IGNORECASE)
            for a in assets:
                if rx.search(a["name"]) and not a["name"].startswith("cudart"):
                    return a
        return None

    async def install(self, tag: str, backend: str, asset: dict[str, Any],
                      progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        if self.install_state.get("active"):
            raise RuntimeError("an engine installation is already running")
        name = f"{tag}-{backend}"
        target = self.engines_dir / name
        self.install_state = {"active": True, "tag": tag, "backend": backend, "phase": "download",
                              "done": 0, "total": asset.get("size", 0), "asset": asset["name"]}
        progress(dict(self.install_state))
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        dl = self.downloads_dir / asset["name"]
        part = dl.with_suffix(dl.suffix + ".part")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=120), follow_redirects=True,
                                         headers={"User-Agent": "WinRunner"}) as c:
                async with c.stream("GET", asset["url"]) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0)) or asset.get("size", 0)
                    self.install_state["total"] = total
                    done = 0
                    last = 0.0
                    with open(part, "wb") as f:
                        async for chunk in r.aiter_bytes(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if time.monotonic() - last > 0.25:
                                last = time.monotonic()
                                self.install_state["done"] = done
                                progress(dict(self.install_state))
            os.replace(part, dl)
            self.install_state.update({"phase": "extract", "done": self.install_state["total"]})
            progress(dict(self.install_state))
            tmp = self.engines_dir / (name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            await asyncio.to_thread(_extract, dl, tmp)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            os.replace(tmp, target)
            server = self._find_server(target)
            if not server:
                raise RuntimeError("archive did not contain llama-server")
            if not IS_WINDOWS:
                for p in server.parent.iterdir():
                    if p.is_file() and (p.name.startswith(("llama-", "rpc-", "ggml-rpc")) or p.name == "llama"):
                        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            manifest = {"tag": tag, "backend": backend, "asset": asset["name"], "installed_at": time.time(),
                        "server": str(server.relative_to(target))}
            (target / "engine.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            try:
                dl.unlink()
            except OSError:
                pass
            self.install_state = {"active": False, "phase": "done", "tag": tag, "backend": backend, "name": name}
            progress(dict(self.install_state))
            return {"name": name, "server": str(server)}
        except Exception as exc:
            self.install_state = {"active": False, "phase": "error", "error": str(exc), "tag": tag, "backend": backend}
            progress(dict(self.install_state))
            raise

    def remove(self, name: str) -> None:
        d = (self.engines_dir / name).resolve()
        if d.parent != self.engines_dir.resolve() or not d.is_dir():
            raise ValueError("unknown engine")
        shutil.rmtree(d)


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                out = (dest / m.filename).resolve()
                if not str(out).startswith(str(root)):
                    raise RuntimeError(f"unsafe path in archive: {m.filename}")
            z.extractall(dest)
    else:
        with tarfile.open(archive, "r:*") as t:
            for m in t.getmembers():
                out = (dest / m.name).resolve()
                if not str(out).startswith(str(root)) or m.issym() and os.path.isabs(m.linkname):
                    raise RuntimeError(f"unsafe path in archive: {m.name}")
            if sys.version_info >= (3, 12):
                t.extractall(dest, filter="tar")
            else:
                t.extractall(dest)


def _tag_num(s: str) -> int:
    m = re.search(r"b(\d{3,})", s or "")
    return int(m.group(1)) if m else 0
