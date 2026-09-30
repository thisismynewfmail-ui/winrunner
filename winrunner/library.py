"""Model library: discovers GGUF files, pairs vision projectors, assigns IDs.

Folders are scanned recursively. The LM Studio layout
(``models/<publisher>/<repo>/<file>.gguf``) is understood, so an existing LM
Studio model folder can be used directly. Parsed headers are cached in
``<data>/cache/gguf_index.json`` and only re-read when size or mtime changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gguf import SPLIT_RE, GGUFError, ModelInfo, summarize
from .templates import analyze as analyze_template
from .util import slugify

log = logging.getLogger("winrunner.library")

INDEX_VERSION = 4  # 4: ModelInfo.layer_parts
_MMPROJ_NAME_RE = re.compile(r"mmproj|vision[-_]?proj|clip[-_]", re.IGNORECASE)
_PRECISION_RANK = ["f16", "bf16", "f32", "q8_0", "q6_k", "q5_k", "q4_k", "q4_0"]


@dataclass
class ModelEntry:
    id: str
    info: ModelInfo
    publisher: str
    repo: str
    mmproj_candidates: list[str] = field(default_factory=list)
    mmproj_default: str = ""
    mmproj_info: dict[str, Any] | None = None
    template: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return self.info.path

    @property
    def has_vision(self) -> bool:
        return bool(self.mmproj_info and self.mmproj_info.get("has_vision"))

    @property
    def has_audio(self) -> bool:
        return bool(self.mmproj_info and self.mmproj_info.get("has_audio"))

    @property
    def lms_type(self) -> str:
        if self.info.kind == "embedding":
            return "embeddings"
        return "vlm" if self.has_vision else "llm"

    def to_summary(self) -> dict[str, Any]:
        i = self.info
        return {
            "id": self.id,
            "path": i.path,
            "file_name": i.file_name,
            "publisher": self.publisher,
            "repo": self.repo,
            "kind": i.kind,
            "type": self.lms_type,
            "name": i.name or Path(i.file_name).stem,
            "architecture": i.architecture,
            "quant": i.quant,
            "size_label": i.size_label,
            "n_params": i.n_params,
            "file_size": i.file_size,
            "n_layer": i.n_layer,
            "context_length": i.context_length,
            "is_moe": i.expert_count > 0,
            "expert_count": i.expert_count,
            "expert_used_count": i.expert_used_count,
            "vision": self.has_vision,
            "audio": self.has_audio,
            "mmproj": self.mmproj_default,
            "mmproj_candidates": self.mmproj_candidates,
            "tools": bool(self.template.get("tools")),
            "reasoning": bool(self.template.get("reasoning")),
            "template_family": self.template.get("family", ""),
            "split_count": max(1, len(i.split_files)),
            "mtime": i.mtime,
            "error": i.error,
        }

    def to_detail(self) -> dict[str, Any]:
        d = self.to_summary()
        i = self.info
        d.update(
            {
                "info": {
                    k: v
                    for k, v in i.to_dict().items()
                    if k not in ("layer_bytes", "layer_expert_bytes", "layer_ffn_bytes", "metadata_brief")
                },
                "metadata": i.metadata_brief,
                "template_analysis": self.template,
                "mmproj_info": self.mmproj_info,
                "mmproj_compatible": self.mmproj_compatible(),
            }
        )
        return d

    def mmproj_compatible(self, mm: dict | None = None) -> bool | None:
        mm = mm if mm is not None else self.mmproj_info
        if not mm or not mm.get("projection_dim") or not self.info.n_embd:
            return None
        return int(mm["projection_dim"]) == int(self.info.n_embd)


def _is_mmproj_file(path: Path) -> bool:
    return bool(_MMPROJ_NAME_RE.search(path.name))


def _precision_score(name: str) -> int:
    n = name.lower()
    for i, tag in enumerate(_PRECISION_RANK):
        if tag in n:
            return i
    return len(_PRECISION_RANK)


_QUANT_TOKEN = re.compile(r"^(i?q\d\w*|f16|bf16|f32|fp16|k|m|s|l|xs|xxs|xl|nl|\d)$")


def _name_similarity(model_file: str, mmproj_file: str) -> int:
    """Number of model-identity tokens shared by a model file and an mmproj file.

    Quantization tokens are ignored: the projector precision is chosen
    separately (F16 preferred for vision quality).
    """
    def tok(s: str) -> set[str]:
        return {t for t in re.split(r"[-_.\s]+", s.lower())
                if t and t not in ("gguf", "mmproj", "model") and not _QUANT_TOKEN.match(t)}

    return len(tok(model_file) & tok(mmproj_file))


class ModelLibrary:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._lock = threading.RLock()
        self._entries: dict[str, ModelEntry] = {}
        self._by_path: dict[str, ModelEntry] = {}
        self._infos: dict[str, ModelInfo] = {}  # all parsed files incl. mmproj
        self._aliases: dict[str, str] = {}  # lowercase alias -> model path
        self._cache: dict[str, dict] = self._load_index()
        self.last_scan: float = 0.0
        self.scan_errors: list[str] = []
        self.scanning = False

    # ----- persistence ----------------------------------------------------------

    def _load_index(self) -> dict[str, dict]:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            if raw.get("version") == INDEX_VERSION:
                return raw.get("files", {})
        except (OSError, ValueError):
            pass
        return {}

    def _save_index(self) -> None:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": INDEX_VERSION, "files": self._cache}), encoding="utf-8")
            os.replace(tmp, self.index_path)
        except OSError as exc:
            log.warning("could not save library index: %s", exc)

    # ----- scanning -------------------------------------------------------------

    @staticmethod
    def _walk(dirs: list[str]) -> list[Path]:
        found: list[Path] = []
        seen: set[str] = set()
        for d in dirs:
            root = Path(os.path.expandvars(d)).expanduser()
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
                dirnames[:] = [x for x in dirnames if not x.startswith(".")]
                for fn in filenames:
                    if not fn.lower().endswith(".gguf"):
                        continue
                    m = SPLIT_RE.search(fn)
                    if m and int(m.group(1)) != 1:
                        continue  # only the first part of split models
                    p = Path(dirpath) / fn
                    key = os.path.normcase(str(p.resolve()))
                    if key not in seen:
                        seen.add(key)
                        found.append(p)
        return found

    def _parse(self, p: Path) -> ModelInfo | None:
        try:
            st = p.stat()
        except OSError:
            return None
        key = str(p)
        cached = self._cache.get(key)
        if cached and cached.get("size") == st.st_size and cached.get("mtime") == st.st_mtime:
            try:
                return ModelInfo.from_dict(cached["info"])
            except TypeError:
                pass
        try:
            info = summarize(p)
        except (GGUFError, OSError, ValueError, KeyError) as exc:
            self.scan_errors.append(f"{p}: {exc}")
            log.warning("failed to read %s: %s", p, exc)
            return None
        except Exception as exc:  # malformed files must never break the scan
            self.scan_errors.append(f"{p}: {type(exc).__name__}: {exc}")
            log.exception("unexpected error reading %s", p)
            return None
        self._cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "info": info.to_dict()}
        return info

    def scan(self, dirs: list[str], aliases: dict[str, str] | None = None) -> dict[str, Any]:
        """Rescan ``dirs``. ``aliases`` maps model path -> user alias."""
        t0 = time.time()
        with self._lock:
            self.scanning = True
            self.scan_errors = []
        try:
            files = self._walk(dirs)
            with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as ex:
                infos = [i for i in ex.map(self._parse, files) if i is not None]
            known = {i.path for i in infos}
            self._cache = {k: v for k, v in self._cache.items() if k in known}
            self._save_index()
            self._build(infos, aliases or {})
        finally:
            with self._lock:
                self.scanning = False
                self.last_scan = time.time()
        return {"models": len(self._entries), "files": len(files), "seconds": round(time.time() - t0, 3),
                "errors": list(self.scan_errors)}

    def _build(self, infos: list[ModelInfo], aliases: dict[str, str]) -> None:
        mmprojs = [i for i in infos if i.kind == "mmproj"]
        models = [i for i in infos if i.kind in ("llm", "embedding")]
        mm_by_dir: dict[str, list[ModelInfo]] = {}
        for m in mmprojs:
            mm_by_dir.setdefault(os.path.normcase(str(Path(m.path).parent)), []).append(m)

        # Stable IDs: lowercase file stem; qualify with publisher on collisions.
        def stem(i: ModelInfo) -> str:
            s = Path(i.file_name).name
            s = SPLIT_RE.sub("", s) if SPLIT_RE.search(s) else s[: -len(".gguf")]
            return slugify(s)

        def pub_repo(i: ModelInfo) -> tuple[str, str]:
            parent = Path(i.path).parent
            return slugify(parent.parent.name), slugify(parent.name)

        counts: dict[str, int] = {}
        for i in models:
            counts[stem(i)] = counts.get(stem(i), 0) + 1

        entries: dict[str, ModelEntry] = {}
        for i in models:
            publisher, repo = pub_repo(i)
            base = stem(i)
            mid = base if counts[base] == 1 else f"{publisher}/{base}"
            n = 2
            while mid in entries:
                mid = f"{base}-{n}"
                n += 1
            cands = mm_by_dir.get(os.path.normcase(str(Path(i.path).parent)), [])
            cands = sorted(
                cands,
                key=lambda m: (-_name_similarity(i.file_name, m.file_name), _precision_score(m.file_name), m.file_name),
            )
            e = ModelEntry(
                id=mid,
                info=i,
                publisher=publisher,
                repo=repo,
                mmproj_candidates=[m.path for m in cands],
                mmproj_default=cands[0].path if cands else "",
                mmproj_info=cands[0].mmproj if cands else None,
                template=analyze_template(i.chat_template),
            )
            entries[mid] = e

        with self._lock:
            self._entries = entries
            self._by_path = {os.path.normcase(e.path): e for e in entries.values()}
            self._infos = {os.path.normcase(i.path): i for i in infos}
            self._aliases = {a.lower(): p for p, a in aliases.items() if a}

    # ----- queries --------------------------------------------------------------

    def entries(self) -> list[ModelEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: e.id)

    def get(self, model_id: str) -> ModelEntry | None:
        with self._lock:
            return self._entries.get(model_id)

    def by_path(self, path: str) -> ModelEntry | None:
        with self._lock:
            return self._by_path.get(os.path.normcase(str(path)))

    def mmproj_info(self, path: str) -> dict | None:
        with self._lock:
            i = self._infos.get(os.path.normcase(str(path)))
        if i is None and path and Path(path).is_file():
            try:
                i = summarize(path)
            except Exception:
                return None
        return i.mmproj if i and i.kind == "mmproj" else None

    def resolve(self, name: str | None) -> ModelEntry | None:
        """Map a client supplied model name to a library entry.

        Accepts IDs, user aliases, file names, paths, LM Studio style
        ``publisher/repo`` keys and unambiguous prefixes.
        """
        if not name:
            return None
        q = name.strip()
        ql = q.lower()
        with self._lock:
            if q in self._entries:
                return self._entries[q]
            alias_path = self._aliases.get(ql)
            if alias_path:
                e = self._by_path.get(os.path.normcase(alias_path))
                if e:
                    return e
            for e in self._entries.values():
                if e.id.lower() == ql:
                    return e
            e = self._by_path.get(os.path.normcase(q))
            if e:
                return e
            fn = ql[:-5] if ql.endswith(".gguf") else ql
            matches = [e for e in self._entries.values() if Path(e.info.file_name).stem.lower() == fn]
            if len(matches) == 1:
                return matches[0]
            # "publisher/repo" or "repo" (LM Studio identifiers)
            tail = ql.split("/")[-1]
            matches = [e for e in self._entries.values() if e.repo == slugify(tail) or e.repo == slugify(ql)]
            if len(matches) == 1:
                return matches[0]
            matches = [e for e in self._entries.values() if e.id.lower().startswith(ql) or e.id.split("/")[-1].startswith(tail)]
            if len(matches) == 1:
                return matches[0]
        return None
