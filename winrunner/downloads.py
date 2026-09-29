"""Hugging Face model search and resumable GGUF downloads."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

from .util import new_id

log = logging.getLogger("winrunner.downloads")

HF = "https://huggingface.co"
_QUANT_RE = re.compile(
    r"(IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_K(?:_[SML]|_XL)?|Q[4-8]_[01]|Q8_K_XL|MXFP4(?:_MOE)?|BF16|F16|F32)",
    re.IGNORECASE)


class DownloadManager:
    def __init__(self, target_dir: Callable[[], Path], token: Callable[[], str],
                 on_event: Callable[[dict[str, Any]], None], on_complete: Callable[[], None]):
        self.target_dir = target_dir
        self.token = token
        self.on_event = on_event
        self.on_complete = on_complete
        self.jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(2)

    def _headers(self) -> dict[str, str]:
        h = {"User-Agent": "WinRunner"}
        t = self.token()
        if t:
            h["Authorization"] = f"Bearer {t}"
        return h

    async def search(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        params = {"search": query, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": str(limit),
                  "full": "false"}
        async with httpx.AsyncClient(timeout=20, headers=self._headers()) as c:
            r = await c.get(f"{HF}/api/models", params=params)
            r.raise_for_status()
            out = []
            for m in r.json():
                out.append({
                    "repo": m.get("id") or m.get("modelId"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "updated": m.get("lastModified", ""),
                    "pipeline": m.get("pipeline_tag", ""),
                    "tags": [t for t in (m.get("tags") or []) if t in ("vision", "image-text-to-text",
                                                                       "conversational", "text-generation")],
                })
            return out

    async def files(self, repo: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20, headers=self._headers(), follow_redirects=True) as c:
            r = await c.get(f"{HF}/api/models/{repo}/tree/main", params={"recursive": "true"})
            r.raise_for_status()
            items = r.json()
        files = []
        for it in items:
            if it.get("type") != "file":
                continue
            path = it.get("path", "")
            if not path.lower().endswith(".gguf"):
                continue
            size = (it.get("lfs") or {}).get("size") or it.get("size") or 0
            name = path.split("/")[-1]
            m = _QUANT_RE.search(name)
            split = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name)
            files.append({
                "path": path,
                "name": name,
                "size": size,
                "quant": m.group(1).upper() if m else "",
                "mmproj": "mmproj" in name.lower(),
                "split": f"{int(split.group(1))}/{int(split.group(2))}" if split else "",
            })
        files.sort(key=lambda f: (f["mmproj"], f["path"]))
        return {"repo": repo, "files": files, "has_mmproj": any(f["mmproj"] for f in files)}

    def start(self, repo: str, paths: list[str], sizes: dict[str, int] | None = None) -> list[str]:
        ids = []
        for p in paths:
            jid = new_id("dl-", 4)
            owner, _, name = repo.partition("/")
            dest = self.target_dir() / owner / name / p
            job = {"id": jid, "repo": repo, "path": p, "dest": str(dest), "state": "queued", "done": 0,
                   "total": (sizes or {}).get(p, 0), "speed": 0.0, "error": "", "t_start": time.time()}
            self.jobs[jid] = job
            self._tasks[jid] = asyncio.create_task(self._run(job))
            ids.append(jid)
            self._emit(job)
        return ids

    def cancel(self, jid: str) -> bool:
        t = self._tasks.get(jid)
        if t and not t.done():
            t.cancel()
            return True
        if jid in self.jobs and self.jobs[jid]["state"] in ("done", "error", "cancelled"):
            self.jobs.pop(jid, None)
            return True
        return False

    def _emit(self, job: dict[str, Any]) -> None:
        self.on_event(dict(job))

    async def _run(self, job: dict[str, Any]) -> None:
        async with self._sem:
            dest = Path(job["dest"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            url = f"{HF}/{job['repo']}/resolve/main/{quote(job['path'])}"
            job["state"] = "downloading"
            self._emit(job)
            try:
                for attempt in range(5):
                    have = part.stat().st_size if part.exists() else 0
                    headers = self._headers()
                    if have:
                        headers["Range"] = f"bytes={have}-"
                    try:
                        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=120), follow_redirects=True,
                                                     headers=headers) as c:
                            async with c.stream("GET", url) as r:
                                if r.status_code == 416:  # already complete
                                    break
                                if r.status_code not in (200, 206):
                                    raise RuntimeError(f"HTTP {r.status_code}")
                                if r.status_code == 200 and have:
                                    have = 0  # server ignored the range: restart
                                total = int(r.headers.get("content-length", 0)) + have
                                job["total"] = total or job["total"]
                                done = have
                                t0, d0, last = time.monotonic(), done, 0.0
                                with open(part, "ab" if have else "wb") as f:
                                    async for chunk in r.aiter_bytes(1 << 20):
                                        f.write(chunk)
                                        done += len(chunk)
                                        now = time.monotonic()
                                        if now - last > 0.5:
                                            job["done"] = done
                                            job["speed"] = (done - d0) / max(1e-3, now - t0)
                                            last = now
                                            self._emit(job)
                        break
                    except (httpx.HTTPError, RuntimeError) as exc:
                        if attempt == 4:
                            raise
                        log.warning("download %s retry %d: %s", job["path"], attempt + 1, exc)
                        await asyncio.sleep(2 * (attempt + 1))
                os.replace(part, dest)
                job.update(state="done", done=job["total"] or dest.stat().st_size, speed=0.0)
                self._emit(job)
                self.on_complete()
            except asyncio.CancelledError:
                job.update(state="cancelled")
                self._emit(job)
            except Exception as exc:
                job.update(state="error", error=str(exc))
                self._emit(job)
