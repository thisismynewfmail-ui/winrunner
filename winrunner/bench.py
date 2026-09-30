"""llama-bench integration (prompt processing / generation throughput)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .engine import engine_env
from .paths import IS_WINDOWS
from .util import new_id, strip_ansi

log = logging.getLogger("winrunner.bench")
_CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


class BenchRunner:
    def __init__(self, results_file: Path, on_event: Callable[[dict[str, Any]], None]):
        self.results_file = results_file
        self.on_event = on_event
        self.current: dict[str, Any] | None = None
        self._proc: subprocess.Popen | None = None

    def history(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.results_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _save(self, run: dict[str, Any]) -> None:
        hist = [run] + self.history()
        self.results_file.write_text(json.dumps(hist[:100], indent=1), encoding="utf-8")

    def clear(self) -> None:
        try:
            self.results_file.unlink()
        except OSError:
            pass

    def cancel(self) -> bool:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            return True
        return False

    async def run(self, args: list[str], meta: dict[str, Any]) -> dict[str, Any]:
        if self.current:
            raise RuntimeError("a benchmark is already running")
        run = {"id": new_id("bench-", 3), "t": time.time(), "state": "running", "meta": meta,
               "command": " ".join(args), "results": [], "log": []}
        self.current = run
        self.on_event({"state": "running", "id": run["id"], "meta": meta})
        loop = asyncio.get_running_loop()

        def work() -> tuple[int, str]:
            self._proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          stdin=subprocess.DEVNULL, cwd=os.path.dirname(args[0]) or None,
                                          env=engine_env(), creationflags=_CREATE_NO_WINDOW)
            out_chunks: list[bytes] = []

            def pump_err() -> None:
                assert self._proc and self._proc.stderr
                for raw in iter(self._proc.stderr.readline, b""):
                    line = strip_ansi(raw.decode("utf-8", errors="replace").rstrip())
                    if line:
                        run["log"].append(line)
                        loop.call_soon_threadsafe(self.on_event, {"state": "log", "id": run["id"], "line": line})

            t = threading.Thread(target=pump_err, daemon=True)
            t.start()
            assert self._proc.stdout
            for raw in iter(self._proc.stdout.readline, b""):
                out_chunks.append(raw)
            rc = self._proc.wait()
            t.join(timeout=2)
            return rc, b"".join(out_chunks).decode("utf-8", errors="replace")

        try:
            rc, out = await asyncio.to_thread(work)
            results = []
            try:
                data = json.loads(out[out.find("["):]) if "[" in out else []
                for r in data:
                    results.append({
                        "test": ("pp" + str(r.get("n_prompt")) if r.get("n_prompt") else "tg" + str(r.get("n_gen")))
                        + (f" @ d{r['n_depth']}" if r.get("n_depth") else ""),
                        "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts"),
                        "n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"), "n_depth": r.get("n_depth", 0),
                        "backends": r.get("backends"), "gpu_info": r.get("gpu_info"), "model_type": r.get("model_type"),
                        "model_size": r.get("model_size"), "n_gpu_layers": r.get("n_gpu_layers"),
                        "flash_attn": r.get("flash_attn"), "type_k": r.get("type_k"), "type_v": r.get("type_v"),
                        "n_batch": r.get("n_batch"), "n_ubatch": r.get("n_ubatch"), "build": r.get("build_number"),
                    })
            except ValueError:
                pass
            run["results"] = results
            run["state"] = "done" if rc == 0 and results else "error"
            if run["state"] == "error":
                run["error"] = f"llama-bench exited with code {rc}: " + " | ".join(run["log"][-3:])
            run["log"] = run["log"][-200:]
            self._save(run)
            self.on_event({"state": run["state"], "id": run["id"], "run": run})
            return run
        finally:
            self.current = None
            self._proc = None
