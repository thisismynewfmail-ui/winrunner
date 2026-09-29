"""Control panel API (``/wr/api``) and live telemetry websocket (``/wr/ws``)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import PRODUCT_NAME, PRODUCT_TAGLINE, __version__
from .cmdline import build_bench_args
from .config import SAMPLING_KEYS, LoadParams
from .context import AppContext
from .manager import ModelError
from .templates import TemplateError, analyze, render
from .util import lan_addresses

log = logging.getLogger("winrunner.admin")


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def build_router(ctx: AppContext) -> APIRouter:
    r = APIRouter(prefix="/wr")

    # ----- status / settings -------------------------------------------------------------

    def server_urls() -> dict[str, Any]:
        s = ctx.store.settings.server
        port = ctx.extras.get("bound_port", s.port)
        host = ctx.extras.get("bound_host", s.host)
        lan = lan_addresses() if host in ("0.0.0.0", "::") else ([host] if host not in ("127.0.0.1", "localhost") else [])
        return {
            "host": host, "port": port,
            "local": f"http://127.0.0.1:{port}/v1",
            "lan": [f"http://{a}:{port}/v1" for a in lan],
            "restart_required": (s.port != port or s.host != host),
        }

    def status_payload() -> dict[str, Any]:
        eng = ctx.manager._engine
        lib = ctx.library.entries()
        return {
            "product": PRODUCT_NAME, "tagline": PRODUCT_TAGLINE, "version": __version__,
            "started_at": ctx.started_at, "uptime": round(time.time() - ctx.started_at, 1),
            "server": {**server_urls(), "api_enabled": ctx.store.settings.server.api_enabled,
                       "jit": ctx.store.settings.server.jit_loading, "auth": bool(ctx.store.settings.server.api_key)},
            "engine": eng.to_dict() if eng else None,
            "instances": [i.status() for i in ctx.manager.instances.values()],
            "library": {"models": len(lib), "vision": sum(1 for e in lib if e.has_vision),
                        "scanning": ctx.library.scanning, "last_scan": ctx.library.last_scan},
            "totals": ctx.tracker.totals,
            "active_requests": len(ctx.tracker.active),
            "system": ctx.extras.get("sysinfo"),
            "window": ctx.extras.get("window_mode", False),
            "can_exit": bool(ctx.extras.get("can_exit") or ctx.extras.get("window_mode")),
            "notice": ctx.extras.get("notice"),
        }

    @r.get("/api/status")
    async def status() -> dict:
        return status_payload()

    @r.get("/api/settings")
    async def get_settings() -> dict:
        snap = ctx.store.snapshot()
        snap.pop("models", None)
        return snap

    @r.put("/api/settings")
    async def put_settings(request: Request) -> Response:
        patch = await request.json()
        if not isinstance(patch, dict):
            return _err(400, "expected an object")
        patch.pop("models", None)
        try:
            ctx.store.update(patch)
        except Exception as exc:
            return _err(400, f"invalid settings: {exc}")
        if "library" in patch and "model_dirs" in patch["library"]:
            asyncio.create_task(rescan())
        if "engine" in patch:
            await ctx.manager.engine_async(refresh=True)
        if "hardware" in patch and "telemetry_interval_s" in patch["hardware"]:
            ctx.monitor.interval = max(0.25, float(ctx.store.settings.hardware.telemetry_interval_s))
        ctx.bus.publish("settings", settings=await get_settings())
        return JSONResponse(await get_settings())

    @r.post("/api/settings/reset")
    async def reset_settings(request: Request) -> dict:
        body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
        section = body.get("section") if isinstance(body, dict) else None
        from .config import Settings

        defaults = Settings().model_dump(mode="json")
        patch = {section: defaults[section]} if section in defaults and section != "models" else {
            k: v for k, v in defaults.items() if k not in ("models", "library", "startup")}
        ctx.store.update(patch)
        return await get_settings()

    # ----- library ------------------------------------------------------------------------

    async def rescan() -> dict:
        aliases = {k: p.alias for k, p in ctx.store.settings.models.items() if p.alias}
        ctx.bus.publish("library_scan", state="scanning")
        res = await asyncio.to_thread(ctx.library.scan, ctx.store.settings.library.model_dirs, aliases)
        ctx.bus.publish("library_scan", state="done", **res)
        ctx.bus.activity_log(f"Library scan: {res['models']} models in {res['seconds']:.2f} s", category="library")
        return res

    ctx.extras["rescan"] = rescan

    @r.get("/api/library")
    async def library() -> dict:
        loaded = {i.model_id: i.state for i in ctx.manager.instances.values()}
        models = []
        for e in ctx.library.entries():
            d = e.to_summary()
            d["state"] = loaded.get(e.id, "")
            prof = ctx.store.settings.models.get(e.path)
            d["alias"] = prof.alias if prof else ""
            d["last_loaded"] = prof.last_loaded if prof else 0
            d["load_count"] = prof.load_count if prof else 0
            models.append(d)
        dirs = [{"path": d, "exists": Path(d).expanduser().is_dir()} for d in ctx.store.settings.library.model_dirs]
        return {"models": models, "scanning": ctx.library.scanning, "last_scan": ctx.library.last_scan,
                "errors": ctx.library.scan_errors[-20:], "dirs": dirs,
                "download_dir": download_dir_str()}

    @r.post("/api/library/rescan")
    async def library_rescan() -> dict:
        return await rescan()

    def entry_or_404(model_id: str):
        e = ctx.library.get(model_id) or ctx.library.resolve(model_id)
        if e is None:
            raise ModelError(f"model '{model_id}' not found", 404)
        return e

    @r.get("/api/library/item")
    async def library_item(id: str) -> Response:
        try:
            e = entry_or_404(id)
        except ModelError as exc:
            return _err(exc.status, str(exc))
        d = e.to_detail()
        prof = ctx.store.profile(e.path)
        d["profile"] = prof.model_dump(mode="json")
        d["effective"] = ctx.store.effective_load_params(e.path).model_dump(mode="json")
        d["defaults"] = ctx.store.settings.defaults.model_dump(mode="json")
        d["chat_template"] = e.info.chat_template
        d["named_templates"] = e.info.named_templates
        d["mmproj_details"] = {p: ctx.library.mmproj_info(p) for p in e.mmproj_candidates}
        drafts = [x.id for x in ctx.library.entries()
                  if x.id != e.id and x.info.kind == "llm" and x.info.tokenizer_model == e.info.tokenizer_model
                  and x.info.n_vocab == e.info.n_vocab and x.info.file_size < e.info.file_size]
        d["draft_candidates"] = drafts
        return JSONResponse(d)

    @r.put("/api/library/profile")
    async def put_profile(request: Request) -> Response:
        body = await request.json()
        try:
            e = entry_or_404(body.get("id", ""))
        except ModelError as exc:
            return _err(exc.status, str(exc))
        prof = ctx.store.profile(e.path)
        if "alias" in body:
            prof.alias = str(body["alias"] or "").strip()
        if "load" in body and isinstance(body["load"], dict):
            load = {k: v for k, v in body["load"].items() if k in LoadParams.model_fields}
            try:
                LoadParams.model_validate({**ctx.store.settings.defaults.model_dump(mode="json"), **load})
            except Exception as exc:
                return _err(400, f"invalid load settings: {exc}")
            prof.load = load
        if "sampling" in body and isinstance(body["sampling"], dict):
            prof.sampling = {k: v for k, v in body["sampling"].items() if k in SAMPLING_KEYS and v not in (None, "")}
        ctx.store.set_profile(e.path, prof)
        if "alias" in body:
            asyncio.create_task(rescan())
        return JSONResponse(prof.model_dump(mode="json"))

    @r.post("/api/library/folder")
    async def add_folder(request: Request) -> Response:
        body = await request.json()
        path = str(body.get("path", "")).strip()
        action = body.get("action", "add")
        dirs = list(ctx.store.settings.library.model_dirs)
        if action == "add":
            if not path or not Path(path).expanduser().is_dir():
                return _err(400, "folder does not exist")
            if path not in dirs:
                dirs.append(path)
        else:
            dirs = [d for d in dirs if d != path]
        ctx.store.update({"library": {"model_dirs": dirs}})
        asyncio.create_task(rescan())
        return JSONResponse({"dirs": dirs})

    # ----- chat templates ------------------------------------------------------------------

    @r.post("/api/template/preview")
    async def template_preview(request: Request) -> Response:
        body = await request.json()
        try:
            e = entry_or_404(body.get("id", ""))
        except ModelError as exc:
            return _err(exc.status, str(exc))
        messages = body.get("messages")
        template = body.get("template") or e.info.chat_template
        kwargs = body.get("kwargs") or {}
        inst = ctx.manager.instance_for_model(e.id)
        if inst and inst.state == "ready" and not body.get("template") and inst.client is not None:
            try:
                payload = {"messages": messages or [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                    {"role": "user", "content": "And of Italy?"}]}
                if kwargs:
                    payload["chat_template_kwargs"] = kwargs
                resp = await inst.client.post("/apply-template", json=payload, timeout=15)
                if resp.status_code == 200:
                    return JSONResponse({"source": "engine", "prompt": resp.json().get("prompt", ""),
                                         "analysis": analyze(template)})
            except Exception:
                pass
        try:
            out = await asyncio.to_thread(render, template, messages, e.info.bos_token, e.info.eos_token, True,
                                          None, kwargs)
            return JSONResponse({"source": "local", "prompt": out, "analysis": analyze(template)})
        except TemplateError as exc:
            return JSONResponse({"source": "local", "error": str(exc), "analysis": analyze(template)})

    # ----- planning / loading ------------------------------------------------------------------

    @r.post("/api/plan")
    async def plan(request: Request) -> Response:
        body = await request.json()
        try:
            pl, extra = await ctx.manager.plan(body.get("id", ""), body.get("overrides") or {},
                                               verify=bool(body.get("verify")))
        except ModelError as exc:
            return _err(exc.status, str(exc))
        except Exception as exc:
            log.exception("plan failed")
            return _err(500, str(exc))
        return JSONResponse({"plan": pl.to_dict(), **extra})

    @r.post("/api/models/load")
    async def load(request: Request) -> Response:
        body = await request.json()
        model_id = body.get("id", "")
        overrides = body.get("overrides") or None
        try:
            entry_or_404(model_id)
        except ModelError as exc:
            return _err(exc.status, str(exc))
        if body.get("save"):
            e = entry_or_404(model_id)
            prof = ctx.store.profile(e.path)
            prof.load = {k: v for k, v in (overrides or {}).items() if k in LoadParams.model_fields}
            ctx.store.set_profile(e.path, prof)

        async def job() -> None:
            try:
                await ctx.manager.load(model_id, overrides, source="ui")
            except ModelError as exc:
                ctx.bus.publish("load_error", model=model_id, error=str(exc))
            except Exception as exc:
                log.exception("load failed")
                ctx.bus.publish("load_error", model=model_id, error=str(exc))

        asyncio.create_task(job())
        return JSONResponse({"started": True, "model": model_id}, status_code=202)

    @r.post("/api/models/unload")
    async def unload(request: Request) -> Response:
        body = await request.json()
        ok = await ctx.manager.unload(body.get("id", ""))
        return JSONResponse({"unloaded": ok})

    @r.get("/api/instances")
    async def instances() -> dict:
        return ctx.manager.status()

    @r.get("/api/instances/log")
    async def instance_log(iid: str, limit: int = 3000) -> Response:
        inst = ctx.manager.instances.get(iid) or ctx.manager.instance_for_model(iid)
        if not inst:
            return _err(404, "instance not found")
        return JSONResponse({"id": inst.id, "model": inst.model_id, "lines": list(inst.log)[-limit:]})

    # ----- requests / metrics / logs ---------------------------------------------------------------

    @r.get("/api/requests")
    async def requests_(limit: int = 200) -> dict:
        return {"requests": ctx.tracker.recent(limit), "totals": ctx.tracker.totals,
                "tps": list(ctx.tracker.tps_history)}

    @r.get("/api/requests/item")
    async def request_item(id: str) -> Response:
        rec = ctx.tracker.get(id)
        return JSONResponse(rec) if rec else _err(404, "request not found")

    @r.get("/api/metrics/history")
    async def metrics_history(seconds: int = 900) -> dict:
        cutoff = time.time() - seconds
        return {"samples": [s for s in ctx.metrics_history if s["t"] >= cutoff]}

    @r.get("/api/logs/app")
    async def app_log(limit: int = 2000) -> dict:
        return {"lines": list(ctx.bus.applog)[-limit:]}

    @r.get("/api/logs/download")
    async def log_download(iid: str = "") -> Response:
        if iid:
            inst = ctx.manager.instances.get(iid)
            if not inst:
                return _err(404, "instance not found")
            lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x['t']))} {x['level'].upper():5} {x['text']}"
                     for x in inst.log]
            name = f"engine-{inst.model_id.replace('/', '_')}.log"
        else:
            lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x['t']))} {x['level'].upper():5} {x['text']}"
                     for x in ctx.bus.applog]
            name = "winrunner.log"
        return PlainTextResponse("\n".join(lines), headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @r.get("/api/activity")
    async def activity() -> dict:
        return {"items": list(ctx.bus.activity)}

    # ----- hardware / engine -------------------------------------------------------------------------

    @r.get("/api/hardware")
    async def hardware() -> dict:
        devices, err = await ctx.manager.devices(max_age=10)
        return {"system": ctx.extras.get("sysinfo"), "sample": ctx.monitor.last,
                "engine_devices": [d.__dict__ for d in devices], "device_error": err,
                "device_map": ctx.manager.device_map(devices),
                "recommendations": ctx.extras.get("recommendations", [])}

    @r.get("/api/engine")
    async def engine() -> dict:
        eng = await ctx.manager.engine_async()
        return {"active": eng.to_dict() if eng else None, "installed": ctx.engines.installed(),
                "install_state": ctx.engines.install_state, "settings": ctx.store.settings.engine.model_dump()}

    @r.post("/api/engine/refresh")
    async def engine_refresh() -> dict:
        eng = await ctx.manager.engine_async(refresh=True)
        devices, err = await ctx.manager.devices(max_age=0)
        return {"active": eng.to_dict() if eng else None, "devices": [d.__dict__ for d in devices], "error": err}

    @r.get("/api/engine/releases")
    async def engine_releases() -> Response:
        try:
            return JSONResponse({"releases": await ctx.engines.releases()})
        except Exception as exc:
            return _err(502, f"could not query GitHub releases: {exc}")

    @r.post("/api/engine/install")
    async def engine_install(request: Request) -> Response:
        body = await request.json()
        tag, backend, asset = body.get("tag"), body.get("backend"), body.get("asset")
        if not (tag and backend and isinstance(asset, dict) and asset.get("url")):
            return _err(400, "tag, backend and asset are required")
        if not re.match(r"^https://github\.com/ggml-org/llama\.cpp/releases/download/", asset["url"]):
            return _err(400, "only official llama.cpp release assets can be installed")

        async def job() -> None:
            try:
                res = await ctx.engines.install(tag, backend, asset,
                                                lambda st: ctx.bus.publish("engine_install", **st))
                ctx.store.update({"engine": {"active_engine": res["name"], "engine_path": "", "backend": backend}})
                await ctx.manager.engine_async(refresh=True)
                ctx.bus.activity_log(f"Installed llama.cpp {tag} ({backend})", level="ok", category="engine")
                ctx.bus.publish("engine_changed")
            except Exception as exc:
                ctx.bus.activity_log(f"Engine installation failed: {exc}", level="error", category="engine")

        asyncio.create_task(job())
        return JSONResponse({"started": True}, status_code=202)

    @r.post("/api/engine/select")
    async def engine_select(request: Request) -> Response:
        body = await request.json()
        patch: dict[str, Any] = {}
        if "name" in body:
            inst = next((e for e in ctx.engines.installed() if e["name"] == body["name"]), None)
            if not inst:
                return _err(404, "engine not installed")
            patch = {"active_engine": inst["name"], "engine_path": "", "backend": inst["backend"]
                     if inst["backend"] in ("vulkan", "rocm", "cpu") else "custom"}
        elif "path" in body:
            p = Path(str(body["path"])).expanduser()
            if not p.exists():
                return _err(400, "path does not exist")
            patch = {"engine_path": str(p), "backend": "custom"}
        ctx.store.update({"engine": patch})
        eng = await ctx.manager.engine_async(refresh=True)
        ctx.bus.publish("engine_changed")
        return JSONResponse({"active": eng.to_dict() if eng else None})

    @r.delete("/api/engine/{name}")
    async def engine_delete(name: str) -> Response:
        try:
            ctx.engines.remove(name)
        except (ValueError, OSError) as exc:
            return _err(400, str(exc))
        if ctx.store.settings.engine.active_engine == name:
            ctx.store.update({"engine": {"active_engine": ""}})
        await ctx.manager.engine_async(refresh=True)
        return JSONResponse({"removed": name})

    # ----- downloads ------------------------------------------------------------------------------

    def download_dir_str() -> str:
        d = ctx.store.settings.library.download_dir
        return d or str(ctx.paths.default_models_dir)

    @r.get("/api/hf/search")
    async def hf_search(q: str) -> Response:
        try:
            return JSONResponse({"results": await ctx.extras["downloads"].search(q)})
        except Exception as exc:
            return _err(502, f"Hugging Face search failed: {exc}")

    @r.get("/api/hf/files")
    async def hf_files(repo: str) -> Response:
        if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
            return _err(400, "invalid repository id")
        try:
            return JSONResponse(await ctx.extras["downloads"].files(repo))
        except Exception as exc:
            return _err(502, f"could not list repository files: {exc}")

    @r.get("/api/downloads")
    async def downloads() -> dict:
        return {"jobs": list(ctx.extras["downloads"].jobs.values()), "dir": download_dir_str()}

    @r.post("/api/downloads")
    async def start_download(request: Request) -> Response:
        body = await request.json()
        repo = body.get("repo", "")
        files = body.get("files") or []
        if not re.match(r"^[\w.-]+/[\w.-]+$", repo) or not files:
            return _err(400, "repo and files are required")
        for f in files:
            if ".." in f or f.startswith(("/", "\\")):
                return _err(400, "invalid file path")
        dl_dir = Path(download_dir_str()).expanduser()
        if str(dl_dir) not in [str(Path(d).expanduser()) for d in ctx.store.settings.library.model_dirs]:
            ctx.store.update({"library": {"model_dirs": ctx.store.settings.library.model_dirs + [str(dl_dir)]}})
        ids = ctx.extras["downloads"].start(repo, files, body.get("sizes") or {})
        return JSONResponse({"ids": ids})

    @r.delete("/api/downloads/{jid}")
    async def cancel_download(jid: str) -> dict:
        return {"cancelled": ctx.extras["downloads"].cancel(jid)}

    # ----- benchmark -------------------------------------------------------------------------------

    @r.get("/api/bench")
    async def bench_history() -> dict:
        b = ctx.extras["bench"]
        return {"history": b.history(), "running": b.current}

    @r.post("/api/bench")
    async def bench_run(request: Request) -> Response:
        body = await request.json()
        b = ctx.extras["bench"]
        if b.current:
            return _err(409, "a benchmark is already running")
        try:
            e = entry_or_404(body.get("id", ""))
            pl, _ = await ctx.manager.plan(e.id, body.get("overrides") or {}, verify=False)
        except ModelError as exc:
            return _err(exc.status, str(exc))
        eng = await ctx.manager.engine_async()
        if not eng or not eng.bench:
            return _err(400, "the active engine does not include llama-bench")
        p = ctx.store.effective_load_params(e.path, body.get("overrides") or {})
        devices, _ = await ctx.manager.devices(max_age=5)
        sel = [d.name for d in devices if not p.devices or d.name in p.devices]
        args = build_bench_args(eng, e.path, p, pl, sel, str(body.get("pp", "512")), str(body.get("tg", "128")),
                                str(body.get("depth", "0")), int(body.get("reps", 3)))
        if not args:
            return _err(400, "could not build benchmark command")
        if body.get("unload") and ctx.manager.instances:
            for inst in list(ctx.manager.instances.values()):
                await ctx.manager.unload(inst.id)
        meta = {"model": e.id, "quant": e.info.quant, "engine": f"{eng.backend} b{eng.build}",
                "ngl": pl.gpu_layers, "fa": pl.flash_attn, "kv": f"{pl.kv_k}/{pl.kv_v}",
                "ts": pl.tensor_split, "ub": p.ubatch_size, "b": p.batch_size}

        async def job() -> None:
            try:
                await b.run(args, meta)
            except Exception as exc:
                ctx.bus.publish("bench", state="error", error=str(exc))

        asyncio.create_task(job())
        return JSONResponse({"started": True, "command": " ".join(args)}, status_code=202)

    @r.post("/api/bench/cancel")
    async def bench_cancel() -> dict:
        return {"cancelled": ctx.extras["bench"].cancel()}

    @r.delete("/api/bench")
    async def bench_clear() -> dict:
        ctx.extras["bench"].clear()
        return {"cleared": True}

    # ----- chats (test console history) ----------------------------------------------------------------

    def chat_path(cid: str) -> Path:
        if not re.match(r"^[\w-]{1,64}$", cid):
            raise ValueError("invalid chat id")
        return ctx.paths.chats_dir / f"{cid}.json"

    @r.get("/api/chats")
    async def chats() -> dict:
        out = []
        for p in sorted(ctx.paths.chats_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append({"id": p.stem, "title": d.get("title", ""), "updated": p.stat().st_mtime,
                            "model": d.get("model", ""), "messages": len(d.get("messages", []))})
            except (OSError, ValueError):
                continue
        return {"chats": out}

    @r.get("/api/chats/{cid}")
    async def chat_get(cid: str) -> Response:
        try:
            return JSONResponse(json.loads(chat_path(cid).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return _err(404, "chat not found")

    @r.put("/api/chats/{cid}")
    async def chat_put(cid: str, request: Request) -> Response:
        body = await request.body()
        if len(body) > 64 * 1024 * 1024:
            return _err(413, "chat too large")
        try:
            data = json.loads(body)
            chat_path(cid).write_text(json.dumps(data), encoding="utf-8")
        except (ValueError, OSError) as exc:
            return _err(400, str(exc))
        return JSONResponse({"saved": cid})

    @r.delete("/api/chats/{cid}")
    async def chat_delete(cid: str) -> Response:
        try:
            chat_path(cid).unlink()
        except (OSError, ValueError):
            return _err(404, "chat not found")
        return JSONResponse({"deleted": cid})

    # ----- application ---------------------------------------------------------------------------------

    @r.post("/api/app/exit")
    async def app_exit() -> dict:
        stop = ctx.extras.get("request_exit")
        if stop:
            asyncio.get_running_loop().call_later(0.3, stop)
        return {"exiting": True}

    # ----- websocket -------------------------------------------------------------------------------------

    @r.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        guard = ctx.extras.get("ws_guard")
        if guard and not guard(websocket):
            await websocket.close(code=4403)
            return
        await websocket.accept()
        q = ctx.bus.subscribe()
        ctx.manager.ui_clients += 1
        try:
            hello = {
                "type": "hello",
                "status": status_payload(),
                "metrics": list(ctx.metrics_history)[-600:],
                "activity": list(ctx.bus.activity)[-200:],
                "requests": ctx.tracker.recent(60),
                "tps": list(ctx.tracker.tps_history),
                "downloads": list(ctx.extras["downloads"].jobs.values()),
                "install_state": ctx.engines.install_state,
            }
            await websocket.send_text(json.dumps(hello, default=str))
            while True:
                ev = await q.get()
                batch = [ev]
                while not q.empty() and len(batch) < 200:
                    batch.append(q.get_nowait())
                await websocket.send_text(json.dumps({"type": "batch", "events": batch}, default=str))
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        except asyncio.CancelledError:
            raise
        finally:
            ctx.manager.ui_clients -= 1
            ctx.bus.unsubscribe(q)

    return r
