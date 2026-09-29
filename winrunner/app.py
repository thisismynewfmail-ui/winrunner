"""FastAPI application assembly."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import PRODUCT_NAME, __version__
from .api_admin import build_router
from .api_openai import OpenAIRouter
from .bench import BenchRunner
from .config import SettingsStore
from .context import AppContext
from .downloads import DownloadManager
from .engine import EngineManager
from .events import BusLogHandler, EventBus, RequestTracker
from .hardware import HardwareMonitor
from .library import ModelLibrary
from .manager import ModelManager
from .paths import STATIC_DIR, DataPaths
from .util import GiB, is_loopback

log = logging.getLogger("winrunner.app")

FORBIDDEN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>WinRunner</title>
<style>body{background:#3e4637;color:#dee5d7;font:13px Tahoma,Verdana,sans-serif;display:grid;place-items:center;
height:100vh;margin:0}div{border:1px solid #889180;background:#4c5844;padding:24px 28px;max-width:520px}
h1{font-size:14px;color:#c4b550;margin:0 0 10px;letter-spacing:1px}</style></head><body><div>
<h1>CONTROL PANEL NOT AVAILABLE ON THE NETWORK</h1>
<p>The WinRunner control panel only accepts connections from the computer it runs on.</p>
<p>The API at <code>/v1</code> is available. To allow the control panel from other computers, enable
<b>Settings &rsaquo; Network &rsaquo; Allow control panel from LAN</b> on the host.</p></div></body></html>"""


class AccessMiddleware:
    """LAN access policy for the control panel + runtime-configurable CORS for the API."""

    API_PREFIXES = ("/v1", "/api/v0")

    def __init__(self, app: Any, ctx: AppContext):
        self.app = app
        self.ctx = ctx

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        client = (scope.get("client") or ("", 0))[0]
        s = self.ctx.store.settings.server
        is_api = path.startswith(self.API_PREFIXES)
        if not is_api and not s.lan_control_panel and not is_loopback(client):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4403})
                return
            body = FORBIDDEN_HTML.encode()
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"text/html; charset=utf-8"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        if scope["type"] == "http" and is_api and s.cors:
            headers = dict(scope.get("headers") or [])
            origin = headers.get(b"origin")
            cors = [(b"access-control-allow-origin", origin or b"*"),
                    (b"access-control-allow-credentials", b"true"),
                    (b"vary", b"Origin")]
            if scope.get("method") == "OPTIONS" and b"access-control-request-method" in headers:
                req_h = headers.get(b"access-control-request-headers", b"*")
                await send({"type": "http.response.start", "status": 204, "headers": cors + [
                    (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                    (b"access-control-allow-headers", req_h), (b"access-control-max-age", b"600")]})
                await send({"type": "http.response.body", "body": b""})
                return

            async def send_cors(msg: dict) -> None:
                if msg["type"] == "http.response.start":
                    msg = dict(msg)
                    msg["headers"] = list(msg.get("headers") or []) + cors
                await send(msg)

            await self.app(scope, receive, send_cors)
            return
        await self.app(scope, receive, send)


def recommendations(sysinfo: dict[str, Any]) -> list[dict[str, str]]:
    """Hardware-specific guidance shown in Settings > Hardware."""
    out: list[dict[str, str]] = []
    phys, logical = sysinfo.get("cores_physical") or 0, sysinfo.get("cores_logical") or 0
    if phys:
        out.append({"title": "CPU threads",
                    "text": f"{sysinfo.get('cpu')}: {phys} cores / {logical} threads. The engine uses {phys} threads "
                            "for generation (one per physical core); with full GPU offload the CPU only schedules "
                            "work, so more threads do not help."})
    gpus = [g for g in sysinfo.get("gpus") or [] if g.get("vendor") in ("AMD", "NVIDIA", "Intel")]
    if len(gpus) >= 2:
        names = {g["name"] for g in gpus}
        out.append({"title": "Multi-GPU layer split",
                    "text": f"{len(gpus)} GPUs detected ({', '.join(sorted(names))}). Models are split by layer "
                            "(pipeline) across GPUs; only small activations cross the PCIe bus, so a secondary slot "
                            "running at x4 costs little. Row split is not recommended on consumer boards. The tensor "
                            "split is computed per model from free VRAM, the output layer and the KV cache."})
    if any(g.get("vendor") == "AMD" for g in gpus):
        out.append({"title": "Backend",
                    "text": "Vulkan is the default for Radeon GPUs on Windows: it needs only the Adrenalin driver, "
                            "supports flash attention, quantized KV cache and multi-GPU. The ROCm (HIP) build can "
                            "be installed alongside it; compare both on the Benchmark tab with your models."})
    ram = sysinfo.get("ram_total") or 0
    if ram >= 48 * GiB:
        out.append({"title": "System memory",
                    "text": f"{ram / GiB:.0f} GiB RAM: large mixture-of-experts models that exceed VRAM keep expert "
                            "weights in system RAM automatically while attention and KV cache stay on the GPUs. "
                            "The prompt cache (8 GiB default) keeps recent conversations for instant reuse."})
    total_vram = sum(g.get("vram_total") or 0 for g in gpus)
    if total_vram:
        out.append({"title": "Context length and KV cache",
                    "text": f"{total_vram / GiB:.0f} GiB total VRAM. WinRunner keeps the KV cache at F16 when the "
                            "requested context fits and switches to Q8_0 (near-lossless, half the size) only when that "
                            "is what allows the whole model to stay on the GPUs."})
    return out


def create_app(paths: DataPaths, window_mode: bool = False) -> tuple[FastAPI, AppContext]:
    paths.ensure()
    store = SettingsStore(paths.settings_file)
    library = ModelLibrary(paths.gguf_index)
    engines = EngineManager(paths.engines_dir, paths.downloads_tmp)
    monitor = HardwareMonitor(store.settings.hardware.telemetry_interval_s)
    bus = EventBus()
    tracker = RequestTracker(bus, store.settings.server.request_history)
    manager = ModelManager(store, paths, library, engines, monitor, bus, tracker)
    http = httpx.AsyncClient(headers={"User-Agent": f"{PRODUCT_NAME}/{__version__}"})
    ctx = AppContext(paths=paths, store=store, library=library, engines=engines, monitor=monitor, bus=bus,
                     tracker=tracker, manager=manager, http=http)
    ctx.metrics_history = deque(maxlen=3600)
    ctx.extras["window_mode"] = window_mode

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        bus.bind(loop)
        handler = BusLogHandler(bus)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("winrunner").addHandler(handler)
        tracker.start()

        def on_sample(s: dict) -> None:
            ctx.metrics_history.append(s)
            bus.publish("metrics", sample=s)

        monitor.subscribe(on_sample)
        monitor.start()
        sysinfo = monitor.system_info()
        ctx.extras["sysinfo"] = sysinfo
        ctx.extras["recommendations"] = recommendations(sysinfo)

        def dl_dir():
            from pathlib import Path

            d = store.settings.library.download_dir
            return Path(d).expanduser() if d else paths.default_models_dir

        def on_download_complete() -> None:
            loop.create_task(ctx.extras["rescan"]())

        ctx.extras["downloads"] = DownloadManager(dl_dir, lambda: store.settings.library.hf_token,
                                                  lambda j: bus.publish("download", job=j), on_download_complete)
        ctx.extras["bench"] = BenchRunner(paths.bench_file, lambda e: bus.publish("bench", **e))
        manager.start_background()
        bus.activity_log(f"{PRODUCT_NAME} {__version__} started", category="server")
        gpu_names = ", ".join(g["name"] for g in sysinfo["gpus"]) or "no GPU telemetry"
        bus.activity_log(f"Hardware: {sysinfo['cpu']} · {sysinfo['ram_total'] / GiB:.0f} GiB RAM · {gpu_names}",
                         category="hardware")

        async def startup_tasks() -> None:
            await ctx.extras["rescan"]()
            eng = await manager.engine_async()
            if eng:
                bus.activity_log(f"Engine: llama.cpp build {eng.build or eng.version} ({eng.backend})",
                                 category="engine")
                devs, err = await manager.devices(max_age=0)
                for d in devs:
                    bus.activity_log(f"{d.name}: {d.description} · {d.free_mib:,} / {d.total_mib:,} MiB free",
                                     category="hardware")
                if err:
                    bus.activity_log(f"Device query: {err[:200]}", level="warn", category="engine")
            else:
                bus.activity_log("No llama.cpp engine installed - open Settings > Engine to download one",
                                 level="warn", category="engine")
            st = store.settings.startup
            if eng and st.autoload_last_model and st.last_model and library.get(st.last_model):
                try:
                    await manager.load(st.last_model, source="startup")
                except Exception as exc:
                    bus.activity_log(f"Auto-load failed: {exc}", level="error", category="model")
            sm = ctx.extras.get("cli_model")
            if eng and sm:
                try:
                    await manager.load(sm, source="startup")
                except Exception as exc:
                    bus.activity_log(f"Startup model load failed: {exc}", level="error", category="model")

        startup = loop.create_task(startup_tasks())
        try:
            yield
        finally:
            startup.cancel()
            await manager.shutdown()
            monitor.stop()
            await tracker.stop()
            await http.aclose()
            logging.getLogger("winrunner").removeHandler(handler)

    app = FastAPI(title=PRODUCT_NAME, version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.include_router(OpenAIRouter(ctx).router)
    app.include_router(build_router(ctx))
    app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")

    no_cache = {"Cache-Control": "no-cache"}

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        return FileResponse(STATIC_DIR / "index.html", headers=no_cache)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return FileResponse(STATIC_DIR / "img" / "icon.svg", media_type="image/svg+xml")

    @app.get("/health", include_in_schema=False)
    async def health() -> Response:
        return HTMLResponse("ok")

    def ws_guard(websocket: Any) -> bool:
        host = websocket.client.host if websocket.client else ""
        return store.settings.server.lan_control_panel or is_loopback(host)

    ctx.extras["ws_guard"] = ws_guard
    wrapped = AccessMiddleware(app, ctx)
    return wrapped, ctx  # type: ignore[return-value]
