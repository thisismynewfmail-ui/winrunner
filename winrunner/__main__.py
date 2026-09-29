"""WinRunner entry point: ``python -m winrunner``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
import webbrowser

from . import PRODUCT_NAME, __version__
from .paths import DataPaths


def _setup_logging(level: str, paths: DataPaths) -> None:
    from logging.handlers import RotatingFileHandler

    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        try:
            if stream is not None:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(paths.logs_dir / "winrunner.log", maxBytes=5 * 1024 * 1024, backupCount=3,
                             encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if sys.stderr is not None:  # pythonw.exe has no console streams
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        root.addHandler(ch)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _install_engine(paths: DataPaths, backend: str) -> int:
    from .config import SettingsStore
    from .engine import EngineManager

    paths.ensure()
    em = EngineManager(paths.engines_dir, paths.downloads_tmp)
    store = SettingsStore(paths.settings_file)

    async def run() -> int:
        rels = await em.releases(limit=3)
        for rel in rels:
            asset = rel["backends"].get(backend)
            if asset:
                print(f"Installing llama.cpp {rel['tag']} ({backend}): {asset['name']}")
                last = [0.0]

                def progress(st: dict) -> None:
                    if st.get("phase") == "download" and st.get("total") and time.time() - last[0] > 1:
                        last[0] = time.time()
                        print(f"  {st['done'] / 1e6:8.1f} / {st['total'] / 1e6:.1f} MB", flush=True)
                    elif st.get("phase") in ("extract", "done", "error"):
                        print(f"  {st['phase']} {st.get('error', '')}", flush=True)

                res = await em.install(rel["tag"], backend, asset, progress)
                store.update({"engine": {"active_engine": res["name"], "engine_path": "", "backend": backend}})
                print(f"Installed: {res['server']}")
                return 0
        print(f"No {backend} build found in the latest releases.", file=sys.stderr)
        return 1

    return asyncio.run(run())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="winrunner", description=f"{PRODUCT_NAME} - local LLM inference server")
    ap.add_argument("--host", help="bind address for the API (default from settings: 0.0.0.0)")
    ap.add_argument("--port", type=int, help="API port (default from settings: 5070)")
    ap.add_argument("--data-dir", help="data directory (settings, engines, cache)")
    ui = ap.add_mutually_exclusive_group()
    ui.add_argument("--window", action="store_true", help="open the control panel in a native window")
    ui.add_argument("--browser", action="store_true", help="open the control panel in the default browser")
    ui.add_argument("--headless", action="store_true", help="do not open the control panel")
    ap.add_argument("--model", help="model id to load at startup")
    ap.add_argument("--install-engine", choices=["vulkan", "rocm", "cpu"],
                    help="download the latest llama.cpp release for this backend and exit")
    ap.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    ap.add_argument("--version", action="version", version=f"{PRODUCT_NAME} {__version__}")
    args = ap.parse_args(argv)

    paths = DataPaths(args.data_dir)
    paths.ensure()
    _setup_logging(args.log_level, paths)
    log = logging.getLogger("winrunner")

    if args.install_engine:
        return _install_engine(paths, args.install_engine)

    import uvicorn

    from .app import create_app
    from .util import port_available

    open_mode = "window" if args.window else "browser" if args.browser else "none" if args.headless else None
    app, ctx = create_app(paths, window_mode=False)
    s = ctx.store.settings
    host = args.host or s.server.host
    port = args.port or s.server.port
    if open_mode is None:
        open_mode = s.startup.open_ui
    if args.model:
        ctx.extras["cli_model"] = args.model
    ctx.extras["bound_host"] = host
    ctx.extras["bound_port"] = port

    if not port_available(port, host):
        log.error("Port %d on %s is already in use. Close the other application or change the port "
                  "(Settings > Network, or --port).", port, host)
        return 2

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False,
                            timeout_keep_alive=30, ws_ping_interval=20, ws_ping_timeout=30,
                            timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    ctx.manager.exiting = lambda: server.should_exit
    ui_url = f"http://127.0.0.1:{port}/"

    webview = None
    if open_mode == "window":
        try:
            import webview  # type: ignore
        except ImportError:
            log.warning("pywebview is not installed; opening the control panel in the browser instead")
            open_mode = "browser"

    log.info("%s %s - API http://%s:%d/v1 - control panel %s", PRODUCT_NAME, __version__, host, port, ui_url)

    if open_mode == "window" and webview is not None:
        ctx.extras["window_mode"] = True
        t = threading.Thread(target=server.run, name="uvicorn", daemon=True)
        t.start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        width, height = 1100, 1400
        try:
            screens = webview.screens
            if screens:
                sc = screens[0]
                width = min(1200, max(820, int(sc.width * 0.9)))
                height = max(700, int(sc.height * 0.92))
        except Exception:
            pass
        window = webview.create_window(f"{PRODUCT_NAME} {__version__}", ui_url, width=width, height=height,
                                       min_size=(760, 640), background_color="#1f241b", text_select=True)

        def request_exit() -> None:
            server.should_exit = True
            try:
                window.destroy()
            except Exception:
                pass

        ctx.extras["request_exit"] = request_exit
        try:
            webview.start(gui="edgechromium" if sys.platform == "win32" else None, private_mode=False)
        except Exception as exc:
            log.error("native window failed (%s); continuing headless - open %s", exc, ui_url)
            t.join()
            return 0
        server.should_exit = True
        t.join(timeout=20)
        return 0

    ctx.extras["request_exit"] = lambda: setattr(server, "should_exit", True)
    if open_mode == "browser":
        def opener() -> None:
            for _ in range(200):
                if server.started:
                    webbrowser.open(ui_url)
                    return
                time.sleep(0.05)

        threading.Thread(target=opener, daemon=True).start()
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
