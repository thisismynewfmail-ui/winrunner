"""WinRunner entry point: ``python -m winrunner``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
import webbrowser

from . import PRODUCT_NAME, __version__
from .paths import STATIC_DIR, DataPaths


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


def _install_engine(paths: DataPaths, backend: str, tag: str | None = None) -> int:
    from .config import SettingsStore
    from .engine import EngineManager

    paths.ensure()
    em = EngineManager(paths.engines_dir, paths.downloads_tmp)
    store = SettingsStore(paths.settings_file)

    async def run() -> int:
        if tag:
            existing = next((e for e in em.installed() if e["tag"] == tag and e["backend"] == backend), None)
            if existing:
                store.update({"engine": {"active_engine": existing["name"], "engine_path": "", "backend": backend}})
                print(f"llama.cpp {tag} ({backend}) is already installed: {existing['server']}")
                return 0
            rels = [await em.release(tag)]
        else:
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
        where = f"release {tag}" if tag else "the latest releases"
        print(f"No {backend} build for this platform found in {where}.", file=sys.stderr)
        return 1

    return asyncio.run(run())


def _check(paths: DataPaths) -> int:
    """Print the engine and the GPUs it can use (setup.sh runs this after installing)."""
    from pathlib import Path

    from .config import SettingsStore
    from .engine import EngineManager
    from .hardware import HardwareMonitor

    store = SettingsStore(paths.settings_file)
    es = store.settings.engine
    em = EngineManager(paths.engines_dir, paths.downloads_tmp)
    server = em.resolve_server(es.engine_path, es.active_engine, es.backend)
    if server is None:
        print("No llama.cpp engine installed.")
        return 1
    info = em.probe(Path(server))
    print(f"Engine:  llama.cpp {info.version} (build {info.build}, {info.backend}) - {server}")
    if info.probe_error:
        print(f"         problem: {info.probe_error}")
    devices, err = em.list_devices(Path(server))
    mon = HardwareMonitor()
    sysinfo = mon.system_info()
    if sysinfo.get("vulkan_driver"):
        print(f"Driver:  Mesa (RADV) {sysinfo['vulkan_driver']}")
    mapping = mon.map_engine_devices([{"name": d.name, "description": d.description, "total_mib": d.total_mib,
                                       "free_mib": d.free_mib} for d in devices])
    gpus = {g.id: g for g in mon.gpus}
    for d in devices:
        g = gpus.get(mapping.get(d.name, ""))
        extra = ""
        if g is not None:
            bits = [g.pci, g.pcie, "display" if g.boot_vga else "",
                    "Resizable BAR on" if g.rebar else "Resizable BAR off" if g.rebar is False else ""]
            extra = " - " + ", ".join(b for b in bits if b)
        print(f"GPU:     {d.name}: {d.description} - {d.free_mib:,} of {d.total_mib:,} MiB free{extra}")
    if not devices:
        print("GPU:     none visible to the engine" + (f" ({err.strip().splitlines()[-1]})" if err.strip() else ""))
        acc = sysinfo.get("render_access") or {}
        if acc.get("nodes") and not acc.get("accessible"):
            print("         No access to /dev/dri/renderD*: log out and back in (new 'render' group membership).")
        return 1
    return 0


WINDOW_LOAD_TIMEOUT = 25.0
WEBVIEW2_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"


LINUX_WINDOW_PACKAGES = "python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1"


def _linux_gui() -> str | None:
    """pywebview backend available on this Linux desktop: 'gtk' (WebKit2GTK), 'qt' or None."""
    try:
        import gi  # type: ignore

        gi.require_version("Gtk", "3.0")
        for ver in ("4.1", "4.0"):
            try:
                gi.require_version("WebKit2", ver)
                from gi.repository import Gtk, WebKit2  # type: ignore  # noqa: F401

                return "gtk"
            except (ValueError, ImportError):
                continue
    except (ImportError, ValueError):
        pass
    try:
        import qtpy  # type: ignore  # noqa: F401
        from qtpy import QtWebEngineWidgets  # type: ignore  # noqa: F401

        return "qt"
    except Exception:
        return None


def _window_icon(gui: str | None) -> str | None:
    """Icon file for the app window that the toolkit can actually read.

    GTK reads SVG only through the optional librsvg pixbuf loader, and pywebview leaves the window hidden when
    the icon cannot be loaded, so the PNG comes first and every candidate is test-loaded.
    """
    if gui not in ("gtk", "qt"):
        return None
    for name in ("icon.png", "icon.svg"):
        path = STATIC_DIR / "img" / name
        if not path.is_file():
            continue
        if gui == "gtk":
            try:
                import gi  # type: ignore

                gi.require_version("GdkPixbuf", "2.0")
                from gi.repository import GdkPixbuf  # type: ignore

                GdkPixbuf.Pixbuf.new_from_file(str(path))
            except Exception:
                continue
        return str(path)
    return None


def _quit_gui(gui: str | None) -> None:
    """End the window toolkit's event loop directly (pywebview cannot close a window that never appeared)."""
    try:
        if gui == "gtk":
            from gi.repository import Gio, GLib  # type: ignore

            def stop() -> bool:
                app = getattr(sys.modules.get("webview.platforms.gtk"), "_app", None) or Gio.Application.get_default()
                if app is not None:
                    app.quit()
                return False

            GLib.idle_add(stop)
        elif gui == "qt":
            from qtpy.QtCore import QCoreApplication, QMetaObject, Qt  # type: ignore

            app = QCoreApplication.instance()
            if app is not None:
                QMetaObject.invokeMethod(app, "quit", Qt.QueuedConnection)
    except Exception as exc:
        logging.getLogger("winrunner").debug("could not stop the window event loop: %s", exc)


def _window_unavailable() -> str | None:
    """Why the native window cannot be used, or ``None`` when it can."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "there is no graphical desktop session (DISPLAY / WAYLAND_DISPLAY not set)"
    try:
        import webview  # type: ignore  # noqa: F401
    except ImportError:
        return "pywebview is not installed (run setup.sh)"
    if sys.platform == "win32":
        from .platform import win32

        if not win32.webview2_version():
            return ("the Microsoft Edge WebView2 Runtime is not installed (needed for the app window; "
                    f"download: {WEBVIEW2_URL} or re-run install.bat)")
    elif sys.platform.startswith("linux") and _linux_gui() is None:
        return ("the app window needs WebKit2GTK for Python "
                f"(sudo apt install {LINUX_WINDOW_PACKAGES}, or run setup.sh)")
    return None


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
                    help="download a llama.cpp release for this backend and exit")
    ap.add_argument("--engine-tag", help="with --install-engine: the llama.cpp release to install (e.g. b11269); "
                                         "default: the latest release")
    ap.add_argument("--check", action="store_true", help="print the engine and the GPUs it can use, then exit")
    ap.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    ap.add_argument("--version", action="version", version=f"{PRODUCT_NAME} {__version__}")
    args = ap.parse_args(argv)

    paths = DataPaths(args.data_dir)
    paths.ensure()
    _setup_logging(args.log_level, paths)
    log = logging.getLogger("winrunner")

    if args.install_engine:
        return _install_engine(paths, args.install_engine, args.engine_tag)
    if args.check:
        return _check(paths)

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
        reason = _window_unavailable()
        if reason:
            log.warning("%s; opening the control panel in the browser instead", reason)
            ctx.extras["notice"] = f"{reason}. The control panel opened in the browser instead."
            open_mode = "browser"
        else:
            import webview  # type: ignore

    # The control panel offers an Exit button whenever it is opened by WinRunner itself: an app window, or a
    # browser tab (started from the desktop menu there is no terminal to press Ctrl+C in).
    ctx.extras["can_exit"] = open_mode in ("window", "browser") or sys.stderr is None
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
        fullscreen = [False]

        def toggle_fullscreen() -> bool:
            """F11 in the control panel (the app window has no browser full-screen mode of its own)."""
            window.toggle_fullscreen()
            fullscreen[0] = not fullscreen[0]
            return fullscreen[0]

        ctx.extras["toggle_fullscreen"] = toggle_fullscreen
        loaded = threading.Event()
        gui_done = threading.Event()
        failed: list[str] = []
        window.events.loaded += lambda *a: loaded.set()
        gui = "edgechromium" if sys.platform == "win32" else _linux_gui() if sys.platform.startswith("linux") else None

        def close_window() -> None:
            # pywebview's destroy() waits until the window has been shown: never call it on a thread that must
            # not block (the server's event loop, the watchdog).
            def destroy() -> None:
                try:
                    window.destroy()
                except Exception:
                    pass

            threading.Thread(target=destroy, name="window-close", daemon=True).start()
            if not gui_done.wait(3.0):
                _quit_gui(gui)

        def watchdog() -> None:
            # A broken browser engine leaves an empty window that never loads the page.
            if not loaded.wait(WINDOW_LOAD_TIMEOUT) and not server.should_exit:
                failed.append(f"the app window did not load within {WINDOW_LOAD_TIMEOUT:.0f} s")
                close_window()

        def request_exit() -> None:
            server.should_exit = True
            threading.Thread(target=close_window, name="window-exit", daemon=True).start()

        ctx.extras["request_exit"] = request_exit
        start_kw: dict = {"gui": gui, "private_mode": False}
        icon = _window_icon(gui)
        if icon:
            start_kw["icon"] = icon
        try:
            try:
                webview.start(watchdog, **start_kw)
            except TypeError:  # older pywebview without the icon parameter
                start_kw.pop("icon", None)
                webview.start(watchdog, **start_kw)
        except Exception as exc:
            failed.append(f"the app window could not be created ({exc})")
        gui_done.set()
        loaded.set()  # release the watchdog thread when the window closes early
        if failed and not server.should_exit:
            log.warning("%s; opening the control panel in the browser instead", failed[0])
            ctx.extras.update(window_mode=False, can_exit=True, request_exit=lambda: setattr(server, "should_exit", True),
                              toggle_fullscreen=None,
                              notice=f"The app window failed: {failed[0]}. Using the browser instead.")
            webbrowser.open(ui_url)
            while t.is_alive():  # keep serving until Exit is pressed in the control panel
                t.join(timeout=1)
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
