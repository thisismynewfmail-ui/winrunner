"""Small shared helpers."""

from __future__ import annotations

import ipaddress
import re
import secrets
import socket
import sys
import time

MiB = 1024 * 1024
GiB = 1024 * MiB

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def human_count(n: float | int | None) -> str:
    """Parameter counts: 7.62B, 596M."""
    if n is None:
        return "-"
    n = float(n)
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(int(n))


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def port_available(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform != "win32":
            # like uvicorn's own socket: connections of a previous run in TIME_WAIT must not block a restart
            # (a port another program listens on is still refused). On Windows the option means port sharing.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def lan_addresses() -> list[str]:
    """IPv4 addresses of this machine that other LAN hosts can reach."""
    addrs: set[str] = set()
    try:
        import psutil

        for ifname, entries in psutil.net_if_addrs().items():
            for e in entries:
                if e.family == socket.AF_INET and e.address:
                    addrs.add(e.address)
    except Exception:
        pass
    # Route-based discovery (no packets are sent for UDP connect).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    out = []
    for a in addrs:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            continue
        out.append(a)

    def rank(a: str) -> tuple:
        ip = ipaddress.ip_address(a)
        # Prefer typical home LAN ranges first.
        return (0 if a.startswith("192.168.") else 1 if ip.is_private else 2, a)

    return sorted(out, key=rank)


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in ("localhost", "testclient"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def new_id(prefix: str = "", n: int = 6) -> str:
    return f"{prefix}{secrets.token_hex(n)}"


def now() -> float:
    return time.time()


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._@/-]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")
