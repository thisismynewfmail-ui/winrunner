#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  WinRunner setup for Linux Mint 22.x (Ubuntu 24.04 "noble" base)
#  with AMD Radeon GPUs (tuned for 2 x RX 6800 + Ryzen 5 3600).
#
#    ./setup.sh                      install / update everything
#    ./setup.sh --engine-tag bNNNNN  install another llama.cpp release
#    ./setup.sh --cpu                CPU-only engine (no GPU)
#    ./setup.sh --no-apt             skip the system packages (already installed)
#    ./setup.sh --firewall           also open the API port in ufw (LAN access)
#
#  Steps:
#    1. system packages (apt): Python venv, the Mesa RADV Vulkan driver and
#       tools, WebKit2GTK for the app window
#    2. adds you to the 'render' and 'video' groups (GPU compute access)
#    3. Python virtual environment in .venv with the required packages
#    4. downloads the pinned llama.cpp Vulkan build into data/engines
#    5. checks that the engine sees your GPUs
#    6. adds WinRunner to the application menu
#
#  Safe to run again: finished steps are skipped or refreshed.
# ---------------------------------------------------------------------------
set -euo pipefail

# llama.cpp release installed by default: the official Ubuntu Vulkan build that
# WinRunner's memory planner was verified with. Override with --engine-tag or
# LLAMA_CPP_TAG=bNNNNN ./setup.sh
LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b11269}"
BACKEND="vulkan"
DO_APT=1
DO_FIREWALL=0
API_PORT=5070

APT_PACKAGES=(
  python3 python3-venv python3-pip
  python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1  # app window (pywebview GTK backend)
  libvulkan1 mesa-vulkan-drivers vulkan-tools                     # Vulkan: Mesa RADV driver + vulkaninfo
  libgomp1 libssl3                                                 # llama.cpp runtime libraries
  pciutils curl ca-certificates tar xdg-utils
)

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engine-tag) LLAMA_CPP_TAG="${2:?--engine-tag needs a release tag, e.g. b11269}"; shift 2 ;;
    --engine-tag=*) LLAMA_CPP_TAG="${1#*=}"; shift ;;
    --cpu) BACKEND="cpu"; shift ;;
    --no-apt) DO_APT=0; shift ;;
    --firewall) DO_FIREWALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$HERE"
ME="$(id -un)"
VENV="$HERE/.venv"
PY="$VENV/bin/python"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;32m[%s]\033[0m %s\n' "$1" "$2"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

echo
bold " WINRUNNER  -  Local Inference Server  -  setup"
echo " =============================================="

if [ "$(id -u)" -eq 0 ]; then
  die "Run setup.sh as your normal user (not with sudo). It asks for your password when it needs it."
fi
[ "$(uname -m)" = "x86_64" ] || die "This setup is for x86-64 PCs (found $(uname -m))."

# ---- 0. system check ----------------------------------------------------------
# shellcheck source=/dev/null
. /etc/os-release 2>/dev/null || true
OS_NAME="${PRETTY_NAME:-unknown Linux}"
if [ "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}" != "noble" ]; then
  warn "This script targets Linux Mint 22.x / Ubuntu 24.04 (noble). Found: $OS_NAME. Package names may differ."
else
  echo " System: $OS_NAME"
fi
echo " Folder: $HERE"

SUDO=""
if [ "$DO_APT" -eq 1 ] || [ "$DO_FIREWALL" -eq 1 ]; then
  command -v sudo >/dev/null 2>&1 || die "sudo is required (or run with --no-apt after installing: ${APT_PACKAGES[*]})"
  SUDO="sudo"
fi

# ---- 1. system packages ---------------------------------------------------------
if [ "$DO_APT" -eq 1 ]; then
  step "1/6" "Installing system packages (your password may be requested)"
  $SUDO apt-get update
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PACKAGES[@]}"
else
  step "1/6" "Skipping system packages (--no-apt)"
fi

# ---- 2. GPU access ------------------------------------------------------------------
step "2/6" "GPU access"
RELOGIN=0
if [ "$BACKEND" != "cpu" ]; then
  for g in render video; do
    if getent group "$g" >/dev/null && ! id -nG "$ME" | tr ' ' '\n' | grep -qx "$g"; then
      if [ -n "$SUDO" ] || sudo -n true 2>/dev/null; then
        sudo usermod -aG "$g" "$ME" && echo " Added $ME to the '$g' group." && RELOGIN=1
      else
        warn "Add yourself to the '$g' group: sudo usermod -aG $g $ME (then log out and back in)"
      fi
    fi
  done
  shopt -s nullglob
  nodes=(/dev/dri/renderD*)
  shopt -u nullglob
  if [ ${#nodes[@]} -eq 0 ]; then
    warn "No GPU render devices (/dev/dri/renderD*) found. Is the amdgpu driver loaded? (lspci -k | grep -A3 VGA)"
  else
    for n in "${nodes[@]}"; do
      if [ -r "$n" ] && [ -w "$n" ]; then echo " $n: accessible"; else echo " $n: no access yet (log out and back in)"; fi
    done
  fi
  if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" | sed 's/^[[:space:]]*/ /' || true
  fi
else
  echo " CPU-only engine selected: no GPU setup needed."
fi

# ---- 3. Python environment ------------------------------------------------------------
step "3/6" "Python environment (.venv)"
# The app window needs the system PyGObject (python3-gi), which is built for the distribution's own
# Python (3.12 on Mint 22). Use the first system interpreter that can import it.
py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; }
py_ver() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }
SYS_PY=""
for c in /usr/bin/python3 /usr/bin/python3.12 /usr/bin/python3.13 /usr/bin/python3.11 /usr/bin/python3.10; do
  if [ -x "$c" ] && py_ok "$c" && "$c" -c 'import gi' >/dev/null 2>&1; then SYS_PY="$c"; break; fi
done
if [ -z "$SYS_PY" ]; then
  for c in /usr/bin/python3 "$(command -v python3 || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then SYS_PY="$c"; break; fi
  done
fi
[ -n "$SYS_PY" ] || die "python3 was not found (sudo apt install python3 python3-venv)"
py_ok "$SYS_PY" || die "Python 3.10 or newer is required (found $("$SYS_PY" --version 2>&1))."
if [ -x "$PY" ]; then
  if ! "$PY" -c 'import sys' >/dev/null 2>&1; then
    warn "The existing .venv is broken (Python was upgraded?); recreating it."
    rm -rf "$VENV"
  elif [ "$(py_ver "$PY")" != "$(py_ver "$SYS_PY")" ] \
       || ! grep -qs "include-system-site-packages = true" "$VENV/pyvenv.cfg"; then
    echo " Recreating .venv with $("$SYS_PY" --version) and the system packages (for the app window)."
    rm -rf "$VENV"
  fi
fi
if [ ! -x "$PY" ]; then
  # --system-site-packages: the app window uses the system's PyGObject / WebKit2GTK (apt packages)
  "$SYS_PY" -m venv --system-site-packages "$VENV" \
    || die "Could not create the virtual environment (sudo apt install python3-venv)"
  echo " Created $VENV with $("$PY" --version)"
else
  echo " Using $VENV ($("$PY" --version))"
fi
"$PY" -m pip install --upgrade --quiet pip wheel
"$PY" -m pip install --upgrade --quiet -r requirements.txt || die "Installing the Python packages failed (see above)."
echo " Python packages installed."
if "$PY" -c 'import gi; gi.require_version("Gtk", "3.0"); gi.require_version("WebKit2", "4.1"); from gi.repository import Gtk, WebKit2' >/dev/null 2>&1; then
  echo " App window: WebKit2GTK available."
else
  warn "WebKit2GTK for Python is missing: the control panel will open in your browser instead."
  warn "Fix: sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1"
fi

# ---- 4. llama.cpp engine ----------------------------------------------------------------
step "4/6" "llama.cpp engine: release $LLAMA_CPP_TAG ($BACKEND)"
"$PY" -m winrunner --install-engine "$BACKEND" --engine-tag "$LLAMA_CPP_TAG" \
  || die "Downloading llama.cpp $LLAMA_CPP_TAG failed. Check the internet connection and the release tag (https://github.com/ggml-org/llama.cpp/releases)."

# ---- 5. check --------------------------------------------------------------------------------
step "5/6" "Checking the engine and GPUs"
if ! "$PY" -m winrunner --check; then
  if [ "$BACKEND" != "cpu" ]; then
    warn "The engine sees no GPU yet."
    [ "$RELOGIN" -eq 1 ] && warn "You were just added to the render/video groups: log out and back in (or reboot), then run ./run.sh"
    warn "Diagnose with: vulkaninfo --summary   (should list 2 x AMD Radeon RX 6800 with driver RADV)"
  fi
fi

# ---- 6. application menu -------------------------------------------------------------------
step "6/6" "Application menu entry"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
cat > "$APPS/winrunner.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=WinRunner
GenericName=Local LLM Server
Comment=Local LLM inference server with an OpenAI / LM Studio compatible API
Exec="$HERE/run.sh"
Icon=$HERE/winrunner/static/img/icon.png
Terminal=false
Categories=Development;Utility;
Keywords=llm;llama;gguf;ai;server;
StartupNotify=true
EOF
chmod +x "$HERE/run.sh" "$HERE/setup.sh" 2>/dev/null || true
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS" >/dev/null 2>&1 || true
fi
echo " Added 'WinRunner' to the application menu ($APPS/winrunner.desktop)."

# ---- optional: firewall -------------------------------------------------------------------------
if [ "$DO_FIREWALL" -eq 1 ]; then
  if command -v ufw >/dev/null 2>&1 && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
    $SUDO ufw allow "$API_PORT/tcp" comment "WinRunner API" && echo " ufw: port $API_PORT/tcp allowed."
  else
    echo " ufw is not active: nothing to open."
  fi
fi

echo
bold " Setup complete."
echo "   Start WinRunner:   ./run.sh        (or 'WinRunner' in the application menu)"
echo "   API only:          ./run.sh --headless"
echo "   API endpoint:      http://$(hostname -I 2>/dev/null | awk '{print $1}'):$API_PORT/v1"
if [ "$RELOGIN" -eq 1 ]; then
  echo
  warn "Log out and back in once so the new 'render'/'video' group membership applies."
fi
