#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Start WinRunner (after ./setup.sh).
#
#    ./run.sh                 control panel in its own app window
#    ./run.sh --browser       control panel in the web browser
#    ./run.sh --headless      API server only (e.g. over SSH)
#    ./run.sh --model ID      also load this model at start
#    ./run.sh --port 5070     other port (default from Settings > Network)
#
#  Without a desktop session (SSH) it starts headless automatically.
#  The API is served at http://<this-pc>:5070/v1
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$HERE"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ] || ! "$PY" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo "WinRunner is not set up yet: run ./setup.sh first." >&2
  if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || [ -t 1 ]; then exit 1; fi
  command -v notify-send >/dev/null 2>&1 && notify-send "WinRunner" "Run setup.sh in the WinRunner folder first."
  exit 1
fi

args=("$@")
mode=""
for a in "$@"; do
  case "$a" in --window|--browser|--headless) mode="$a" ;; esac
done
if [ -z "$mode" ] && [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  args+=(--headless)  # no desktop session: API server only
fi

# Started from the application menu there is no terminal: keep a log of the console output.
if [ ! -t 1 ]; then
  mkdir -p "$HERE/data/logs"
  exec >>"$HERE/data/logs/console.log" 2>&1
fi

exec "$PY" -m winrunner "${args[@]}"
