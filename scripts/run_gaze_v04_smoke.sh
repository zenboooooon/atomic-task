#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
"$repo_dir/.tools/xvfb/usr/bin/Xvfb" :119 -screen 0 1280x720x24 -nolisten tcp -ac \
  > "$repo_dir/gaze-v04-smoke-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:119
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/.venv-gpu/bin/python" "$repo_dir/scripts/smoke_gaze_v04_players.py"
