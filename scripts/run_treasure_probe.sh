#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :103 -screen 0 1280x720x24 -nolisten tcp -ac > "$repo_dir/treasure-probe-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:103
export PYTHONPATH="$repo_dir/python:$repo_dir/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/.venv/bin/python" "$repo_dir/scripts/task_probe.py" \
  --task treasure --policy treasure --worker-id 43 "$@"
