#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
display_id=98
"$xvfb" ":$display_id" -screen 0 1280x720x24 -nolisten tcp -ac > "$repo_dir/.tools/task-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
if ! kill -0 "$xvfb_pid" 2>/dev/null; then
  echo "Xvfb failed; see .tools/task-xvfb.log" >&2
  exit 2
fi
probe_script="${TPS_PROBE_SCRIPT:-task_probe.py}"
DISPLAY=":$display_id" "$repo_dir/.venv/bin/python" "$repo_dir/scripts/$probe_script" "$@"
