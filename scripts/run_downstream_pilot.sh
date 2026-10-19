#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :98 -screen 0 1280x720x24 -nolisten tcp -ac > "$repo_dir/.tools/downstream-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
if ! kill -0 "$xvfb_pid" 2>/dev/null; then
  echo "Xvfb failed; see .tools/downstream-xvfb.log" >&2
  exit 2
fi
DISPLAY=:98 "$repo_dir/scripts/with_gpu.sh" "$repo_dir/.venv-gpu/bin/python" \
  "$repo_dir/scripts/train_downstream_pilot.py" "$@"
