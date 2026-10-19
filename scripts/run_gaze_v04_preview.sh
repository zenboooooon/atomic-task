#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :111 -screen 0 1280x720x24 -nolisten tcp -ac \
  > "$repo_dir/gaze-v04-preview-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:111
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"

for mode in look approach treasure-static treasure-moving; do
  "$repo_dir/.venv-gpu/bin/python" "$repo_dir/scripts/record_gaze_v04_preview.py" \
    --mode "$mode" --worker-id 86 \
    --output-dir "$repo_dir/runs/gaze_v04/preview"
done
