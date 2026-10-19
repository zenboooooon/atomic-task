#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
out=runs/gaze_v07/preview
mkdir -p "$out"
"$repo_dir/.tools/xvfb/usr/bin/Xvfb" :160 -screen 0 1280x720x24 \
  -nolisten tcp -ac > "$out/xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:160
export PYTHONPATH="$repo_dir/python"
for index in 0 1; do
  side=left
  if [[ "$index" == 1 ]]; then side=right; fi
  scripts/with_gpu.sh .venv-gpu/bin/python scripts/record_gaze_v07_preview.py --mode look \
    --layout-start "$index" --name "look-$side" --worker-id 160 --output-dir "$out"
done
for motion in static moving; do
  scripts/with_gpu.sh .venv-gpu/bin/python scripts/record_gaze_v07_preview.py \
    --mode "treasure-$motion" --worker-id 160 --output-dir "$out"
done
scripts/with_gpu.sh .venv-gpu/bin/python scripts/record_gaze_v07_preview.py --mode look \
  --look-movement --layout-start 0 --name look-moving --worker-id 160 --output-dir "$out"
scripts/with_gpu.sh .venv-gpu/bin/python scripts/check_gaze_v07_players.py
