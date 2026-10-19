#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
out=runs/gaze_v05_look1p5/preview
mkdir -p "$out"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :130 -screen 0 1280x720x24 -nolisten tcp -ac > "$out/xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:130
export PYTHONPATH="$repo_dir/python"
"$repo_dir/.venv-gpu/bin/python" scripts/smoke_gaze_v05_look1p5_players.py > "$out/smoke.log" 2>&1
for mode in look approach treasure-static treasure-moving; do
  "$repo_dir/.venv-gpu/bin/python" scripts/record_gaze_v05_look1p5_preview.py --mode "$mode" --worker-id 130 --output-dir "$out" > "$out/$mode.record.log" 2>&1
done
"$repo_dir/.venv-gpu/bin/python" scripts/record_gaze_v05_look1p5_contact.py > "$out/treasure-static-contact.record.log" 2>&1
