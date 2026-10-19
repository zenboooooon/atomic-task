#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :110 -screen 0 1280x720x24 -nolisten tcp -ac \
  > "$repo_dir/treasure-record-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:110
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
base=runs/treasure/rewardreset_v03/downstream/P_seed0
scripts/with_gpu.sh .venv-gpu/bin/python scripts/record_treasure_validation.py \
  --checkpoint "$base/checkpoint_2000.pt" \
  --reference "$base/validation_2000.json" \
  --output-dir runs/treasure/rewardreset_v03/videos/P_2000
