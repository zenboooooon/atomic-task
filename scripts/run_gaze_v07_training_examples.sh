#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/python" DISPLAY=:220 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
parent="$repo_dir/runs/gaze_v07_2000/qualitative"
mkdir -p "$parent"
"$repo_dir/.tools/xvfb/usr/bin/Xvfb" "$DISPLAY" -screen 0 1280x720x24 \
  -nolisten tcp -ac > "$parent/xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
checkpoint=$(find "$repo_dir/runs/gaze_v07_2000/seed1/pretrain2_seed1" \
  -maxdepth 1 -name 'checkpoint_*.pt' -type f | sort | tail -n 1)
[[ -n "$checkpoint" ]]
update="$(basename "$checkpoint" .pt)"
scripts/with_gpu.sh .venv-gpu/bin/python -u scripts/record_gaze_v07_training_examples.py \
  --checkpoint "$checkpoint" --output-dir "$parent/seed1_${update}_$(date +%Y%m%d_%H%M%S)" \
  --worker-id 220
