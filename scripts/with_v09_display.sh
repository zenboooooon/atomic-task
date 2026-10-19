#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
export TZ=Asia/Tokyo PYTHONPATH="$repo_dir/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export DISPLAY=":${TPS_DISPLAY:-450}"
mkdir -p runs/v09_runtime
"$repo_dir/.tools/xvfb/usr/bin/Xvfb" "$DISPLAY" -screen 0 1280x720x24 \
  -nolisten tcp -ac > "runs/v09_runtime/xvfb_${TPS_DISPLAY:-450}.log" 2>&1 &
xvfb_pid=$!
trap 'kill -TERM "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
scripts/with_gpu.sh "$@"
