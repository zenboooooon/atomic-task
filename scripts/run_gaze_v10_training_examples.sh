#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python" DISPLAY=:630 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
parent="$PWD/runs/qualitative"
mkdir -p "$parent"
[[ ! -e /tmp/.X11-unix/X630 ]] || { echo 'Video display :630 is already in use' >&2; exit 2; }
"$PWD/.tools/xvfb/usr/bin/Xvfb" "$DISPLAY" -screen 0 1280x720x24 -nolisten tcp -ac > "$parent/xvfb_630.log" 2>&1 &
display_pid=$!
trap 'kill "$display_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$display_pid"
scripts/with_gpu.sh .venv-gpu/bin/python -u scripts/record_gaze_v10_training_examples.py "$@"
