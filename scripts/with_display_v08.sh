#!/usr/bin/env bash
# Launch one command on a temporary display, using the bundled Xvfb runtime.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
display_number="${TPS_V08_DISPLAY_NUMBER:-285}"
export DISPLAY=":$display_number"
if [[ -e "/tmp/.X11-unix/X$display_number" ]]; then
  echo "Display $DISPLAY is already in use" >&2
  exit 2
fi
mkdir -p "$repo_dir/runs/gaze_v08"
"$repo_dir/.tools/xvfb/usr/bin/Xvfb" "$DISPLAY" -screen 0 1280x720x24 \
  -nolisten tcp -ac > "$repo_dir/runs/gaze_v08/xvfb-$display_number.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
for attempt in {1..30}; do
  [[ -e "/tmp/.X11-unix/X$display_number" ]] && break
  kill -0 "$xvfb_pid"
  sleep 0.1
done
[[ -e "/tmp/.X11-unix/X$display_number" ]]
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
"$@"
