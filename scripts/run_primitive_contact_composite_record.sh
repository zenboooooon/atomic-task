#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :111 -screen 0 1280x720x24 -nolisten tcp -ac \
  > "$repo_dir/runs/primitive_contact_v03/composite_record_xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:111
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"

for condition in P S F; do
  base="runs/primitive_contact_v03/downstream/${condition}_seed0"
  "$repo_dir/scripts/with_gpu.sh" "$repo_dir/.venv-gpu/bin/python" \
    "$repo_dir/scripts/record_primitive_contact_composite.py" \
    --condition "$condition" --worker-id 76 \
    --checkpoint "$base/checkpoint_2000.pt" \
    --reference "$base/validation_2000.json" \
    --output-dir "runs/primitive_contact_v03/composite_policy_videos/${condition}_2000"
done
