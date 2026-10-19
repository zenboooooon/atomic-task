#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base="$1"
seed="$2"
output="$base/seed${seed}/pretrain_shared_seed${seed}"
seed_status="$base/seed${seed}/pipeline.status"
display=$((290 + seed))
worker=$((300 + seed))
export TZ=Asia/Tokyo
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
export DISPLAY=":$display" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$output"
stage=pretraining
xvfb_pid=
cleanup() {
  if [[ -n "$xvfb_pid" ]]; then kill -TERM "$xvfb_pid" 2>/dev/null || true; fi
}
failed() { printf '%s FAILED %s\n' "$(date -Is)" "$stage" > "$seed_status"; }
trap cleanup EXIT
trap failed ERR
trap 'exit 143' TERM INT
"$repo_dir/.tools/xvfb/usr/bin/Xvfb" "$DISPLAY" -screen 0 1280x720x24 \
  -nolisten tcp -ac > "$output/xvfb.log" 2>&1 &
xvfb_pid=$!
sleep 1
kill -0 "$xvfb_pid"
printf '%s RUNNING pretraining\n' "$(date -Is)" > "$seed_status"
scripts/with_gpu.sh .venv-gpu/bin/python -u scripts/train_pretraining_pilot.py \
  --seed "$seed" --task-set gaze6 --share-heads-by-task --model-width 256 --worker-id "$worker" \
  --rollout-per-task 128 --updates 2000 --checkpoint-every 100 \
  --epochs 2 --minibatch-per-task 16 --shared-lr 3e-5 --head-lr 1e-4 \
  --look-move-log-std -0.5 \
  --layout-manifest manifests/gaze_v08_layouts.json --output "$output" \
  > "$output/train.log" 2>&1
stage=primitive_validation
printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$seed_status"
scripts/with_gpu.sh .venv-gpu/bin/python -u scripts/evaluate_multi_pilot.py \
  --checkpoint "$output/checkpoint_2000.pt" \
  --layout-manifest manifests/gaze_v08_layouts.json --episodes-per-agent 6 \
  --seed 9000 --worker-id "$worker" \
  --output "$base/seed${seed}/pretrain6_validation.json" \
  > "$base/seed${seed}/pretrain6_validation.log" 2>&1
printf '%s COMPLETE\n' "$(date -Is)" > "$seed_status"
