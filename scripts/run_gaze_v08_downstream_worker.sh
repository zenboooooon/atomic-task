#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
mode="$1"; base="$2"; condition="$3"; seed="$4"; index="$5"
model_dir="$base/${condition}_seed${seed}"
export TZ=Asia/Tokyo PYTHONPATH="$repo_dir/python" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ "$mode" == train ]]; then
  display=$((340 + index)); worker=$((340 + index))
else
  display=370; worker=380
fi
export DISPLAY=":$display"
mkdir -p "$model_dir"
[[ ! -e "/tmp/.X11-unix/X${display}" ]] || { echo "Display already in use: $DISPLAY" >&2; exit 2; }
"$repo_dir/.tools/xvfb/usr/bin/Xvfb" "$DISPLAY" -screen 0 1280x720x24 \
  -nolisten tcp -ac > "$model_dir/${mode}_xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM INT
sleep 1
kill -0 "$xvfb_pid"
if [[ "$mode" == train ]]; then
  trap 'printf "%s FAILED training\n" "$(date -Is)" > "$model_dir/pipeline.status"' ERR
  printf '%s RUNNING training\n' "$(date -Is)" > "$model_dir/pipeline.status"
  scripts/with_gpu.sh .venv-gpu/bin/python -u scripts/train_downstream_pilot.py \
    --pretrained "runs/gaze_v08_2000/seed${seed}/pretrain6_seed${seed}/checkpoint_2000.pt" \
    --downstream-task treasure --conditions "$condition" --seeds "$seed" --worker-id "$worker" \
    --rollout 256 --updates 2000 --checkpoint-every 200 --epochs 2 --minibatch 32 \
    --shared-lr 3e-5 --head-lr 1e-4 --output "$base" \
    --summary-output "$model_dir/summary.json"
  printf '%s COMPLETE training\n' "$(date -Is)" > "$model_dir/pipeline.status"
elif [[ "$mode" == validate ]]; then
  checkpoint=$(printf '%04d' "$index")
  scripts/with_gpu.sh .venv-gpu/bin/python -u scripts/evaluate_multi_pilot.py \
    --checkpoint "$model_dir/checkpoint_${checkpoint}.pt" \
    --layout-manifest manifests/gaze_v08_layouts.json --episodes-per-agent 2 \
    --seed 9000 --worker-id "$worker" --output "$model_dir/validation_${checkpoint}.json"
else
  echo "Unknown worker mode: $mode" >&2
  exit 2
fi
