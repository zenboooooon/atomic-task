#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

base=runs/phase3/stage2_transfer_pilot_51200
source=runs/phase2/staged_v06_stage2/checkpoint_0500.pt
lock=runs/phase3/stage2_transfer_pilot_51200.lock
mkdir -p runs/phase3
mkdir "$lock"
trap 'rmdir "$lock"' EXIT
if test -e "$base/S_seed0/checkpoint_0000.pt"; then
  echo "Refusing to restart existing transfer pilot: $base" >&2
  exit 2
fi
mkdir -p "$base"

echo "$(date -Is) matched composite pilot training start" > "$base/pipeline.log"
scripts/run_downstream_pilot.sh \
  --pretrained "$source" --seeds 0,1 --conditions S,R,P,F \
  --rollout 256 --updates 200 --epochs 2 --minibatch 32 \
  --shared-lr 3e-5 --head-lr 1e-4 --output "$base" \
  > "$base/train.log" 2>&1
echo "$(date -Is) matched composite pilot training complete" >> "$base/pipeline.log"

for seed in 0 1; do
  for condition in S R P F; do
    run="$base/${condition}_seed${seed}"
    echo "$(date -Is) evaluating ${condition} seed ${seed}" >> "$base/pipeline.log"
    scripts/run_multi_evaluation.sh \
      --checkpoint "$run/checkpoint_0200.pt" --episodes-per-agent 2 \
      --output "$run/validation_32.json" \
      > "$run/validation_32.log" 2>&1
  done
done
echo "$(date -Is) matched composite pilot validation complete" >> "$base/pipeline.log"
