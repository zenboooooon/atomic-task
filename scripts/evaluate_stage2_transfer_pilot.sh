#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

base=runs/phase3/stage2_transfer_pilot_51200
lock=runs/phase3/stage2_transfer_pilot_validation.lock
mkdir "$lock"
trap 'rmdir "$lock"' EXIT

for seed in 0 1; do
  for condition in S R P F; do
    run="$base/${condition}_seed${seed}"
    test -f "$run/checkpoint_0200.pt"
    if test -f "$run/validation_32.json"; then
      echo "$(date -Is) validation already complete: ${condition} seed ${seed}" >> "$base/pipeline.log"
      continue
    fi
    echo "$(date -Is) evaluating ${condition} seed ${seed}" >> "$base/pipeline.log"
    scripts/run_multi_evaluation.sh \
      --checkpoint "$run/checkpoint_0200.pt" --episodes-per-agent 2 \
      --output "$run/validation_32.json" \
      > "$run/validation_32.log" 2>&1
  done
done
echo "$(date -Is) matched composite pilot validation complete" >> "$base/pipeline.log"
