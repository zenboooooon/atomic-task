#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

base=runs/phase2
lock="$base/staged_v06.lock"
mkdir "$lock"
trap 'rmdir "$lock"' EXIT

run_stage() {
  local index="$1" source="$2" output="$3"
  if test -e "$output/checkpoint_0000.pt"; then
    echo "Refusing to restart existing stage: $output" >&2
    return 2
  fi
  echo "$(date -Is) stage $index training start, source=$source"
  scripts/run_pretraining_pilot.sh \
    --seed 0 --shared-from "$source" --stage-index "$index" \
    --rollout-per-task 256 --updates 500 --epochs 2 \
    --minibatch-per-task 32 --look-move-log-std -2.5 \
    --stop-camera-log-std -2.5 --output "$output" \
    > "$base/staged_v06_stage${index}.log" 2>&1
  echo "$(date -Is) stage $index training complete"
  scripts/run_multi_evaluation.sh \
    --checkpoint "$output/checkpoint_0500.pt" --episodes-per-agent 8 \
    --output "$output/validation_32.json" \
    > "$base/staged_v06_stage${index}_validation.log" 2>&1
  echo "$(date -Is) stage $index validation complete"
}

run_stage 2 "$base/pretrain_v06_512000/checkpoint_0500.pt" "$base/staged_v06_stage2"
if [[ "${1:-}" == "--through-stage3" ]]; then
  run_stage 3 "$base/staged_v06_stage2/checkpoint_0500.pt" "$base/staged_v06_stage3"
fi
echo "$(date -Is) requested stages complete"
