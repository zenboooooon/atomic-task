#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base=runs/treasure/rewardreset_v03
source_run=runs/treasure/resetfix_v02
source_checkpoint="$source_run/pretrain3_seed0/checkpoint_1333.pt"

test -f "$source_checkpoint"
if test -e "$base/pipeline.status"; then
  echo "Refusing to overwrite an existing run: $base" >&2
  exit 2
fi
mkdir -p "$base/downstream"
{
  printf 'source_pretraining=%s\n' "$source_checkpoint"
  printf 'source_validation=%s\n' "$source_run/pretrain3_validation.json"
  sha256sum "$source_checkpoint" unity/Build/TpsTreasure_Data/Managed/Assembly-CSharp.dll
} > "$base/provenance.txt"

stage=treasure_downstream
trap 'printf "%s FAILED %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"' ERR
printf "%s RUNNING %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/run_treasure_downstream.sh \
  --pretrained "$source_checkpoint" \
  --conditions P,S --seeds 0 --rollout 256 --updates 2000 \
  --checkpoint-every 200 --output "$base/downstream" \
  > "$base/downstream/train.log" 2>&1

stage=treasure_validation
printf "%s RUNNING %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/evaluate_treasure_series.sh "$base/downstream" \
  > "$base/downstream/validation.log" 2>&1

printf "%s COMPLETE\n" "$(date -Is)" > "$base/pipeline.status"
