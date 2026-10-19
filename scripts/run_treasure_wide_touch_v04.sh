#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base=runs/treasure/wide_touch_v04_600
source_run=runs/treasure/resetfix_v02
source_checkpoint="$source_run/pretrain3_seed0/checkpoint_1333.pt"
source_manifest=manifests/pretrain3_v01_layouts.json

test -f "$source_checkpoint"
test -f "$source_manifest"
if test -e "$base/pipeline.status"; then
  echo "Refusing to overwrite an existing run: $base" >&2
  exit 2
fi
mkdir -p "$base/downstream"
{
  printf 'source_pretraining=%s\n' "$source_checkpoint"
  printf 'source_manifest=%s\n' "$source_manifest"
  sha256sum "$source_checkpoint" "$source_manifest" \
    manifests/pilot_layouts.json unity/Build/TpsTreasure_Data/Managed/Assembly-CSharp.dll
} > "$base/provenance.txt"

stage=treasure_downstream
trap 'printf "%s FAILED %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"' ERR
printf "%s RUNNING %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/run_treasure_downstream.sh \
  --pretrained "$source_checkpoint" \
  --pretrain-layout-manifest "$source_manifest" \
  --conditions P,S --seeds 0 --rollout 256 --updates 600 \
  --checkpoint-every 200 --output "$base/downstream" \
  > "$base/downstream/train.log" 2>&1

stage=treasure_validation
printf "%s RUNNING %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/evaluate_treasure_series.sh "$base/downstream" 600 \
  > "$base/downstream/validation.log" 2>&1

printf "%s COMPLETE\n" "$(date -Is)" > "$base/pipeline.status"
