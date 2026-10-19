#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
old_run=/home/s6323038/atomic-task-treasure-wide-v02/runs/treasure/wide_touch_v04_600
base=runs/primitive_contact_v03
source_manifest=manifests/primitive_contact_v03_layouts.json
source_checkpoint="$base/pretrain3_seed0/checkpoint_1333.pt"

if ! test -f "$old_run/pipeline.status" || ! grep -q ' COMPLETE$' "$old_run/pipeline.status"; then
  echo "Current P/S comparison has not completed; refusing to start training" >&2
  exit 2
fi
if test -e "$base/pipeline.status"; then
  echo "Refusing to overwrite an existing run: $base" >&2
  exit 2
fi
test -f "$source_manifest"
mkdir -p "$base/pretrain3_seed0" "$base/downstream"
{
  printf 'old_comparison=%s\n' "$old_run"
  printf 'source_manifest=%s\n' "$source_manifest"
  sha256sum "$source_manifest" manifests/pilot_layouts.json \
    unity/Build/TpsPretrainThree_Data/Managed/Assembly-CSharp.dll \
    unity/Build/TpsTreasure_Data/Managed/Assembly-CSharp.dll
} > "$base/provenance.txt"

stage=primitive_pretraining
trap 'printf "%s FAILED %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"' ERR
printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/run_treasure_pretraining.sh \
  --seed 0 --rollout-per-task 128 --updates 1333 --checkpoint-every 100 \
  --layout-manifest "$source_manifest" --output "$base/pretrain3_seed0" \
  > "$base/pretrain3_seed0/train.log" 2>&1

stage=primitive_validation
printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/run_treasure_validation.sh \
  --checkpoint "$source_checkpoint" --layout-manifest "$source_manifest" \
  --episodes-per-agent 8 --output "$base/pretrain3_validation.json" \
  > "$base/pretrain3_validation.log" 2>&1

stage=treasure_camera_verification
printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/.venv/bin/python" scripts/verify_treasure_fixed_camera_v03.py \
  > "$base/treasure_camera_verification.log" 2>&1

stage=treasure_downstream
printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/run_treasure_downstream.sh \
  --pretrained "$source_checkpoint" --pretrain-layout-manifest "$source_manifest" \
  --conditions P,S,F --seeds 0 --rollout 256 --updates 2000 \
  --checkpoint-every 200 --output "$base/downstream" \
  > "$base/downstream/train.log" 2>&1

stage=treasure_validation
printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
scripts/evaluate_treasure_series.sh "$base/downstream" 2000 \
  > "$base/downstream/validation.log" 2>&1
printf '%s COMPLETE\n' "$(date -Is)" > "$base/pipeline.status"
