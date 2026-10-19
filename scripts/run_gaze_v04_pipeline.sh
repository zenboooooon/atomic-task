#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base=runs/gaze_v04
if [[ ! -f "$base/APPROVED" ]]; then
  echo "Training requires the user's approval after reviewing preview videos." >&2
  exit 2
fi
if [[ -e "$base/pipeline.status" ]]; then
  echo "Refusing to overwrite an existing gaze v0.4 run." >&2
  exit 2
fi
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/.venv-gpu/bin/python" - <<'PY'
from pathlib import Path
from tps_transfer.build_identity import layout_manifest_sha256
manifest = Path('manifests/gaze_v04_layouts.json')
for name in ('TpsPretrainTwo', 'TpsTreasure'):
    layout_manifest_sha256(Path('unity/Build') / (name + '.x86_64'), manifest)
PY

stage=pretraining
trap 'printf "%s FAILED %s\n" "$(date -Is)" "$stage" > "$base/pipeline.status"' ERR
printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
for seed in 0 1 2; do
  seed_dir="$base/seed${seed}"
  source="$seed_dir/pretrain2_seed${seed}"
  mkdir -p "$source" "$seed_dir/downstream"
  stage="pretraining_seed${seed}"
  printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
  scripts/run_treasure_pretraining.sh \
    --seed "$seed" --task-set gaze2 --rollout-per-task 128 \
    --updates 1333 --checkpoint-every 100 \
    --layout-manifest manifests/gaze_v04_layouts.json --output "$source" \
    > "$source/train.log" 2>&1

  stage="primitive_validation_seed${seed}"
  printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
  scripts/run_treasure_validation.sh \
    --checkpoint "$source/checkpoint_1333.pt" \
    --layout-manifest manifests/gaze_v04_layouts.json \
    --episodes-per-agent 6 --output "$seed_dir/pretrain2_validation.json" \
    > "$seed_dir/pretrain2_validation.log" 2>&1

  stage="downstream_training_seed${seed}"
  printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
  scripts/run_treasure_downstream.sh \
    --pretrained "$source/checkpoint_1333.pt" \
    --conditions P,S,F --seeds "$seed" --rollout 256 \
    --updates 2000 --checkpoint-every 200 \
    --output "$seed_dir/downstream" \
    > "$seed_dir/downstream/train.log" 2>&1

  stage="downstream_validation_seed${seed}"
  printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
  for condition in P S F; do
    for update in 200 400 600 800 1000 1200 1400 1600 1800 2000; do
      index=$(printf '%04d' "$update")
      model_dir="$seed_dir/downstream/${condition}_seed${seed}"
      scripts/run_treasure_validation.sh \
        --checkpoint "$model_dir/checkpoint_${index}.pt" \
        --episodes-per-agent 2 \
        --output "$model_dir/validation_${index}.json" \
        >> "$seed_dir/downstream/validation.log" 2>&1
    done
  done
done
printf '%s COMPLETE\n' "$(date -Is)" > "$base/pipeline.status"
