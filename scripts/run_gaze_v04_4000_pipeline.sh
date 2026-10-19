#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base=runs/gaze_v04_4000
previous=runs/gaze_v04
resume_checkpoint="$previous/seed0/pretrain2_seed0/checkpoint_1333.pt"

[[ -f "$base/APPROVED" ]] || { echo "The 4000-update continuation needs approval." >&2; exit 2; }
[[ ! -e "$base/pipeline.status" ]] || { echo "Refusing to overwrite an existing run." >&2; exit 2; }
[[ -f "$resume_checkpoint" ]] || { echo "The seed-0 resume checkpoint is missing." >&2; exit 2; }
[[ "$(cat "$previous/pipeline.status")" == *"STOPPED superseded_by_gaze_v04_4000" ]] || {
  echo "The previous pipeline has not stopped." >&2; exit 2;
}

export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/.venv-gpu/bin/python" - <<'PY'
from pathlib import Path
from tps_transfer.build_identity import layout_manifest_sha256
manifest = Path('manifests/gaze_v04_layouts.json')
for name in ('TpsPretrainTwo', 'TpsTreasure'):
    layout_manifest_sha256(Path('unity/Build') / (name + '.x86_64'), manifest)
PY

stage=pretraining_parallel
pid0= pid1= pid2=
cleanup() {
  for pid in "$pid0" "$pid1" "$pid2"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
}
failed() {
  printf '%s FAILED %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
  cleanup
}
trap failed ERR
trap cleanup EXIT

launch_pretraining() {
  local seed="$1" display="$2" worker="$3" output="$base/seed${1}/pretrain2_seed${1}"
  mkdir -p "$output" "$base/seed${seed}/downstream"
  local resume=()
  if [[ "$seed" == 0 ]]; then resume=(--resume "$resume_checkpoint"); fi
  (
    set -Eeuo pipefail
    xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
    "$xvfb" ":$display" -screen 0 1280x720x24 -nolisten tcp -ac > "$output/xvfb.log" 2>&1 &
    xvfb_pid=$!
    train_pid=
    close_children() {
      if [[ -n "$train_pid" ]]; then
        pkill -TERM -P "$train_pid" 2>/dev/null || true
        kill -TERM "$train_pid" 2>/dev/null || true
      fi
      kill -TERM "$xvfb_pid" 2>/dev/null || true
    }
    trap close_children EXIT
    trap 'exit 143' TERM INT
    sleep 1
    kill -0 "$xvfb_pid"
    export DISPLAY=":$display" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    "$repo_dir/scripts/with_gpu.sh" "$repo_dir/.venv-gpu/bin/python" \
      "$repo_dir/scripts/train_pretraining_pilot.py" \
      --seed "$seed" --task-set gaze2 --worker-id "$worker" \
      --rollout-per-task 128 --updates 4000 --checkpoint-every 100 \
      --layout-manifest manifests/gaze_v04_layouts.json \
      --output "$output" "${resume[@]}" &
    train_pid=$!
    wait "$train_pid"
    train_pid=
  ) > "$output/train.log" 2>&1 &
  launched_pid=$!
}

printf '%s RUNNING pretraining_parallel_seed0_seed1\n' "$(date -Is)" > "$base/pipeline.status"
launch_pretraining 0 114 64; pid0=$launched_pid
launch_pretraining 1 115 65; pid1=$launched_pid
wait "$pid0"; pid0=
printf '%s RUNNING pretraining_parallel_seed1_seed2\n' "$(date -Is)" > "$base/pipeline.status"
launch_pretraining 2 117 66; pid2=$launched_pid
wait "$pid1"; pid1=
wait "$pid2"; pid2=

for seed in 0 1 2; do
  seed_dir="$base/seed${seed}"
  source="$seed_dir/pretrain2_seed${seed}"
  stage="primitive_validation_seed${seed}"
  printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
  scripts/run_treasure_validation.sh \
    --checkpoint "$source/checkpoint_4000.pt" \
    --layout-manifest manifests/gaze_v04_layouts.json \
    --episodes-per-agent 6 --output "$seed_dir/pretrain2_validation.json" \
    > "$seed_dir/pretrain2_validation.log" 2>&1

  stage="downstream_training_seed${seed}"
  printf '%s RUNNING %s\n' "$(date -Is)" "$stage" > "$base/pipeline.status"
  scripts/run_treasure_downstream.sh \
    --pretrained "$source/checkpoint_4000.pt" \
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
