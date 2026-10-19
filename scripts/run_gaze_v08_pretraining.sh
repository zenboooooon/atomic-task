#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base="$repo_dir/runs/gaze_v08_2000"
export TZ=Asia/Tokyo
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$base/APPROVED" ]] || { echo "Missing training authorization record" >&2; exit 2; }
[[ ! -e "$base/pipeline.status" ]] || { echo "Refusing to overwrite an existing run" >&2; exit 2; }
.venv-gpu/bin/python - <<'PY'
import json
from pathlib import Path
from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.primitive_variants import COLOR_TASKS, appearance_protocol
base=Path('runs/gaze_v08_2000')
plan=json.loads((base/'run.plan.json').read_text())
assert plan['appearance_revision']=='color2_brown_target_white_avatar'
assert plan['training_seeds']==[0,1,2] and plan['training_updates']==2000
assert plan['decisions_per_update']==768 and plan['decisions_per_seed']==1536000
checks=json.loads(Path('runs/gaze_v08/preflight/report.json').read_text())
assert checks['palettes']==appearance_protocol()['variants']
assert checks['palettes'][2]['target_rgb']==[.45,.22,.08]
assert checks['palettes'][2]['avatar_rgb']==[.94,.94,.94]
assert checks['initial_look_layouts_match_across_colors']
assert all(checks['look_scripted_successes_by_head'][f'look_color{i}']==6 for i in range(3))
assert checks['look_hold_ticks']==25 and not checks['look_movement_penalties']
assert set(checks['behaviors'])==set(COLOR_TASKS)
manifest=Path('manifests/gaze_v08_layouts.json')
player=Path('unity/Build/TpsPretrainColors.x86_64')
assert layout_manifest_sha256(player,manifest)==plan['verification']['layout_manifest_sha256']
assert unity_build_sha256(player)==plan['verification']['unity_build_sha256']
assert plan['verification']['brown_gpu_smoke_updates']==2
assert plan['verification']['brown_gpu_smoke_agent_decisions']==1536
assert plan['verification']['all_heads_updated']
PY
tar --exclude='__pycache__' --exclude='._*' -czf "$base/source_snapshot.tar.gz" \
  python scripts configs manifests unity/Assets/Scripts unity/Assets/Editor
git diff --binary > "$base/source_changes.diff"
printf '%s\n' "$$" > "$base/pipeline.pid"
declare -A active=()
active_count=0
next=0
completed=0
cleanup() {
  for pid in "${!active[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
interrupted() {
  printf '%s STOPPED interrupted\n' "$(date -Is)" > "$base/pipeline.status"
  cleanup
  exit 143
}
trap cleanup EXIT
trap interrupted TERM INT
trap 'printf "%s FAILED supervisor\n" "$(date -Is)" > "$base/pipeline.status"' ERR
while (( next < 3 || active_count > 0 )); do
  parallel=$(cat "$base/parallelism")
  [[ "$parallel" =~ ^[123]$ ]] || { echo "parallelism must be 1, 2 or 3" >&2; exit 2; }
  while (( next < 3 && active_count < parallel )); do
    seed="$next"
    mkdir -p "$base/seed${seed}"
    setsid bash scripts/run_gaze_v08_pretrain_seed.sh "$base" "$seed" \
      > "$base/seed${seed}/pipeline.log" 2>&1 &
    pid=$!
    active[$pid]="$seed"
    active_count=$((active_count + 1))
    printf '%s\n' "$pid" > "$base/seed${seed}/pipeline.pid"
    printf '%s LAUNCHED seed=%s pid=%s\n' "$(date -Is)" "$seed" "$pid"
    next=$((next + 1))
  done
  for pid in "${!active[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      seed="${active[$pid]}"
      if wait "$pid"; then
        unset "active[$pid]"
        active_count=$((active_count - 1))
        completed=$((completed + 1))
        printf '%s COMPLETED seed=%s\n' "$(date -Is)" "$seed"
      else
        printf '%s FAILED seed%s\n' "$(date -Is)" "$seed" > "$base/pipeline.status"
        exit 1
      fi
    fi
  done
  if (( next < 3 || active_count > 0 )); then
    printf '%s RUNNING completed=%s/3 active=%s parallel_limit=%s\n' \
      "$(date -Is)" "$completed" "$active_count" "$parallel" > "$base/pipeline.status"
    sleep 5
  fi
done
printf '%s COMPLETE primitive_training_and_validation\n' "$(date -Is)" > "$base/pipeline.status"
