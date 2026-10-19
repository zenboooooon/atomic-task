#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base="$repo_dir/runs/gaze_v06_2000"
export TZ=Asia/Tokyo
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$base/APPROVED" ]] || { echo "Missing training authorization record" >&2; exit 2; }
[[ ! -e "$base/pipeline.status" ]] || { echo "Refusing to overwrite an existing run" >&2; exit 2; }
.venv-gpu/bin/python - <<'PY'
import json
from pathlib import Path
from tps_transfer.build_identity import layout_manifest_sha256
checks=json.loads(Path('runs/gaze_v06/preview/player_checks.json').read_text())
assert sum(x['unique_layouts_checked'] for x in checks['look']) == 160
assert all(x['all_targets_visible_at_edges'] for x in checks['look'])
layout_manifest_sha256(Path('unity/Build/TpsPretrainTwo.x86_64'),
                       Path('manifests/gaze_v06_layouts.json'))
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
    setsid bash scripts/run_gaze_v06_pretrain_seed.sh "$base" "$seed" \
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
