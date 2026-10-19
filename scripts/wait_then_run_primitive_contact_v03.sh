#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
old_status=/home/s6323038/atomic-task-treasure-wide-v02/runs/treasure/wide_touch_v04_600/pipeline.status
base=runs/primitive_contact_v03
mkdir -p "$base"
if test -e "$base/wait.status"; then
  echo "Watcher already registered: $base/wait.status" >&2
  exit 2
fi
printf '%s WAITING old_P_S_comparison\n' "$(date -Is)" > "$base/wait.status"
while true; do
  if test -f "$old_status" && grep -q ' COMPLETE$' "$old_status"; then
    printf '%s STARTING primitive_contact_v03\n' "$(date -Is)" > "$base/wait.status"
    exec scripts/run_primitive_contact_v03.sh
  fi
  if test -f "$old_status" && grep -q ' FAILED ' "$old_status"; then
    printf '%s BLOCKED old_P_S_comparison_failed\n' "$(date -Is)" > "$base/wait.status"
    exit 1
  fi
  sleep 60
done
