#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
base=runs/treasure/rewardreset_v03
run_dir="$base/downstream/P_seed0"

stage=P_validation
trap 'printf "%s FAILED %s\n" "$(date -Is)" "$stage" > "$base/P_validation.status"' ERR
printf "%s RUNNING %s\n" "$(date -Is)" "$stage" > "$base/P_validation.status"
for update in 2000 200 400 600 800 1000 1200 1400 1600 1800; do
  checkpoint="$run_dir/checkpoint_$(printf '%04d' "$update").pt"
  output="$run_dir/validation_$(printf '%04d' "$update").json"
  test -f "$checkpoint"
  if [[ -s "$output" ]] && python3 -m json.tool "$output" > /dev/null; then
    continue
  fi
  printf '%s evaluating P update=%d\n' "$(date -Is)" "$update"
  scripts/run_treasure_validation.sh \
    --checkpoint "$checkpoint" --episodes-per-agent 2 --output "$output"
done
printf "%s COMPLETE %s\n" "$(date -Is)" "$stage" > "$base/P_validation.status"
