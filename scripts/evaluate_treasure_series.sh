#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
run_dir="${1:?pass the downstream run directory}"
last_update="${2:-2000}"
conditions=(P S)
if [[ -f "${run_dir}.plan.json" ]]; then
  plan_line=$(python3 - "${run_dir}.plan.json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
assert plan["experiment"] == "primitive_contact_v03"
assert plan["validation_episodes_per_checkpoint"] == 32
assert plan["conditions"] == ["P", "S", "F"]
assert plan["checkpoint_every"] == 200
assert plan["updates"] % plan["checkpoint_every"] == 0
print(",".join(plan["conditions"]), plan["updates"], plan["checkpoint_every"])
PY
  )
  read -r plan_conditions plan_updates plan_every <<< "$plan_line"
  if [[ -z "$plan_conditions" || -z "$plan_updates" || -z "$plan_every" ]]; then
    echo "Invalid evaluation plan: ${run_dir}.plan.json" >&2
    exit 2
  fi
  IFS=, read -r -a conditions <<< "$plan_conditions"
  last_update="$plan_updates"
  echo "Using validation plan: conditions=$plan_conditions updates=$last_update checkpoint_every=$plan_every" >&2
fi
if (( last_update < 200 || last_update % 200 != 0 )); then
  echo "last update must be a positive multiple of 200" >&2
  exit 2
fi
for condition in "${conditions[@]}"; do
  for (( update=200; update<=last_update; update+=200 )); do
    checkpoint="$run_dir/${condition}_seed0/checkpoint_$(printf '%04d' "$update").pt"
    output="$run_dir/${condition}_seed0/validation_$(printf '%04d' "$update").json"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 2
    fi
    if [[ -s "$output" ]] && python3 -m json.tool "$output" > /dev/null; then
      echo "Validation already complete: $output"
      continue
    fi
    "$repo_dir/scripts/run_treasure_validation.sh" \
      --checkpoint "$checkpoint" --episodes-per-agent 2 --output "$output"
  done
done
