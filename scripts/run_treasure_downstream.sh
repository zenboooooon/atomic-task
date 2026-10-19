#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
args=("$@")
output=""
for (( index=0; index<${#args[@]}-1; index++ )); do
  if [[ "${args[index]}" == "--output" ]]; then
    output="${args[index+1]}"
  fi
done
if [[ -n "$output" && -f "${output}.plan.json" ]]; then
  plan_line=$(python3 - "${output}.plan.json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
assert plan["experiment"] == "primitive_contact_v03"
assert plan["downstream_task"] == "treasure"
assert plan["seed"] == 0 and plan["rollout"] == 256
assert plan["conditions"] == ["P", "S", "F"]
assert plan["agent_decisions_per_condition"] == plan["updates"] * plan["rollout"]
print(",".join(plan["conditions"]), plan["updates"], plan["checkpoint_every"])
PY
  )
  read -r plan_conditions plan_updates plan_every <<< "$plan_line"
  if [[ -z "$plan_conditions" || -z "$plan_updates" || -z "$plan_every" ]]; then
    echo "Invalid downstream plan: ${output}.plan.json" >&2
    exit 2
  fi
  echo "Using downstream plan: conditions=$plan_conditions updates=$plan_updates checkpoint_every=$plan_every" >&2
  set -- "$@" --conditions "$plan_conditions" --updates "$plan_updates" \
    --checkpoint-every "$plan_every"
fi
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :138 -screen 0 1280x720x24 -nolisten tcp -ac > "$repo_dir/treasure-downstream-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:138
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/scripts/with_gpu.sh" "$repo_dir/.venv-gpu/bin/python" \
  "$repo_dir/scripts/train_downstream_pilot.py" \
  --downstream-task treasure --worker-id 145 "$@"
