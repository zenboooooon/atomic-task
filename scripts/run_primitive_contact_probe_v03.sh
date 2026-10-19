#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
"$xvfb" :115 -screen 0 1280x720x24 -nolisten tcp -ac > "$repo_dir/primitive-contact-probe-xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$xvfb_pid"
export DISPLAY=:115
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/.venv-gpu/bin/python" "$repo_dir/scripts/probe_primitive_contact_v03.py"
