#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
xvfb="$repo_dir/.tools/xvfb/usr/bin/Xvfb"
player="$repo_dir/unity/Build/TpsSmoke.x86_64"
python="$repo_dir/.venv/bin/python"
display_id=99

if [[ ! -x "$xvfb" || ! -x "$player" || ! -x "$python" ]]; then
  echo "Xvfb, Unity player, or Python environment is missing" >&2
  exit 2
fi

"$xvfb" ":$display_id" -screen 0 1280x720x24 -nolisten tcp -ac \
  > "$repo_dir/.tools/xvfb.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
sleep 1
if ! kill -0 "$xvfb_pid" 2>/dev/null; then
  echo "Xvfb failed; see $repo_dir/.tools/xvfb.log" >&2
  exit 3
fi
DISPLAY=":$display_id" "$python" -m tps_transfer.smoke \
  --executable "$player" --steps 120
