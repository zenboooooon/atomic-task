#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
project="$repo_dir/unity"
log_file="$repo_dir/.tools/build-smoke.log"

if [[ ! -x "$editor" ]]; then
  echo "Unity Editor missing: $editor" >&2
  exit 2
fi

"$editor" \
  -batchmode -nographics -quit \
  -projectPath "$project" \
  -executeMethod TpsTransfer.Editor.TpsSmokeSceneBuilder.BuildLinuxPlayer \
  -logFile "$log_file"

if [[ ! -x "$project/Build/TpsSmoke.x86_64" ]]; then
  echo "Unity build did not produce a player; see $log_file" >&2
  exit 3
fi
echo "$project/Build/TpsSmoke.x86_64"
