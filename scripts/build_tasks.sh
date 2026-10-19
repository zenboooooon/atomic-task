#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
source_layouts="$repo_dir/manifests/pilot_layouts.json"
streamed_layouts="$repo_dir/unity/Assets/StreamingAssets/pilot_layouts.json"
if ! cmp -s "$source_layouts" "$streamed_layouts"; then
  echo "Unity StreamingAssets layouts differ from manifest" >&2
  exit 2
fi
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" -executeMethod TpsTransfer.Editor.TpsTaskSceneBuilder.BuildLinuxPlayer -logFile "$repo_dir/.tools/build-tasks.log"
if [[ ! -x "$repo_dir/unity/Build/TpsTask.x86_64" ]]; then
  echo "Unity task build failed; see .tools/build-tasks.log" >&2
  exit 3
fi
echo "$repo_dir/unity/Build/TpsTask.x86_64"
