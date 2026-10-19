#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
cmp manifests/gaze_v07_layouts.json manifests/pilot_layouts.json
cmp manifests/gaze_v07_layouts.json unity/Assets/StreamingAssets/pilot_layouts.json
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
  -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildEdgeAppearanceLinuxPlayers \
  -logFile "$repo_dir/build-gaze-v07.log"
