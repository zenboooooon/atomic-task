#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cmp "$repo_dir/manifests/gaze_v08_layouts.json" "$repo_dir/manifests/pilot_layouts.json"
cmp "$repo_dir/manifests/gaze_v08_layouts.json" "$repo_dir/unity/Assets/StreamingAssets/pilot_layouts.json"
cmp "$repo_dir/manifests/primitive_color_variants.json" "$repo_dir/unity/Assets/StreamingAssets/primitive_color_variants.json"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
  -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildColorVariantLinuxPlayers \
  -logFile "$repo_dir/build-gaze-v08.log"
