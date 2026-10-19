#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
if ! cmp -s "$repo_dir/manifests/pilot_layouts.json" \
  "$repo_dir/unity/Assets/StreamingAssets/pilot_layouts.json"; then
  echo "Unity StreamingAssets layouts differ from manifest" >&2
  exit 2
fi
for kind in pretraining composite; do
  if [[ "$kind" == pretraining ]]; then
    method=TpsTransfer.Editor.TpsMultiSceneBuilder.BuildPretrainingLinuxPlayer
    player=TpsPretrain
  else
    method=TpsTransfer.Editor.TpsMultiSceneBuilder.BuildCompositeLinuxPlayer
    player=TpsComposite
  fi
  "$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
    -executeMethod "$method" -logFile "$repo_dir/.tools/build-$kind.log"
  if [[ ! -x "$repo_dir/unity/Build/$player.x86_64" ]]; then
    echo "Unity $kind build failed; see .tools/build-$kind.log" >&2
    exit 3
  fi
  echo "$repo_dir/unity/Build/$player.x86_64"
done
