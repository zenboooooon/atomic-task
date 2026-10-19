#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
"$repo_dir/.venv/bin/python" "$repo_dir/scripts/generate_layouts.py" \
  --protocol-version treasure-v0.1 --seed 20261013 \
  --train 128 --validation 32 --test 32 \
  --output "$repo_dir/manifests/pilot_layouts.json"
cp "$repo_dir/manifests/pilot_layouts.json" \
  "$repo_dir/unity/Assets/StreamingAssets/pilot_layouts.json"
for kind in pretrain-three treasure task; do
  case "$kind" in
    pretrain-three)
      method=TpsTransfer.Editor.TpsMultiSceneBuilder.BuildPretrainThreeLinuxPlayer
      player=TpsPretrainThree ;;
    treasure)
      method=TpsTransfer.Editor.TpsMultiSceneBuilder.BuildTreasureLinuxPlayer
      player=TpsTreasure ;;
    task)
      method=TpsTransfer.Editor.TpsTaskSceneBuilder.BuildLinuxPlayer
      player=TpsTask ;;
  esac
  "$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
    -executeMethod "$method" -logFile "$repo_dir/build-$kind.log"
  test -x "$repo_dir/unity/Build/$player.x86_64"
  echo "Built $player"
done
