#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
source_manifest="$repo_dir/manifests/pretrain3_v01_layouts.json"
if [[ ! -f "$source_manifest" ]]; then
  cp "$repo_dir/manifests/pilot_layouts.json" "$source_manifest"
fi
"$repo_dir/.venv/bin/python" "$repo_dir/scripts/generate_layouts.py" \
  --protocol-version treasure-v0.2 --seed 20261013 \
  --train 128 --validation 32 --test 32 \
  --output "$repo_dir/manifests/pilot_layouts.json"
cp "$repo_dir/manifests/pilot_layouts.json" \
  "$repo_dir/unity/Assets/StreamingAssets/pilot_layouts.json"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
  -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildTreasureLinuxPlayer \
  -logFile "$repo_dir/build-treasure-v02.log"
test -x "$repo_dir/unity/Build/TpsTreasure.x86_64"
echo "Built TpsTreasure for treasure-v0.2"
