#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
source_manifest=manifests/gaze_v05_look1p5_layouts.json
cmp "$source_manifest" manifests/pilot_layouts.json
cmp "$source_manifest" unity/Assets/StreamingAssets/pilot_layouts.json
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity"   -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildPretrainTwoLinuxPlayer   -logFile "$repo_dir/build-gaze-v05-pretrain.log"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity"   -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildTreasureLinuxPlayer   -logFile "$repo_dir/build-gaze-v05-treasure.log"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity"   -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildGazePreviewLinuxPlayers   -logFile "$repo_dir/build-gaze-v05-preview.log"
PYTHONPATH="$repo_dir/python" "$repo_dir/.venv-gpu/bin/python" - <<'VERIFY'
from pathlib import Path
from tps_transfer.build_identity import layout_manifest_sha256
root=Path.cwd()
source=root/'manifests/gaze_v05_look1p5_layouts.json'
for name in ('TpsPretrainTwo','TpsTreasure','TpsLookPreview',
             'TpsApproachPreview','TpsTreasurePreview'):
    exe=root/'unity/Build'/f'{name}.x86_64'
    print(name,layout_manifest_sha256(exe,source),flush=True)
VERIFY
