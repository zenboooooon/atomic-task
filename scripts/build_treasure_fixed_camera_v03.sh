#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
python="$repo_dir/.venv/bin/python"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
source_player=unity/Build/TpsPretrainThree.x86_64
treasure_player=unity/Build/TpsTreasure.x86_64
composite_player=unity/Build/TpsComposite.x86_64
treasure_manifest=manifests/pilot_layouts.json
marker=manifests/treasure_fixed_camera_v03.json

"$python" - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("manifests/pilot_layouts.json").read_text())
assert manifest["protocol_version"] == "treasure-v0.2"
assert any(item["task"] == "treasure" for item in manifest["layouts"])
PY
source_hash_before=$("$python" -c \
  'from pathlib import Path; from tps_transfer.build_identity import unity_build_sha256; print(unity_build_sha256(Path("unity/Build/TpsPretrainThree.x86_64")))')

"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
  -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildTreasureLinuxPlayer \
  -logFile "$repo_dir/build-treasure-fixed-camera-v03.log"
test -x "$treasure_player"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
  -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildCompositeLinuxPlayer \
  -logFile "$repo_dir/build-composite-fixed-camera-v03.log"
test -x "$composite_player"

"$python" - "$source_hash_before" "$marker" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256

source_before, marker_name = sys.argv[1:]
source_after = unity_build_sha256(Path("unity/Build/TpsPretrainThree.x86_64"))
if source_before != source_after:
    raise RuntimeError("source pretraining player changed during treasure build")
treasure = Path("unity/Build/TpsTreasure.x86_64")
composite = Path("unity/Build/TpsComposite.x86_64")
manifest = Path("manifests/pilot_layouts.json")
report = {
    "camera_fix": "fixed yaw boom; pitch rotates view only",
    "source_build_sha256": source_after,
    "treasure_build_sha256": unity_build_sha256(treasure),
    "treasure_layout_sha256": layout_manifest_sha256(treasure, manifest),
    "composite_build_sha256": unity_build_sha256(composite),
    "composite_layout_sha256": layout_manifest_sha256(composite, manifest),
    "scene_builder_sha256": hashlib.sha256(
        Path("unity/Assets/Editor/TpsMultiSceneBuilder.cs").read_bytes()
    ).hexdigest(),
}
Path(marker_name).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
