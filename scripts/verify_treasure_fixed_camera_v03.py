"""Refuse downstream training unless the fixed-camera Unity build is intact."""

import hashlib
import json
from pathlib import Path

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256


ROOT = Path(__file__).resolve().parents[1]
marker = json.loads((ROOT / "manifests/treasure_fixed_camera_v03.json").read_text())
actual = {
    "source_build_sha256": unity_build_sha256(
        ROOT / "unity/Build/TpsPretrainThree.x86_64"),
    "treasure_build_sha256": unity_build_sha256(
        ROOT / "unity/Build/TpsTreasure.x86_64"),
    "treasure_layout_sha256": layout_manifest_sha256(
        ROOT / "unity/Build/TpsTreasure.x86_64",
        ROOT / "manifests/pilot_layouts.json"),
    "composite_build_sha256": unity_build_sha256(
        ROOT / "unity/Build/TpsComposite.x86_64"),
    "composite_layout_sha256": layout_manifest_sha256(
        ROOT / "unity/Build/TpsComposite.x86_64",
        ROOT / "manifests/pilot_layouts.json"),
    "scene_builder_sha256": hashlib.sha256(
        (ROOT / "unity/Assets/Editor/TpsMultiSceneBuilder.cs").read_bytes()
    ).hexdigest(),
}
if any(marker.get(key) != value for key, value in actual.items()):
    raise RuntimeError("fixed-camera build identity differs from the verified build")
if marker.get("camera_fix") != "fixed yaw boom; pitch rotates view only":
    raise RuntimeError("wrong camera-fix marker")
print(json.dumps(actual, indent=2))
