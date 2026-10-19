"""Version the free-movement 0.5-second look rules; retain v0.6 layouts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "gaze-v0.7-free-look0p5"


def main() -> None:
    document = json.loads((ROOT / "manifests/gaze_v06_layouts.json").read_text())
    for row in document["layouts"]:
        row["layout_id"] = hashlib.sha256(
            f"{VERSION}|{row['layout_id']}".encode()).hexdigest()[:20]
    document["protocol_version"] = VERSION
    document["look_success"] = {
        "continuous_center_seconds": 0.5, "center_angle_degrees": 7,
        "movement_allowed": True, "movement_penalties": False,
    }
    document.pop("sha256", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    document["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    for name in ("manifests/gaze_v07_layouts.json", "manifests/pilot_layouts.json",
                 "unity/Assets/StreamingAssets/pilot_layouts.json"):
        (ROOT / name).write_text(content)
    print(json.dumps({"protocol": VERSION, "sha256": document["sha256"]}))


if __name__ == "__main__":
    main()
