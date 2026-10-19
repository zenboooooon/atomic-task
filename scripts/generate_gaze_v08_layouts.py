"""Retain v0.7 geometry and rewards; version the three primitive appearances."""

import hashlib
import json
from pathlib import Path

from tps_transfer.primitive_variants import PROTOCOL_VERSION, appearance_protocol

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    document = json.loads((ROOT / "manifests/gaze_v07_layouts.json").read_text())
    for row in document["layouts"]:
        row["layout_id"] = hashlib.sha256(
            f"{PROTOCOL_VERSION}|{row['layout_id']}".encode()).hexdigest()[:20]
    document["protocol_version"] = PROTOCOL_VERSION
    document["appearance_protocol"] = appearance_protocol()
    document["appearance"]["atomic"] = {
        "avatar": "cylinder; palette-specific color",
        "target": "ellipsoid; palette-specific color",
        "variants": appearance_protocol()["variants"],
    }
    document.pop("sha256", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    document["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    for name in ("manifests/gaze_v08_layouts.json", "manifests/pilot_layouts.json",
                 "unity/Assets/StreamingAssets/pilot_layouts.json"):
        (ROOT / name).write_text(content)
    colors = json.dumps(appearance_protocol(), indent=2) + "\n"
    for name in ("manifests/primitive_color_variants.json",
                 "unity/Assets/StreamingAssets/primitive_color_variants.json"):
        (ROOT / name).write_text(colors)
    print(json.dumps({"protocol": PROTOCOL_VERSION, "sha256": document["sha256"]}))


if __name__ == "__main__":
    main()
