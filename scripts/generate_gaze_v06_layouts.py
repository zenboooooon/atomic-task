"""Keep v0.5 geometry and put every look target visibly near a horizontal edge."""

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "gaze-v0.6-edge-appearance"


def viewport(distance: float, yaw_offset: float) -> tuple[float, float]:
    # The avatar's horizontal scale is .7; yaw rig (1.5, 2, -3.6),
    # initial camera pitch 10 degrees, square RGB, vertical FOV 70 degrees.
    a = math.radians(yaw_offset)
    x = -distance * math.sin(a) - 1.5 * 0.7
    z = distance * math.cos(a) + 3.6 * 0.7
    y = 1.6 - 3.0
    pitch = math.radians(10)
    forward = z * math.cos(pitch) - y * math.sin(pitch)
    up = z * math.sin(pitch) + y * math.cos(pitch)
    scale = forward * math.tan(math.radians(35))
    return 0.5 + x / (2 * scale), 0.5 + up / (2 * scale)


def main() -> None:
    document = json.loads((ROOT / "manifests/gaze_v05_look1p5_layouts.json").read_text())
    counters = {}
    checks = []
    for row in document["layouts"]:
        old_id = row["layout_id"]
        if row["task"] == "look":
            split = row["split"]
            index = counters.get(split, 0)
            counters[split] = index + 1
            rng = random.Random(int(old_id[:16], 16))
            desired_x = rng.uniform(0.11, 0.15)
            if index % 2:
                desired_x = 1.0 - desired_x
            dx = row["target_x"] - row["player_x"]
            dz = row["target_z"] - row["player_z"]
            distance = math.hypot(dx, dz)
            lower, upper = -85.0, 85.0
            for _ in range(70):
                middle = (lower + upper) / 2
                if viewport(distance, middle)[0] > desired_x:
                    lower = middle
                else:
                    upper = middle
            offset = (lower + upper) / 2
            row["camera_yaw_degrees"] = math.degrees(math.atan2(dx, dz)) + offset
            row["look_start_viewport_x"] = desired_x
            x, y = viewport(distance, offset)
            assert 0.10 < x < 0.16 or 0.84 < x < 0.90
            assert 0.1 < y < 0.9
            checks.append({"split": split, "index": index, "viewport_x": x,
                           "viewport_y": y, "yaw_offset_degrees": offset})
        row["layout_id"] = hashlib.sha256(f"{VERSION}|{old_id}".encode()).hexdigest()[:20]
    document["protocol_version"] = VERSION
    document["look_initial_target_viewport_x"] = [[0.11, 0.15], [0.85, 0.89]]
    document["appearance"] = {
        "atomic": {"target": "orange ellipsoid", "avatar": "cyan cylinder"},
        "treasure": {"target": "magenta diamond", "avatar": "yellow box"},
    }
    document.pop("sha256", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    document["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    for name in ("manifests/gaze_v06_layouts.json", "manifests/pilot_layouts.json",
                 "unity/Assets/StreamingAssets/pilot_layouts.json"):
        (ROOT / name).write_text(content)
    print(json.dumps({"look_layouts_checked": len(checks), "sha256": document["sha256"],
                      "first_validation_layouts": [c for c in checks if c["split"] == "validation"][:2]}, indent=2))


if __name__ == "__main__":
    main()
