"""Generate paired wide-field look/approach and treasure layouts for gaze v0.4."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

VERSION = "gaze-v0.4"
SEED = 20261015
COUNTS = {"train": 128, "validation": 32, "test": 0}
HALF_WIDTH = 9.0


def new_layout(task: str, split: str, index: int) -> dict:
    identity = f"{VERSION}|{SEED}|{task}|{split}|{index}"
    digest = hashlib.sha256(identity.encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    for _ in range(1000):
        px, pz = rng.uniform(-1.3, 1.3), rng.uniform(-1.3, 1.3)
        distance = rng.uniform(6.0, 9.0)
        bearing = rng.uniform(-math.pi, math.pi)
        tx = px + distance * math.cos(bearing)
        tz = pz + distance * math.sin(bearing)
        if max(abs(tx), abs(tz)) > HALF_WIDTH - 0.6:
            continue
        target_yaw = math.degrees(math.atan2(tx - px, tz - pz))
        offset = (rng.uniform(95.0, 125.0) * rng.choice((-1, 1))
                  if task == "look" else rng.uniform(-8.0, 8.0))
        return {
            "layout_id": digest.hex()[:20], "task": task, "split": split,
            "player_x": px, "player_z": pz,
            "target_x": tx, "target_z": tz,
            "camera_yaw_degrees": target_yaw + offset,
            "initial_speed_mps": 0.0, "obstacles": [],
        }
    raise RuntimeError(f"could not place {task} layout {split} {index}")


def main() -> None:
    source = Path("manifests/pilot_layouts.json")
    reference = json.loads(source.read_text())
    if reference["protocol_version"] != "treasure-v0.2":
        raise ValueError("expected treasure-v0.2 reference layouts")
    treasure = [item for item in reference["layouts"] if item["task"] == "treasure"]
    if len([item for item in treasure if item["split"] == "train"]) != COUNTS["train"]:
        raise ValueError("unexpected number of treasure training layouts")
    if len([item for item in treasure if item["split"] == "validation"]) != COUNTS["validation"]:
        raise ValueError("unexpected number of treasure validation layouts")
    layouts = [new_layout(task, split, index)
               for task in ("look", "approach")
               for split in ("train", "validation")
               for index in range(COUNTS[split])]
    for item in treasure:
        copied = dict(item)
        copied["layout_id"] = hashlib.sha256(
            f"{VERSION}|{item['layout_id']}".encode()).hexdigest()[:20]
        layouts.append(copied)
    if len({item["layout_id"] for item in layouts}) != len(layouts):
        raise RuntimeError("duplicate layout ID")
    document = {"protocol_version": VERSION, "seed": SEED,
                "counts_per_task": COUNTS, "layouts": layouts}
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    document["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    output = Path("manifests/gaze_v04_layouts.json")
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"{output}: {document['sha256']} ({len(layouts)} layouts)")


if __name__ == "__main__":
    main()
