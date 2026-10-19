"""Version the source-task rules while keeping the treasure comparison layouts fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tps_transfer.protocol import Layout, Obstacle, validate_layout

SOURCE_TASKS = {"look", "approach", "avoid"}
VERSION = "primitive-contact-v0.3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    if source["protocol_version"] != "treasure-v0.2":
        raise ValueError("expected the unchanged treasure-v0.2 layout manifest")
    for item in source["layouts"]:
        if item["task"] in SOURCE_TASKS:
            layout = Layout(**{**item, "obstacles": tuple(
                Obstacle(**obstacle) for obstacle in item["obstacles"]
            )})
            validate_layout(layout)
        item["layout_id"] = hashlib.sha256(
            f"{VERSION}|{item['layout_id']}".encode()
        ).hexdigest()[:20]
    source["protocol_version"] = VERSION
    source.pop("sha256", None)
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"))
    source["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(source, sort_keys=True, indent=2) + "\n")
    print(f"{args.output}: sha256 {source['sha256']}")


if __name__ == "__main__":
    main()
