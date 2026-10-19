"""Create a versioned manifest of explicit TPS pilot layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from tps_transfer.protocol import write_layout_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("manifests/pilot_layouts.json"))
    parser.add_argument("--protocol-version", default="pilot-v0.2")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--validation", type=int, default=32)
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()
    digest = write_layout_manifest(
        args.output,
        args.protocol_version,
        args.seed,
        {"train": args.train, "validation": args.validation, "test": args.test},
    )
    print(f"{args.output}: sha256 {digest}")


if __name__ == "__main__":
    main()
