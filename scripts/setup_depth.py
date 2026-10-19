"""Install the pinned official Depth Anything V2 Small source and checkpoint."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from urllib.request import urlopen

from tps_transfer.depth import CHECKPOINT, CHECKPOINT_SHA256, REPOSITORY, REPOSITORY_COMMIT

SOURCE_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
WEIGHT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/"
    "resolve/main/depth_anything_v2_vits.pth?download=true"
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    if not REPOSITORY.exists():
        REPOSITORY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", SOURCE_URL, str(REPOSITORY)], check=True)
        subprocess.run(
            ["git", "-C", str(REPOSITORY), "checkout", REPOSITORY_COMMIT], check=True
        )
    actual_commit = subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != REPOSITORY_COMMIT:
        raise SystemExit(f"source commit mismatch: {actual_commit}")
    if not CHECKPOINT.exists():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(WEIGHT_URL, timeout=120) as response, CHECKPOINT.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    actual_hash = digest(CHECKPOINT)
    if actual_hash != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {actual_hash}")
    print(f"Depth Anything V2 Small ready: {actual_commit}, {actual_hash}")


if __name__ == "__main__":
    main()
