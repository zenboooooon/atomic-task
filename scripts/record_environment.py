"""Write a reproducibility manifest without copying account credentials."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> dict[str, object]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    result: dict[str, object] = {"ok": completed.returncode == 0}
    if completed.returncode == 0:
        result["output"] = completed.stdout.strip()
    else:
        result["error"] = completed.stderr.strip() or completed.stdout.strip()
    return result


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    manifest = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "packages": {
            name: package_version(name)
            for name in ("mlagents-envs", "numpy", "torch", "torchvision", "opencv-python-headless", "pytest", "tps-transfer")
        },
        "unity_editor": command(
            str(ROOT / ".tools/unity-editors/6000.0.84f1/Editor/Unity"), "-version"
        ),
        "unity_cli": command(str(ROOT / ".tools/unity-cli/bin/unity"), "--version"),
        "unity_package": json.loads(
            (ROOT / "unity/Packages/manifest.json").read_text()
        )["dependencies"]["com.unity.ml-agents"],
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
        "loaded_nvidia_module": command("cat", "/proc/driver/nvidia/version"),
        "installed_nvidia_module": command("modinfo", "-F", "version", "nvidia"),
        "git_commit": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
    }
    output = ROOT / "manifests/environment.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
