"""Content identity of a Unity standalone player, including its Data directory."""

from __future__ import annotations

import hashlib
from pathlib import Path


def unity_build_sha256(executable: Path) -> str:
    executable = executable.resolve()
    data = executable.with_name(executable.stem + "_Data")
    if not executable.is_file() or not data.is_dir():
        raise FileNotFoundError("Unity player executable or Data directory is missing")
    files = [executable] + sorted(
        path for path in data.rglob("*")
        if path.is_file() and "ML-Agents/Timers" not in path.relative_to(data).as_posix()
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(executable.parent).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def layout_manifest_sha256(executable: Path, source: Path) -> str:
    """Require the built StreamingAssets layout file to equal the source manifest."""
    executable = executable.resolve()
    bundled = executable.with_name(executable.stem + "_Data") / "StreamingAssets/pilot_layouts.json"
    if not bundled.is_file() or not source.is_file():
        raise FileNotFoundError("source or bundled layout manifest is missing")
    source_bytes = source.read_bytes()
    if bundled.read_bytes() != source_bytes:
        raise ValueError("Unity Player layout manifest differs from Python source")
    return hashlib.sha256(source_bytes).hexdigest()
