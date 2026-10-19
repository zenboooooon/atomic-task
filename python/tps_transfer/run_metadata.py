"""Small, explicit provenance for local experiment runs."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_state(root: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    )
    return {"code_commit": commit, "code_dirty": bool(status.strip())}
