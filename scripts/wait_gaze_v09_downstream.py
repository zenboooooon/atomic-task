"""Wait for all primitive training and final validation, then run downstream."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HEADS = tuple(f"{task}_color{i}" for task in ("look", "approach") for i in range(3))
JOBS = [f"{condition}_seed{seed}" for seed in range(3) for condition in ("P", "S", "F")]
TZ = timezone(timedelta(hours=9))


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def state(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    parts = path.read_text().split()
    return parts[1] if len(parts) > 1 else "MISSING"


def primitive_process_alive(pid_file: Path) -> bool:
    if not pid_file.is_file():
        return False
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc = Path(f"/proc/{pid}")
    if proc.is_dir():
        try:
            process_state = (proc / "stat").read_text().rpartition(")")[2].split()[0]
            command = (proc / "cmdline").read_bytes()
        except FileNotFoundError:
            return False
        return process_state != "Z" and b"run_gaze_v09_" in command
    return True


def primitive_readiness(base: Path) -> tuple[bool, str]:
    overall = state(base / "pipeline.status")
    if overall in ("FAILED", "STOPPED"):
        raise RuntimeError(f"primitive pipeline {overall}")
    for seed in range(3):
        current = state(base / f"seed{seed}/pipeline.status")
        if current in ("FAILED", "STOPPED"):
            raise RuntimeError(f"primitive seed{seed} {current}")
    if overall != "COMPLETE":
        return False, "waiting for all primitive training and validation"
    for seed in range(3):
        directory = base / f"seed{seed}"
        if state(directory / "pipeline.status") != "COMPLETE":
            return False, f"waiting for primitive seed{seed} completion"
        training = directory / f"pretrain_shared_seed{seed}"
        checkpoint = training / "checkpoint_2000.pt"
        report_path = training / "report.json"
        validation_path = directory / "pretrain6_validation.json"
        if not all(p.is_file() for p in (checkpoint, report_path, validation_path)):
            return False, f"waiting for primitive seed{seed} final artifacts"
        report = json.loads(report_path.read_text())
        updates = {int(row["update"]): row for row in report["updates"]}
        if set(updates) != set(range(1, 2001)):
            return False, f"waiting for primitive seed{seed} all 2000 updates"
        if updates[2000]["agent_decisions_trained"] != 1536000:
            raise ValueError(f"primitive seed{seed} decision count differs")
        evaluation = json.loads(validation_path.read_text())
        if (evaluation.get("environment_seed") != 9000
                or evaluation.get("agent_decisions_trained") != 1536000):
            raise ValueError(f"primitive seed{seed} validation identity differs")
        summary = evaluation.get("summary", {})
        if set(summary) != set(HEADS) or any(summary[h].get("episodes") != 36 for h in HEADS):
            return False, f"waiting for primitive seed{seed} all six final validations"
        if primitive_process_alive(directory / "pipeline.pid"):
            return False, f"waiting for primitive seed{seed} process exit"
    if primitive_process_alive(base / "pipeline.pid"):
        return False, "waiting for primitive supervisor exit"
    return True, "all primitive training and final validation complete"


def main() -> None:
    base = ROOT / "runs/gaze_v09_2000"
    downstream = base / "downstream"
    lock = os.open(downstream / "chain.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(lock, "w") as stream:
        stream.write(str(os.getpid()) + "\n")
    for name in ("pipeline.pid", "chain.pid"):
        (downstream / name).write_text(str(os.getpid()) + "\n")

    def interrupt(signum, frame):
        raise InterruptedError("downstream waiting process stopped")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    previous = None
    try:
        while True:
            ready, detail = primitive_readiness(base)
            if detail != previous:
                print(f"{now()} {'READY' if ready else 'WAITING'} {detail}", flush=True)
                previous = detail
            if ready:
                os.execv(sys.executable, [sys.executable, "-u",
                                         str(ROOT / "scripts/run_gaze_v09_downstream.py")])
            (downstream / "pipeline.status").write_text(f"{now()} WAITING primitive_training_and_validation\n")
            progress = {"time_jst": now(), "stage": "waiting_for_primitives", "wait_reason": detail,
                        "trained_jobs": [], "running_jobs": [], "queued_jobs": JOBS,
                        "validations_complete": 0, "validations_total": 90, "active_validation": None}
            pending = downstream / "progress.json.pending"
            pending.write_text(json.dumps(progress, indent=2) + "\n")
            pending.replace(downstream / "progress.json")
            time.sleep(30)
    except BaseException as error:
        outcome = "STOPPED" if isinstance(error, InterruptedError) else "FAILED"
        (downstream / "pipeline.status").write_text(f"{now()} {outcome} {error}\n")
        print(f"{now()} {outcome} {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
