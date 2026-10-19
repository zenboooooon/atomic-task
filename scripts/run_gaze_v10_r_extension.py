"""Append R0..2 without interrupting the running P/S/F supervisor or workers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

from run_gaze_v10_downstream import launch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/gaze_v10_2000/downstream_wide24"
CONTROL = BASE / "r_extension"
ORIGINAL_JOBS = [f"{c}_seed{s}" for s in range(3) for c in ("P", "S", "F")]
R_JOBS = [f"R_seed{s}" for s in range(3)]
UPDATES = range(200, 2001, 200)
JST = timezone(timedelta(hours=9))


def now():
    return datetime.now(JST).isoformat(timespec="seconds")


def save(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def load(path):
    return json.loads(path.read_text())


def alive(pid):
    try:
        p = Path("/proc") / str(int(pid))
        return (p / "stat").read_text().rpartition(")")[2].split()[0] != "Z" and bool((p / "cmdline").read_bytes())
    except (OSError, ValueError):
        return False


def training_gate(original, original_workers_alive):
    return (set(original.get("trained_jobs", [])) == set(ORIGINAL_JOBS)
            and not original.get("running_jobs") and not original.get("queued_jobs")
            and not original_workers_alive)


def evaluation_gate(original):
    return original.get("validations_complete") == 90 and original.get("active_validation") is None


def combined_progress(original, trained, running, launched, validated, evaluator, stage):
    return {
        "time_jst": now(), "stage": "complete" if len(trained) == 3 and len(validated) == 30 else "running",
        "training_jobs_total": 12,
        "trained_jobs": original.get("trained_jobs", []) + [f"R_seed{s}" for s in sorted(trained)],
        "running_jobs": original.get("running_jobs", []) + [
            {"condition": "R", "seed": s, "pid": p.pid} for s, p in running.items()],
        "queued_jobs": original.get("queued_jobs", []) + [f"R_seed{s}" for s in range(3) if s not in launched],
        "validations_complete": original.get("validations_complete", 0) + len(validated),
        "validations_total": 120,
        "active_validation": original.get("active_validation") or (
            {"condition": "R", "seed": evaluator[0][0], "update": evaluator[0][1], "pid": evaluator[1].pid}
            if evaluator else None),
        "R_stage": stage,
        "original_progress": "progress.json", "R_progress": "r_extension/progress.json",
    }


def preflight():
    assert (CONTROL / "APPROVED").is_file()
    for name in R_JOBS:
        assert not list((BASE / name).glob("checkpoint_*.pt")), "existing R training cannot be overwritten"
    audit = ROOT / "runs/r_extension_change"
    fixed = load(audit / "fixed_files_before.json")
    for name, digest in fixed.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    smoke = load(audit / "verification.json")
    assert smoke["R_shared_matches_S_initial"] and smoke["R_shared_frozen_after_update"]
    assert smoke["R_actor_and_critic_updated"] and smoke["all_heads_match_existing_initial_heads"]
    from tps_transfer.build_identity import unity_build_sha256, layout_manifest_sha256
    plan = load(CONTROL / "run.plan.json")
    assert plan["conditions"] == ["R"] and plan["seeds"] == [0, 1, 2]
    assert plan["updates_per_condition_seed"] == 2000 and plan["rollout"] == 256
    exe = ROOT / plan["unity_executable"]
    assert unity_build_sha256(exe) == plan["unity_build_sha256"]
    assert layout_manifest_sha256(exe, ROOT / plan["layout_manifest"]) == plan["layout_manifest_sha256"]
    plan.update(status="queued_after_existing_PSF_training", preflight_time_jst=now(), sources=load(BASE / "run.plan.json")["sources"])
    save(CONTROL / "run.plan.json", plan)
    with tarfile.open(CONTROL / "source_snapshot.tar.gz", "w:gz") as archive:
        for folder in ("python", "scripts", "configs", "manifests", "environment_sources/wide24"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"):
                    archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight()
        (CONTROL / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        running, trained, launched, validated = {}, set(), set(), set()
        evaluator = None
        stopped = False

        def interrupt(signum, frame):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)

        def event(message, filename="train.log"):
            text = f"{now()} {message}"
            print(text, flush=True)
            with (CONTROL / filename).open("a") as stream:
                stream.write(text + "\n")

        try:
            while True:
                if stopped:
                    raise InterruptedError("R extension stopped")
                original = load(BASE / "progress.json")
                original_state = (BASE / "pipeline.status").read_text().split()[1]
                for seed, process in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"R_seed{seed} training exited {code}")
                    report = load(BASE / f"R_seed{seed}/report.json")
                    assert report["condition"] == "R" and report["seed"] == seed
                    assert report["updates"][-1]["update"] == 2000 and report["shared_frozen"]
                    assert report["initial_shared_hash"] == report["final_shared_hash"]
                    trained.add(seed)
                    del running[seed]
                    event(f"TRAINING_COMPLETE R_seed{seed}")
                original_workers_alive = any(
                    alive((BASE / job / "pipeline.pid").read_text().strip())
                    for job in ORIGINAL_JOBS if (BASE / job / "pipeline.pid").exists())
                ready = original_state not in ("FAILED", "STOPPED") and training_gate(original, original_workers_alive)
                parallelism = int((BASE / "parallelism").read_text())
                assert parallelism in (1, 2, 3)
                if ready:
                    for seed in range(3):
                        if len(running) >= parallelism:
                            break
                        if seed in launched:
                            continue
                        # The original worker derives display/worker 540+index.
                        # R uses 650..652; existing P/S/F and evaluation IDs stay intact.
                        process = launch("train", "R", seed, 110 + seed)
                        running[seed] = process
                        launched.add(seed)
                        (BASE / f"R_seed{seed}/pipeline.pid").write_text(str(process.pid) + "\n")
                        event(f"TRAINING_STARTED R_seed{seed} pid={process.pid}")
                if evaluator:
                    (seed, update), process = evaluator
                    code = process.poll()
                    if code is not None:
                        if code:
                            raise RuntimeError(f"R validation {seed}/{update} exited {code}")
                        report = load(BASE / f"R_seed{seed}/validation_{update:04d}.json")
                        summary = report["summary"]["treasure"]
                        assert report["condition"] == "R" and report["environment_seed"] == 9000
                        assert report["agent_decisions_trained"] == update * 256 and summary["episodes"] == 32
                        assert sum(bool(v["success"]) for v in report["results"]["treasure"]) == summary["successes"]
                        validated.add((seed, update))
                        evaluator = None
                        event(f"VALIDATION_COMPLETE R_seed{seed} update={update} successes={summary['successes']}/32", "validation.log")
                if evaluator is None and original_state not in ("FAILED", "STOPPED") and evaluation_gate(original):
                    pending = next(((seed, update) for update in UPDATES for seed in range(3)
                                    if (seed, update) not in validated
                                    and (BASE / f"R_seed{seed}/checkpoint_{update:04d}.pt").exists()), None)
                    if pending:
                        seed, update = pending
                        process = launch("validate", "R", seed, update)
                        evaluator = pending, process
                        event(f"VALIDATION_STARTED R_seed{seed} update={update} pid={process.pid}", "validation.log")
                stage = ("waiting_for_PSF_training" if not launched else "training_R" if running
                         else "evaluating_R" if evaluator else "waiting_for_PSF_validation" if not evaluation_gate(original)
                         else "waiting_for_R_checkpoints")
                complete = len(trained) == 3 and len(validated) == 30
                if complete:
                    stage = "complete"
                own = {"time_jst": now(), "stage": stage, "trained_jobs": [f"R_seed{s}" for s in sorted(trained)],
                       "running_jobs": [{"condition": "R", "seed": s, "pid": p.pid} for s, p in running.items()],
                       "queued_jobs": [f"R_seed{s}" for s in range(3) if s not in launched],
                       "validations_complete": len(validated), "validations_total": 30,
                       "active_validation": ({"seed": evaluator[0][0], "update": evaluator[0][1], "pid": evaluator[1].pid} if evaluator else None)}
                save(CONTROL / "progress.json", own)
                own_state = "COMPLETE" if complete else "WAITING" if stage.startswith("waiting") else "RUNNING"
                (CONTROL / "pipeline.status").write_text(f"{now()} {own_state} {stage}\n")
                combined = combined_progress(original, trained, running, launched, validated, evaluator, stage)
                save(BASE / "combined.progress.json", combined)
                status = "COMPLETE" if complete else "FAILED" if original_state in ("FAILED", "STOPPED") else "RUNNING"
                (BASE / "combined.pipeline.status").write_text(
                    f"{now()} {status} training={len(combined['trained_jobs'])}/12 validation={combined['validations_complete']}/120\n")
                if complete:
                    event("COMPLETE R training 3/3 and validation 30/30; combined 12/12 and 120/120")
                    break
                time.sleep(5)
        except BaseException as error:
            status = "STOPPED" if isinstance(error, InterruptedError) else "FAILED"
            (CONTROL / "pipeline.status").write_text(f"{now()} {status} {error}\n")
            (BASE / "combined.pipeline.status").write_text(f"{now()} {status} R_extension {error}\n")
            event(f"{status} {error}")
            raise
        finally:
            # Only children launched by this R extension are signalled.
            # The original P/S/F supervisor and workers are never touched.
            for process in list(running.values()) + ([evaluator[1]] if evaluator else []):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        CONTROL.mkdir(parents=True, exist_ok=True)
        status = "STOPPED" if isinstance(error, InterruptedError) else "FAILED"
        (CONTROL / "pipeline.status").write_text(f"{now()} {status} {type(error).__name__}: {error}\n")
        raise
