"""Run nine matched downstream jobs with three trainers and one checkpoint evaluator."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/gaze_v07_2000/downstream"
TZ = datetime.timezone(datetime.timedelta(hours=9))
JOBS = [(condition, seed) for seed in range(3) for condition in ("P", "S", "F")]
UPDATES = list(range(200, 2001, 200))


def now() -> str:
    return datetime.datetime.now(TZ).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def event(message: str, log: str = "train.log") -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with (BASE / log).open("a") as stream:
        stream.write(line + "\n")


def preflight() -> None:
    import torch
    from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256

    assert (BASE / "APPROVED").is_file()
    assert not (BASE / "pipeline.status").exists(), "existing run cannot be overwritten"
    plan = json.loads((BASE / "run.plan.json").read_text())
    assert plan["conditions"] == ["P", "S", "F"] and plan["seeds"] == [0, 1, 2]
    assert plan["updates_per_condition_seed"] == 2000 and plan["rollout"] == 256
    assert plan["validation_episodes_per_checkpoint"] == 32
    manifest = ROOT / "manifests/gaze_v07_layouts.json"
    source_build = ROOT / "unity/Build/TpsPretrainTwo.x86_64"
    downstream_build = ROOT / "unity/Build/TpsTreasure.x86_64"
    source_sha = unity_build_sha256(source_build)
    downstream_sha = unity_build_sha256(downstream_build)
    layout_sha = layout_manifest_sha256(source_build, manifest)
    assert layout_manifest_sha256(downstream_build, manifest) == layout_sha
    sources = {}
    for seed in range(3):
        directory = BASE.parent / f"seed{seed}"
        assert "COMPLETE" in (directory / "pipeline.status").read_text()
        path = directory / f"pretrain2_seed{seed}/checkpoint_2000.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["seed"] == seed and state["agent_decisions"] == 512000
        assert state["model_config"]["width"] == 256
        assert state["pretraining_tasks"] == ["look", "approach"]
        assert state["layout_manifest_sha256"] == layout_sha
        assert state["unity_build_sha256"] == source_sha
        evaluation = json.loads((directory / "pretrain2_validation.json").read_text())
        assert all(evaluation["summary"][t]["episodes"] == 36 for t in ("look", "approach"))
        sources[str(seed)] = {"checkpoint": str(path.relative_to(ROOT)),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    plan.update(preflight_time_jst=now(), sources=sources, unity_build_sha256=downstream_sha,
                source_unity_build_sha256=source_sha, layout_manifest_sha256=layout_sha)
    save(BASE / "run.plan.json", plan)
    with tarfile.open(BASE / "source_snapshot.tar.gz", "w:gz") as archive:
        for folder in ("python", "scripts", "configs", "manifests", "unity/Assets/Scripts", "unity/Assets/Editor"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"):
                    archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    (BASE / "source_changes.diff").write_bytes(diff)


def launch(mode: str, condition: str, seed: int, index: int) -> subprocess.Popen:
    directory = BASE / f"{condition}_seed{seed}"
    directory.mkdir(exist_ok=True)
    name = "train.log" if mode == "train" else f"validation_{index:04d}.log"
    with (directory / name).open("ab") as stream:
        return subprocess.Popen(["bash", "scripts/run_gaze_v07_downstream_worker.sh", mode,
                                 str(BASE), condition, str(seed), str(index)], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)


def main() -> None:
    preflight()
    (BASE / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    running = {}
    trained = set()
    validated = set()
    next_job = 0
    evaluator = None
    stopped = False

    def interrupt(signum, unused_frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while True:
            if stopped:
                raise InterruptedError("supervisor stopped")
            parallelism = int((BASE / "parallelism").read_text())
            assert parallelism in (1, 2, 3)
            for job, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"training {job} exited {code}")
                condition, seed = job
                report = json.loads((BASE / f"{condition}_seed{seed}/report.json").read_text())
                assert report["updates"][-1]["update"] == 2000
                assert report["condition"] == condition and report["seed"] == seed
                assert report["shared_frozen"] == (condition == "P")
                trained.add(job)
                del running[job]
                event(f"TRAINING_COMPLETE {condition}_seed{seed}")
            while next_job < len(JOBS) and len(running) < parallelism:
                condition, seed = JOBS[next_job]
                process = launch("train", condition, seed, next_job)
                running[(condition, seed)] = process
                (BASE / f"{condition}_seed{seed}/pipeline.pid").write_text(str(process.pid) + "\n")
                event(f"TRAINING_STARTED {condition}_seed{seed} pid={process.pid}")
                next_job += 1
            if evaluator is not None:
                key, process = evaluator
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"validation {key} exited {code}")
                    condition, seed, update = key
                    report = json.loads((BASE / f"{condition}_seed{seed}/validation_{update:04d}.json").read_text())
                    summary = report["summary"]["treasure"]
                    assert report["condition"] == condition and report["agent_decisions_trained"] == update * 256
                    assert report["environment_seed"] == 9000 and summary["episodes"] == 32
                    assert sum(bool(r["success"]) for r in report["results"]["treasure"]) == summary["successes"]
                    validated.add(key)
                    event(f"VALIDATION_COMPLETE {condition}_seed{seed} update={update} successes={summary['successes']}/32", "validation.log")
                    evaluator = None
            if evaluator is None:
                for update in UPDATES:
                    for condition, seed in JOBS:
                        key = (condition, seed, update)
                        checkpoint = BASE / f"{condition}_seed{seed}/checkpoint_{update:04d}.pt"
                        if key not in validated and checkpoint.exists():
                            process = launch("validate", condition, seed, update)
                            evaluator = key, process
                            event(f"VALIDATION_STARTED {condition}_seed{seed} update={update} pid={process.pid}", "validation.log")
                            break
                    if evaluator is not None:
                        break
            progress = {"time_jst": now(), "trained_jobs": [f"{c}_seed{s}" for c, s in JOBS if (c, s) in trained],
                        "running_jobs": [{"condition": c, "seed": s, "pid": p.pid} for (c, s), p in running.items()],
                        "queued_jobs": [f"{c}_seed{s}" for c, s in JOBS[next_job:]],
                        "validations_complete": len(validated), "validations_total": 90,
                        "active_validation": ({"condition": evaluator[0][0], "seed": evaluator[0][1],
                                               "update": evaluator[0][2], "pid": evaluator[1].pid} if evaluator else None)}
            save(BASE / "progress.json", progress)
            complete = len(trained) == 9 and len(validated) == 90
            (BASE / "pipeline.status").write_text(
                f"{now()} {'COMPLETE' if complete else 'RUNNING'} training={len(trained)}/9 validation={len(validated)}/90\n")
            if complete:
                event("COMPLETE all nine training jobs and 90 validations")
                break
            time.sleep(5)
    except BaseException as error:
        status = "STOPPED" if isinstance(error, InterruptedError) else "FAILED"
        (BASE / "pipeline.status").write_text(f"{now()} {status} {error}\n")
        event(f"{status} {error}")
        raise
    finally:
        processes = list(running.values()) + ([evaluator[1]] if evaluator else [])
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
