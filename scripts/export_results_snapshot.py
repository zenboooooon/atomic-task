"""Read a running v0.10 experiment and export a portable result snapshot.

Standard library only. This never starts, stops, edits or resumes training.
Outputs are written only below --output; experiment files are read only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

TASKS = ("look", "approach", "jump")


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def unique_updates(rows):
    return sorted({int(row["update"]): row for row in rows}.values(), key=lambda r: r["update"])


def load_updates(directory):
    report = directory / "report.json"
    if report.exists():
        return unique_updates(json.loads(report.read_text())["updates"]), "report.json"
    rows = []
    log = directory / "train.log"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "update" in row:
                rows.append(row)
    return unique_updates(rows), "complete JSON lines from train.log"


def totals(rows, task=None):
    counts = dict(successes=0, failures=0, interruptions=0)
    for row in rows:
        value = row["tasks"][task] if task else row
        for key in counts:
            counts[key] += int(value[key])
    counts["ended"] = sum(counts.values())
    counts["success_rate"] = counts["successes"] / counts["ended"] if counts["ended"] else None
    return counts


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = root / "runs/gaze_v10_2000"
    downstream = base / "downstream_wide24"
    plan_path = downstream / "combined.run.plan.json"
    if not plan_path.exists():
        plan_path = downstream / "run.plan.json"
    plan = json.loads(plan_path.read_text())
    conditions = plan["conditions"]
    copied = {}

    def copy(relative, target=None):
        source = root / relative
        if not source.is_file():
            return None
        data = source.read_bytes()
        destination = out / (target or Path("raw") / relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        copied[str(destination.relative_to(out))] = {
            "source": str(relative), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "mtime_jst": datetime.fromtimestamp(source.stat().st_mtime, ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        }
        return destination

    snapshot = {"captured_start_jst": now(), "experiment": plan["experiment"],
                "primitive_experiment": "gaze-jump-v0.10-single-color", "source_root": str(root),
                "live_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "live_code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
                "snapshot_consistency": "Files read sequentially while downstream training continues; see per-file times.",
                "training_denominator": "successes + failures + interruptions, over unique updates; unfinished episodes and discarded surplus transitions excluded",
                "primitive": {}, "downstream": {"conditions": conditions, "plan": plan, "jobs": {}}}
    primitive_csv, downstream_csv, summary_csv = [], [], []
    for seed in range(3):
        job = base / f"seed{seed}"
        directory = job / f"pretrain3_seed{seed}"
        rows, origin = load_updates(directory)
        update = rows[-1]["update"] if rows else 0
        assert [row["update"] for row in rows] == list(range(1, update + 1))
        evaluation = json.loads((job / "pretrain3_validation.json").read_text()) if (job / "pretrain3_validation.json").exists() else None
        checkpoint = max(directory.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[-1]), default=None)
        value = {"update": update, "unique_updates": len(rows), "decisions": update * 768,
                 "status": (job / "pipeline.status").read_text().strip(), "data_source": origin,
                 "latest_checkpoint": checkpoint.name if checkpoint else None,
                 "totals": {task: totals(rows, task) for task in TASKS},
                 "validation": evaluation["summary"] if evaluation else None,
                 "final_checkpoint_sha256": evaluation.get("checkpoint_sha256") if evaluation else None}
        snapshot["primitive"][str(seed)] = value
        for task in TASKS:
            v = value["totals"][task]
            summary_csv.append(dict(stage="primitive", condition="", seed=seed, task=task, update=update,
                                    decisions=update*768, decisions_scope="per_seed_all_three_tasks", task_decisions=update*256, **v,
                                    validation_successes=value["validation"][task]["successes"] if evaluation else "",
                                    validation_episodes=36 if evaluation else ""))
        for row in rows:
            for task in TASKS:
                primitive_csv.append(dict(seed=seed, task=task, update=row["update"],
                                          agent_decisions_trained=row["agent_decisions_trained"],
                                          surplus_transitions_discarded=row["surplus_transitions_discarded"][task],
                                          rollout_seconds=row["rollout_seconds"], training_seconds=row["training_seconds"],
                                          **row["tasks"][task]))
        copy(directory.relative_to(root) / "report.json", f"raw/primitive_seed{seed}/report.json")
        copy(job.relative_to(root) / "pretrain3_validation.json", f"raw/primitive_seed{seed}/validation.json")
        copy(job.relative_to(root) / "pipeline.status", f"raw/primitive_seed{seed}/pipeline.status")
    for seed in range(3):
        for condition in conditions:
            key = f"{condition}_seed{seed}"
            directory = downstream / key
            rows, origin = load_updates(directory)
            update = rows[-1]["update"] if rows else 0
            assert [row["update"] for row in rows] == list(range(1, update + 1))
            checkpoint = max(directory.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[-1]), default=None)
            validations = {}
            for path in sorted(directory.glob("validation_*.json")):
                report = json.loads(path.read_text())
                validations[str(int(path.stem.split("_")[-1]))] = report["summary"]["treasure"]
                copy(path.relative_to(root), f"raw/downstream/{key}/{path.name}")
            count = totals(rows)
            status = (directory / "pipeline.status").read_text().strip() if (directory / "pipeline.status").exists() else "WAITING"
            snapshot["downstream"]["jobs"][key] = dict(update=update, unique_updates=len(rows), decisions=update*256,
                status=status, data_source=origin, totals=count, latest_checkpoint=checkpoint.name if checkpoint else None, validations=validations)
            summary_csv.append(dict(stage="downstream", condition=condition, seed=seed, task="treasure", update=update, decisions=update*256,
                                    decisions_scope="per_condition_seed", task_decisions=update*256, **count,
                                    validation_successes="", validation_episodes=""))
            for row in rows:
                downstream_csv.append({"condition": condition, "seed": seed, **row})
            if rows:
                p = out / f"raw/downstream/{key}/updates.jsonl"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for label, directory in (("primitive", base), ("downstream", downstream)):
        for name in ("run.plan.json", "progress.json", "pipeline.status"):
            copy((directory / name).relative_to(root), f"raw/{label}_{name}")
    for name in ("combined.run.plan.json", "combined.progress.json", "combined.pipeline.status"):
        copy((downstream / name).relative_to(root), f"raw/downstream_{name}")
    for name in ("run.plan.json", "progress.json", "pipeline.status"):
        copy((downstream / "r_extension" / name).relative_to(root), f"raw/R_extension_{name}")
    progress = downstream / "combined.progress.json"
    if not progress.exists():
        progress = downstream / "progress.json"
    snapshot["downstream"]["progress"] = json.loads(progress.read_text()) if progress.exists() else None
    snapshot["downstream"]["completed_validation_files"] = sum(len(j["validations"]) for j in snapshot["downstream"]["jobs"].values())
    for path in sorted((root / "runs/wide24_change").glob("*.json")):
        copy(path.relative_to(root), f"audit/{path.name}")
    for path in sorted((root / "runs/r_extension_change").glob("*.json")):
        copy(path.relative_to(root), f"audit/R_{path.name}")
    copy((base / "downstream_original18_unstarted/run.plan.json").relative_to(root), "audit/original18_unstarted_plan.json")
    snapshot["captured_end_jst"] = now()
    csv_write(out / "primitive_updates.csv", primitive_csv)
    csv_write(out / "downstream_updates.csv", downstream_csv)
    csv_write(out / "training_summary.csv", summary_csv)
    write_json(out / "summary.json", snapshot)
    write_json(out / "artifact_provenance.json", copied)
    source_hashes = {}
    for folder in ("python", "scripts", "configs", "manifests", "unity/Assets/Scripts", "unity/Assets/Editor", "unity/ProjectSettings", "environment_sources"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"):
                source_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out / "live_source_sha256.json", source_hashes)
    print(json.dumps({"output": str(out), "captured": snapshot["captured_end_jst"],
                      "primitive_updates": {k:v["update"] for k,v in snapshot["primitive"].items()},
                      "downstream_updates": {k:v["update"] for k,v in snapshot["downstream"]["jobs"].items()},
                      "validations": snapshot["downstream"]["completed_validation_files"]}))


if __name__ == "__main__":
    main()
