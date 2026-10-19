"""Check task reachability on several fixed pilot layouts with diagnostic policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_probe import run


POLICIES = {
    "look": "look",
    "approach": "forward",
    "stop": "stop",
    "avoid": "avoid",
    "composite": "composite",
    "treasure": "treasure",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1")
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--tasks", default=",".join(POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--worker-id", type=int, default=0)
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",")]
    splits = args.splits.split(",")
    tasks = args.tasks.split(",")
    results = []
    for task in tasks:
        for split in splits:
            for index in indices:
                result = run(task, split, index, POLICIES[task], 450,
                             args.capture_dir, args.worker_id)
                results.append(result)
                print(f"{task} {split} {index}: {result['success']} "
                      f"({result['decisions_sent']} decisions)", flush=True)
    output = {
        "probe_type": "scripted reachability; actions excluded from RL",
        "successes": sum(result["success"] for result in results),
        "attempts": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "results"}))


if __name__ == "__main__":
    main()
