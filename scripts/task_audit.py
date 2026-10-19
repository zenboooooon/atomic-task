"""Audit waiting and collision policies against the first train layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_probe import run

CASES = (
    ("look", "zero", True, "time_limit"),
    ("look", "forward", False, "look_moved"),
    ("approach", "zero", True, "time_limit"),
    ("stop", "zero", True, "time_limit"),
    ("avoid", "forward", False, "collision_Obstacle0"),
    ("composite", "zero", True, "time_limit"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for task, policy, should_interrupt, expected_reason in CASES:
        result = run(task, "train", 0, policy, 450)
        if (result["success"] or result["interrupted"] != should_interrupt
                or result["diagnostics"].get("reason") != expected_reason):
            raise RuntimeError(f"unexpected outcome for {task}/{policy}: {result}")
        if result["component_total"] is None:
            raise RuntimeError(f"missing Unity reward ledger for {task}/{policy}")
        print(f"{task}/{policy}: {result['diagnostics'].get('reason')} "
              f"reward={result['cumulative_reward']:.4f}", flush=True)
        results.append(result)
    output = {"audit": "fixed negative policies, excluded from RL", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"cases": len(results), "all_expected": True}))


if __name__ == "__main__":
    main()
