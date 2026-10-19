"""Time pinned Depth Anything V2 Small on a rendered pilot RGB frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from tps_transfer.depth import DepthEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("runs/smoke/sample.png"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase0/depth_benchmark.json"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    encoded = args.image.read_bytes()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode rendered RGB image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    estimator = DepthEstimator(input_size=518)
    rows = []
    maps = {}
    for size in (518, 392, 266):
        estimator.input_size = size
        estimator.infer_rgb(rgb)  # warm up the exact inference path
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            depth = estimator.infer_rgb(rgb)
            durations.append(time.perf_counter() - started)
        maps[size] = depth
        rows.append({
            "input_size": size,
            "seconds_median": float(np.median(durations)),
            "seconds_min": float(np.min(durations)),
            "seconds_max": float(np.max(durations)),
            "raw_min": float(depth.min()),
            "raw_max": float(depth.max()),
            "raw_std": float(depth.std()),
        })
    reference = maps[518].ravel()
    for row in rows[1:]:
        candidate = maps[row["input_size"]].ravel()
        row["pixel_pearson_vs_518_one_frame"] = float(np.corrcoef(reference, candidate)[0, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.parent / "depth_518_raw.npy", maps[518])
    # This contrast stretch is solely for visual inspection; it is not model input.
    low, high = np.quantile(maps[518], (0.01, 0.99))
    preview = np.clip((maps[518] - low) / (high - low), 0, 1)
    cv2.imwrite(str(args.output.parent / "depth_518_preview.png"), np.rint(preview * 255).astype(np.uint8))
    result = {
        "rgb_image": str(args.image.resolve()),
        "rgb_sha256": hashlib.sha256(encoded).hexdigest(),
        "image_shape": list(rgb.shape),
        "device": str(estimator.device),
        "repeats": args.repeats,
        "max_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "measurements": rows,
        "scope": "one rendered smoke frame; timing and visual diagnostic only",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
