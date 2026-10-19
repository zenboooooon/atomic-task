"""Compare fixed-depth resolutions on train pilot RGB and fit one calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from tps_transfer.depth import DepthEstimator
from tps_transfer.preprocess import fit_depth_calibration

ROOT = Path(__file__).resolve().parents[1]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def _edge_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def gradients(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        return np.stack((cv2.Sobel(image, cv2.CV_32F, 1, 0),
                         cv2.Sobel(image, cv2.CV_32F, 0, 1)), axis=-1)
    return _correlation(gradients(first), gradients(second))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=518, help="size used for fixed calibration")
    args = parser.parse_args()
    args.captures = args.captures.resolve()
    args.output = args.output.resolve()
    records = json.loads(args.captures.read_text())["results"]
    files = sorted({ROOT / name for record in records for name in record["captures"]
                    if record["split"] == "train"})
    if not files:
        raise ValueError("no train-side captures")
    if any(not file.is_file() or "train" not in file.name for file in files):
        raise ValueError("capture set contains missing or non-training images")
    sizes = sorted({252, 392, 518, args.size})
    estimators = {size: DepthEstimator(size) for size in sizes}
    if not torch.cuda.is_available():
        raise RuntimeError("GPU depth calibration was requested but CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    digest = hashlib.sha256()
    timing = {size: [] for size in sizes}
    pixel_correlation = {size: [] for size in sizes if size != 518}
    edge_correlation = {size: [] for size in sizes if size != 518}
    calibration_depths = []
    for file in files:
        image_bytes = file.read_bytes()
        digest.update(str(file.relative_to(ROOT)).encode())
        digest.update(hashlib.sha256(image_bytes).digest())
        bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (224, 224):
            raise ValueError(f"invalid capture {file}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictions = {}
        for size in sizes:
            torch.cuda.synchronize()
            started = time.perf_counter()
            predictions[size] = estimators[size].infer_rgb(rgb)
            torch.cuda.synchronize()
            timing[size].append(time.perf_counter() - started)
        reference = predictions[518]
        for size in pixel_correlation:
            pixel_correlation[size].append(_correlation(reference, predictions[size]))
            edge_correlation[size].append(_edge_correlation(reference, predictions[size]))
        calibration_depths.append(predictions[args.size])
    calibration = fit_depth_calibration(calibration_depths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output.with_name(args.output.stem + "_normalization.json")
    calibration.save(calibration_path)
    output = {
        "capture_count": len(files),
        "capture_set_sha256": digest.hexdigest(),
        "source_sweep": str(args.captures.relative_to(ROOT)),
        "device": device_name,
        "reference_size": 518,
        "calibration_size": args.size,
        "normalization": str(calibration_path.relative_to(ROOT)),
        "comparisons": {
            str(size): {
                "median_seconds": float(np.median(timing[size])),
                "pixel_correlation_median": (float(np.median(pixel_correlation[size]))
                                             if size != 518 else 1.0),
                "edge_correlation_median": (float(np.median(edge_correlation[size]))
                                            if size != 518 else 1.0),
            }
            for size in sizes
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
