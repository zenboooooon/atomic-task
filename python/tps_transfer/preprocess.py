"""Shared RGB/depth geometry and fixed relative-depth calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class DepthCalibration:
    low: float
    high: float
    lower_quantile: float
    upper_quantile: float
    pixel_count: int
    training_digest: str

    def __post_init__(self) -> None:
        if not np.isfinite([self.low, self.high]).all() or self.low >= self.high:
            raise ValueError("depth calibration requires a finite nonzero range")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("invalid quantiles")
        if self.pixel_count < 1:
            raise ValueError("pixel count must be positive")

    def normalize(self, depth: np.ndarray) -> np.ndarray:
        raw = np.asarray(depth, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("depth map contains nonfinite values")
        return np.clip((raw - self.low) / (self.high - self.low), 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> DepthCalibration:
        return cls(**json.loads(path.read_text()))


def fit_depth_calibration(
    training_depths: Iterable[np.ndarray],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> DepthCalibration:
    """Pool all pixels from the training-side pilot images, then fix limits."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("invalid quantiles")
    arrays: list[np.ndarray] = []
    digest = hashlib.sha256()
    for item in training_depths:
        depth = np.asarray(item, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0 or not np.isfinite(depth).all():
            raise ValueError("calibration inputs must be finite nonempty 2D maps")
        contiguous = np.ascontiguousarray(depth)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
        arrays.append(contiguous.ravel())
    if not arrays:
        raise ValueError("no training depth maps")
    pooled = np.concatenate(arrays)
    low, high = np.quantile(pooled, [lower_quantile, upper_quantile])
    return DepthCalibration(
        low=float(low),
        high=float(high),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        pixel_count=int(pooled.size),
        training_digest=digest.hexdigest(),
    )


def rgbd_tensor(
    rgb: np.ndarray,
    relative_depth: np.ndarray,
    calibration: DepthCalibration,
    output_size: int = 96,
) -> Tensor:
    """Return CHW float RGBD using one aligned resize for all four channels."""
    color = np.asarray(rgb)
    depth = np.asarray(relative_depth)
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError("RGB must have shape (H, W, 3)")
    if depth.shape != color.shape[:2]:
        raise ValueError("RGB and depth must have the same image geometry")
    if output_size < 1:
        raise ValueError("output size must be positive")
    if color.dtype == np.uint8:
        color_float = color.astype(np.float32) / 255.0
    elif np.issubdtype(color.dtype, np.floating):
        color_float = color.astype(np.float32)
        if not np.isfinite(color_float).all() or color_float.min() < 0 or color_float.max() > 1:
            raise ValueError("floating RGB must be finite and scaled to [0, 1]")
    else:
        raise ValueError("RGB must be uint8 or floating point")
    depth_float = calibration.normalize(depth)
    channels = np.concatenate((color_float, depth_float[..., None]), axis=-1)
    tensor = torch.from_numpy(np.ascontiguousarray(channels)).permute(2, 0, 1)[None]
    return F.interpolate(
        tensor, size=(output_size, output_size), mode="bilinear", align_corners=False
    )[0]
