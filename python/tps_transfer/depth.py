"""Pinned official Depth Anything V2 Small relative-depth inference."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPOSITORY_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
CHECKPOINT_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT / ".tools/depth-anything-v2"
CHECKPOINT = ROOT / ".tools/checkpoints/depth_anything_v2_vits.pth"


def _check_source() -> None:
    if not REPOSITORY.is_dir():
        raise FileNotFoundError(f"official Depth Anything V2 source missing: {REPOSITORY}")
    revision = subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REPOSITORY_COMMIT:
        raise ValueError(f"Depth Anything V2 source commit mismatch: {revision}")


def _check_checkpoint() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Depth Anything V2 Small checkpoint missing: {CHECKPOINT}")
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Depth Anything V2 Small checkpoint SHA-256 mismatch")


class DepthEstimator:
    def __init__(self, input_size: int = 518) -> None:
        if input_size < 14:
            raise ValueError("input size must be at least one 14-pixel patch")
        _check_source()
        _check_checkpoint()
        if str(REPOSITORY) not in sys.path:
            sys.path.insert(0, str(REPOSITORY))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)

    def infer_rgb(self, rgb: np.ndarray) -> np.ndarray:
        color = np.asarray(rgb)
        if color.ndim != 3 or color.shape[-1] != 3:
            raise ValueError("RGB must have shape (H, W, 3)")
        if color.dtype == np.uint8:
            color8 = color
        elif np.issubdtype(color.dtype, np.floating):
            if not np.isfinite(color).all() or color.min() < 0 or color.max() > 1:
                raise ValueError("floating RGB must be finite and in [0, 1]")
            color8 = np.rint(color * 255).astype(np.uint8)
        else:
            raise ValueError("RGB must be uint8 or floating point")
        bgr = cv2.cvtColor(color8, cv2.COLOR_RGB2BGR)
        with torch.inference_mode():
            depth = self.model.infer_image(bgr, input_size=self.input_size)
        if depth.shape != color.shape[:2] or not np.isfinite(depth).all():
            raise RuntimeError("depth inference returned an invalid map")
        return depth.astype(np.float32, copy=False)
