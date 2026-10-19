"""Build model inputs from RGB histories with one depth inference per frame."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from .history import HistoryWindow
from .preprocess import DepthCalibration, rgbd_tensor


class RelativeDepthModel(Protocol):
    def infer_rgb(self, rgb: np.ndarray) -> np.ndarray: ...


class FrameProcessor:
    """Cache transformed frames only for the lifetime of one rollout batch."""

    def __init__(
        self, depth: RelativeDepthModel, calibration: DepthCalibration,
        output_size: int = 96,
    ) -> None:
        self.depth = depth
        self.calibration = calibration
        self.output_size = output_size
        self._frames: dict[int, tuple[np.ndarray, Tensor]] = {}

    def process(self, rgb: np.ndarray) -> Tensor:
        key = id(rgb)
        cached = self._frames.get(key)
        if cached is not None and cached[0] is rgb:
            return cached[1]
        depth = self.depth.infer_rgb(rgb)
        frame = rgbd_tensor(rgb, depth, self.calibration, self.output_size)
        self._frames[key] = (rgb, frame)
        return frame

    def clear(self) -> None:
        self._frames.clear()


def history_batch(
    histories: Sequence[HistoryWindow], processor: FrameProcessor,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Tensor]:
    if not histories:
        raise ValueError("empty history batch")
    zero_image = torch.zeros(4, processor.output_size, processor.output_size)
    images = []
    actions = []
    masks = []
    for history in histories:
        tokens = history.tokens()
        images.append(torch.stack([
            processor.process(rgb) if rgb is not None else zero_image
            for rgb in tokens.images
        ]))
        actions.append(torch.from_numpy(np.stack([
            np.asarray(action, dtype=np.float32) if action is not None
            else np.zeros(5, dtype=np.float32)
            for action in tokens.actions
        ])))
        masks.append(torch.tensor(tokens.valid, dtype=torch.bool))
    return (
        torch.stack(images).to(device),
        torch.stack(actions).to(device),
        torch.stack(masks).to(device),
    )
