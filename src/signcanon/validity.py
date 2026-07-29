from __future__ import annotations

import numpy as np

from .skeleton import LEFT_HAND, RIGHT_HAND


def frame_presence(clips: np.ndarray, block: slice) -> np.ndarray:
    return clips[:, :, :, block].std(axis=3).sum(axis=1) >= 1e-6


def clip_validity(clips: np.ndarray, min_fraction: float = 0.5) -> np.ndarray:
    left = frame_presence(clips, LEFT_HAND).mean(axis=1) >= min_fraction
    right = frame_presence(clips, RIGHT_HAND).mean(axis=1) >= min_fraction
    return left & right


def dropout_census(clips: np.ndarray) -> dict[str, float]:
    left = frame_presence(clips, LEFT_HAND).mean(axis=1) < 0.5
    right = frame_presence(clips, RIGHT_HAND).mean(axis=1) < 0.5
    return {
        "left_hand_missing": float(left.mean()),
        "right_hand_missing": float(right.mean()),
        "either_hand_missing": float((left | right).mean()),
    }
