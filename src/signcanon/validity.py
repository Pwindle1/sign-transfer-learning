from __future__ import annotations

import numpy as np

from .skeleton import LEFT_HAND, RIGHT_HAND


def frame_presence(clips: np.ndarray, block: slice) -> np.ndarray:
    """(N, T) boolean: the hand block has any non-constant coordinate in that frame."""
    return clips[:, :, :, block].std(axis=3).sum(axis=1) >= 1e-6


def clip_validity(clips: np.ndarray, min_fraction: float = 0.5) -> np.ndarray:
    left = frame_presence(clips, LEFT_HAND).mean(axis=1) >= min_fraction
    right = frame_presence(clips, RIGHT_HAND).mean(axis=1) >= min_fraction
    return left & right


def dropout_census(clips: np.ndarray) -> dict[str, float]:
    """Whole-clip hand loss (paper Table 3). A hand counts as missing when it is present in fewer
    than half of the clip's frames. On every staged corpus in the paper per-clip presence is exactly
    0 or 1 for each hand block, so this is identical to 'never detected in any frame';
    `fractional_presence_clips` reports how many clips fall strictly between 0 and 1 so that the
    equivalence can be checked on any new corpus (it is 0 on all seven in the paper)."""
    left_frac = frame_presence(clips, LEFT_HAND).mean(axis=1)
    right_frac = frame_presence(clips, RIGHT_HAND).mean(axis=1)
    left, right = left_frac < 0.5, right_frac < 0.5
    fractional = ((left_frac > 0) & (left_frac < 1)) | ((right_frac > 0) & (right_frac < 1))
    return {
        "left_hand_missing": float(left.mean()),
        "right_hand_missing": float(right.mean()),
        "either_hand_missing": float((left | right).mean()),
        "fractional_presence_clips": int(fractional.sum()),
    }
