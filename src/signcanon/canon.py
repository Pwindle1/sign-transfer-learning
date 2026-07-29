from __future__ import annotations

import numpy as np

from .skeleton import LEFT_HAND, RIGHT_HAND, mirror
from .validity import frame_presence


def hand_energies(clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    motion = np.abs(np.diff(clips, axis=2))
    left = motion[:, :, :, LEFT_HAND].sum(axis=(1, 2, 3))
    right = motion[:, :, :, RIGHT_HAND].sum(axis=(1, 2, 3))
    return left, right


def flip_decisions(clips: np.ndarray, threshold: float = 1.2) -> np.ndarray:
    left, right = hand_energies(clips)
    return left > threshold * right


def canonicalize(clips: np.ndarray, threshold: float = 1.2) -> np.ndarray:
    flip = flip_decisions(clips, threshold)
    out = clips.copy()
    out[flip] = mirror(clips[flip])
    return out


def controlled_mixture(clips: np.ndarray, left_fraction: float, seed: int = 0) -> np.ndarray:
    left, right = hand_energies(clips)
    out = clips.copy()
    is_left = left > right
    out[is_left] = mirror(clips[is_left])
    n_flip_back = int(round(left_fraction * len(out)))
    if n_flip_back:
        chosen = np.random.default_rng(seed).permutation(len(out))[:n_flip_back]
        out[chosen] = mirror(out[chosen])
    return out


def robust_flip_decisions(
    clips: np.ndarray, threshold: float = 1.2, min_covisible: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    left_seen = frame_presence(clips, LEFT_HAND)
    right_seen = frame_presence(clips, RIGHT_HAND)
    covisible = left_seen & right_seen
    pair_ok = covisible[:, :-1] & covisible[:, 1:]
    motion = np.abs(np.diff(clips, axis=2))
    mask = pair_ok[:, None, :, None]
    left = (motion[:, :, :, LEFT_HAND] * mask).sum(axis=(1, 2, 3))
    right = (motion[:, :, :, RIGHT_HAND] * mask).sum(axis=(1, 2, 3))
    decidable = pair_ok.sum(axis=1) >= min_covisible
    flip = decidable & (left > threshold * right)
    return flip, decidable


def robust_canonicalize(
    clips: np.ndarray, threshold: float = 1.2, min_covisible: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    flip, decidable = robust_flip_decisions(clips, threshold, min_covisible)
    out = clips.copy()
    out[flip] = mirror(clips[flip])
    return out, decidable
