from __future__ import annotations

import numpy as np

NOSE = 0
BODY = slice(0, 7)
LEFT_HAND = slice(7, 28)
RIGHT_HAND = slice(28, 49)
NUM_NODES = 49

_HAND_INWARD = [
    (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9),
    (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19),
]


def _hand_edges(offset: int) -> list[tuple[int, int]]:
    return [(a + offset, b + offset) for a, b in _HAND_INWARD]


INWARD_EDGES = (
    [(1, 0), (2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (28, 6)]
    + _hand_edges(7)
    + _hand_edges(28)
)

_BODY_SWAP = [0, 2, 1, 4, 3, 6, 5]
MIRROR_PERM = np.array(_BODY_SWAP + list(range(28, 49)) + list(range(7, 28)))


def mirror(clips: np.ndarray) -> np.ndarray:
    out = clips[..., MIRROR_PERM].copy()
    out[:, 0] *= -1
    return out


def mirror_is_automorphism() -> bool:
    edges = {(min(a, b), max(a, b)) for a, b in INWARD_EDGES}
    mapped = {
        (min(int(MIRROR_PERM[a]), int(MIRROR_PERM[b])), max(int(MIRROR_PERM[a]), int(MIRROR_PERM[b])))
        for a, b in INWARD_EDGES
    }
    involution = bool(np.all(MIRROR_PERM[MIRROR_PERM] == np.arange(NUM_NODES)))
    return edges == mapped and involution
