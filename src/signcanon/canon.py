from __future__ import annotations

import numpy as np

from .skeleton import LEFT_HAND, RIGHT_HAND, mirror
from .validity import frame_presence


def hand_energies(clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip motion energy of each hand block: summed absolute frame-to-frame displacement."""
    motion = np.abs(np.diff(clips, axis=2))
    left = motion[:, :, :, LEFT_HAND].sum(axis=(1, 2, 3))
    right = motion[:, :, :, RIGHT_HAND].sum(axis=(1, 2, 3))
    return left, right


def flip_decisions(clips: np.ndarray, threshold: float = 1.2) -> np.ndarray:
    """The paper's rule: mirror a clip iff e_L > threshold * e_R (threshold 1.2 = the margin)."""
    left, right = hand_energies(clips)
    return left > threshold * right


def canonicalize(clips: np.ndarray, threshold: float = 1.2) -> np.ndarray:
    """Dominance canonicalization — the operator every headline number uses (paper Section 3)."""
    flip = flip_decisions(clips, threshold)
    out = clips.copy()
    out[flip] = mirror(clips[flip])
    return out


def _canonicalize_no_margin(clips: np.ndarray) -> np.ndarray:
    """Step 1 shared by every constructed variant: mirror iff e_L > e_R (no 1.2 margin), so the
    construction can set the corpus flip rate exactly. Identical to production canon_entropy step 1."""
    left, right = hand_energies(clips)
    out = clips.copy()
    is_left = left > right
    out[is_left] = mirror(clips[is_left])
    return out


def controlled_mixture(clips: np.ndarray, left_fraction: float, seed: int = 0) -> np.ndarray:
    """Random-fraction variant (paper Table 2 'random f', seeds 50-57): canonicalize with no margin,
    then mirror back a seeded, uniformly random subset of exactly round(f * N) clips, so the corpus is
    left-dominant at rate f with the assignment independent of signer and sign. f = 1.0 is the
    all-left mirror-image donor (seed 91)."""
    out = _canonicalize_no_margin(clips)
    n_flip_back = int(round(left_fraction * len(out)))
    if n_flip_back:
        chosen = np.random.default_rng(seed).permutation(len(out))[:n_flip_back]
        out[chosen] = mirror(out[chosen])
    return out


def all_left(clips: np.ndarray) -> np.ndarray:
    """Every clip left-dominant: the mirror image of the canonical corpus (seed 91)."""
    return controlled_mixture(clips, 1.0)


def unconditional_flip(clips: np.ndarray) -> np.ndarray:
    """Mirror EVERY clip, no energy comparison and no gate (paper 'unconditional flip', seeds 43-45).
    Every chirality statistic is identical to the natural corpus; only the storage side changes."""
    return mirror(clips)


def _greedy_blocks(counts: np.ndarray, target: float, seed: int) -> list[int]:
    """Choose blocks (signers or glosses) in a seeded random order, taking a block iff doing so moves
    the running clip count closer to the target. Returns the chosen block indices."""
    order = np.random.default_rng(seed).permutation(len(counts))
    chosen, run = [], 0
    for i in order:
        if abs(run + counts[i] - target) < abs(run - target):
            chosen.append(int(i))
            run += int(counts[i])
    return chosen


def signer_blocked(clips: np.ndarray, signers: np.ndarray, left_fraction: float, seed: int = 0) -> np.ndarray:
    """Signer-blocked variant (paper Table 2, seeds 70-72): canonicalize with no margin, then mirror
    back EVERY clip of a greedily chosen set of signers so the corpus left fraction lands as close as
    possible to left_fraction. Chirality is constant within each signer, which drives the
    chirality-signer coupling to its maximum at that marginal rate."""
    out = _canonicalize_no_margin(clips)
    unique = np.array(sorted(set(signers.tolist())))
    counts = np.array([int((signers == s).sum()) for s in unique], dtype=np.int64)
    chosen = _greedy_blocks(counts, left_fraction * len(signers), seed)
    pick = np.isin(signers, unique[chosen])
    out[pick] = mirror(out[pick])
    return out


def gloss_blocked(clips: np.ndarray, labels: np.ndarray, left_fraction: float, seed: int = 0) -> np.ndarray:
    """Gloss-blocked variant (paper Table 2, seeds 75-77): the sign-axis twin of signer_blocked —
    whole glosses are mirrored back instead of whole signers, so chirality is constant within each
    sign and the chirality-sign coupling is maximal at that marginal rate."""
    out = _canonicalize_no_margin(clips)
    unique = np.array(sorted(set(int(g) for g in labels)))
    counts = np.array([int((labels == g).sum()) for g in unique], dtype=np.int64)
    chosen = _greedy_blocks(counts, left_fraction * len(labels), seed)
    pick = np.isin(labels, unique[chosen])
    out[pick] = mirror(out[pick])
    return out


def signer_majority(clips: np.ndarray, signers: np.ndarray, threshold: float = 1.2) -> np.ndarray:
    """Signer-majority variant (paper Table 2, seeds 80-82): decide dominance per SIGNER by the
    majority of the paper's rule over that signer's clips, then mirror all clips of left-majority
    signers. Within a signer, clips keep their natural relative chirality."""
    flip = flip_decisions(clips, threshold)
    out = clips.copy()
    for s in sorted(set(signers.tolist())):
        member = signers == s
        if flip[member].mean() > 0.5:
            out[member] = mirror(clips[member])
    return out


def robust_flip_decisions(
    clips: np.ndarray, threshold: float = 1.2, min_covisible: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Auxiliary tool, NOT used for any number in the paper: judge dominance only from frames in
    which both hands are present, and abstain below min_covisible such frames."""
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
