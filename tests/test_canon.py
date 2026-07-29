import numpy as np

from signcanon.canon import (
    canonicalize,
    controlled_mixture,
    flip_decisions,
    hand_energies,
    robust_flip_decisions,
)
from signcanon.skeleton import LEFT_HAND, RIGHT_HAND, mirror


def make_clip(active: slice, rng, n=6):
    clips = np.zeros((n, 2, 32, 49), np.float32)
    clips[:, :, :, active] = rng.normal(0, 0.05, size=(n, 2, 32, 21)).cumsum(axis=2)
    clips += rng.normal(0, 1e-4, size=clips.shape)
    return clips


def test_right_dominant_clips_are_left_alone():
    clips = make_clip(RIGHT_HAND, np.random.default_rng(0))
    assert not flip_decisions(clips).any()
    assert np.allclose(canonicalize(clips), clips)


def test_left_dominant_clips_are_mirrored():
    clips = make_clip(LEFT_HAND, np.random.default_rng(1))
    assert flip_decisions(clips).all()
    left, right = hand_energies(canonicalize(clips))
    assert (right > left).all()


def test_canonical_form_is_mirror_invariant():
    clips = make_clip(LEFT_HAND, np.random.default_rng(2))
    assert np.allclose(canonicalize(clips), canonicalize(mirror(clips)), atol=1e-5)


def test_missing_right_hand_forces_a_flip():
    clips = make_clip(LEFT_HAND, np.random.default_rng(3))
    clips[:, :, :, RIGHT_HAND] = 0.0
    assert flip_decisions(clips).all()


def test_robust_rule_abstains_when_hands_are_never_covisible():
    clips = make_clip(LEFT_HAND, np.random.default_rng(4))
    clips[:, :, :, RIGHT_HAND] = 0.0
    flips, decidable = robust_flip_decisions(clips)
    assert not decidable.any()
    assert not flips.any()


def test_robust_rule_matches_naive_rule_on_clean_clips():
    rng = np.random.default_rng(5)
    clips = np.zeros((8, 2, 32, 49), np.float32)
    clips[:, :, :, LEFT_HAND] = rng.normal(0, 0.05, size=(8, 2, 32, 21)).cumsum(axis=2)
    clips[:, :, :, RIGHT_HAND] = rng.normal(0, 0.01, size=(8, 2, 32, 21)).cumsum(axis=2)
    naive = flip_decisions(clips)
    robust, decidable = robust_flip_decisions(clips)
    assert decidable.all()
    assert np.array_equal(naive, robust)


def test_controlled_mixture_hits_the_requested_rate():
    rng = np.random.default_rng(6)
    halves = [make_clip(LEFT_HAND, rng, n=50), make_clip(RIGHT_HAND, rng, n=50)]
    clips = np.concatenate(halves)
    for fraction in (0.0, 0.3, 0.5):
        mixed = controlled_mixture(clips, fraction, seed=0)
        left, right = hand_energies(mixed)
        assert abs((left > right).mean() - fraction) < 0.05
