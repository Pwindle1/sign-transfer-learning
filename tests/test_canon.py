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
    flips, covisible_ok = robust_flip_decisions(clips)
    assert not covisible_ok.any()
    assert not flips.any()


def test_robust_rule_matches_naive_rule_on_clean_clips():
    rng = np.random.default_rng(5)
    clips = np.zeros((8, 2, 32, 49), np.float32)
    clips[:, :, :, LEFT_HAND] = rng.normal(0, 0.05, size=(8, 2, 32, 21)).cumsum(axis=2)
    clips[:, :, :, RIGHT_HAND] = rng.normal(0, 0.01, size=(8, 2, 32, 21)).cumsum(axis=2)
    naive = flip_decisions(clips)
    robust, covisible_ok = robust_flip_decisions(clips)
    assert covisible_ok.all()
    assert np.array_equal(naive, robust)


def test_controlled_mixture_hits_the_requested_rate():
    rng = np.random.default_rng(6)
    halves = [make_clip(LEFT_HAND, rng, n=50), make_clip(RIGHT_HAND, rng, n=50)]
    clips = np.concatenate(halves)
    for fraction in (0.0, 0.3, 0.5):
        mixed = controlled_mixture(clips, fraction, seed=0)
        left, right = hand_energies(mixed)
        assert abs((left > right).mean() - fraction) < 0.05


def test_weak_hand_share_and_decidability_follow_the_paper_definition():
    from signcanon.canon import is_decidable, weak_hand_share

    rng = np.random.default_rng(7)
    one_handed = make_clip(RIGHT_HAND, rng)          # one hand clearly leads -> w near 0, decidable
    two_handed = np.zeros((4, 2, 32, 49), np.float32)
    motion = rng.normal(0, 0.05, size=(4, 2, 32, 21)).cumsum(axis=2)
    two_handed[:, :, :, LEFT_HAND] = motion          # both hands equal energy -> w near 1, undecidable
    two_handed[:, :, :, RIGHT_HAND] = motion
    assert (weak_hand_share(one_handed) < 0.5).all()
    assert is_decidable(one_handed).all()
    assert (weak_hand_share(two_handed) > 0.5).all()
    assert not is_decidable(two_handed).any()


def test_untracked_clip_is_undecidable():
    from signcanon.canon import is_decidable, weak_hand_share

    dead = np.zeros((2, 2, 32, 49), np.float32)       # 0/0 energies
    assert (weak_hand_share(dead) == 1.0).all()
    assert not is_decidable(dead).any()
