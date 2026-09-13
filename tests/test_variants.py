import numpy as np

from signcanon.canon import (
    all_left,
    canonicalize,
    controlled_mixture,
    flip_decisions,
    gloss_blocked,
    hand_energies,
    signer_blocked,
    signer_majority,
    unconditional_flip,
)
from signcanon.skeleton import LEFT_HAND, RIGHT_HAND, mirror


def make_clip(active: slice, rng, n=6):
    clips = np.zeros((n, 2, 32, 49), np.float32)
    clips[:, :, :, active] = rng.normal(0, 0.05, size=(n, 2, 32, 21)).cumsum(axis=2)
    clips += rng.normal(0, 1e-4, size=clips.shape)
    return clips


def mixed_corpus(rng, n_left=60, n_right=40):
    clips = np.concatenate([make_clip(LEFT_HAND, rng, n_left), make_clip(RIGHT_HAND, rng, n_right)])
    signers = np.array([f"s{i % 10}" for i in range(len(clips))])
    labels = np.array([i % 7 for i in range(len(clips))])
    return clips, signers, labels


def test_unconditional_flip_is_an_involution_that_swaps_energies():
    clips, _, _ = mixed_corpus(np.random.default_rng(0))
    flipped = unconditional_flip(clips)
    assert np.allclose(unconditional_flip(flipped), clips)
    left, right = hand_energies(clips)
    left_f, right_f = hand_energies(flipped)
    assert np.allclose(left, right_f, rtol=1e-5) and np.allclose(right, left_f, rtol=1e-5)


def test_unconditional_flip_inverts_the_flip_rate_exactly():
    clips, _, _ = mixed_corpus(np.random.default_rng(1))
    before, after = flip_decisions(clips), flip_decisions(unconditional_flip(clips))
    assert abs(before.mean() - (1 - after.mean())) < 1e-9


def test_all_left_is_the_mirror_image_of_canonical():
    clips, _, _ = mixed_corpus(np.random.default_rng(2))
    left, right = hand_energies(all_left(clips))
    assert (left > right).all()
    assert np.allclose(all_left(clips), controlled_mixture(clips, 1.0))


def test_signer_blocked_is_constant_within_signer_and_hits_the_rate():
    clips, signers, _ = mixed_corpus(np.random.default_rng(3))
    out = signer_blocked(clips, signers, 0.5, seed=0)
    left, right = hand_energies(out)
    is_left = left > right
    for s in set(signers.tolist()):
        assert len(set(is_left[signers == s].tolist())) == 1
    assert abs(is_left.mean() - 0.5) <= 0.1


def test_gloss_blocked_is_constant_within_gloss_and_hits_the_rate():
    clips, _, labels = mixed_corpus(np.random.default_rng(4))
    out = gloss_blocked(clips, labels, 0.5, seed=0)
    left, right = hand_energies(out)
    is_left = left > right
    for g in set(labels.tolist()):
        assert len(set(is_left[labels == g].tolist())) == 1
    assert abs(is_left.mean() - 0.5) <= 0.15


def test_signer_majority_mirrors_only_left_majority_signers():
    rng = np.random.default_rng(5)
    left_signer = make_clip(LEFT_HAND, rng, 8)
    right_signer = make_clip(RIGHT_HAND, rng, 8)
    clips = np.concatenate([left_signer, right_signer])
    signers = np.array(["a"] * 8 + ["b"] * 8)
    out = signer_majority(clips, signers)
    assert np.allclose(out[:8], mirror(left_signer))
    assert np.allclose(out[8:], right_signer)


def test_every_variant_preserves_the_information_of_the_natural_corpus():
    clips, signers, labels = mixed_corpus(np.random.default_rng(6))
    for variant in (
        canonicalize(clips),
        unconditional_flip(clips),
        all_left(clips),
        controlled_mixture(clips, 0.3),
        signer_blocked(clips, signers, 0.5),
        gloss_blocked(clips, labels, 0.5),
        signer_majority(clips, signers),
    ):
        # each clip is either itself or its mirror image, so mirroring back recovers the original
        recovered = np.where(
            np.isclose(variant, clips, atol=1e-6).all(axis=(1, 2, 3))[:, None, None, None], variant, mirror(variant)
        )
        assert np.allclose(recovered, clips, atol=1e-5)
