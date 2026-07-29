import numpy as np

from signcanon.skeleton import MIRROR_PERM, NUM_NODES, mirror, mirror_is_automorphism


def test_mirror_permutation_is_graph_automorphism():
    assert mirror_is_automorphism()


def test_mirror_is_involution():
    assert np.all(MIRROR_PERM[MIRROR_PERM] == np.arange(NUM_NODES))


def test_nose_is_the_only_fixed_point():
    assert [i for i in range(NUM_NODES) if MIRROR_PERM[i] == i] == [0]


def test_double_mirror_is_identity():
    clips = np.random.default_rng(0).normal(size=(4, 2, 32, NUM_NODES)).astype(np.float32)
    assert np.allclose(mirror(mirror(clips)), clips)
