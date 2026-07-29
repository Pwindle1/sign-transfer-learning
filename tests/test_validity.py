import numpy as np

from signcanon.skeleton import LEFT_HAND, RIGHT_HAND
from signcanon.validity import clip_validity, dropout_census, frame_presence


def test_present_hands_are_detected():
    clips = np.random.default_rng(0).normal(size=(3, 2, 32, 49)).astype(np.float32)
    assert frame_presence(clips, LEFT_HAND).all()
    assert clip_validity(clips).all()


def test_a_zeroed_hand_is_missing():
    clips = np.random.default_rng(1).normal(size=(3, 2, 32, 49)).astype(np.float32)
    clips[0, :, :, LEFT_HAND] = 0.0
    census = dropout_census(clips)
    assert census["left_hand_missing"] == 1 / 3
    assert census["right_hand_missing"] == 0.0
    assert not clip_validity(clips)[0]
    assert clip_validity(clips)[1:].all()


def test_partial_dropout_below_threshold_keeps_the_clip():
    clips = np.random.default_rng(2).normal(size=(2, 2, 32, 49)).astype(np.float32)
    clips[0, :, :10, RIGHT_HAND] = 0.0
    assert clip_validity(clips).all()
