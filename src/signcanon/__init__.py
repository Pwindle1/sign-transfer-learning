"""signcanon - dominance canonicalization for cross-lingual sign language transfer.

Companion code for "Using Sign Phonology to Improve Few-Shot Cross-Lingual Transfer Learning for
Sign Language Recognition" (WSLP 2026)."""

from .adapt import classify_accuracy, embed, full_finetune, prototype_accuracy
from .canon import (
    all_left,
    canonicalize,
    controlled_mixture,
    flip_decisions,
    gloss_blocked,
    hand_energies,
    robust_canonicalize,
    robust_flip_decisions,
    signer_blocked,
    signer_majority,
    unconditional_flip,
)
from .data import (
    Corpus,
    Episode,
    donor_split,
    episodes,
    load_corpus,
    sample_support,
    signer_disjoint_split,
    subsample_balanced,
    to_model_input,
)
from .evaluate import exact_permutation_test, pooled_within_sd, treatment_effect
from .skeleton import INWARD_EDGES, LEFT_HAND, MIRROR_PERM, RIGHT_HAND, mirror, mirror_is_automorphism
from .validity import clip_validity, dropout_census, frame_presence

__all__ = [name for name in dir() if not name.startswith("_")]
