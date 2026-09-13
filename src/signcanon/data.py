from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Corpus:
    name: str
    clips: np.ndarray
    labels: np.ndarray
    signers: np.ndarray


def load_corpus(path: str | Path, name: str | None = None) -> Corpus:
    path = Path(path)
    archive = np.load(path, allow_pickle=False)
    return Corpus(
        name=name or path.stem,
        clips=archive["X"].astype(np.float32),
        labels=archive["y"],
        signers=archive["sg"],
    )


def signer_disjoint_split(
    labels: np.ndarray, signers: np.ndarray, n_test_signers: int, n_val_signers: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Recipient-side split used by every episode. Corpora without usable signer ids fall back to a
    clip-level split. n_val_signers = 0 reserves no validation signers."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(signers.tolist()))
    if len(unique) <= 2 or unique == ["NA"]:
        order = rng.permutation(len(labels))
        a, b = int(0.34 * len(labels)), int(0.5 * len(labels))
        return order[b:], order[:a], order[a:b], "clip_only"
    rng.shuffle(unique)
    n_test = max(1, min(n_test_signers, len(unique) - 2))
    n_val = 0 if n_val_signers == 0 else max(1, min(n_val_signers, len(unique) - n_test - 1))
    test_set, val_set = set(unique[:n_test]), set(unique[n_test : n_test + n_val])
    test = np.array([i for i in range(len(labels)) if signers[i] in test_set])
    val = np.array([i for i in range(len(labels)) if signers[i] in val_set], dtype=int)
    adapt = np.array([i for i in range(len(labels)) if signers[i] not in test_set | val_set])
    return adapt, test, val, "signer_disjoint"


def donor_split(
    signers: np.ndarray, n_held: int = 9, seed: int = 42, val_cap: int = 3000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The donor split used for every donor in the paper: hold n_held signers out entirely (train on
    all clips of the remaining signers - 28 signers / 26,157 clips on AUTSL), and take the first
    val_cap held-signer clips as the in-training validation set. Returns (train, held, val) indices."""
    unique = sorted(set(signers.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    held = set(unique[:n_held])
    train = np.array([i for i in range(len(signers)) if signers[i] not in held])
    held_idx = np.array([i for i in range(len(signers)) if signers[i] in held])
    return train, held_idx, held_idx[:val_cap]


def sample_support(
    adapt_indices: np.ndarray, labels: np.ndarray, k: int, seed: int, classes: set[int] | None = None
) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i in adapt_indices:
        if classes is None or labels[i] in classes:
            by_class[int(labels[i])].append(int(i))
    support: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        support += indices[:k]
    return np.array(support)


def cap_test_set(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    return np.sort(np.random.default_rng(9000 + seed).choice(indices, limit, replace=False))


def subsample_balanced(labels: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Class-balanced donor subsample for the scaling arms (seeds 60-63): an equal per-class quota of
    n // n_classes clips, shuffled per class with the seed, capped at n. Returns clip indices."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        by_class[int(c)].append(i)
    per = max(1, n // len(by_class))
    take: list[int] = []
    for indices in by_class.values():
        indices = list(indices)
        rng.shuffle(indices)
        take += indices[:per]
    return np.array(sorted(take[:n]))


@dataclass(frozen=True)
class Episode:
    eval_seed: int
    k: int
    support: np.ndarray
    test: np.ndarray
    classes: list[int]


def episodes(
    corpus: Corpus,
    ks: tuple[int, ...] = (1, 5, 10),
    eval_seeds: tuple[int, ...] = (0, 1, 2),
    test_cap: int = 1200,
):
    """k-shot episodes. The headline grid uses eval_seeds (0, 1, 2, 3, 4); the mechanism variants
    use (0, 1, 2). Support and test draws depend only on (corpus, k, eval_seed), so every arm is
    scored on exactly the same cells."""
    unique_signers = sorted(set(corpus.signers.tolist()))
    for seed in eval_seeds:
        adapt, test, val, _ = signer_disjoint_split(
            corpus.labels,
            corpus.signers,
            max(1, int(len(unique_signers) * 0.25)),
            max(1, int(len(unique_signers) * 0.2)),
            seed,
        )
        for k in ks:
            probe = sample_support(adapt, corpus.labels, 1, seed)
            classes = sorted(
                set(corpus.labels[probe].tolist())
                & set(corpus.labels[test].tolist())
                & set(corpus.labels[val].tolist())
            )
            if len(classes) < 2:
                continue
            class_set = set(classes)
            support = sample_support(adapt, corpus.labels, k, seed, classes=class_set)
            capped = cap_test_set(
                np.array([i for i in test if corpus.labels[i] in class_set]), test_cap, seed
            )
            yield Episode(eval_seed=seed, k=k, support=support, test=capped, classes=classes)


def to_model_input(clips: np.ndarray):
    """(N, 2, T, 49) -> the CTR-GCN input tensor: a zero third channel is appended (the model expects
    three input channels; only x and y carry data) and the axes are laid out as (N, T, 49*3)."""
    import torch

    n, c, t, v = clips.shape
    padded = np.concatenate([clips, np.zeros((n, 1, t, v), clips.dtype)], axis=1)
    return torch.tensor(np.ascontiguousarray(padded.transpose(0, 2, 3, 1).reshape(n, t, v * 3)))
