import numpy as np

from signcanon.data import Corpus, episodes, sample_support, signer_disjoint_split


def make_corpus(n_signers=12, n_classes=10, clips_per=4, seed=0):
    rng = np.random.default_rng(seed)
    labels, signers = [], []
    for s in range(n_signers):
        for c in range(n_classes):
            for _ in range(clips_per):
                labels.append(c)
                signers.append(f"s{s}")
    clips = rng.normal(size=(len(labels), 2, 32, 49)).astype(np.float32)
    return Corpus("toy", clips, np.array(labels), np.array(signers))


def test_split_is_signer_disjoint():
    corpus = make_corpus()
    adapt, test, val, kind = signer_disjoint_split(corpus.labels, corpus.signers, 3, 2, seed=0)
    assert kind == "signer_disjoint"
    groups = [set(corpus.signers[i] for i in part) for part in (adapt, test, val)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


def test_support_respects_k():
    corpus = make_corpus()
    adapt, *_ = signer_disjoint_split(corpus.labels, corpus.signers, 3, 2, seed=0)
    support = sample_support(adapt, corpus.labels, 2, seed=0)
    counts = np.bincount(corpus.labels[support])
    assert (counts <= 2).all()


def test_episode_test_sets_never_share_signers_with_support():
    corpus = make_corpus()
    for episode in episodes(corpus, ks=(2,), eval_seeds=(0, 1)):
        support_signers = set(corpus.signers[i] for i in episode.support)
        test_signers = set(corpus.signers[i] for i in episode.test)
        assert not support_signers & test_signers


def test_zero_validation_signers_reserves_none():
    corpus = make_corpus()
    adapt, test, val, _ = signer_disjoint_split(corpus.labels, corpus.signers, 3, 0, seed=0)
    assert len(val) == 0
    assert len(adapt) + len(test) == len(corpus.labels)


def test_donor_split_holds_out_exactly_the_requested_signers():
    from signcanon.data import donor_split

    corpus = make_corpus(n_signers=37)
    train, held, val = donor_split(corpus.signers, n_held=9, seed=42, val_cap=100)
    assert len(set(corpus.signers[train].tolist())) == 28
    assert len(set(corpus.signers[held].tolist())) == 9
    assert not set(corpus.signers[train].tolist()) & set(corpus.signers[held].tolist())
    assert len(val) == 100 and set(val.tolist()) <= set(held.tolist())


def test_balanced_subsample_is_class_balanced_and_capped():
    from signcanon.data import subsample_balanced

    corpus = make_corpus(n_classes=10, clips_per=4)
    keep = subsample_balanced(corpus.labels, 200, seed=0)
    counts = np.bincount(corpus.labels[keep])
    assert len(keep) <= 200 and counts.max() - counts.min() <= 1
