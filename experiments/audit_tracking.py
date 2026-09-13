"""The tracking census (paper Table 3): whole-clip hand loss, the decidable fraction, and the
corruption share - the fraction of rule-decidable clips whose decision was forced by a missing hand.

Conventions follow the paper: a clip is decidable when the weak-hand share min(e_L, e_R)/max(e_L, e_R)
is below 0.5; a clip in which neither hand was tracked (0/0) counts as undecidable ('strict').
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from signcanon import dropout_census, load_corpus
from signcanon.canon import flip_decisions, hand_energies
from signcanon.skeleton import LEFT_HAND, RIGHT_HAND
from signcanon.validity import frame_presence


def decidability(clips: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    left, right = hand_energies(clips)
    hi, lo = np.maximum(left, right), np.minimum(left, right)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(hi > 0, lo / np.where(hi > 0, hi, 1.0), 1.0)   # 0/0 -> undecidable
    return share < threshold


def census(clips: np.ndarray) -> dict[str, float]:
    left_missing = frame_presence(clips, LEFT_HAND).mean(axis=1) < 0.5
    right_missing = frame_presence(clips, RIGHT_HAND).mean(axis=1) < 0.5
    lost = left_missing | right_missing
    decidable = decidability(clips)
    flips = flip_decisions(clips)
    n_flips = max(int(flips.sum()), 1)
    n_keeps = max(int((~flips).sum()), 1)
    return dropout_census(clips) | {
        "decidable_fraction": float(decidable.mean()),
        "corruption_share": float((decidable & lost).sum() / max(int(decidable.sum()), 1)),
        "flip_rate": float(flips.mean()),
        "flip_rate_on_undecidable": float(flips[~decidable].mean()) if (~decidable).any() else 0.0,
        "flips_forced_by_missing_right": float((flips & right_missing).sum() / n_flips),
        "non_flips_forced_by_missing_left": float(((~flips) & left_missing).sum() / n_keeps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = {}
    print(f"{'corpus':12s} {'loss %':>7s} {'dec. %':>7s} {'corr. %':>8s} {'flip %':>7s} {'fractional':>10s}")
    for path in args.corpora:
        corpus = load_corpus(path)
        r = report[corpus.name] = census(corpus.clips)
        print(f"{corpus.name:12s} {100 * r['either_hand_missing']:7.2f} {100 * r['decidable_fraction']:7.2f} "
              f"{100 * r['corruption_share']:8.2f} {100 * r['flip_rate']:7.2f} {r['fractional_presence_clips']:10d}")
    if args.out:
        open(args.out, "w").write(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
