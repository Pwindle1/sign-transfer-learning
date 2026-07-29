from __future__ import annotations

import argparse
import json

import numpy as np

from signcanon import dropout_census, load_corpus
from signcanon.canon import flip_decisions
from signcanon.skeleton import LEFT_HAND, RIGHT_HAND
from signcanon.validity import frame_presence


def forced_decision_rates(clips: np.ndarray) -> dict[str, float]:
    left_missing = frame_presence(clips, LEFT_HAND).mean(axis=1) < 0.5
    right_missing = frame_presence(clips, RIGHT_HAND).mean(axis=1) < 0.5
    flips = flip_decisions(clips)
    n_flips = max(int(flips.sum()), 1)
    n_keeps = max(int((~flips).sum()), 1)
    return {
        "flip_rate": float(flips.mean()),
        "flips_forced_by_missing_right": float((flips & right_missing).sum() / n_flips),
        "non_flips_forced_by_missing_left": float(((~flips) & left_missing).sum() / n_keeps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = {}
    for path in args.corpora:
        corpus = load_corpus(path)
        report[corpus.name] = dropout_census(corpus.clips) | forced_decision_rates(corpus.clips)
        r = report[corpus.name]
        print(
            f"{corpus.name:15s} either-hand-missing {100 * r['either_hand_missing']:5.1f}%   "
            f"flips forced {100 * r['flips_forced_by_missing_right']:5.1f}%   "
            f"non-flips forced {100 * r['non_flips_forced_by_missing_left']:5.1f}%"
        )
    if args.out:
        open(args.out, "w").write(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
