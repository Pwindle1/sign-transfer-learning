"""Donor-level comparison of two arms (paper Section 5): exact permutation test over donor means,
the pooled same-recipe noise floor, and a two-sample t confidence interval, plus per-recipient gains.

Example (the headline, four standard vs four canonical donors on the 90-cell grid):
  python experiments/compare_donors.py --results paper_results/cells/*.json \\
      --control 0 1 2 3 --treatment 90 94 12 13
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from signcanon import treatment_effect


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--control", type=int, nargs="+", required=True)
    parser.add_argument("--treatment", type=int, nargs="+", required=True)
    parser.add_argument("--method", default="full_ft")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None,
                        help="restrict to these episode seeds (e.g. 0 1 2 for the 54-cell basis)")
    args = parser.parse_args()

    rows = []
    for path in args.results:
        rows += json.loads(open(path).read())
    if args.eval_seeds is not None:
        rows = [r for r in rows if r["eval_seed"] in set(args.eval_seeds)]

    result = treatment_effect(rows, args.control, args.treatment, args.method)
    print(f"effect {result['effect']:+.2f}  noise floor {result['noise_floor']:.2f}  "
          f"({result['effect_in_floor_units']:.1f}x)  95% CI [{result['ci95'][0]:+.2f}, {result['ci95'][1]:+.2f}]")
    print(f"exact permutation p: one-sided {result['p_one_sided']:.4f}  two-sided {result['p_two_sided']:.4f}  "
          f"(floor {result['attainable_floor']:.4f}, {result['n_permutations']} relabellings)  "
          f"complete separation: {result['complete_separation']}  common cells: {result['n_common_cells']}")
    print("donor means:", {k: round(v, 2) for k, v in result["donor_means"].items()})

    per_target = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["method"] != args.method or row["donor"] not in set(args.control) | set(args.treatment):
            continue
        group = "treatment" if row["donor"] in args.treatment else "control"
        per_target[row["target"]][group].append(row["accuracy"])
    print(f"\n{'recipient':15s} {'control':>9s} {'treatment':>10s} {'gain':>8s}")
    for target, groups in sorted(per_target.items()):
        control = np.mean(groups["control"])
        treatment = np.mean(groups["treatment"])
        print(f"{target:15s} {control:9.2f} {treatment:10.2f} {treatment - control:+8.2f}")


if __name__ == "__main__":
    main()
