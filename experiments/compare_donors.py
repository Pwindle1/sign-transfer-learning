from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from signcanon import treatment_effect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--control", type=int, nargs="+", required=True)
    parser.add_argument("--treatment", type=int, nargs="+", required=True)
    parser.add_argument("--method", default="full_ft")
    args = parser.parse_args()

    rows = []
    for path in args.results:
        rows += json.loads(open(path).read())

    result = treatment_effect(rows, args.control, args.treatment, args.method)
    print(json.dumps({k: v for k, v in result.items() if k != "donor_means"}, indent=2))

    per_target = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["method"] != args.method:
            continue
        group = "treatment" if row["donor"] in args.treatment else "control"
        per_target[row["target"]][group].append(row["accuracy"])
    print(f"\n{'target':15s} {'control':>9s} {'treatment':>10s} {'gain':>8s}")
    for target, groups in sorted(per_target.items()):
        control = np.mean(groups["control"])
        treatment = np.mean(groups["treatment"])
        print(f"{target:15s} {control:9.2f} {treatment:10.2f} {treatment - control:+8.2f}")


if __name__ == "__main__":
    main()
