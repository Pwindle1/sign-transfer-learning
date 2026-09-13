from __future__ import annotations

from itertools import combinations

import numpy as np

# two-sided 97.5% Student-t quantiles; the paper's headline uses df = 6 (four vs four donors)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def _t975(df: int) -> float:
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    return _T975[max(k for k in keys if k <= df)] if df > keys[0] else _T975[keys[0]]


def donor_level_means(
    rows: list[dict], donors: list, method: str
) -> tuple[dict, list[tuple]]:
    """Mean accuracy per donor over the cells (target, k, eval_seed) that EVERY donor has - the unit
    of replication in the paper is the donor checkpoint, never the cell."""
    per_donor: dict = {d: {} for d in donors}
    for row in rows:
        if row["method"] == method and row["donor"] in per_donor:
            per_donor[row["donor"]][(row["target"], row["k"], row["eval_seed"])] = row["accuracy"]
    common = sorted(set.intersection(*(set(cells) for cells in per_donor.values())))
    means = {d: float(np.mean([per_donor[d][c] for c in common])) for d in donors}
    return means, common


def exact_permutation_test(values: np.ndarray, is_treatment: np.ndarray) -> dict:
    """Exact test of 'treatment mean - control mean' over every relabelling of the donors. With four
    vs four donors there are 70 relabellings, so the smallest attainable one-sided p is 1/70."""
    observed = values[is_treatment].mean() - values[~is_treatment].mean()
    n, k = len(values), int(is_treatment.sum())
    gaps = []
    for chosen in combinations(range(n), k):
        mask = np.zeros(n, bool)
        mask[list(chosen)] = True
        gaps.append(values[mask].mean() - values[~mask].mean())
    gaps = np.array(gaps)
    return {
        "effect": float(observed),
        "p_one_sided": float((gaps >= observed - 1e-12).sum() / len(gaps)),
        "p_two_sided": float((np.abs(gaps) >= abs(observed) - 1e-12).sum() / len(gaps)),
        "n_permutations": len(gaps),
        "attainable_floor": float(1.0 / len(gaps)),
        "complete_separation": bool(values[is_treatment].min() > values[~is_treatment].max()),
    }


def pooled_within_sd(values: np.ndarray, is_treatment: np.ndarray) -> float:
    """The paper's noise floor: pooled standard deviation across donors trained with the identical
    recipe and differing only in seed."""
    ss, df = 0.0, 0
    for group in (values[is_treatment], values[~is_treatment]):
        if len(group) >= 2:
            ss += float(((group - group.mean()) ** 2).sum())
            df += len(group) - 1
    return float(np.sqrt(ss / df)) if df else float("nan")


def treatment_effect(rows: list[dict], control: list, treatment: list, method: str) -> dict:
    """Donor-level treatment effect with the exact permutation test, the pooled noise floor, and a
    two-sample t confidence interval on donor-level means - the three quantities the paper reports."""
    means, common = donor_level_means(rows, control + treatment, method)
    values = np.array([means[d] for d in control + treatment])
    labels = np.array([d in treatment for d in control + treatment])
    result = exact_permutation_test(values, labels)
    floor = pooled_within_sd(values, labels)
    n_c, n_t = int((~labels).sum()), int(labels.sum())
    df = n_c + n_t - 2
    se = floor * np.sqrt(1 / n_c + 1 / n_t) if df > 0 else float("nan")
    half = _t975(df) * se if df > 0 else float("nan")
    result.update({
        "noise_floor": floor,
        "effect_in_floor_units": result["effect"] / floor if floor else float("nan"),
        "ci95": [result["effect"] - half, result["effect"] + half],
        "n_common_cells": len(common),
        "donor_means": {str(d): means[d] for d in control + treatment},
    })
    return result
