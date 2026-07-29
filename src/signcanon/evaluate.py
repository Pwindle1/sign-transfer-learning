from __future__ import annotations

from itertools import combinations

import numpy as np


def donor_level_means(
    rows: list[dict], donors: list[int], method: str
) -> tuple[dict[int, float], list[tuple]]:
    per_donor: dict[int, dict[tuple, float]] = {d: {} for d in donors}
    for row in rows:
        if row["method"] == method and row["donor"] in per_donor:
            per_donor[row["donor"]][(row["target"], row["k"], row["eval_seed"])] = row["accuracy"]
    common = sorted(set.intersection(*(set(cells) for cells in per_donor.values())))
    means = {d: float(np.mean([per_donor[d][c] for c in common])) for d in donors}
    return means, common


def exact_permutation_test(values: np.ndarray, is_treatment: np.ndarray) -> dict:
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
        "p_one_sided": float((gaps >= observed).sum() / len(gaps)),
        "n_permutations": len(gaps),
        "attainable_floor": float(1.0 / len(gaps)),
    }


def treatment_effect(rows: list[dict], control: list[int], treatment: list[int], method: str) -> dict:
    means, common = donor_level_means(rows, control + treatment, method)
    values = np.array([means[d] for d in control + treatment])
    labels = np.array([d in treatment for d in control + treatment])
    result = exact_permutation_test(values, labels)
    result["control_sd"] = float(np.std(values[~labels], ddof=1))
    result["effect_in_sd_units"] = result["effect"] / result["control_sd"]
    result["n_common_cells"] = len(common)
    result["donor_means"] = {str(d): means[d] for d in control + treatment}
    return result
