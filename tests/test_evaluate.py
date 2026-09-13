import numpy as np

from signcanon.evaluate import exact_permutation_test, pooled_within_sd, treatment_effect


def rows_for(donor_means):
    rows = []
    for donor, mean in donor_means.items():
        for target in ("a", "b"):
            for k in (1, 5):
                for seed in (0, 1):
                    rows.append(dict(donor=donor, target=target, k=k, eval_seed=seed,
                                     method="full_ft", accuracy=mean + 0.1 * seed))
    return rows


def test_four_vs_four_separation_hits_the_design_floor():
    values = np.array([38.0, 39.0, 37.5, 38.5, 44.0, 45.0, 44.5, 45.5])
    treat = np.array([False] * 4 + [True] * 4)
    result = exact_permutation_test(values, treat)
    assert result["n_permutations"] == 70
    assert result["complete_separation"]
    assert abs(result["p_one_sided"] - 1 / 70) < 1e-9
    assert abs(result["p_two_sided"] - 2 / 70) < 1e-9


def test_pooled_floor_matches_the_hand_calculation():
    values = np.array([1.0, 3.0, 2.0, 4.0])
    treat = np.array([False, False, True, True])
    assert abs(pooled_within_sd(values, treat) - np.sqrt(2.0)) < 1e-9


def test_treatment_effect_reports_floor_units_and_interval():
    rows = rows_for({0: 38.0, 1: 39.0, 2: 37.5, 3: 38.5, 90: 44.0, 94: 45.0, 12: 44.5, 13: 45.5})
    result = treatment_effect(rows, [0, 1, 2, 3], [90, 94, 12, 13], "full_ft")
    assert abs(result["effect"] - 6.5) < 1e-9
    assert result["n_common_cells"] == 8
    assert result["ci95"][0] < result["effect"] < result["ci95"][1]
    assert result["effect_in_floor_units"] > 5
