import numpy as np

from models.distribution import (
    bivariate_poisson_pmf,
    compute_win_probabilities,
    expected_margin,
    expected_scores,
    margin_pmf,
)


def test_bivariate_poisson_normalization():
    joint = bivariate_poisson_pmf(110 / 3, 108 / 3, 5.0, max_points=120)
    assert joint.sum() <= 1.0
    assert np.isclose(joint.sum(), 1.0, atol=1e-2)


def test_margin_expectation_matches_joint():
    joint = bivariate_poisson_pmf(105 / 3, 100 / 3, 3.0, max_points=100)
    margin = margin_pmf(joint, (-40, 40))
    exp_home, exp_away = expected_scores(joint)
    exp_margin = expected_margin(margin, (-40, 40))
    assert np.isclose(exp_margin, exp_home - exp_away, atol=1e-2)


def test_win_probabilities_sum_to_one():
    joint = bivariate_poisson_pmf(112 / 3, 112 / 3, 2.5, max_points=90)
    win_home, win_away, tie = compute_win_probabilities(joint)
    assert np.isclose(win_home + win_away + tie, 1.0, atol=1e-3)
