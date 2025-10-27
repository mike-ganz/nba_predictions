import numpy as np

from calibration.calibrator import CalibrationConfig, Calibrator


def test_calibrator_fit_and_transform():
    config = CalibrationConfig(apply_isotonic=True, apply_temperature=True)
    calibrator = Calibrator(config)

    margin_cdf = np.linspace(0.1, 0.9, 10)[None, :].repeat(5, axis=0)
    joint_probs = np.ones((5, 5, 5)) / 25
    margin_outcomes = np.array([0, 1, 2, 3, 4])
    score_outcomes_home = np.array([2, 2, 2, 2, 2])
    score_outcomes_away = np.array([1, 2, 3, 1, 0])

    calibrator.fit(
        margin_cdf=margin_cdf,
        joint_probs=joint_probs,
        margin_outcomes=margin_outcomes,
        score_outcomes_home=score_outcomes_home,
        score_outcomes_away=score_outcomes_away,
    )

    transformed_cdf = calibrator.calibrate_margin_cdf(margin_cdf)
    transformed_probs = calibrator.calibrate_joint_probs(joint_probs)

    assert transformed_cdf.shape == margin_cdf.shape
    assert transformed_probs.shape == joint_probs.shape
