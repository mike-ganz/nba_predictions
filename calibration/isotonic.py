"""Isotonic regression calibrator for probability distributions."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.regressor = IsotonicRegression(out_of_bounds="clip")

    def fit(self, predicted_cdf: np.ndarray, indicators: np.ndarray) -> None:
        x = predicted_cdf.ravel()
        y = indicators.ravel()
        self.regressor.fit(x, y)

    def transform(self, predicted_cdf: np.ndarray) -> np.ndarray:
        flat = self.regressor.transform(predicted_cdf.ravel())
        reshaped = flat.reshape(predicted_cdf.shape)
        return np.clip(reshaped, 0.0, 1.0)


