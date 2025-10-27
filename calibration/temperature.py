"""Temperature scaling calibrator."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureCalibrator:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def fit(self, predicted_probs: np.ndarray, outcomes: np.ndarray) -> None:
        probs = np.clip(predicted_probs, 1e-12, 1.0)

        def loss(temp: float) -> float:
            scaled = self._apply_temperature(probs, temp)
            loss = 0.0
            for col in range(scaled.shape[1]):
                indices = np.clip(outcomes[:, col], 0, scaled.shape[2] - 1)
                loss += -np.log(scaled[:, col])[np.arange(len(scaled)), indices].mean()
            return loss / scaled.shape[1]

        result = minimize_scalar(loss, bounds=(0.25, 4.0), method="bounded")
        self.temperature = float(result.x)

    def transform(self, predicted_probs: np.ndarray) -> np.ndarray:
        probs = np.clip(predicted_probs, 1e-12, 1.0)
        return self._apply_temperature(probs, self.temperature)

    @staticmethod
    def _apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
        scaled = np.power(probs, 1.0 / max(temp, 1e-6))
        scaled[~np.isfinite(scaled)] = 0.0
        scaled_sum = scaled.sum(axis=2, keepdims=True)
        scaled_sum = np.clip(scaled_sum, 1e-12, None)
        return scaled / scaled_sum


