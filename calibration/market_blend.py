"""Blend model expectations with market baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MarketBlendConfig:
    weight: float = 0.5
    weight_bounds: tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    regularization: float = 0.0


class MarketBlender:
    def __init__(self, config: MarketBlendConfig | None = None) -> None:
        self.config = config or MarketBlendConfig()
        self.weight: float = self.config.weight

    def fit(
        self,
        model_home: np.ndarray,
        model_away: np.ndarray,
        market_home: np.ndarray,
        market_away: np.ndarray,
        actual_home: np.ndarray,
        actual_away: np.ndarray,
    ) -> float:
        """Learn blend weight by minimizing squared error to actual scores."""

        if not (
            model_home.shape == model_away.shape == market_home.shape == market_away.shape == actual_home.shape == actual_away.shape
        ):
            raise ValueError("All input arrays must share the same shape")

        mh = model_home.ravel()
        ma = model_away.ravel()
        kh = market_home.ravel()
        ka = market_away.ravel()
        yh = actual_home.ravel()
        ya = actual_away.ravel()

        # Solve for w in least squares: minimize ||w*m + (1-w)*k - y||^2
        # Equivalent to minimize ||w*(m-k) + k - y||^2
        delta_home = mh - kh
        delta_away = ma - ka
        rhs_home = yh - kh
        rhs_away = ya - ka

        numer = np.dot(delta_home, rhs_home) + np.dot(delta_away, rhs_away)
        denom = np.dot(delta_home, delta_home) + np.dot(delta_away, delta_away)

        if denom <= 1e-12:
            self.weight = self.config.weight
            return self.weight

        w = numer / denom

        if self.config.regularization > 0:
            w /= (1.0 + self.config.regularization)

        low, high = self.config.weight_bounds
        self.weight = float(np.clip(w, low, high))
        return self.weight

    def blend(
        self,
        model_home: np.ndarray,
        model_away: np.ndarray,
        market_home: np.ndarray,
        market_away: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        w = self.weight
        blended_home = w * model_home + (1 - w) * market_home
        blended_away = w * model_away + (1 - w) * market_away
        return blended_home, blended_away


