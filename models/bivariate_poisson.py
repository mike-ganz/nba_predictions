"""Bivariate Poisson outcome model with elastic-net components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BivariatePoissonConfig:
    alpha_home: float = 1.0
    alpha_away: float = 1.0
    alpha_shared: float = 1.0
    enable_overdispersion: bool = False
    overdispersion_alpha: float = 1.0


class BivariatePoissonModel:
    def __init__(self, config: BivariatePoissonConfig | None = None) -> None:
        self.config = config or BivariatePoissonConfig()
        self.model_home = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=self.config.alpha_home))
        ])
        self.model_away = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=self.config.alpha_away))
        ])
        self.model_shared = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=self.config.alpha_shared))
        ])
        self.model_overdispersion_home = None
        self.model_overdispersion_away = None
        if self.config.enable_overdispersion:
            self.model_overdispersion_home = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=self.config.overdispersion_alpha))
            ])
            self.model_overdispersion_away = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=self.config.overdispersion_alpha))
            ])

    def fit(
        self,
        x_home: np.ndarray,
        x_away: np.ndarray,
        x_shared: np.ndarray,
        y_home_resid: np.ndarray,
        y_away_resid: np.ndarray,
        y_shared: np.ndarray,
    ) -> None:
        self.model_home.fit(x_home, y_home_resid)
        self.model_away.fit(x_away, y_away_resid)
        self.model_shared.fit(x_shared, y_shared)

        if self.config.enable_overdispersion and self.model_overdispersion_home:
            residual_home = y_home_resid - self.model_home.predict(x_home)
            residual_away = y_away_resid - self.model_away.predict(x_away)
            self.model_overdispersion_home.fit(x_home, residual_home ** 2)
            self.model_overdispersion_away.fit(x_away, residual_away ** 2)

    def predict_rates(
        self,
        x_home: np.ndarray,
        x_away: np.ndarray,
        x_shared: np.ndarray,
        baseline_home: np.ndarray,
        baseline_away: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        log_baseline_home = np.log(np.maximum(baseline_home, 1e-6))
        log_baseline_away = np.log(np.maximum(baseline_away, 1e-6))

        log_lambda_home = log_baseline_home + self.model_home.predict(x_home)
        log_lambda_away = log_baseline_away + self.model_away.predict(x_away)
        log_kappa = self.model_shared.predict(x_shared)

        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)
        kappa = np.exp(np.clip(log_kappa, -10, 5))

        sigma_home = None
        sigma_away = None
        if self.config.enable_overdispersion and self.model_overdispersion_home:
            sigma_home = np.sqrt(np.maximum(self.model_overdispersion_home.predict(x_home), 1e-6))
            sigma_away = np.sqrt(np.maximum(self.model_overdispersion_away.predict(x_away), 1e-6))

        return lambda_home, lambda_away, kappa, sigma_home, sigma_away


__all__ = [
    "BivariatePoissonConfig",
    "BivariatePoissonModel",
]

