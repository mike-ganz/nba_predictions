"""Bivariate Poisson outcome model with elastic-net components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import ElasticNet


@dataclass
class BivariatePoissonConfig:
    alpha_home: float = 0.5
    l1_ratio_home: float = 0.1
    alpha_away: float = 0.5
    l1_ratio_away: float = 0.1
    alpha_shared: float = 0.5
    l1_ratio_shared: float = 0.0
    enable_overdispersion: bool = False
    overdispersion_alpha: float = 0.1
    overdispersion_l1_ratio: float = 0.0


class BivariatePoissonModel:
    def __init__(self, config: BivariatePoissonConfig | None = None) -> None:
        self.config = config or BivariatePoissonConfig()
        self.model_home = ElasticNet(
            alpha=self.config.alpha_home,
            l1_ratio=self.config.l1_ratio_home,
            max_iter=5000,
        )
        self.model_away = ElasticNet(
            alpha=self.config.alpha_away,
            l1_ratio=self.config.l1_ratio_away,
            max_iter=5000,
        )
        self.model_shared = ElasticNet(
            alpha=self.config.alpha_shared,
            l1_ratio=self.config.l1_ratio_shared,
            max_iter=5000,
        )
        self.model_overdispersion_home = None
        self.model_overdispersion_away = None
        if self.config.enable_overdispersion:
            self.model_overdispersion_home = ElasticNet(
                alpha=self.config.overdispersion_alpha,
                l1_ratio=self.config.overdispersion_l1_ratio,
                max_iter=5000,
            )
            self.model_overdispersion_away = ElasticNet(
                alpha=self.config.overdispersion_alpha,
                l1_ratio=self.config.overdispersion_l1_ratio,
                max_iter=5000,
            )

    def fit(
        self,
        x_home: np.ndarray,
        x_away: np.ndarray,
        x_shared: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
    ) -> None:
        self.model_home.fit(x_home, np.log(np.maximum(y_home, 1)))
        self.model_away.fit(x_away, np.log(np.maximum(y_away, 1)))
        shared_target = np.log(np.maximum(np.minimum(y_home, y_away), 1))
        self.model_shared.fit(x_shared, shared_target)

        if self.config.enable_overdispersion and self.model_overdispersion_home:
            residual_home = np.log1p(np.maximum(y_home, 0)) - self.model_home.predict(x_home)
            residual_away = np.log1p(np.maximum(y_away, 0)) - self.model_away.predict(x_away)
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
        log_lambda_home = np.log(np.maximum(baseline_home, 1e-6)) + self.model_home.predict(x_home)
        log_lambda_away = np.log(np.maximum(baseline_away, 1e-6)) + self.model_away.predict(x_away)
        log_kappa = self.model_shared.predict(x_shared)
        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)
        kappa = np.exp(log_kappa)

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

