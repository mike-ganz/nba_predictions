"""Training orchestration for the bivariate Poisson model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from models.bivariate_poisson import BivariatePoissonConfig, BivariatePoissonModel
from models.distribution import bivariate_poisson_pmf, margin_pmf
from .dataset import TrainingDataset


@dataclass
class TrainerConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_score: int = 150
    margin_low: int = -60
    margin_high: int = 60


class Trainer:
    def __init__(
        self,
        dataset: TrainingDataset,
        config: TrainerConfig | None = None,
        model_config: BivariatePoissonConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config or TrainerConfig()
        self.model = BivariatePoissonModel(model_config)
        self.margin_range: Tuple[int, int] = (self.config.margin_low, self.config.margin_high)

    def fit(self) -> Tuple[BivariatePoissonModel, dict[str, np.ndarray]]:
        batch = self.dataset.build()

        # If test_size is 0, use all data for training (no internal split)
        if self.config.test_size == 0 or self.config.test_size == 0.0:
            x_home_train = batch.x_home
            x_away_train = batch.x_away
            x_shared_train = batch.x_shared
            y_home_train = batch.y_home
            y_away_train = batch.y_away
            y_shared_train = batch.y_shared
            baseline_home_train = batch.baseline_home
            baseline_away_train = batch.baseline_away
            
            # Use same data for validation payload (calibration will use external holdout)
            x_home_val = batch.x_home
            x_away_val = batch.x_away
            x_shared_val = batch.x_shared
            y_home_val = batch.y_home
            y_away_val = batch.y_away
            baseline_home_val = batch.baseline_home
            baseline_away_val = batch.baseline_away
        else:
            # Normal train/test split
            x_home_train, x_home_val, y_home_train, y_home_val = train_test_split(
                batch.x_home,
                batch.y_home,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            x_away_train, x_away_val, y_away_train, y_away_val = train_test_split(
                batch.x_away,
                batch.y_away,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            x_shared_train, x_shared_val, y_shared_train, y_shared_val = train_test_split(
                batch.x_shared,
                batch.y_shared,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            baseline_home_train, baseline_home_val = train_test_split(
                batch.baseline_home,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            baseline_away_train, baseline_away_val = train_test_split(
                batch.baseline_away,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

        self.model.fit(
            x_home_train,
            x_away_train,
            x_shared_train,
            y_home_train,
            y_away_train,
            y_shared_train,
        )

        lambda_home_val, lambda_away_val, kappa_val, sigma_home_val, sigma_away_val = self.model.predict_rates(
            x_home_val,
            x_away_val,
            x_shared_val,
            baseline_home_val,
            baseline_away_val,
        )
        joint_pmfs = np.stack(
            [
                bivariate_poisson_pmf(h, a, k, max_points=self.config.max_score)
                for h, a, k in zip(lambda_home_val, lambda_away_val, kappa_val)
            ]
        )
        margin_cdfs = np.stack(
            [
                np.cumsum(
                    margin_pmf(joint, self.margin_range)
                )
                for joint in joint_pmfs
            ]
        )

        margin_outcomes = np.exp(y_home_val) * baseline_home_val - np.exp(y_away_val) * baseline_away_val

        validation_payload = {
            "joint_pmfs": joint_pmfs,
            "margin_cdf": margin_cdfs,
            "margin_outcomes": margin_outcomes,
            "score_outcomes_home": np.exp(y_home_val) * baseline_home_val,
            "score_outcomes_away": np.exp(y_away_val) * baseline_away_val,
            "market_home": baseline_home_val,
            "market_away": baseline_away_val,
            "sigma_home": sigma_home_val,
            "sigma_away": sigma_away_val,
        }

        return self.model, validation_payload


__all__ = [
    "Trainer",
    "TrainerConfig",
]

