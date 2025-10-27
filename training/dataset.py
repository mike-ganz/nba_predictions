"""Dataset preparation for bivariate Poisson training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from data.schema import GameRecord
from features.builder import (
    FeatureBuilder,
    HOME_FEATURE_KEYS,
    AWAY_FEATURE_KEYS,
    SHARED_FEATURE_KEYS,
)


@dataclass
class TrainingBatch:
    x_home: np.ndarray
    x_away: np.ndarray
    x_shared: np.ndarray
    baseline_home: np.ndarray
    baseline_away: np.ndarray
    y_home: np.ndarray
    y_away: np.ndarray
    y_shared: np.ndarray


class TrainingDataset:
    def __init__(self, records: Iterable[GameRecord], builder: FeatureBuilder | None = None):
        self.records: List[GameRecord] = list(records)
        self.builder = builder or FeatureBuilder()

    def build(self) -> TrainingBatch:
        x_home_list = []
        x_away_list = []
        x_shared_list = []
        baseline_home = []
        baseline_away = []
        y_home_raw = []
        y_away_raw = []

        for record in self.records:
            features = self.builder.build(record)
            x_home_list.append([features.x_home[k] for k in HOME_FEATURE_KEYS])
            x_away_list.append([features.x_away[k] for k in AWAY_FEATURE_KEYS])
            x_shared_list.append([features.x_shared[k] for k in SHARED_FEATURE_KEYS])
            baseline_home.append(features.baseline_home)
            baseline_away.append(features.baseline_away)
            if not record.outcome:
                raise ValueError("GameRecord missing outcome data for training")
            y_home_raw.append(record.outcome.home_final)
            y_away_raw.append(record.outcome.away_final)

        y_home_raw = np.array(y_home_raw)
        y_away_raw = np.array(y_away_raw)
        baseline_home = np.array(baseline_home)
        baseline_away = np.array(baseline_away)
        y_home_resid = np.log(np.maximum(y_home_raw, 1)) - np.log(np.maximum(baseline_home, 1))
        y_away_resid = np.log(np.maximum(y_away_raw, 1)) - np.log(np.maximum(baseline_away, 1))
        # Shared component learns baseline-relative game-level deviation, not absolute level
        shared_target = 0.5 * (y_home_resid + y_away_resid)

        return TrainingBatch(
            x_home=np.array(x_home_list),
            x_away=np.array(x_away_list),
            x_shared=np.array(x_shared_list),
            baseline_home=baseline_home,
            baseline_away=baseline_away,
            y_home=y_home_resid,
            y_away=y_away_resid,
            y_shared=shared_target,
        )


__all__ = [
    "TrainingDataset",
    "TrainingBatch",
]

