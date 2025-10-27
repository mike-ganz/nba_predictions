"""High-level calibration orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .isotonic import IsotonicCalibrator
from .market_blend import MarketBlendConfig, MarketBlender
from .temperature import TemperatureCalibrator


@dataclass
class CalibrationResult:
    isotonic: Optional[IsotonicCalibrator]
    temperature: Optional[TemperatureCalibrator]
    blender: MarketBlender


@dataclass
class CalibrationConfig:
    apply_isotonic: bool = True
    apply_temperature: bool = True
    market_weight: float = 0.5
    margin_low: int = -60
    margin_high: int = 60


class Calibrator:
    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self.isotonic: Optional[IsotonicCalibrator] = (
            IsotonicCalibrator() if self.config.apply_isotonic else None
        )
        self.temperature: Optional[TemperatureCalibrator] = (
            TemperatureCalibrator() if self.config.apply_temperature else None
        )
        self.blender = MarketBlender(MarketBlendConfig(weight=self.config.market_weight))
        self.margin_range: Tuple[int, int] = (self.config.margin_low, self.config.margin_high)
        self.sigma_home: Optional[np.ndarray] = None
        self.sigma_away: Optional[np.ndarray] = None

    def fit(
        self,
        margin_cdf: np.ndarray,
        joint_probs: np.ndarray,
        margin_outcomes: np.ndarray,
        score_outcomes_home: np.ndarray,
        score_outcomes_away: np.ndarray,
        sigma_home: Optional[np.ndarray] = None,
        sigma_away: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        if margin_cdf.ndim != 2:
            raise ValueError("margin_cdf must be [n_samples, n_margin_bins]")
        if joint_probs.ndim != 3:
            raise ValueError("joint_probs must be [n_samples, score_max+1, score_max+1]")
        if self.isotonic:
            indicators = np.zeros_like(margin_cdf)
            for i, outcome in enumerate(margin_outcomes):
                idx = np.clip(outcome - self.margin_range[0], 0, margin_cdf.shape[1] - 1)
                indicators[i, idx:] = 1.0
            self.isotonic.fit(margin_cdf, indicators)
        if self.temperature:
            home_pmfs = joint_probs.sum(axis=2)
            away_pmfs = joint_probs.sum(axis=1)
            stacked_pmfs = np.stack([home_pmfs, away_pmfs], axis=1)
            stacked_outcomes = np.stack([score_outcomes_home, score_outcomes_away], axis=1)
            self.temperature.fit(stacked_pmfs, stacked_outcomes)
        self.sigma_home = sigma_home
        self.sigma_away = sigma_away
        return CalibrationResult(
            isotonic=self.isotonic,
            temperature=self.temperature,
            blender=self.blender,
        )

    def calibrate_margin_cdf(self, margin_cdf: np.ndarray) -> np.ndarray:
        if self.isotonic is None:
            return margin_cdf
        return self.isotonic.transform(margin_cdf)

    def calibrate_joint_probs(self, joint_probs: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            return joint_probs
        home_pmfs = joint_probs.sum(axis=2)
        away_pmfs = joint_probs.sum(axis=1)
        stacked_pmfs = np.stack([home_pmfs, away_pmfs], axis=1)
        transformed = self.temperature.transform(stacked_pmfs)
        adjusted_joint = joint_probs.copy()
        for idx in range(joint_probs.shape[0]):
            target_home = transformed[idx, 0, :]
            target_away = transformed[idx, 1, :]
            joint = adjusted_joint[idx]
            row_sum = joint.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1e-8
            joint *= (target_home / row_sum)
            col_sum = joint.sum(axis=0, keepdims=True)
            col_sum[col_sum == 0] = 1e-8
            joint *= (target_away / col_sum)
            total_sum = joint.sum()
            if not np.isfinite(total_sum) or total_sum <= 0:
                joint = np.outer(target_home, target_away)
                total_sum = joint.sum()
            adjusted_joint[idx] = joint / total_sum
        return adjusted_joint

    def blend_expectations(
        self,
        model_home: np.ndarray,
        model_away: np.ndarray,
        market_home: np.ndarray,
        market_away: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.blender.blend(model_home, model_away, market_home, market_away)


__all__ = [
    "Calibrator",
    "CalibrationConfig",
    "CalibrationResult",
]

