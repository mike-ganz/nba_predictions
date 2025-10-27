"""Blend model expectations with market baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarketBlendConfig:
    weight: float = 0.5


class MarketBlender:
    def __init__(self, config: MarketBlendConfig | None = None) -> None:
        self.config = config or MarketBlendConfig()

    def blend(
        self,
        model_home: np.ndarray,
        model_away: np.ndarray,
        market_home: np.ndarray,
        market_away: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        w = self.config.weight
        blended_home = w * model_home + (1 - w) * market_home
        blended_away = w * model_away + (1 - w) * market_away
        return blended_home, blended_away


