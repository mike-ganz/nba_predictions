"""Market derived features and offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from data.schema import MarketInfo


def implied_prob_from_moneyline(moneyline: int) -> float:
    if moneyline < 0:
        return (-moneyline) / ((-moneyline) + 100.0)
    return 100.0 / (moneyline + 100.0)


def devig_two_way(home_raw: float, away_raw: float) -> tuple[float, float]:
    total = home_raw + away_raw
    if total <= 0:
        raise ValueError("Moneyline probabilities result in non-positive sum")
    home = home_raw / total
    away = away_raw / total
    return home, away


@dataclass
class MarketFeatures:
    baseline_home: float
    baseline_away: float
    implied_home_winprob: float
    implied_away_winprob: float
    spread_line_movement: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "baseline_home": self.baseline_home,
            "baseline_away": self.baseline_away,
            "implied_home_winprob": self.implied_home_winprob,
            "implied_away_winprob": self.implied_away_winprob,
            "spread_line_movement": self.spread_line_movement,
        }


def compute_market_features(market: MarketInfo) -> MarketFeatures:
    home_raw = implied_prob_from_moneyline(market.moneyline_home)
    away_raw = implied_prob_from_moneyline(market.moneyline_away)
    home_winprob, away_winprob = devig_two_way(home_raw, away_raw)

    baseline_home = 0.5 * (market.total + market.spread_home)
    baseline_away = 0.5 * (market.total - market.spread_home)

    # NOTE: spread_line_movement feature removed after testing
    # Testing showed it decreased ATS accuracy from 53.31% to 48.37%
    # Hypothesis: closing spread already incorporates all line movement information
    
    return MarketFeatures(
        baseline_home=baseline_home,
        baseline_away=baseline_away,
        implied_home_winprob=home_winprob,
        implied_away_winprob=away_winprob,
        spread_line_movement=0.0,  # Disabled - kept for backward compatibility
    )


__all__ = [
    "MarketFeatures",
    "compute_market_features",
    "implied_prob_from_moneyline",
    "devig_two_way",
]

