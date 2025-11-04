"""Orchestrates feature construction for the direct prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from data.schema import GameRecord

from .availability import AvailabilityFeatures, compute_availability_features
from .market import MarketFeatures, compute_market_features
from .matchup import MatchupFeatures, compute_matchup_features

HOME_FEATURE_KEYS: List[str] = [
    "edge",
    "orb_edge",
    "tov_edge",
    "tpar",
    "ftr",
    "rest_days",
    "minutes_missing_top2",
    "star_out",
    "usage_share_top2",
]
AWAY_FEATURE_KEYS: List[str] = HOME_FEATURE_KEYS.copy()
SHARED_FEATURE_KEYS: List[str] = [
    "pace_mean",
    "pace_diff",  # NEW: Home pace - Away pace (provides orthogonal info to pace_mean)
    "implied_home_winprob",
    "implied_away_winprob",
    "team_weighted_ts_home",
    "team_weighted_ts_away",
]


@dataclass
class BuiltFeatures:
    x_home: Dict[str, float]
    x_away: Dict[str, float]
    x_shared: Dict[str, float]
    baseline_home: float
    baseline_away: float


class FeatureBuilder:
    def build(self, record: GameRecord) -> BuiltFeatures:
        market = compute_market_features(record.market)
        matchup = compute_matchup_features(record.teams)

        availability_home = AvailabilityFeatures(0.0, 0, 0.53, 0.0)
        availability_away = AvailabilityFeatures(0.0, 0, 0.53, 0.0)
        if record.players:
            availability_home = compute_availability_features(record.players.H)
            availability_away = compute_availability_features(record.players.A)

        x_shared = {
            "pace_mean": matchup.pace_mean,
            "pace_diff": matchup.pace_diff,  # NEW: Home pace - Away pace
            "implied_home_winprob": market.implied_home_winprob,
            "implied_away_winprob": market.implied_away_winprob,
            "team_weighted_ts_home": availability_home.team_weighted_ts,
            "team_weighted_ts_away": availability_away.team_weighted_ts,
        }

        x_home = {
            "edge": matchup.edge_home,
            "orb_edge": matchup.orb_edge_home,
            "tov_edge": matchup.tov_edge_home,
            "tpar": matchup.tpar_home,
            "ftr": matchup.ftr_home,
            "rest_days": matchup.rest_home or 0.0,
            "minutes_missing_top2": availability_home.minutes_missing_top2,
            "star_out": float(availability_home.star_out),
            "usage_share_top2": availability_home.usage_share_top2,
        }

        x_away = {
            "edge": matchup.edge_away,
            "orb_edge": matchup.orb_edge_away,
            "tov_edge": matchup.tov_edge_away,
            "tpar": matchup.tpar_away,
            "ftr": matchup.ftr_away,
            "rest_days": matchup.rest_away or 0.0,
            "minutes_missing_top2": availability_away.minutes_missing_top2,
            "star_out": float(availability_away.star_out),
            "usage_share_top2": availability_away.usage_share_top2,
        }

        return BuiltFeatures(
            x_home=x_home,
            x_away=x_away,
            x_shared=x_shared,
            baseline_home=market.baseline_home,
            baseline_away=market.baseline_away,
        )


__all__ = [
    "BuiltFeatures",
    "FeatureBuilder",
    "HOME_FEATURE_KEYS",
    "AWAY_FEATURE_KEYS",
    "SHARED_FEATURE_KEYS",
]

