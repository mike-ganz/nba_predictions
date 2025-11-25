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
    "b2b",
    "three_in_four",
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
    # spread_line_movement removed - empirically harmful to ATS performance
]

# Context features (league-wide volatility) - optional, added via include_context flag
CONTEXT_FEATURE_KEYS: List[str] = [
    "ctx_std_oeff",
    "ctx_std_deff",
    "ctx_std_pace",
    "ctx_std_orb",
]


@dataclass
class BuiltFeatures:
    x_home: Dict[str, float]
    x_away: Dict[str, float]
    x_shared: Dict[str, float]
    baseline_home: float
    baseline_away: float


class FeatureBuilder:
    def __init__(self, include_context: bool = False):
        """
        Initialize the feature builder.
        
        Args:
            include_context: If True, include league context features (volatility metrics).
        """
        self.include_context = include_context
    
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
            # spread_line_movement excluded - decreased ATS from 53.31% to 48.37%
        }
        
        # Add league context features if requested and available
        if self.include_context and record.league_context:
            x_shared["ctx_std_oeff"] = record.league_context.ctx_std_oeff or 3.5
            x_shared["ctx_std_deff"] = record.league_context.ctx_std_deff or 3.5
            x_shared["ctx_std_pace"] = record.league_context.ctx_std_pace or 2.0
            x_shared["ctx_std_orb"] = record.league_context.ctx_std_orb or 0.03

        x_home = {
            "edge": matchup.edge_home,
            "orb_edge": matchup.orb_edge_home,
            "tov_edge": matchup.tov_edge_home,
            "tpar": matchup.tpar_home,
            "ftr": matchup.ftr_home,
            "rest_days": matchup.rest_home or 0.0,
            # Explicit schedule density flags
            "b2b": 1.0 if record.teams.H.b2b else 0.0,
            "three_in_four": 1.0 if record.teams.H.three_in_four else 0.0,
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
            "b2b": 1.0 if record.teams.A.b2b else 0.0,
            "three_in_four": 1.0 if record.teams.A.three_in_four else 0.0,
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
    "CONTEXT_FEATURE_KEYS",
]

