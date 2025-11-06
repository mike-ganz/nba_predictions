"""Extended feature builder with additional categorical features for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from data.schema import GameRecord

from .availability import AvailabilityFeatures, compute_availability_features
from .market import MarketFeatures, compute_market_features
from .matchup import MatchupFeatures, compute_matchup_features

# Extended feature keys that include new experimental features
HOME_FEATURE_KEYS_EXTENDED: List[str] = [
    "edge",
    "orb_edge",
    "tov_edge",
    "tpar",
    "ftr",  # THIS IS THE KEY FEATURE WE'RE TESTING
    "rest_days",
    "minutes_missing_top2",
    "star_out",
    "usage_share_top2",
]

AWAY_FEATURE_KEYS_EXTENDED: List[str] = HOME_FEATURE_KEYS_EXTENDED.copy()

SHARED_FEATURE_KEYS_EXTENDED: List[str] = [
    "pace_mean",
    "pace_diff",
    "implied_home_winprob",
    "implied_away_winprob",
    "team_weighted_ts_home",
    "team_weighted_ts_away",
    # NEW: Game situation indicators
    "is_home_favorite",
    "is_home_underdog",
    "is_away_favorite",
    "is_away_underdog",
]


@dataclass
class BuiltFeaturesExtended:
    x_home: Dict[str, float]
    x_away: Dict[str, float]
    x_shared: Dict[str, float]
    baseline_home: float
    baseline_away: float


class FeatureBuilderExtended:
    """
    Extended feature builder that includes favorite/underdog indicators.
    
    New features:
    - is_home_favorite: 1.0 if home team is favored (spread < 0), else 0.0
    - is_home_underdog: 1.0 if home team is underdog (spread > 0), else 0.0
    - is_away_favorite: 1.0 if away team is favored (spread > 0), else 0.0
    - is_away_underdog: 1.0 if away team is underdog (spread < 0), else 0.0
    
    These are mutually exclusive binary indicators based on closing spread.
    """
    
    def __init__(self, include_fav_underdog_features: bool = True):
        """
        Args:
            include_fav_underdog_features: Whether to include the favorite/underdog indicators.
                                           Set to False to match original feature builder.
        """
        self.include_fav_underdog_features = include_fav_underdog_features
    
    def build(self, record: GameRecord) -> BuiltFeaturesExtended:
        market = compute_market_features(record.market)
        matchup = compute_matchup_features(record.teams)

        availability_home = AvailabilityFeatures(0.0, 0, 0.53, 0.0)
        availability_away = AvailabilityFeatures(0.0, 0, 0.53, 0.0)
        if record.players:
            availability_home = compute_availability_features(record.players.H)
            availability_away = compute_availability_features(record.players.A)

        # Base shared features
        x_shared = {
            "pace_mean": matchup.pace_mean,
            "pace_diff": matchup.pace_diff,
            "implied_home_winprob": market.implied_home_winprob,
            "implied_away_winprob": market.implied_away_winprob,
            "team_weighted_ts_home": availability_home.team_weighted_ts,
            "team_weighted_ts_away": availability_away.team_weighted_ts,
        }
        
        # Add favorite/underdog indicators if enabled
        if self.include_fav_underdog_features:
            spread = record.market.spread_home
            
            # Determine favorite/underdog status
            # Negative spread means home is favored
            # Positive spread means away is favored
            # Zero spread (pick'em) is treated as neither favorite nor underdog
            
            is_home_favorite = 1.0 if spread < -0.5 else 0.0  # -0.5 threshold to avoid pick'em
            is_home_underdog = 1.0 if spread > 0.5 else 0.0
            is_away_favorite = 1.0 if spread > 0.5 else 0.0
            is_away_underdog = 1.0 if spread < -0.5 else 0.0
            
            x_shared["is_home_favorite"] = is_home_favorite
            x_shared["is_home_underdog"] = is_home_underdog
            x_shared["is_away_favorite"] = is_away_favorite
            x_shared["is_away_underdog"] = is_away_underdog

        x_home = {
            "edge": matchup.edge_home,
            "orb_edge": matchup.orb_edge_home,
            "tov_edge": matchup.tov_edge_home,
            "tpar": matchup.tpar_home,
            "ftr": matchup.ftr_home,  # THIS IS INCLUDED IN EXTENDED VERSION
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

        return BuiltFeaturesExtended(
            x_home=x_home,
            x_away=x_away,
            x_shared=x_shared,
            baseline_home=market.baseline_home,
            baseline_away=market.baseline_away,
        )


__all__ = [
    "BuiltFeaturesExtended",
    "FeatureBuilderExtended",
    "HOME_FEATURE_KEYS_EXTENDED",
    "AWAY_FEATURE_KEYS_EXTENDED",
    "SHARED_FEATURE_KEYS_EXTENDED",
]

