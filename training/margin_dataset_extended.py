"""Extended dataset preparation for margin prediction with experimental features.

This module extends the standard margin dataset to support:
- Including/excluding home_ftr feature
- Including/excluding favorite/underdog × home/away indicators
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from data.schema import GameRecord
from features.builder_extended import (
    FeatureBuilderExtended,
    HOME_FEATURE_KEYS_EXTENDED,
    AWAY_FEATURE_KEYS_EXTENDED,
    SHARED_FEATURE_KEYS_EXTENDED,
)


@dataclass
class MarginTrainingBatchExtended:
    """Training batch for direct margin prediction with extended features."""
    x: np.ndarray  # Combined features (N, D)
    y_margin: np.ndarray  # Actual margins (N,)
    baseline_margin: np.ndarray  # Market-implied margins (N,)
    actual_home: np.ndarray  # For evaluation (N,)
    actual_away: np.ndarray  # For evaluation (N,)
    market_spread_home: np.ndarray  # For evaluation (N,)
    game_ids: List[str]  # For tracking


class MarginTrainingDatasetExtended:
    """
    Extended dataset builder for margin prediction experiments.
    
    Supports:
    1. Including/excluding home_ftr feature
    2. Including/excluding favorite/underdog indicators
    3. Standard exclude_features mechanism
    """
    
    def __init__(
        self, 
        records: Iterable[GameRecord], 
        exclude_features: List[str] | None = None,
        include_diff_features: bool = True,
        include_fav_underdog_features: bool = False,
    ):
        """
        Args:
            records: Game records to build dataset from
            exclude_features: List of features to exclude (e.g., ['home_ftr'])
            include_diff_features: Whether to include home-away difference features
            include_fav_underdog_features: Whether to include favorite/underdog indicators
        """
        self.records: List[GameRecord] = list(records)
        self.exclude_features = exclude_features or []
        self.include_diff_features = include_diff_features
        self.include_fav_underdog_features = include_fav_underdog_features
        
        # Initialize builder with appropriate feature flags
        self.builder = FeatureBuilderExtended(
            include_fav_underdog_features=include_fav_underdog_features
        )
    
    def build(self) -> MarginTrainingBatchExtended:
        x_list = []
        y_margin = []
        baseline_margin = []
        actual_home = []
        actual_away = []
        market_spread_home = []
        game_ids = []
        
        for record in self.records:
            features = self.builder.build(record)
            
            # Combine all features into single vector
            # Format: [home_features, away_features, shared_features, home-away diffs]
            # Filter out excluded features
            home_feats = [features.x_home[k] for k in HOME_FEATURE_KEYS_EXTENDED 
                         if f'home_{k}' not in self.exclude_features]
            away_feats = [features.x_away[k] for k in AWAY_FEATURE_KEYS_EXTENDED 
                         if f'away_{k}' not in self.exclude_features]
            
            # For shared features, only include fav/underdog if enabled
            shared_keys_to_use = SHARED_FEATURE_KEYS_EXTENDED.copy()
            if not self.include_fav_underdog_features:
                # Remove fav/underdog features from list
                shared_keys_to_use = [k for k in shared_keys_to_use 
                                     if k not in ['is_home_favorite', 'is_home_underdog', 
                                                  'is_away_favorite', 'is_away_underdog']]
            
            shared_feats = [features.x_shared[k] for k in shared_keys_to_use 
                           if f'shared_{k}' not in self.exclude_features 
                           and k in features.x_shared]  # Check existence for fav/underdog
            
            # Add difference features (capture asymmetry)
            diff_feats = []
            if self.include_diff_features:
                for k in HOME_FEATURE_KEYS_EXTENDED:
                    if k in AWAY_FEATURE_KEYS_EXTENDED:  # If both sides have this feature
                        if f'diff_{k}' not in self.exclude_features:
                            # Only add diff if both features are included
                            if f'home_{k}' not in self.exclude_features and f'away_{k}' not in self.exclude_features:
                                diff_feats.append(features.x_home[k] - features.x_away.get(k, 0))
            
            combined = home_feats + away_feats + shared_feats + diff_feats
            x_list.append(combined)
            
            # Target: actual margin (optional for future games)
            if record.outcome:
                margin = record.outcome.home_final - record.outcome.away_final
                actual_home_val = record.outcome.home_final
                actual_away_val = record.outcome.away_final
            else:
                # Future game without outcome - use NaN placeholders
                margin = float('nan')
                actual_home_val = float('nan')
                actual_away_val = float('nan')
            
            y_margin.append(margin)
            
            # Baseline from market (spread implies expected margin)
            # Market spread = -3 means home expected to win by 3
            baseline = -record.market.spread_home
            baseline_margin.append(baseline)
            
            # Store for evaluation (will be NaN for future games)
            actual_home.append(actual_home_val)
            actual_away.append(actual_away_val)
            market_spread_home.append(record.market.spread_home)
            game_ids.append(record.game_id)
        
        return MarginTrainingBatchExtended(
            x=np.array(x_list),
            y_margin=np.array(y_margin),
            baseline_margin=np.array(baseline_margin),
            actual_home=np.array(actual_home),
            actual_away=np.array(actual_away),
            market_spread_home=np.array(market_spread_home),
            game_ids=game_ids,
        )


__all__ = [
    "MarginTrainingDatasetExtended",
    "MarginTrainingBatchExtended",
]

