"""Dataset preparation for direct margin prediction.

This module replaces the bivariate training dataset with a simpler approach:
- Single target: margin = home_score - away_score
- Combined features: home + away + shared + differences
"""

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
class MarginTrainingBatch:
    """Training batch for direct margin prediction."""
    x: np.ndarray  # Combined features (N, D)
    y_margin: np.ndarray  # Actual margins (N,)
    baseline_margin: np.ndarray  # Market-implied margins (N,)
    actual_home: np.ndarray  # For evaluation (N,)
    actual_away: np.ndarray  # For evaluation (N,)
    market_spread_home: np.ndarray  # For evaluation (N,)
    game_ids: List[str]  # For tracking


class MarginTrainingDataset:
    """
    Dataset builder for direct margin prediction.
    
    Combines home, away, and shared features into single feature vector.
    Target is simply: margin = home_score - away_score
    
    Key differences from bivariate approach:
    - Single target (margin) instead of three (home_resid, away_resid, shared)
    - Combined feature vector instead of separate home/away/shared
    - Added difference features to capture relative advantages
    """
    
    def __init__(self, records: Iterable[GameRecord], builder: FeatureBuilder | None = None, exclude_features: List[str] | None = None):
        self.records: List[GameRecord] = list(records)
        self.builder = builder or FeatureBuilder()
        self.exclude_features = exclude_features or []
    
    def build(self) -> MarginTrainingBatch:
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
            home_feats = [features.x_home[k] for k in HOME_FEATURE_KEYS 
                         if f'home_{k}' not in self.exclude_features]
            away_feats = [features.x_away[k] for k in AWAY_FEATURE_KEYS 
                         if f'away_{k}' not in self.exclude_features]
            shared_feats = [features.x_shared[k] for k in SHARED_FEATURE_KEYS 
                           if f'shared_{k}' not in self.exclude_features]
            
            # Add difference features (capture asymmetry)
            # These help the model learn relative advantages
            # Only compute diffs for features that exist on both sides
            diff_feats = []
            for k in HOME_FEATURE_KEYS:
                if k in AWAY_FEATURE_KEYS:  # If both sides have this feature
                    if f'diff_{k}' not in self.exclude_features:
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
        
        return MarginTrainingBatch(
            x=np.array(x_list),
            y_margin=np.array(y_margin),
            baseline_margin=np.array(baseline_margin),
            actual_home=np.array(actual_home),
            actual_away=np.array(actual_away),
            market_spread_home=np.array(market_spread_home),
            game_ids=game_ids,
        )


__all__ = [
    "MarginTrainingDataset",
    "MarginTrainingBatch",
]

