"""Dataset preparation for spread coverage prediction.

This module prepares data for binary classification: predicting whether the 
home team covers the spread (1) or not (0).
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
    """Training batch for spread coverage prediction."""
    x: np.ndarray  # Combined features (N, D)
    y_home_covers: np.ndarray  # Binary: 1 if home covers, 0 if away covers (N,)
    actual_home: np.ndarray  # For evaluation (N,)
    actual_away: np.ndarray  # For evaluation (N,)
    actual_margin: np.ndarray  # For evaluation (N,)
    market_spread_home: np.ndarray  # For evaluation and features (N,)
    game_ids: List[str]  # For tracking


class MarginTrainingDataset:
    """
    Dataset builder for spread coverage prediction.
    
    Combines home, away, and shared features into single feature vector.
    Target is binary: 1 if home team covers, 0 if away team covers.
    
    Coverage definition:
    - Home covers if actual_margin > -spread_home
    - For spread_home = -3 (home favored by 3), home covers if they win by more than 3
    """
    
    def __init__(self, records: Iterable[GameRecord], builder: FeatureBuilder | None = None, exclude_features: List[str] | None = None):
        self.records: List[GameRecord] = list(records)
        self.builder = builder or FeatureBuilder()
        self.exclude_features = exclude_features or []
    
    def build(self) -> MarginTrainingBatch:
        x_list = []
        y_home_covers = []
        actual_home = []
        actual_away = []
        actual_margin = []
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
            
            # Target: binary outcome (home covers = 1, away covers = 0)
            if record.outcome:
                margin = record.outcome.home_final - record.outcome.away_final
                # Home covers if: actual_margin > -spread_home
                # Example: spread_home = -3, home covers if margin > 3
                home_covers = 1 if margin > -record.market.spread_home else 0
                actual_home_val = record.outcome.home_final
                actual_away_val = record.outcome.away_final
                actual_margin_val = margin
            else:
                # Future game without outcome - use NaN placeholders
                home_covers = float('nan')
                actual_home_val = float('nan')
                actual_away_val = float('nan')
                actual_margin_val = float('nan')
            
            y_home_covers.append(home_covers)
            
            # Store for evaluation (will be NaN for future games)
            actual_home.append(actual_home_val)
            actual_away.append(actual_away_val)
            actual_margin.append(actual_margin_val)
            market_spread_home.append(record.market.spread_home)
            game_ids.append(record.game_id)
        
        return MarginTrainingBatch(
            x=np.array(x_list),
            y_home_covers=np.array(y_home_covers),
            actual_home=np.array(actual_home),
            actual_away=np.array(actual_away),
            actual_margin=np.array(actual_margin),
            market_spread_home=np.array(market_spread_home),
            game_ids=game_ids,
        )


__all__ = [
    "MarginTrainingDataset",
    "MarginTrainingBatch",
]
