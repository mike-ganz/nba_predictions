"""Feature engineering package for direct game outcome prediction."""

from .availability import (
    MinutesBaseline,
    AvailabilityFeatures,
    compute_availability_features,
)
from .market import compute_market_features
from .matchup import compute_matchup_features
from .builder import FeatureBuilder

__all__ = [
    "MinutesBaseline",
    "AvailabilityFeatures",
    "compute_availability_features",
    "compute_market_features",
    "compute_matchup_features",
    "FeatureBuilder",
]

