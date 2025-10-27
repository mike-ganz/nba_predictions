"""Tests for market feature utilities."""

import math

from data.schema import MarketInfo
from data.validators import GameDataValidationError, validate_market_vigorish
from features.market import (
    MarketFeatures,
    compute_market_features,
    devig_two_way,
    implied_prob_from_moneyline,
)


def test_implied_probabilities():
    assert math.isclose(implied_prob_from_moneyline(-150), 0.6, rel_tol=1e-6)
    assert math.isclose(implied_prob_from_moneyline(150), 0.4, rel_tol=1e-6)


def test_devig_two_way():
    home, away = devig_two_way(0.55, 0.45)
    assert math.isclose(home, 0.55 / (0.55 + 0.45))
    assert math.isclose(away, 0.45 / (0.55 + 0.45))


def test_compute_market_features_and_vigorish():
    market = MarketInfo.model_validate(
        {
            "spread_home": -5.5,
            "total": 225.5,
            "moneyline_home": -210,
            "moneyline_away": 180,
        }
    )
    validate_market_vigorish(market)
    features = compute_market_features(market)
    assert isinstance(features, MarketFeatures)
    assert features.baseline_home + features.baseline_away == market.total
    assert 0 < features.implied_home_winprob < 1

