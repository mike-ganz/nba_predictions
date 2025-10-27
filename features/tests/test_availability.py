"""Unit tests for availability feature computation."""

from data.schema import PlayerAvailability
from features.availability import AvailabilityFeatures, compute_availability_features, baseline_cache, load_baselines


def make_player(**kwargs):
    payload = {
        "player_id": "p1",
        "baseline_minutes": 30.0,
        "projected_minutes": 30.0,
        "baseline_ts_pct": 0.56,
        "baseline_usage_rate": 0.28,
    }
    payload.update(kwargs)
    return PlayerAvailability.model_validate(payload)


def test_compute_availability_features_baseline():
    baseline_cache.clear()
    players = [
        make_player(player_id="p1", baseline_minutes=34, projected_minutes=34),
        make_player(player_id="p2", baseline_minutes=32, projected_minutes=32, baseline_ts_pct=0.6),
        make_player(player_id="p3", baseline_minutes=20, projected_minutes=18, baseline_ts_pct=0.55),
    ]
    features = compute_availability_features(players)
    assert isinstance(features, AvailabilityFeatures)
    assert features.minutes_missing_top2 == 0
    assert features.star_out == 0
    assert 0.55 < features.team_weighted_ts < 0.58


def test_compute_availability_features_missing_minutes_and_star_out():
    baseline_cache.clear()
    players = [
        make_player(player_id="p1", baseline_minutes=36, projected_minutes=0),
        make_player(player_id="p2", baseline_minutes=34, projected_minutes=30),
        make_player(player_id="p3", baseline_minutes=24, projected_minutes=26),
    ]
    features = compute_availability_features(players)
    assert features.star_out == 1
    assert features.minutes_missing_top2 == 40  # 36 missing from top 1 plus 4 from top 2


def test_baseline_cache_backfill():
    baseline_cache.clear()
    load_baselines(
        {
            "p1": {"baseline_minutes": 32, "ts_pct": 0.58, "usage_rate": 0.27},
            "p2": {"baseline_minutes": 30, "ts_pct": 0.55, "usage_rate": 0.25},
        }
    )
    players = [
        make_player(player_id="p1", baseline_minutes=0, projected_minutes=None, baseline_ts_pct=None, baseline_usage_rate=None),
        make_player(player_id="p2", baseline_minutes=30, projected_minutes=None, baseline_ts_pct=None, baseline_usage_rate=None),
    ]
    features = compute_availability_features(players)
    assert abs(features.team_weighted_ts - 0.565483870967742) < 1e-6
    assert features.usage_share_top2 > 0.5


