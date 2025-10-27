"""Tests for the FeatureBuilder orchestration."""

from data.schema import GameRecord
from features.builder import FeatureBuilder, BuiltFeatures


def make_game_record():
    return GameRecord.model_validate(
        {
            "game_id": "2024-12-01-MIA-BOS",
            "season": "2024-2025",
            "date": "2024-12-01",
            "teams": {
                "A": {
                    "team_id": "MIA",
                    "off_rating": 114.0,
                    "def_rating": 109.0,
                    "pace": 99.0,
                    "three_pt_rate": 0.39,
                    "free_throw_rate": 0.25,
                    "off_reb_rate": 0.28,
                    "def_reb_rate": 0.73,
                    "assist_rate": 0.59,
                    "turnover_rate": 0.14,
                    "rest_days": 1,
                },
                "H": {
                    "team_id": "BOS",
                    "off_rating": 118.0,
                    "def_rating": 107.0,
                    "pace": 101.0,
                    "three_pt_rate": 0.42,
                    "free_throw_rate": 0.26,
                    "off_reb_rate": 0.29,
                    "def_reb_rate": 0.72,
                    "assist_rate": 0.61,
                    "turnover_rate": 0.13,
                    "rest_days": 2,
                },
            },
            "market": {
                "spread_home": -4.5,
                "total": 226.0,
                "moneyline_home": -180,
                "moneyline_away": 160,
            },
            "players": {
                "A": [
                    {
                        "player_id": "A1",
                        "baseline_minutes": 34,
                        "projected_minutes": 32,
                        "baseline_ts_pct": 0.58,
                        "baseline_usage_rate": 0.28,
                    },
                    {
                        "player_id": "A2",
                        "baseline_minutes": 32,
                        "projected_minutes": 31,
                        "baseline_ts_pct": 0.56,
                        "baseline_usage_rate": 0.26,
                    },
                    {
                        "player_id": "A3",
                        "baseline_minutes": 28,
                        "projected_minutes": 30,
                        "baseline_ts_pct": 0.55,
                        "baseline_usage_rate": 0.24,
                    },
                    {
                        "player_id": "A4",
                        "baseline_minutes": 20,
                        "projected_minutes": 22,
                        "baseline_ts_pct": 0.54,
                        "baseline_usage_rate": 0.20,
                    },
                    {
                        "player_id": "A5",
                        "baseline_minutes": 18,
                        "projected_minutes": 18,
                        "baseline_ts_pct": 0.53,
                        "baseline_usage_rate": 0.18,
                    },
                ],
                "H": [
                    {
                        "player_id": "H1",
                        "baseline_minutes": 36,
                        "projected_minutes": 36,
                        "baseline_ts_pct": 0.61,
                        "baseline_usage_rate": 0.30,
                    },
                    {
                        "player_id": "H2",
                        "baseline_minutes": 34,
                        "projected_minutes": 33,
                        "baseline_ts_pct": 0.6,
                        "baseline_usage_rate": 0.27,
                    },
                    {
                        "player_id": "H3",
                        "baseline_minutes": 28,
                        "projected_minutes": 30,
                        "baseline_ts_pct": 0.59,
                        "baseline_usage_rate": 0.24,
                    },
                    {
                        "player_id": "H4",
                        "baseline_minutes": 22,
                        "projected_minutes": 20,
                        "baseline_ts_pct": 0.55,
                        "baseline_usage_rate": 0.19,
                    },
                    {
                        "player_id": "H5",
                        "baseline_minutes": 18,
                        "projected_minutes": 18,
                        "baseline_ts_pct": 0.54,
                        "baseline_usage_rate": 0.17,
                    },
                ],
            },
        }
    )


def test_feature_builder_outputs():
    builder = FeatureBuilder()
    record = make_game_record()
    built = builder.build(record)

    assert isinstance(built, BuiltFeatures)
    assert isinstance(built.baseline_home, float)
    assert isinstance(built.baseline_away, float)
    assert "edge" in built.x_home
    assert "edge" in built.x_away
    assert "pace_mean" in built.x_shared

