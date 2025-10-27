"""Unit tests for matchup feature generation."""

from data.schema import GameTeams, TeamSnapshot
from features.matchup import MatchupFeatures, compute_matchup_features


def test_compute_matchup_features_basic_edges():
    home = TeamSnapshot.model_validate(
        {
            "team_id": "HOME",
            "off_rating": 118.0,
            "def_rating": 110.0,
            "pace": 100.0,
            "three_pt_rate": 0.4,
            "free_throw_rate": 0.28,
            "off_reb_rate": 0.3,
            "def_reb_rate": 0.7,
            "assist_rate": 0.6,
            "turnover_rate": 0.13,
            "rest_days": 1,
        }
    )
    away = TeamSnapshot.model_validate(
        {
            "team_id": "AWAY",
            "off_rating": 112.0,
            "def_rating": 115.0,
            "pace": 98.0,
            "three_pt_rate": 0.37,
            "free_throw_rate": 0.25,
            "off_reb_rate": 0.29,
            "def_reb_rate": 0.72,
            "assist_rate": 0.58,
            "turnover_rate": 0.15,
            "rest_days": 2,
        }
    )
    teams = GameTeams(A=away, H=home)
    features = compute_matchup_features(teams)

    assert isinstance(features, MatchupFeatures)
    assert features.edge_home == home.off_rating - away.def_rating
    assert features.edge_away == away.off_rating - home.def_rating
    assert features.pace_diff == home.pace - away.pace

