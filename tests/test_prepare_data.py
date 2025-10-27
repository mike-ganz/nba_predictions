import pandas as pd
import pytest

import generate_team_stats as gts
from prepare_data import build_game_record


def _make_team_frame(team_name: str, dates, oeff: float) -> pd.DataFrame:
    rows = []
    for date in dates:
        rows.append(
            {
                "TEAM": team_name,
                "DATE": pd.to_datetime(date),
                "FGA": 80,
                "FG": 40,
                "3PA": 30,
                "FTA": 18,
                "OR": 10,
                "DR": 28,
                "A": 24,
                "TO": 12,
                "OEFF": oeff,
                "DEFF": 105.0,
                "PACE": 99.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    return frame


def test_generate_team_stats_prior_season_fallback():
    original_season_year = gts.SEASON_YEAR
    try:
        gts._team_data_cache.clear()
        gts._stats_cache.clear()
        gts.df = None
        gts.SEASON_YEAR = "2023-2024"

        team_city = "LA Lakers"
        current = _make_team_frame(team_city, ["2023-10-18", "2023-10-20"], 111.0)
        fallback = _make_team_frame(team_city, ["2023-02-01", "2023-02-05", "2023-02-10"], 118.0)

        gts._team_data_cache["2023-2024"] = current
        gts._team_data_cache["2022-2023"] = fallback

        stats = gts.generate_team_stats("Los Angeles Lakers", "2023-10-25", fallback_season="2023-2024")

        assert stats["STAT_SOURCE"] == "prior_season"
        assert stats["SEASON_USED"] == "2022-2023"
        assert stats["GAMES_CURRENT_SEASON"] == len(current)
        assert stats["GAMES_PLAYED"] == len(fallback)
        assert stats["OEFF"] == pytest.approx(118.0)
    finally:
        gts.SEASON_YEAR = original_season_year
        gts._team_data_cache.clear()
        gts._stats_cache.clear()
        gts.df = None


def test_build_game_record_includes_team_stat_metadata(monkeypatch):
    import prepare_data

    def fake_generate(team_name: str, target_date=None, fallback_season=None):
        if "Lakers" in team_name:
            return {
                "OEFF": 118.0,
                "DEFF": 106.0,
                "PACE": 100.0,
                "3PAr": 0.38,
                "FTr": 0.22,
                "ORr": 0.26,
                "DRr": 0.74,
                "ASTr": 0.62,
                "TOr": 0.13,
                "REST_DAYS": 4.0,
                "STAT_SOURCE": "home_source",
                "SEASON_USED": fallback_season,
                "PRIMARY_SEASON": fallback_season,
                "GAMES_PLAYED": 20,
                "GAMES_CURRENT_SEASON": 5,
            }
        return {
            "OEFF": 112.0,
            "DEFF": 109.0,
            "PACE": 97.0,
            "3PAr": 0.36,
            "FTr": 0.21,
            "ORr": 0.24,
            "DRr": 0.75,
            "ASTr": 0.6,
            "TOr": 0.12,
            "REST_DAYS": 1.0,
            "STAT_SOURCE": "away_source",
            "SEASON_USED": fallback_season,
            "PRIMARY_SEASON": fallback_season,
            "GAMES_PLAYED": 22,
            "GAMES_CURRENT_SEASON": 6,
        }

    monkeypatch.setattr(prepare_data, "generate_team_stats", fake_generate)

    game_rows = pd.DataFrame(
        [
            {
                "GAME-ID": "0001",
                "DATE": pd.Timestamp("2024-10-25"),
                "TEAM": "Los Angeles Lakers",
                "VENUE": "H",
                "CLOSING SPREAD": -5.5,
                "CLOSING TOTAL": 225.5,
                "MONEYLINE": -210,
                "F": 112,
            },
            {
                "GAME-ID": "0001",
                "DATE": pd.Timestamp("2024-10-25"),
                "TEAM": "Boston Celtics",
                "VENUE": "R",
                "CLOSING SPREAD": 5.5,
                "CLOSING TOTAL": 225.5,
                "MONEYLINE": 175,
                "F": 107,
            },
        ]
    )

    record = build_game_record(game_rows, "2024-2025", "VENUE")

    assert record["teams"]["H"]["off_rating"] != record["teams"]["A"]["off_rating"]
    assert record["teams"]["H"]["rest_days"] == 3
    assert record["teams"]["A"]["rest_days"] == 1

    home_meta = record["metadata"]["team_stats"]["home"]
    away_meta = record["metadata"]["team_stats"]["away"]

    assert home_meta["stat_source"] == "home_source"
    assert away_meta["stat_source"] == "away_source"
    assert home_meta["games_current_season"] == 5
    assert away_meta["games_current_season"] == 6
