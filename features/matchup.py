"""Matchup derived features for offensive/defensive edges and context."""

from __future__ import annotations

from dataclasses import dataclass

from data.schema import GameTeams


@dataclass
class MatchupFeatures:
    edge_home: float
    edge_away: float
    orb_edge_home: float
    orb_edge_away: float
    tov_edge_home: float
    tov_edge_away: float
    pace_mean: float
    pace_diff: float
    tpar_home: float
    tpar_away: float
    ftr_home: float
    ftr_away: float
    rest_home: float | None
    rest_away: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "edge_home": self.edge_home,
            "edge_away": self.edge_away,
            "orb_edge_home": self.orb_edge_home,
            "orb_edge_away": self.orb_edge_away,
            "tov_edge_home": self.tov_edge_home,
            "tov_edge_away": self.tov_edge_away,
            "pace_mean": self.pace_mean,
            "pace_diff": self.pace_diff,
            "tpar_home": self.tpar_home,
            "tpar_away": self.tpar_away,
            "ftr_home": self.ftr_home,
            "ftr_away": self.ftr_away,
            "rest_home": self.rest_home,
            "rest_away": self.rest_away,
        }


def compute_matchup_features(teams: GameTeams) -> MatchupFeatures:
    home = teams.H
    away = teams.A

    edge_home = home.off_rating - away.def_rating
    edge_away = away.off_rating - home.def_rating
    orb_edge_home = home.off_reb_rate - away.def_reb_rate
    orb_edge_away = away.off_reb_rate - home.def_reb_rate
    tov_edge_home = -(home.turnover_rate - away.turnover_rate)
    tov_edge_away = -(away.turnover_rate - home.turnover_rate)
    pace_mean = 0.5 * (home.pace + away.pace)
    pace_diff = home.pace - away.pace

    return MatchupFeatures(
        edge_home=edge_home,
        edge_away=edge_away,
        orb_edge_home=orb_edge_home,
        orb_edge_away=orb_edge_away,
        tov_edge_home=tov_edge_home,
        tov_edge_away=tov_edge_away,
        pace_mean=pace_mean,
        pace_diff=pace_diff,
        tpar_home=home.three_pt_rate,
        tpar_away=away.three_pt_rate,
        ftr_home=home.free_throw_rate,
        ftr_away=away.free_throw_rate,
        rest_home=home.rest_days,
        rest_away=away.rest_days,
    )


__all__ = [
    "MatchupFeatures",
    "compute_matchup_features",
]

