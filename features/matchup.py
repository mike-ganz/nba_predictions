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

    # Use normalized features if available, otherwise fall back to raw features
    h_off = home.off_rating_norm if home.off_rating_norm is not None else home.off_rating
    h_def = home.def_rating_norm if home.def_rating_norm is not None else home.def_rating
    a_off = away.off_rating_norm if away.off_rating_norm is not None else away.off_rating
    a_def = away.def_rating_norm if away.def_rating_norm is not None else away.def_rating
    
    h_orb = home.off_reb_rate_norm if home.off_reb_rate_norm is not None else home.off_reb_rate
    h_drb = home.def_reb_rate_norm if home.def_reb_rate_norm is not None else home.def_reb_rate
    a_orb = away.off_reb_rate_norm if away.off_reb_rate_norm is not None else away.off_reb_rate
    a_drb = away.def_reb_rate_norm if away.def_reb_rate_norm is not None else away.def_reb_rate
    
    h_tov = home.turnover_rate_norm if home.turnover_rate_norm is not None else home.turnover_rate
    a_tov = away.turnover_rate_norm if away.turnover_rate_norm is not None else away.turnover_rate
    
    h_pace = home.pace_norm if home.pace_norm is not None else home.pace
    a_pace = away.pace_norm if away.pace_norm is not None else away.pace
    
    h_tpar = home.three_pt_rate_norm if home.three_pt_rate_norm is not None else home.three_pt_rate
    a_tpar = away.three_pt_rate_norm if away.three_pt_rate_norm is not None else away.three_pt_rate
    
    h_ftr = home.free_throw_rate_norm if home.free_throw_rate_norm is not None else home.free_throw_rate
    a_ftr = away.free_throw_rate_norm if away.free_throw_rate_norm is not None else away.free_throw_rate

    # Calculate edges using league-relative stats
    edge_home = h_off - a_def
    edge_away = a_off - h_def
    orb_edge_home = h_orb - a_drb
    orb_edge_away = a_orb - h_drb
    tov_edge_home = -(h_tov - a_tov)
    tov_edge_away = -(a_tov - h_tov)
    pace_mean = 0.5 * (h_pace + a_pace)
    pace_diff = h_pace - a_pace

    return MatchupFeatures(
        edge_home=edge_home,
        edge_away=edge_away,
        orb_edge_home=orb_edge_home,
        orb_edge_away=orb_edge_away,
        tov_edge_home=tov_edge_home,
        tov_edge_away=tov_edge_away,
        pace_mean=pace_mean,
        pace_diff=pace_diff,
        tpar_home=h_tpar,
        tpar_away=a_tpar,
        ftr_home=h_ftr,
        ftr_away=a_ftr,
        rest_home=home.rest_days,
        rest_away=away.rest_days,
    )


__all__ = [
    "MatchupFeatures",
    "compute_matchup_features",
]

