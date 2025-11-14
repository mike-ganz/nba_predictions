"""Player availability aggregations used in feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from data.schema import PlayerAvailability


@dataclass
class PlayerBaseline:
    baseline_minutes: float
    ts_pct: float
    usage_rate: float


@dataclass
class BaselineCache:
    baselines: Dict[str, PlayerBaseline] = field(default_factory=dict)

    def get(self, player_id: str) -> Optional[PlayerBaseline]:
        return self.baselines.get(player_id)

    def update(self, player_id: str, baseline: PlayerBaseline) -> None:
        self.baselines[player_id] = baseline

    def clear(self) -> None:
        self.baselines.clear()


baseline_cache = BaselineCache()


@dataclass
class MinutesBaseline:
    """Historical baseline information for a single player."""

    player_id: str
    baseline_minutes: float
    ts_pct: float
    usage_rate: float


@dataclass
class AvailabilityFeatures:
    minutes_missing_top2: float
    star_out: int
    team_weighted_ts: float
    usage_share_top2: float


def compute_availability_features(
    players: Iterable[PlayerAvailability],
) -> AvailabilityFeatures:
    player_list: List[PlayerAvailability] = list(players)
    if not player_list:
        return AvailabilityFeatures(0.0, 0, 0.0, 0.0)

    # Availability features should depend only on the per-game baseline fields
    # carried on each PlayerAvailability record. We deliberately ignore and do
    # not mutate the global baseline_cache here so that the features for a
    # given game do not depend on which other games were processed earlier.

    # Identify the top-2 players by their baseline minutes for this game.
    sorted_players = sorted(
        player_list,
        key=lambda p: p.baseline_minutes,
        reverse=True,
    )
    top2 = sorted_players[:2]

    missing_minutes = 0.0
    star_out = 0
    usage_top2 = 0.0
    for player in top2:
        baseline_minutes = float(player.baseline_minutes or 0.0)
        projected = (
            float(player.projected_minutes)
            if player.projected_minutes is not None
            else baseline_minutes
        )
        missing = max(0.0, baseline_minutes - projected)
        missing_minutes += missing
        # Star is out if projected < 10% of their usual minutes (injury/rest/etc)
        if baseline_minutes > 0.0 and projected < (baseline_minutes * 0.10):
            star_out = 1
        # Use the per-game baseline usage rates for the top-2 share.
        usage_top2 += float(player.baseline_usage_rate or 0.25)

    total_minutes = 0.0
    weighted_ts = 0.0
    for player in player_list:
        baseline_minutes = float(player.baseline_minutes or 0.0)
        projected = (
            float(player.projected_minutes)
            if player.projected_minutes is not None
            else baseline_minutes
        )
        total_minutes += projected
        ts_pct = float(player.baseline_ts_pct or 0.53)
        weighted_ts += projected * ts_pct

    total_minutes = max(total_minutes, 1.0)
    weighted_ts /= total_minutes

    return AvailabilityFeatures(
        minutes_missing_top2=missing_minutes,
        star_out=star_out,
        team_weighted_ts=weighted_ts,
        usage_share_top2=usage_top2,
    )


def load_baselines(baseline_data: Dict[str, Dict[str, float]]) -> None:
    for player_id, payload in baseline_data.items():
        baseline_cache.update(
            player_id,
            PlayerBaseline(
                baseline_minutes=payload.get("baseline_minutes", 0.0),
                ts_pct=payload.get("ts_pct", 0.53),
                usage_rate=payload.get("usage_rate", 0.25),
            ),
        )


def load_baselines_from_csv(path: str) -> None:
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        load_baselines(
            {
                row["player_id"]: {
                    "baseline_minutes": row["baseline_minutes"],
                    "ts_pct": row.get("ts_pct", 0.53),
                    "usage_rate": row.get("usage_rate", 0.25),
                }
            }
        )


__all__ = [
    "MinutesBaseline",
    "AvailabilityFeatures",
    "compute_availability_features",
    "baseline_cache",
    "load_baselines",
    "load_baselines_from_csv",
]

