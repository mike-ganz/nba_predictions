"""
Champion ATS Drift Analysis for 2025-2026 Season.

This module builds a per-game evaluation dataset for the current season
and computes time-segmented ATS performance broken out by basic context:
- Home vs away picks
- Favorites vs underdogs
- Road favorites vs road underdogs
- Spread buckets

Outputs are written to an evaluation directory so they can be inspected
in CSV form or pulled into notebooks for further analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class AtsSummary:
    segment: str
    n_games: int
    ats_accuracy: float
    roi_pct: float


def _compute_ats_summary(df: pd.DataFrame, segment: str) -> AtsSummary:
    """Compute simple ATS summary for a slice of the data."""
    if df.empty:
        return AtsSummary(segment=segment, n_games=0, ats_accuracy=0.0, roi_pct=0.0)

    n_games = len(df)
    ats_accuracy = df["ats_correct"].mean() * 100.0

    # ROI per bet using the same convention as evaluate_margin.py:
    # +0.91 units for a correct pick, -1.0 units for an incorrect pick.
    # Note: pushes are treated as incorrect here, matching existing scripts.
    profit_per_bet = df["ats_correct"].apply(lambda x: 0.91 if x else -1.0).mean()
    roi_pct = profit_per_bet * 100.0

    return AtsSummary(
        segment=segment,
        n_games=int(n_games),
        ats_accuracy=float(ats_accuracy),
        roi_pct=float(roi_pct),
    )


def build_per_game_dataset(predictions_csv: Path) -> pd.DataFrame:
    """
    Build a per-game ATS evaluation dataset from the Champion predictions CSV.

    The input is expected to be the output of `predict_margin.py` for the
    Champion model, which includes:
      - game_id, date, away_team, home_team
      - market_spread_home, baseline_margin, pred_margin_mu
      - actual_home, actual_away, actual_margin (for completed games)
    """
    df = pd.read_csv(predictions_csv)

    # Restrict to completed games
    df = df[df["actual_margin"].notna()].copy()

    if df.empty:
        return df

    # Normalize types
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "game_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["game_index"] = np.arange(1, len(df) + 1)

    # Core ATS logic (match evaluate_margin.py)
    df["actual_home_covers"] = (df["actual_margin"] > -df["market_spread_home"]).astype(int)
    df["pred_home_covers"] = (df["pred_margin_mu"] > -df["market_spread_home"]).astype(int)
    df["ats_correct"] = (df["pred_home_covers"] == df["actual_home_covers"]).astype(int)

    # Moneyline-style win/loss from margin (for reference)
    df["actual_home_wins"] = (df["actual_margin"] > 0).astype(int)
    df["pred_home_wins"] = (df["pred_margin_mu"] > 0).astype(int)
    df["ml_correct"] = (df["pred_home_wins"] == df["actual_home_wins"]).astype(int)

    # Helper fields for segmentation
    df["spread_abs"] = df["market_spread_home"].abs()
    df["home_favored"] = df["market_spread_home"] < 0
    df["away_favored"] = df["market_spread_home"] > 0

    # Which side the model is implicitly betting on (home vs away)
    df["pick_side"] = np.where(df["pred_home_covers"] == 1, "home", "away")
    df["pick_team"] = np.where(df["pick_side"] == "home", df["home_team"], df["away_team"])

    # Favorite / underdog classification for the picked side
    # Market convention: negative spread = home favorite, positive = home underdog (away favorite)
    df["pick_is_favorite"] = (
        (df["pick_side"] == "home") & df["home_favored"]
    ) | (
        (df["pick_side"] == "away") & df["away_favored"]
    )
    df["pick_is_underdog"] = (
        (df["pick_side"] == "home") & df["away_favored"]
    ) | (
        (df["pick_side"] == "away") & df["home_favored"]
    )

    # Location + favorite/underdog buckets for the pick
    def _pick_location_bucket(row: pd.Series) -> str:
        if row["pick_side"] == "home" and row["home_favored"]:
            return "home_favorite"
        if row["pick_side"] == "home" and row["away_favored"]:
            return "home_underdog"
        if row["pick_side"] == "away" and row["away_favored"]:
            return "road_favorite"
        if row["pick_side"] == "away" and row["home_favored"]:
            return "road_underdog"
        return "other"

    df["pick_location_bucket"] = df.apply(_pick_location_bucket, axis=1)

    # Spread buckets for analysis (0-3, 3-6, 6-10, 10+)
    bins = [0.0, 3.0, 6.0, 10.0, np.inf]
    labels = ["0-3", "3-6", "6-10", "10+"]
    df["spread_bucket"] = pd.cut(df["spread_abs"], bins=bins, labels=labels, right=False)

    # Absolute prediction error (margin)
    df["abs_error"] = (df["pred_margin_mu"] - df["actual_margin"]).abs()

    return df


def summarize_time_buckets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute ATS performance over time:
      - By calendar week
      - By calendar month
      - By sequential 50-game blocks

    Returns a dict of name -> summary dataframe.
    """
    summaries: Dict[str, pd.DataFrame] = {}

    if df.empty:
        return summaries

    # Weekly aggregation
    weekly = []
    df_week = df.copy()
    df_week["week"] = df_week["date"].dt.to_period("W").dt.start_time
    for week, group in df_week.groupby("week", sort=True):
        s = _compute_ats_summary(group, segment=str(week.date()))
        weekly.append(
            {
                "week_start": week.date(),
                "n_games": s.n_games,
                "ats_accuracy": s.ats_accuracy,
                "roi_pct": s.roi_pct,
            }
        )
    summaries["by_week"] = pd.DataFrame(weekly).sort_values("week_start")

    # Monthly aggregation
    monthly = []
    df_month = df.copy()
    df_month["month"] = df_month["date"].dt.to_period("M").dt.to_timestamp()
    for month, group in df_month.groupby("month", sort=True):
        s = _compute_ats_summary(group, segment=str(month.date()))
        monthly.append(
            {
                "month": month.date(),
                "n_games": s.n_games,
                "ats_accuracy": s.ats_accuracy,
                "roi_pct": s.roi_pct,
            }
        )
    summaries["by_month"] = pd.DataFrame(monthly).sort_values("month")

    # Sequential 50-game blocks
    block_size = 50
    max_index = int(df["game_index"].max())
    blocks: List[Dict[str, object]] = []
    for start_idx in range(1, max_index + 1, block_size):
        end_idx = min(start_idx + block_size - 1, max_index)
        block_df = df[(df["game_index"] >= start_idx) & (df["game_index"] <= end_idx)]
        if block_df.empty:
            continue
        s = _compute_ats_summary(block_df, segment=f"{start_idx}-{end_idx}")
        blocks.append(
            {
                "block_start_index": start_idx,
                "block_end_index": end_idx,
                "n_games": s.n_games,
                "ats_accuracy": s.ats_accuracy,
                "roi_pct": s.roi_pct,
                "first_date": block_df["date"].min().date(),
                "last_date": block_df["date"].max().date(),
            }
        )
    summaries["by_game_block_50"] = pd.DataFrame(blocks).sort_values("block_start_index")

    return summaries


def summarize_context_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ATS performance for basic context segments over the full season.

    Segments include:
      - All picks
      - Home picks vs road picks
      - Favorites vs underdogs
      - Road favorites vs road underdogs
      - Spread buckets
    """
    rows: List[Dict[str, object]] = []

    def add_segment(name: str, subset: pd.DataFrame) -> None:
        s = _compute_ats_summary(subset, segment=name)
        rows.append(
            {
                "segment": name,
                "n_games": s.n_games,
                "ats_accuracy": s.ats_accuracy,
                "roi_pct": s.roi_pct,
            }
        )

    # Overall
    add_segment("overall", df)

    # Home vs road picks
    add_segment("pick_home", df[df["pick_side"] == "home"])
    add_segment("pick_away", df[df["pick_side"] == "away"])

    # Favorites vs underdogs
    add_segment("pick_favorite", df[df["pick_is_favorite"]])
    add_segment("pick_underdog", df[df["pick_is_underdog"]])

    # Road favorites vs road underdogs
    add_segment("road_favorite_picks", df[df["pick_location_bucket"] == "road_favorite"])
    add_segment("road_underdog_picks", df[df["pick_location_bucket"] == "road_underdog"])

    # Spread buckets (overall)
    for bucket in df["spread_bucket"].dropna().unique():
        bucket_df = df[df["spread_bucket"] == bucket]
        add_segment(f"spread_{bucket}", bucket_df)

    return pd.DataFrame(rows).sort_values("segment")


def summarize_road_underdogs(
    df: pd.DataFrame, block_size: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Deep-dive on road underdog picks:
      - How performance evolves over pick order (blocks of N bets)
      - Which teams account for most wins/losses as road dogs
      - Overall impact of excluding road underdogs
    """
    mask_rd = df["pick_location_bucket"] == "road_underdog"
    rd = df[mask_rd].copy()

    if rd.empty:
        empty_blocks = pd.DataFrame(
            columns=[
                "block_start_pick",
                "block_end_pick",
                "n_games",
                "ats_accuracy",
                "roi_pct",
                "first_date",
                "last_date",
            ]
        )
        empty_teams = pd.DataFrame(columns=["pick_team", "n_games", "ats_accuracy", "roi_pct"])
        overview = {
            "overall_ats": df["ats_correct"].mean() * 100.0 if not df.empty else 0.0,
            "overall_excl_road_dogs_ats": df["ats_correct"].mean() * 100.0 if not df.empty else 0.0,
            "road_dogs_ats": 0.0,
            "n_total": len(df),
            "n_road_dogs": 0,
        }
        return empty_blocks, empty_teams, overview

    # Overview metrics
    overall_ats = df["ats_correct"].mean() * 100.0
    excl_rd_ats = df.loc[~mask_rd, "ats_correct"].mean() * 100.0
    rd_ats = rd["ats_correct"].mean() * 100.0

    overview = {
        "overall_ats": overall_ats,
        "overall_excl_road_dogs_ats": excl_rd_ats,
        "road_dogs_ats": rd_ats,
        "n_total": len(df),
        "n_road_dogs": int(mask_rd.sum()),
    }

    # Performance over pick order (blocks of N road-dog bets)
    rd = rd.sort_values("game_index").reset_index(drop=True)
    road_blocks: List[Dict[str, object]] = []
    for start_idx in range(0, len(rd), block_size):
        end_idx = min(start_idx + block_size, len(rd))
        block = rd.iloc[start_idx:end_idx]
        s = _compute_ats_summary(block, segment=f"{start_idx+1}-{end_idx}")
        road_blocks.append(
            {
                "block_start_pick": start_idx + 1,
                "block_end_pick": end_idx,
                "n_games": s.n_games,
                "ats_accuracy": s.ats_accuracy,
                "roi_pct": s.roi_pct,
                "first_date": block["date"].min().date(),
                "last_date": block["date"].max().date(),
            }
        )
    by_pick_order = pd.DataFrame(road_blocks).sort_values("block_start_pick")

    # Team-level breakdown for road dogs
    team_rows: List[Dict[str, object]] = []
    for team, g in rd.groupby("pick_team"):
        s = _compute_ats_summary(g, segment=str(team))
        team_rows.append(
            {
                "pick_team": team,
                "n_games": s.n_games,
                "ats_accuracy": s.ats_accuracy,
                "roi_pct": s.roi_pct,
            }
        )
    by_team = pd.DataFrame(team_rows).sort_values("ats_accuracy")

    return by_pick_order, by_team, overview


def save_outputs(
    df_per_game: pd.DataFrame,
    time_summaries: Dict[str, pd.DataFrame],
    context_summary: pd.DataFrame,
    rd_pick_blocks: pd.DataFrame,
    rd_team_breakdown: pd.DataFrame,
    rd_overview: Dict[str, float],
    output_dir: Path,
) -> None:
    """Persist per-game and aggregated summaries to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-game dataset
    per_game_path = output_dir / "champion_2526_per_game_ats.csv"
    df_per_game.to_csv(per_game_path, index=False)

    # Time-based summaries
    for name, summary_df in time_summaries.items():
        summary_path = output_dir / f"champion_2526_ats_{name}.csv"
        summary_df.to_csv(summary_path, index=False)

    # Context segments (full-season)
    context_path = output_dir / "champion_2526_ats_context_segments.csv"
    context_summary.to_csv(context_path, index=False)

    # Road underdog deep-dive
    rd_blocks_path = output_dir / "champion_2526_ats_road_underdogs_by_pick_order.csv"
    rd_pick_blocks.to_csv(rd_blocks_path, index=False)

    rd_teams_path = output_dir / "champion_2526_ats_road_underdogs_by_team.csv"
    rd_team_breakdown.to_csv(rd_teams_path, index=False)

    # Simple overview as a one-row CSV for easy inspection
    rd_overview_path = output_dir / "champion_2526_ats_road_underdogs_overview.csv"
    pd.DataFrame([rd_overview]).to_csv(rd_overview_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Champion ATS performance drift for 2025-2026 season"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="predictions/current_season_champion_2025_2026_predictions.csv",
        help="Path to Champion predictions CSV for 2025-2026 season",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/champion_current_2526",
        help="Directory to write per-game and summary CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df_per_game = build_per_game_dataset(predictions_path)

    if df_per_game.empty:
        print("No completed games with actuals found in predictions file.")
        return

    time_summaries = summarize_time_buckets(df_per_game)
    context_summary = summarize_context_segments(df_per_game)
    rd_pick_blocks, rd_team_breakdown, rd_overview = summarize_road_underdogs(df_per_game)

    save_outputs(
        df_per_game,
        time_summaries,
        context_summary,
        rd_pick_blocks,
        rd_team_breakdown,
        rd_overview,
        output_dir,
    )

    # Brief console summary for quick inspection
    overall = context_summary[context_summary["segment"] == "overall"].iloc[0]
    print("=" * 70)
    print("Champion 2025-2026 ATS Drift Analysis - Summary")
    print("=" * 70)
    print(f"Games evaluated: {int(overall['n_games'])}")
    print(f"Overall ATS accuracy: {overall['ats_accuracy']:.2f}%")
    print(f"Overall ROI per bet: {overall['roi_pct']:.2f}%")

    # Road underdog overview
    print()
    print("Road underdog impact:")
    print(
        f"  Road-dog picks: {rd_overview['road_dogs_ats']:.2f}% ATS "
        f"(n={int(rd_overview['n_road_dogs'])})"
    )
    print(
        f"  Excluding road dogs: {rd_overview['overall_excl_road_dogs_ats']:.2f}% ATS "
        f"(n={int(rd_overview['n_total'] - rd_overview['n_road_dogs'])})"
    )
    print()
    print(f"Per-game dataset: {output_dir / 'champion_2526_per_game_ats.csv'}")
    print(f"Time summaries:")
    for name in time_summaries.keys():
        print(f"  - {name}: {output_dir / f'champion_2526_ats_{name}.csv'}")
    print(f"Context segments: {output_dir / 'champion_2526_ats_context_segments.csv'}")
    print(f"Road dogs (by pick order): {output_dir / 'champion_2526_ats_road_underdogs_by_pick_order.csv'}")
    print(f"Road dogs (by team): {output_dir / 'champion_2526_ats_road_underdogs_by_team.csv'}")


if __name__ == "__main__":
    main()


