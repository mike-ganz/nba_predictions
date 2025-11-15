"""
Feature drift analysis for the 2025-2026 current season vs training data.

This script compares:
- Early vs late 2025-2026 games (feature distributions & missingness)
- Current season vs training distribution for key champion features
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so we can import `data.*` when run as a script
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import load_game_collection


KEY_FEATURES = [
    # Offense / defense (normalized)
    "home_off_rating_norm",
    "away_off_rating_norm",
    "home_def_rating_norm",
    "away_def_rating_norm",
    # Pace (normalized)
    "home_pace_norm",
    "away_pace_norm",
    # 3P rate, rebounding, assists, turnovers (normalized)
    "home_three_pt_rate_norm",
    "away_three_pt_rate_norm",
    "home_def_reb_rate_norm",
    "away_def_reb_rate_norm",
    "home_assist_rate_norm",
    "away_assist_rate_norm",
    "home_turnover_rate_norm",
    "away_turnover_rate_norm",
    # Rest and schedule
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_three_in_four",
    "away_three_in_four",
]


def _records_to_feature_frame(records) -> pd.DataFrame:
    """Flatten GameRecord objects into a feature dataframe."""
    rows: List[Dict[str, object]] = []
    for rec in records:
        home = rec.teams.H
        away = rec.teams.A
        row: Dict[str, object] = {
            "game_id": rec.game_id,
            "season": rec.season,
            "date": rec.date,
            "home_team": home.team_id,
            "away_team": away.team_id,
            # Normalized team metrics
            "home_off_rating_norm": home.off_rating_norm,
            "away_off_rating_norm": away.off_rating_norm,
            "home_def_rating_norm": home.def_rating_norm,
            "away_def_rating_norm": away.def_rating_norm,
            "home_pace_norm": home.pace_norm,
            "away_pace_norm": away.pace_norm,
            "home_three_pt_rate_norm": home.three_pt_rate_norm,
            "away_three_pt_rate_norm": away.three_pt_rate_norm,
            "home_def_reb_rate_norm": home.def_reb_rate_norm,
            "away_def_reb_rate_norm": away.def_reb_rate_norm,
            "home_assist_rate_norm": home.assist_rate_norm,
            "away_assist_rate_norm": away.assist_rate_norm,
            "home_turnover_rate_norm": home.turnover_rate_norm,
            "away_turnover_rate_norm": away.turnover_rate_norm,
            # Rest and schedule context
            "home_rest_days": home.rest_days,
            "away_rest_days": away.rest_days,
            "home_b2b": home.b2b,
            "away_b2b": away.b2b,
            "home_three_in_four": home.three_in_four,
            "away_three_in_four": away.three_in_four,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "game_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def summarize_features(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """Compute summary stats and missingness for key features."""
    summary = df[KEY_FEATURES].agg(["mean", "std", "min", "max"])
    missing = df[KEY_FEATURES].isna().mean().rename("missing_frac")
    summary = summary.transpose()
    summary["missing_frac"] = missing
    summary.insert(0, "feature", summary.index)
    summary.insert(1, "group", group_label)
    return summary.reset_index(drop=True)


def summarize_rest_distribution(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Simple rest-days histograms for home/away for quick inspection."""
    rows: List[Dict[str, object]] = []
    for side in ["home", "away"]:
        col = f"{side}_rest_days"
        if col not in df.columns:
            continue
        series = df[col].dropna()
        # Bucket into 0,1,2,3+ days
        buckets = {
            "0": (series == 0).mean(),
            "1": (series == 1).mean(),
            "2": (series == 2).mean(),
            "3_plus": (series >= 3).mean(),
        }
        rows.append(
            {
                "group": label,
                "side": side,
                **{f"rest_{k}_frac": v for k, v in buckets.items()},
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature drift for 2025-2026 current season vs training data"
    )
    parser.add_argument(
        "--current-games",
        type=str,
        default="data/games_2025_2026_current_norm.jsonl",
        help="Current season games JSONL (normalized)",
    )
    parser.add_argument(
        "--training-games",
        type=str,
        default="data/games_train_2021_2025_combined_norm.jsonl",
        help="Training games JSONL (normalized)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/champion_feature_drift_2526",
        help="Directory to write feature drift summaries",
    )
    parser.add_argument(
        "--early-cutoff",
        type=int,
        default=100,
        help="Number of earliest games to treat as 'early-season'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load current-season games
    current_collection = load_game_collection(args.current_games)
    current_df = _records_to_feature_frame(current_collection.games)
    current_df["game_index"] = np.arange(1, len(current_df) + 1)

    # Split into early vs late season based on game index
    early = current_df[current_df["game_index"] <= args.early_cutoff]
    late = current_df[current_df["game_index"] > args.early_cutoff]

    # Load training data
    training_collection = load_game_collection(args.training_games)
    training_df = _records_to_feature_frame(training_collection.games)

    # Summaries for early, late, and full current season
    summaries = []
    summaries.append(summarize_features(early, "current_early"))
    summaries.append(summarize_features(late, "current_late"))
    summaries.append(summarize_features(current_df, "current_all"))

    # Training summary (for comparison)
    summaries.append(summarize_features(training_df, "training_all"))

    feature_summary = pd.concat(summaries, ignore_index=True)
    feature_summary.to_csv(out_dir / "feature_summary_early_late_vs_training.csv", index=False)

    # Rest distribution snapshots
    rest_rows = []
    rest_rows.append(summarize_rest_distribution(early, "current_early"))
    rest_rows.append(summarize_rest_distribution(late, "current_late"))
    rest_rows.append(summarize_rest_distribution(current_df, "current_all"))
    rest_rows.append(summarize_rest_distribution(training_df, "training_all"))
    rest_summary = pd.concat(rest_rows, ignore_index=True)
    rest_summary.to_csv(out_dir / "rest_distribution_early_late_vs_training.csv", index=False)

    # Quick console summary focused on rest-days and a couple of core features
    print("=" * 70)
    print("Champion Feature Drift - 2025-2026 vs Training")
    print("=" * 70)
    print(f"Current-season games: {len(current_df)} (early={len(early)}, late={len(late)})")
    print(f"Training games:       {len(training_df)}")
    print()

    for feat in ["home_off_rating_norm", "home_def_rating_norm", "home_pace_norm"]:
        fs = feature_summary[feature_summary["feature"] == feat]
        print(f"{feat}:")
        for grp in ["training_all", "current_all", "current_early", "current_late"]:
            row = fs[fs["group"] == grp]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"  {grp:14s} mean={r['mean']:+6.3f} std={r['std']:+6.3f} "
                f"min={r['min']:+6.3f} max={r['max']:+6.3f}"
            )
        print()

    print("Rest-days distribution (fraction of games by rest bucket):")
    print(rest_summary.to_string(index=False))


if __name__ == "__main__":
    main()


