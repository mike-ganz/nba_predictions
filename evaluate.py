"""Evaluation script for direct prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import joblib
import numpy as np
import pandas as pd
import yaml

from calibration.calibrator import Calibrator
from data.loaders import GameDataLoader
from evaluation.metrics import (
    brier_score,
    bucket_metrics,
    compute_crps,
    compute_log_loss,
    mean_absolute_error_margin,
    pit_histogram,
)
from evaluation.reporting import (
    save_bucket_bar,
    save_pit_histogram,
    save_reliability_curve,
    save_summary_table,
)
from features.builder import (
    FeatureBuilder,
    HOME_FEATURE_KEYS,
    AWAY_FEATURE_KEYS,
    SHARED_FEATURE_KEYS,
)
from features.availability import load_baselines_from_csv
from models.distribution import (
    bivariate_poisson_pmf,
    compute_win_probabilities,
    expected_scores,
    expected_margin,
    margin_pmf,
    bivariate_poisson_log_normal_pmf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--data", type=str, required=True, help="Path to game jsonl with outcomes")
    parser.add_argument("--model", type=str, required=True, help="Model directory")
    parser.add_argument("--max-score", type=int, default=120)
    parser.add_argument("--margin-low", type=int, default=-50)
    parser.add_argument("--margin-high", type=int, default=50)
    parser.add_argument("--gh-samples", type=int, default=25)
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save evaluation reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    model_dir = Path(args.model)
    reports_path = Path(args.reports_dir)

    loader = GameDataLoader(data_path.parent)
    builder = FeatureBuilder()
    model = joblib.load(model_dir / "model.joblib")
    calibrator: Calibrator = joblib.load(model_dir / "calibrator.joblib")

    config_path = model_dir / "config.yaml"
    baselines_path = None
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        baselines_path = config.get("data", {}).get("availability_baselines")
        if baselines_path and Path(baselines_path).exists():
            load_baselines_from_csv(baselines_path)

    games = list(loader.iter_games(data_path.name))

    joint_pmfs = []
    margin_pmfs = []
    home_scores = []
    away_scores = []
    margin_outcomes = []
    market_home = []
    market_away = []

    for idx, game in enumerate(games):
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx}/{len(games)} games for evaluation")
        if not game.outcome:
            continue
        features = builder.build(game)
        x_home = np.array([[features.x_home[k] for k in HOME_FEATURE_KEYS]])
        x_away = np.array([[features.x_away[k] for k in AWAY_FEATURE_KEYS]])
        x_shared = np.array([[features.x_shared[k] for k in SHARED_FEATURE_KEYS]])
        baseline_home = np.array([features.baseline_home])
        baseline_away = np.array([features.baseline_away])

        lambda_home, lambda_away, kappa, sigma_home, sigma_away = model.predict_rates(
            x_home, x_away, x_shared, baseline_home, baseline_away
        )

        joint = (
            bivariate_poisson_log_normal_pmf(
                lambda_home=lambda_home[0],
                lambda_away=lambda_away[0],
                kappa=kappa[0],
                sigma_home=sigma_home[0] if sigma_home is not None else 0.0,
                sigma_away=sigma_away[0] if sigma_away is not None else 0.0,
                max_points=args.max_score,
                n_samples=args.gh_samples,
            )
            if sigma_home is not None and sigma_away is not None
            else bivariate_poisson_pmf(
                lambda_home=lambda_home[0],
                lambda_away=lambda_away[0],
                kappa=kappa[0],
                max_points=args.max_score,
            )
        )
        joint_pmfs.append(joint)
        margin_pmfs.append(margin_pmf(joint, (args.margin_low, args.margin_high)))
        exp_home, exp_away = expected_scores(joint)
        home_scores.append(exp_home)
        away_scores.append(exp_away)
        margin_outcomes.append(game.outcome.home_final - game.outcome.away_final)
        market_home.append(features.baseline_home)
        market_away.append(features.baseline_away)

    joint_stack = np.stack(joint_pmfs)
    margin_stack = np.stack(margin_pmfs)
    outcomes_home = np.array([g.outcome.home_final for g in games if g.outcome])
    outcomes_away = np.array([g.outcome.away_final for g in games if g.outcome])
    margin_range = (args.margin_low, args.margin_high)

    calibrated_joint = calibrator.calibrate_joint_probs(joint_stack)
    margin_cdfs = np.cumsum(margin_stack, axis=1)
    calibrated_margin_cdfs = calibrator.calibrate_margin_cdf(margin_cdfs)
    calibrated_margin_pmfs = np.diff(
        np.concatenate([np.zeros((len(calibrated_margin_cdfs), 1)), calibrated_margin_cdfs], axis=1),
        axis=1,
    )

    blended_home, blended_away = calibrator.blend_expectations(
        np.array(home_scores),
        np.array(away_scores),
        np.array(market_home),
        np.array(market_away),
    )

    home_pmfs = calibrated_joint.sum(axis=2)
    away_pmfs = calibrated_joint.sum(axis=1)
    home_cdfs = np.cumsum(home_pmfs, axis=1)
    away_cdfs = np.cumsum(away_pmfs, axis=1)

    log_loss_home = compute_log_loss(home_pmfs, outcomes_home)
    log_loss_away = compute_log_loss(away_pmfs, outcomes_away)
    crps_home = compute_crps(home_cdfs, outcomes_home)
    crps_away = compute_crps(away_cdfs, outcomes_away)
    pit_home = pit_histogram(home_cdfs, outcomes_home)
    pit_away = pit_histogram(away_cdfs, outcomes_away)

    mae_margin = mean_absolute_error_margin(
        calibrated_margin_pmfs,
        np.array(margin_outcomes),
        margin_range,
    )

    margin_expectations = np.array(
        [expected_margin(calibrated_margin_pmfs[i], margin_range) for i in range(len(calibrated_margin_pmfs))]
    )

    home_win_probs = np.array([compute_win_probabilities(joint)[0] for joint in calibrated_joint])
    home_win_actual = np.array([1 if m > 0 else 0.5 if m == 0 else 0 for m in margin_outcomes])
    win_brier = brier_score(home_win_probs, home_win_actual)

    print("Evaluation Summary")
    print("==================")
    print(f"Games evaluated: {len(joint_pmfs)}")
    print(f"Log Loss Home: {log_loss_home:.4f}")
    print(f"Log Loss Away: {log_loss_away:.4f}")
    print(f"CRPS Home: {crps_home:.4f}")
    print(f"CRPS Away: {crps_away:.4f}")
    print(f"Mean Absolute Error (Margin): {mae_margin:.3f}")
    print(f"Average Expected Home Score (blended): {mean(blended_home):.2f}")
    print(f"Average Expected Away Score (blended): {mean(blended_away):.2f}")
    print(f"Average Expected Margin: {mean(margin_expectations):.2f}")
    print(f"Win Probability Brier Score: {win_brier:.4f}")
    print("PIT Histogram Home:", pit_home)
    print("PIT Histogram Away:", pit_away)

    pit_home_values = [
        home_cdfs[i, int(np.clip(outcomes_home[i], 0, home_cdfs.shape[1] - 1))]
        for i in range(len(outcomes_home))
    ]
    pit_away_values = [
        away_cdfs[i, int(np.clip(outcomes_away[i], 0, away_cdfs.shape[1] - 1))]
        for i in range(len(outcomes_away))
    ]

    save_pit_histogram(pit_home_values, reports_path / "pit_home.png", "Home Score PIT")
    save_pit_histogram(pit_away_values, reports_path / "pit_away.png", "Away Score PIT")

    reliability_df = pd.DataFrame({
        "pred": home_win_probs,
        "actual": home_win_actual,
    })
    reliability_df["pred_bin"] = pd.cut(reliability_df["pred"], bins=np.linspace(0, 1, 11), include_lowest=True)
    reliability_summary = reliability_df.groupby("pred_bin").agg(
        pred_rate=("pred", "mean"),
        actual_rate=("actual", "mean"),
        count=("pred", "size"),
    ).reset_index()
    reliability_summary["pred_bin_mid"] = reliability_summary["pred_bin"].apply(lambda x: x.mid)
    save_reliability_curve(reliability_summary, reports_path / "reliability.png")
    save_summary_table(reliability_summary, reports_path / "reliability.csv")

    margin_outcomes_arr = np.array(margin_outcomes)
    margin_errors = np.abs(margin_expectations - margin_outcomes_arr)
    spread_summary, total_summary = bucket_metrics(
        margin_outcomes_arr,
        np.array([g.market.spread_home for g in games if g.outcome]),
        np.array([g.market.total for g in games if g.outcome]),
        margin_errors,
    )
    save_bucket_bar(spread_summary, "spread_bin", "metric", reports_path / "spread_error.png", "MAE by Spread Bin")
    save_bucket_bar(total_summary, "total_bin", "metric", reports_path / "total_error.png", "MAE by Total Bin")
    save_summary_table(spread_summary, reports_path / "spread_error.csv")
    save_summary_table(total_summary, reports_path / "total_error.csv")

    season_summary = pd.DataFrame({
        "season": [g.season for g in games if g.outcome],
        "margin_error": margin_errors,
    }).groupby("season").mean().reset_index()
    save_bucket_bar(season_summary, "season", "margin_error", reports_path / "season_error.png", "MAE by Season")
    save_summary_table(season_summary, reports_path / "season_error.csv")


if __name__ == "__main__":
    main()

