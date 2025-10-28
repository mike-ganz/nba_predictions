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
from scipy.special import gammaln


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--data", type=str, required=True, help="Path to game jsonl with outcomes")
    parser.add_argument("--model", type=str, required=True, help="Model directory")
    parser.add_argument("--max-score", type=int, default=120)
    parser.add_argument("--margin-low", type=int, default=-50)
    parser.add_argument("--margin-high", type=int, default=50)
    parser.add_argument("--gh-samples", type=int, default=25)
    parser.add_argument("--fast-eval", action="store_true", help="Use fast 1D mixtures (no full joint)")
    parser.add_argument("--sharpen", type=float, default=1.0, help="Extra temperature (<1 sharpens, >1 flattens) for 1D marginals")
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
    home_pmfs_1d = []
    away_pmfs_1d = []
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

        if args.fast_eval and (sigma_home is not None and sigma_away is not None):
            # Fast path: compute 1D Poisson-lognormal mixtures for team scores and
            # Skellam-lognormal mixture for margin. Avoids building full 2D joint.
            import numpy as _np
            from scipy.special import iv as _besseli

            size = args.max_score + 1
            gh_x, gh_w = _np.polynomial.hermite.hermgauss(args.gh_samples)
            gh_x = gh_x.astype(_np.float64)
            gh_w = gh_w.astype(_np.float64)

            lam_h = float(lambda_home[0])
            lam_a = float(lambda_away[0])
            kap = float(kappa[0])
            s_h = max(float(sigma_home[0]), 1e-6)
            s_a = max(float(sigma_away[0]), 1e-6)

            # 1D Poisson-lognormal mixture for team score marginals
            n = _np.arange(size)
            home_pmf = _np.zeros(size, dtype=_np.float64)
            away_pmf = _np.zeros(size, dtype=_np.float64)
            for i in range(len(gh_x)):
                # Adjusted lambdas preserve mean via -sigma^2 term
                adj_lh = lam_h * _np.exp(_np.sqrt(2) * s_h * gh_x[i] - s_h ** 2)
                adj_la = lam_a * _np.exp(_np.sqrt(2) * s_a * gh_x[i] - s_a ** 2)
                rate_h = kap + max(adj_lh, 1e-9)
                rate_a = kap + max(adj_la, 1e-9)
                # Vectorized Poisson PMF
                log_h = n * _np.log(rate_h) - rate_h - gammaln(n + 1)
                log_a = n * _np.log(rate_a) - rate_a - gammaln(n + 1)
                home_pmf += gh_w[i] * _np.exp(log_h)
                away_pmf += gh_w[i] * _np.exp(log_a)
            # Normalize GH (1D) and PMFs
            home_pmf = _np.clip(home_pmf / _np.sqrt(_np.pi), 0.0, None)
            away_pmf = _np.clip(away_pmf / _np.sqrt(_np.pi), 0.0, None)
            home_sum = home_pmf.sum()
            away_sum = away_pmf.sum()
            if home_sum > 0:
                home_pmf /= home_sum
            if away_sum > 0:
                away_pmf /= away_sum
            home_pmfs_1d.append(home_pmf)
            away_pmfs_1d.append(away_pmf)

            # Expected scores without full joint (means preserved under our parametrization)
            home_scores.append(lam_h + kap)
            away_scores.append(lam_a + kap)

            # Skellam-lognormal mixture for margin distribution (shared component cancels)
            m_low, m_high = args.margin_low, args.margin_high
            m_vals = _np.arange(m_low, m_high + 1)
            margin_p = _np.zeros_like(m_vals, dtype=_np.float64)
            for i in range(len(gh_x)):
                for j in range(len(gh_x)):
                    adj_lh = lam_h * _np.exp(_np.sqrt(2) * s_h * gh_x[i] - s_h ** 2)
                    adj_la = lam_a * _np.exp(_np.sqrt(2) * s_a * gh_x[j] - s_a ** 2)
                    mu1 = max(adj_lh, 1e-9)
                    mu2 = max(adj_la, 1e-9)
                    # Skellam PMF vectorized across margins
                    # P(D=d) = exp(-(mu1+mu2)) * (mu1/mu2)^{d/2} * I_{|d|}(2*sqrt(mu1*mu2))
                    t = 2.0 * _np.sqrt(mu1 * mu2)
                    # Handle vector of orders via iv for each |d|
                    orders = _np.abs(m_vals)
                    bessel_vals = _besseli(orders, t)
                    pow_term = _np.power(mu1 / mu2, 0.5 * m_vals)
                    pmf_vec = _np.exp(-(mu1 + mu2)) * pow_term * bessel_vals
                    margin_p += gh_w[i] * gh_w[j] * pmf_vec
            # Normalize GH (2D) and PMF
            margin_p = _np.clip(margin_p / _np.pi, 0.0, None)
            s = margin_p.sum()
            if s > 0:
                margin_p /= s
            margin_pmfs.append(margin_p)
        else:
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

    margin_stack = np.stack(margin_pmfs)
    outcomes_home = np.array([g.outcome.home_final for g in games if g.outcome])
    outcomes_away = np.array([g.outcome.away_final for g in games if g.outcome])
    margin_range = (args.margin_low, args.margin_high)

    margin_cdfs = np.cumsum(margin_stack, axis=1)
    # Use per-bin temperatures by passing spreads (baseline_home - baseline_away)
    spreads_all = np.array(market_home) - np.array(market_away)
    calibrated_margin_cdfs = calibrator.calibrate_margin_cdf(margin_cdfs, spreads=spreads_all)
    calibrated_margin_pmfs = np.diff(
        np.concatenate([np.zeros((len(calibrated_margin_cdfs), 1)), calibrated_margin_cdfs], axis=1),
        axis=1,
    )

    if args.fast_eval and home_pmfs_1d and away_pmfs_1d and calibrator.temperature is not None:
        # Calibrate 1D marginals with temperature scaler
        home_stack = np.stack(home_pmfs_1d)
        away_stack = np.stack(away_pmfs_1d)
        stacked_pmfs = np.stack([home_stack, away_stack], axis=1)
        transformed = calibrator.temperature.transform(stacked_pmfs)
        home_pmfs = transformed[:, 0, :]
        away_pmfs = transformed[:, 1, :]
    elif args.fast_eval and home_pmfs_1d and away_pmfs_1d:
        home_pmfs = np.stack(home_pmfs_1d)
        away_pmfs = np.stack(away_pmfs_1d)
    else:
        joint_stack = np.stack(joint_pmfs)
        home_pmfs = joint_stack.sum(axis=2)
        away_pmfs = joint_stack.sum(axis=1)

    # Optional extra sharpening/flattening at evaluation time
    if abs(args.sharpen - 1.0) > 1e-9:
        t = max(args.sharpen, 1e-6)
        home_pmfs = np.power(np.clip(home_pmfs, 1e-12, 1.0), 1.0 / t)
        home_pmfs /= np.clip(home_pmfs.sum(axis=1, keepdims=True), 1e-12, None)
        away_pmfs = np.power(np.clip(away_pmfs, 1e-12, 1.0), 1.0 / t)
        away_pmfs /= np.clip(away_pmfs.sum(axis=1, keepdims=True), 1e-12, None)

    blended_home, blended_away = calibrator.blend_expectations(
        np.array(home_scores),
        np.array(away_scores),
        np.array(market_home),
        np.array(market_away),
    )

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

    if args.fast_eval:
        # Win prob from margin distribution
        m_idx0 = -margin_range[0]
        home_win_probs = np.array([
            pmf[m_idx0 + 1 :].sum() + 0.5 * pmf[m_idx0]
            for pmf in calibrated_margin_pmfs
        ])
    else:
        calibrated_joint = calibrator.calibrate_joint_probs(joint_stack)
        home_win_probs = np.array([compute_win_probabilities(joint)[0] for joint in calibrated_joint])

    # Apply win-probability scalar calibration if available
    home_win_probs = calibrator.calibrate_win_prob(home_win_probs)
    home_win_actual = np.array([1 if m > 0 else 0.5 if m == 0 else 0 for m in margin_outcomes])
    win_brier = brier_score(home_win_probs, home_win_actual)

    print("Evaluation Summary")
    print("==================")
    num_games = len(margin_pmfs)
    print(f"Games evaluated: {num_games}")
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
    reliability_summary = reliability_df.groupby("pred_bin", observed=False).agg(
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

    # Per-game table: predictions vs actuals
    games_with_outcomes = [g for g in games if g.outcome]
    spreads_home = np.array([g.market.spread_home for g in games_with_outcomes], dtype=float)
    m_low, m_high = margin_range
    m_vals = np.arange(m_low, m_high + 1)
    # ATS probabilities from calibrated margin PMFs
    cover_prob_home = np.zeros(len(calibrated_margin_pmfs), dtype=float)
    push_prob = np.zeros(len(calibrated_margin_pmfs), dtype=float)
    for i, pmf in enumerate(calibrated_margin_pmfs):
        s = spreads_home[i]
        cover_prob_home[i] = pmf[m_vals > -s].sum()
        # Push only when spread is effectively integer
        if abs(s - round(s)) < 1e-9:
            idx_eq = int(round(-s)) - m_low
            if 0 <= idx_eq < len(pmf):
                push_prob[i] = pmf[idx_eq]
    # Apply ATS scalar calibration by |spread| bucket if available (on home side)
    cover_prob_home = calibrator.calibrate_cover_prob(cover_prob_home, spreads_home)
    cover_prob_away = np.clip(1.0 - push_prob - cover_prob_home, 0.0, 1.0)
    payout = 100.0 / 110.0
    # Break-even given pushes (conditional on action)
    be = 110.0 / 210.0
    adj_cover_home = np.divide(cover_prob_home, np.maximum(1.0 - push_prob, 1e-12))
    edge_home = adj_cover_home - be
    ev_home = cover_prob_home * payout - (1.0 - push_prob - cover_prob_home)
    per_game = pd.DataFrame({
        "game_id": [g.game_id for g in games_with_outcomes],
        "date": [str(g.date) for g in games_with_outcomes],
        "season": [g.season for g in games_with_outcomes],
        "away_team": [g.teams.A.team_name or g.teams.A.team_id for g in games_with_outcomes],
        "home_team": [g.teams.H.team_name or g.teams.H.team_id for g in games_with_outcomes],
        "pred_home": blended_home,
        "pred_away": blended_away,
        "actual_home": [g.outcome.home_final for g in games_with_outcomes],
        "actual_away": [g.outcome.away_final for g in games_with_outcomes],
        "win_prob_home": home_win_probs,
        "market_spread_home": spreads_home,
        "market_total": [g.market.total for g in games_with_outcomes],
        "cover_prob_home": cover_prob_home,
        "cover_prob_away": cover_prob_away,
        "push_prob": push_prob,
        "ev_home_-110": ev_home,
        "edge_home_vs_-110": edge_home,
    })
    per_game_path = reports_path / "per_game_predictions.csv"
    per_game.to_csv(per_game_path, index=False)
    # Season-specific export (2024-2025)
    per_game_2425 = per_game[per_game["season"] == "2024-2025"]
    if not per_game_2425.empty:
        per_game_2425.to_csv(reports_path / "per_game_predictions_2024_2025.csv", index=False)


if __name__ == "__main__":
    main()

