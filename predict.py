"""Inference script for direct prediction pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import yaml

from calibration.calibrator import Calibrator
from data.loaders import GameDataLoader
from features.builder import (
    FeatureBuilder,
    HOME_FEATURE_KEYS,
    AWAY_FEATURE_KEYS,
    SHARED_FEATURE_KEYS,
)
from features.availability import load_baselines_from_csv
from models.distribution import (
    bivariate_poisson_log_normal_pmf,
    bivariate_poisson_pmf,
    compute_win_probabilities,
    expected_scores,
    expected_margin,
    margin_pmf,
)


@dataclass
class PredictionResult:
    game_id: str
    lambda_home: float
    lambda_away: float
    kappa: float
    win_prob_home: float
    win_prob_away: float
    expected_home: float
    expected_away: float
    expected_margin: float
    margin_pmf: list[float]
    sigma_home: float | None = None
    sigma_away: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict game outcomes")
    parser.add_argument("--data", type=str, required=True, help="Path to game jsonl")
    parser.add_argument("--model", type=str, required=True, help="Model directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--max-score", type=int, default=120)
    parser.add_argument("--margin-low", type=int, default=-50)
    parser.add_argument("--margin-high", type=int, default=50)
    parser.add_argument("--gh-samples", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    model_dir = Path(args.model)
    output_path = Path(args.output)

    loader = GameDataLoader(data_path.parent)
    builder = FeatureBuilder()
    model = joblib.load(model_dir / "model.joblib")
    calibrator: Calibrator = joblib.load(model_dir / "calibrator.joblib")

    config = {}
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
    elif Path("configs/default.yaml").exists():
        config = yaml.safe_load(Path("configs/default.yaml").read_text())
    baselines_path = config.get("data", {}).get("availability_baselines")
    if baselines_path and Path(baselines_path).exists():
        load_baselines_from_csv(baselines_path)

    games = list(loader.iter_games(data_path.name))
    results: list[PredictionResult] = []

    raw_joint = []
    margin_cdfs = []
    market_home = []
    market_away = []
    lambda_home_list = []
    lambda_away_list = []
    kappa_list = []
    sigma_home_list: list[float] | None = [] if calibrator.sigma_home is not None else None
    sigma_away_list: list[float] | None = [] if calibrator.sigma_away is not None else None

    for idx, game in enumerate(games):
        features = builder.build(game)
        x_home = np.array([[features.x_home[k] for k in HOME_FEATURE_KEYS]])
        x_away = np.array([[features.x_away[k] for k in AWAY_FEATURE_KEYS]])
        x_shared = np.array([[features.x_shared[k] for k in SHARED_FEATURE_KEYS]])
        baseline_home = np.array([features.baseline_home])
        baseline_away = np.array([features.baseline_away])

        lambda_home, lambda_away, kappa, sigma_home, sigma_away = model.predict_rates(
            x_home, x_away, x_shared, baseline_home, baseline_away
        )
        lambda_home_list.append(lambda_home[0])
        lambda_away_list.append(lambda_away[0])
        kappa_list.append(kappa[0])
        if sigma_home_list is not None and sigma_home is not None:
            sigma_home_list.append(float(sigma_home[0]))
        if sigma_away_list is not None and sigma_away is not None:
            sigma_away_list.append(float(sigma_away[0]))

        joint = (
            bivariate_poisson_log_normal_pmf(
                lambda_home=lambda_home[0],
                lambda_away=lambda_away[0],
                kappa=kappa[0],
                sigma_home=sigma_home[0],
                sigma_away=sigma_away[0],
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
        raw_joint.append(joint)
        margin = margin_pmf(joint, (args.margin_low, args.margin_high))
        margin_cdfs.append(np.cumsum(margin))
        market_home.append(features.baseline_home)
        market_away.append(features.baseline_away)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(games)} games")

    raw_joint_array = np.stack(raw_joint)
    margin_cdf_array = np.stack(margin_cdfs)

    calibrated_joint = calibrator.calibrate_joint_probs(raw_joint_array)
    calibrated_margin_cdf = calibrator.calibrate_margin_cdf(margin_cdf_array)

    for idx, game in enumerate(games):
        joint = calibrated_joint[idx]
        win_home, win_away, _ = compute_win_probabilities(joint)
        # Apply scalar win-probability calibration if available
        win_home = float(calibrator.calibrate_win_prob(np.array([win_home]))[0])
        win_away = 1.0 - win_home
        exp_home, exp_away = expected_scores(joint)
        blended_home, blended_away = calibrator.blend_expectations(
            np.array([exp_home]),
            np.array([exp_away]),
            np.array([market_home[idx]]),
            np.array([market_away[idx]]),
        )
        margin_pmf_vals = np.diff(np.concatenate([[0.0], calibrated_margin_cdf[idx]]))
        results.append(
            PredictionResult(
                game_id=game.game_id,
                lambda_home=float(lambda_home_list[idx]),
                lambda_away=float(lambda_away_list[idx]),
                kappa=float(kappa_list[idx]),
                win_prob_home=float(win_home),
                win_prob_away=float(win_away),
                expected_home=float(blended_home[0]),
                expected_away=float(blended_away[0]),
                expected_margin=expected_margin(margin_pmf_vals, (args.margin_low, args.margin_high)),
                margin_pmf=margin_pmf_vals.tolist(),
                sigma_home=sigma_home_list[idx] if sigma_home_list else None,
                sigma_away=sigma_away_list[idx] if sigma_away_list else None,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(res) for res in results], f, indent=2)


if __name__ == "__main__":
    main()

