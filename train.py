"""Training entry point for the direct prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from features.availability import load_baselines_from_csv

from calibration.calibrator import Calibrator, CalibrationConfig
from models.bivariate_poisson import BivariatePoissonConfig
from data.loaders import GameDataLoader
from training.dataset import TrainingDataset
from training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bivariate Poisson model")
    parser.add_argument("--data", type=str, required=True, help="Path to game jsonl")
    parser.add_argument("--output", type=str, required=True, help="Directory to save model")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config")
    parser.add_argument("--calibrate-only", action="store_true", help="Fit calibrators only using an existing model")
    parser.add_argument("--model", type=str, default=None, help="Existing model directory (for --calibrate-only)")
    parser.add_argument("--fast-calibrate", action="store_true", help="Use fast margin-only path for calibrators (no full 2D joint)")
    parser.add_argument("--gh-samples", type=int, default=11, help="Gauss-Hermite nodes for --fast-calibrate")
    return parser.parse_args()


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)

    dataset = TrainingDataset(collection.games)
    baselines_path = cfg.get("data", {}).get("availability_baselines")
    if baselines_path and Path(baselines_path).exists():
        load_baselines_from_csv(baselines_path)
    if args.calibrate_only:
        # Calibrate-only path: load existing model, build calibration payload, fit calibrators
        from models.distribution import bivariate_poisson_pmf, margin_pmf

        model_dir = Path(args.model or args.output)
        model = joblib.load(model_dir / "model.joblib")

        batch = dataset.build()
        trainer_cfg = TrainerConfig(**cfg.get("training", {}))
        margin_range = (trainer_cfg.margin_low, trainer_cfg.margin_high)

        # Predict rates on entire dataset
        lambda_home, lambda_away, kappa, sigma_home, sigma_away = model.predict_rates(
            batch.x_home, batch.x_away, batch.x_shared, batch.baseline_home, batch.baseline_away
        )
        if args.fast_calibrate:
            # Fast path: 1D mixtures and Skellam-lognormal margins; joint as outer product of marginals
            import numpy as _np
            from scipy.special import gammaln as _gammaln, iv as _besseli

            size = int(trainer_cfg.max_score) + 1
            gh_x, gh_w = _np.polynomial.hermite.hermgauss(int(args.gh_samples))
            gh_x = gh_x.astype(_np.float64)
            gh_w = gh_w.astype(_np.float64)

            def _poisson_log_pmf(n: _np.ndarray, rate: float) -> _np.ndarray:
                return n * _np.log(rate) - rate - _gammaln(n + 1)

            n = _np.arange(size)
            m_vals = _np.arange(margin_range[0], margin_range[1] + 1)
            joints = []
            margin_pmfs = []
            for i in range(len(lambda_home)):
                lam_h = float(lambda_home[i])
                lam_a = float(lambda_away[i])
                kap = float(kappa[i])
                sh = float(sigma_home[i]) if sigma_home is not None else 1e-6
                sa = float(sigma_away[i]) if sigma_away is not None else 1e-6
                sh = max(sh, 1e-6)
                sa = max(sa, 1e-6)

                # 1D Poisson-lognormal mixtures for team scores
                home = _np.zeros(size, dtype=_np.float64)
                away = _np.zeros(size, dtype=_np.float64)
                for j in range(len(gh_x)):
                    adj_lh = lam_h * _np.exp(_np.sqrt(2) * sh * gh_x[j] - sh ** 2)
                    adj_la = lam_a * _np.exp(_np.sqrt(2) * sa * gh_x[j] - sa ** 2)
                    rate_h = kap + max(adj_lh, 1e-9)
                    rate_a = kap + max(adj_la, 1e-9)
                    home += gh_w[j] * _np.exp(_poisson_log_pmf(n, rate_h))
                    away += gh_w[j] * _np.exp(_poisson_log_pmf(n, rate_a))
                home = _np.clip(home / _np.sqrt(_np.pi), 0.0, None)
                away = _np.clip(away / _np.sqrt(_np.pi), 0.0, None)
                hs = home.sum(); as_ = away.sum()
                if hs > 0: home /= hs
                if as_ > 0: away /= as_
                joints.append(_np.outer(home, away))

                # Margin Skellam-lognormal mixture
                marg = _np.zeros_like(m_vals, dtype=_np.float64)
                for j in range(len(gh_x)):
                    for k2 in range(len(gh_x)):
                        adj_lh = lam_h * _np.exp(_np.sqrt(2) * sh * gh_x[j] - sh ** 2)
                        adj_la = lam_a * _np.exp(_np.sqrt(2) * sa * gh_x[k2] - sa ** 2)
                        mu1 = max(adj_lh, 1e-9)
                        mu2 = max(adj_la, 1e-9)
                        t = 2.0 * _np.sqrt(mu1 * mu2)
                        bvals = _besseli(_np.abs(m_vals), t)
                        marg += gh_w[j] * gh_w[k2] * (_np.exp(-(mu1 + mu2)) * _np.power(mu1 / mu2, 0.5 * m_vals) * bvals)
                marg = _np.clip(marg / _np.pi, 0.0, None)
                ssum = marg.sum()
                if ssum > 0:
                    marg /= ssum
                margin_pmfs.append(marg)

            joint_pmfs = _np.stack(joints)
            margin_cdfs = _np.cumsum(_np.stack(margin_pmfs), axis=1)
        else:
            # Joint PMFs and margin CDFs
            joint_pmfs = np.stack([
                bivariate_poisson_pmf(h, a, k, max_points=trainer_cfg.max_score)
                for h, a, k in zip(lambda_home, lambda_away, kappa)
            ])
            margin_cdfs = np.stack([
                np.cumsum(margin_pmf(joint, margin_range))
                for joint in joint_pmfs
            ])
        margin_outcomes = np.exp(batch.y_home) * batch.baseline_home - np.exp(batch.y_away) * batch.baseline_away
        score_outcomes_home = np.exp(batch.y_home) * batch.baseline_home
        score_outcomes_away = np.exp(batch.y_away) * batch.baseline_away

        calibration_cfg = CalibrationConfig(**cfg.get("calibration", {}))
        calibrator = Calibrator(calibration_cfg)
        calibrator.fit(
            margin_cdf=margin_cdfs,
            joint_probs=joint_pmfs,
            margin_outcomes=margin_outcomes,
            score_outcomes_home=score_outcomes_home,
            score_outcomes_away=score_outcomes_away,
            market_home=batch.baseline_home,
            market_away=batch.baseline_away,
            sigma_home=sigma_home,
            sigma_away=sigma_away,
        )

        output_dir = Path(args.output or args.model)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save only calibrator and config
        joblib.dump(calibrator, output_dir / "calibrator.joblib")
        (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    else:
        trainer_cfg = TrainerConfig(**cfg.get("training", {}))
        model_cfg = BivariatePoissonConfig(**cfg.get("model", {}))
        trainer = Trainer(dataset, config=trainer_cfg, model_config=model_cfg)
        model, validation_payload = trainer.fit()

        calibration_cfg = CalibrationConfig(**cfg.get("calibration", {}))
        calibrator = Calibrator(calibration_cfg)
        calibrator.fit(
            margin_cdf=validation_payload["margin_cdf"],
            joint_probs=validation_payload["joint_pmfs"],
            margin_outcomes=validation_payload["margin_outcomes"],
            score_outcomes_home=validation_payload["score_outcomes_home"],
            score_outcomes_away=validation_payload["score_outcomes_away"],
            market_home=validation_payload["market_home"],
            market_away=validation_payload["market_away"],
            sigma_home=validation_payload.get("sigma_home"),
            sigma_away=validation_payload.get("sigma_away"),
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_dir / "model.joblib")
        joblib.dump(calibrator, output_dir / "calibrator.joblib")
        (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg))


if __name__ == "__main__":
    main()

