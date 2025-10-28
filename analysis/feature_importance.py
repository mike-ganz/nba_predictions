"""Feature importance analysis for spread-focused evaluation.

Outputs:
- Coefficient-based importances per target (home/away/shared)
- Permutation importance on ATS Brier (home side) using fast-eval margin path
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.special import gammaln, iv as bessel_iv

from data.loaders import GameDataLoader
from features.builder import (
    FeatureBuilder,
    HOME_FEATURE_KEYS,
    AWAY_FEATURE_KEYS,
    SHARED_FEATURE_KEYS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute feature importances")
    p.add_argument("--data", required=True, help="Path to jsonl with outcomes")
    p.add_argument("--model", required=True, help="Model directory")
    p.add_argument("--reports-dir", default="reports/feature_importance", help="Output directory")
    p.add_argument("--gh-samples", type=int, default=9)
    p.add_argument("--max-score", type=int, default=120)
    p.add_argument("--margin-low", type=int, default=-50)
    p.add_argument("--margin-high", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _poisson_log_pmf(n: np.ndarray, rate: float) -> np.ndarray:
    return n * np.log(rate) - rate - gammaln(n + 1)


def _team_pmfs_1d(lam_h: float, lam_a: float, kap: float, sh: float, sa: float, size: int, gh_x: np.ndarray, gh_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.arange(size)
    home = np.zeros(size, dtype=np.float64)
    away = np.zeros(size, dtype=np.float64)
    sh = max(float(sh), 1e-6)
    sa = max(float(sa), 1e-6)
    for i in range(len(gh_x)):
        adj_lh = lam_h * np.exp(np.sqrt(2) * sh * gh_x[i] - sh ** 2)
        adj_la = lam_a * np.exp(np.sqrt(2) * sa * gh_x[i] - sa ** 2)
        rate_h = kap + max(adj_lh, 1e-9)
        rate_a = kap + max(adj_la, 1e-9)
        home += gh_w[i] * np.exp(_poisson_log_pmf(n, rate_h))
        away += gh_w[i] * np.exp(_poisson_log_pmf(n, rate_a))
    home = np.clip(home / np.sqrt(np.pi), 0.0, None)
    away = np.clip(away / np.sqrt(np.pi), 0.0, None)
    home /= max(home.sum(), 1e-18)
    away /= max(away.sum(), 1e-18)
    return home, away


def _margin_pmf_skellam_mix(lam_h: float, lam_a: float, sh: float, sa: float, m_vals: np.ndarray, gh_x: np.ndarray, gh_w: np.ndarray) -> np.ndarray:
    # Skellam-lognormal mixture; shared component cancels
    sh = max(float(sh), 1e-6)
    sa = max(float(sa), 1e-6)
    pmf = np.zeros_like(m_vals, dtype=np.float64)
    for i in range(len(gh_x)):
        for j in range(len(gh_x)):
            adj_lh = lam_h * np.exp(np.sqrt(2) * sh * gh_x[i] - sh ** 2)
            adj_la = lam_a * np.exp(np.sqrt(2) * sa * gh_x[j] - sa ** 2)
            mu1 = max(adj_lh, 1e-9)
            mu2 = max(adj_la, 1e-9)
            t = 2.0 * np.sqrt(mu1 * mu2)
            orders = np.abs(m_vals)
            bvals = bessel_iv(orders, t)
            pmf += gh_w[i] * gh_w[j] * (np.exp(-(mu1 + mu2)) * np.power(mu1 / mu2, 0.5 * m_vals) * bvals)
    pmf = np.clip(pmf / np.pi, 0.0, None)
    pmf_sum = pmf.sum()
    if pmf_sum > 0:
        pmf /= pmf_sum
    return pmf


def _build_matrices(games, builder: FeatureBuilder) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xh, xa, xs = [], [], []
    bh, ba = [], []
    margins, spreads = [], []
    for g in games:
        if not g.outcome:
            continue
        f = builder.build(g)
        xh.append([f.x_home[k] for k in HOME_FEATURE_KEYS])
        xa.append([f.x_away[k] for k in AWAY_FEATURE_KEYS])
        xs.append([f.x_shared[k] for k in SHARED_FEATURE_KEYS])
        bh.append(f.baseline_home)
        ba.append(f.baseline_away)
        margins.append(g.outcome.home_final - g.outcome.away_final)
        spreads.append(g.market.spread_home)
    return (
        np.asarray(xh, dtype=float),
        np.asarray(xa, dtype=float),
        np.asarray(xs, dtype=float),
        np.asarray(bh, dtype=float),
        np.asarray(ba, dtype=float),
        np.asarray(margins, dtype=float),
        np.asarray(spreads, dtype=float),
    )


def _ats_brier_from_features(model, calibrator, xh, xa, xs, bh, ba, margins, spreads, gh_x, gh_w, m_vals) -> float:
    # Predict rates
    lam_h, lam_a, kappa, sigma_h, sigma_a = model.predict_rates(xh, xa, xs, bh, ba)
    # Fast margin path
    pmfs = []
    for i in range(len(lam_h)):
        pmf = _margin_pmf_skellam_mix(float(lam_h[i]), float(lam_a[i]),
                                      float(sigma_h[i]) if sigma_h is not None else 1e-6,
                                      float(sigma_a[i]) if sigma_a is not None else 1e-6,
                                      m_vals, gh_x, gh_w)
        pmfs.append(pmf)
    pmfs = np.stack(pmfs)
    # Calibrate margin CDF (with per-bin temps if available)
    cdfs = np.cumsum(pmfs, axis=1)
    cdfs_cal = calibrator.calibrate_margin_cdf(cdfs, spreads=spreads)
    pmfs_cal = np.diff(np.concatenate([np.zeros((len(cdfs_cal), 1)), cdfs_cal], axis=1), axis=1)
    # Compute cover probabilities
    cover = np.zeros(len(pmfs_cal), dtype=float)
    m_low = m_vals[0]
    for i, pmf in enumerate(pmfs_cal):
        s = spreads[i]
        cover[i] = pmf[(m_vals > -s)].sum()
    push = np.zeros(len(pmfs_cal), dtype=float)
    int_spread = np.isclose(spreads, np.round(spreads))
    idx_eq = (np.round(-spreads[int_spread]) - m_low).astype(int)
    push[int_spread] = pmfs_cal[int_spread, idx_eq]
    # Actual outcome (home cover; push=0.5)
    actual = np.where(margins > -spreads, 1.0, np.where(np.isclose(margins, -spreads), 0.5, 0.0))
    # Calibrate cover prob by ATS calibrator
    cover_cal = calibrator.calibrate_cover_prob(cover, spreads)
    brier = float(np.mean((cover_cal - actual) ** 2))
    return brier


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    model_dir = Path(args.model)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    loader = GameDataLoader(data_path.parent)
    games = list(loader.iter_games(data_path.name))
    builder = FeatureBuilder()
    model = joblib.load(model_dir / "model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")

    # Matrices
    xh, xa, xs, bh, ba, margins, spreads = _build_matrices(games, builder)

    # Coefficient-based importances (standardized in pipeline)
    rows = []
    def _coef_df(pipe, keys: List[str], target: str) -> pd.DataFrame:
        reg = pipe.named_steps["reg"]
        coefs = reg.coef_.ravel()
        df = pd.DataFrame({"feature": keys, "coef": coefs})
        df["abs_coef"] = df["coef"].abs()
        df.sort_values("abs_coef", ascending=False, inplace=True)
        df.insert(0, "target", target)
        return df

    df_home = _coef_df(model.model_home, HOME_FEATURE_KEYS, "home_rate_resid")
    df_away = _coef_df(model.model_away, AWAY_FEATURE_KEYS, "away_rate_resid")
    df_shared = _coef_df(model.model_shared, SHARED_FEATURE_KEYS, "shared_kappa_log")
    coef_out = pd.concat([df_home, df_away, df_shared], ignore_index=True)
    coef_out.to_csv(reports_dir / "feature_importance_coefficients.csv", index=False)

    # Permutation importance on ATS Brier (home side)
    rng = np.random.default_rng(args.seed)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(args.gh_samples)
    gh_x = gh_x.astype(np.float64)
    gh_w = gh_w.astype(np.float64)
    m_vals = np.arange(args.margin_low, args.margin_high + 1)

    baseline_brier = _ats_brier_from_features(model, calibrator, xh, xa, xs, bh, ba, margins, spreads, gh_x, gh_w, m_vals)

    def _perm_delta(mat: np.ndarray, col: int, scope: str) -> float:
        mat_perm = mat.copy()
        perm = rng.permutation(mat.shape[0])
        mat_perm[:, col] = mat_perm[perm, col]
        if scope == "home":
            brier = _ats_brier_from_features(model, calibrator, mat_perm, xa, xs, bh, ba, margins, spreads, gh_x, gh_w, m_vals)
        elif scope == "away":
            brier = _ats_brier_from_features(model, calibrator, xh, mat_perm, xs, bh, ba, margins, spreads, gh_x, gh_w, m_vals)
        else:
            brier = _ats_brier_from_features(model, calibrator, xh, xa, mat_perm, bh, ba, margins, spreads, gh_x, gh_w, m_vals)
        return float(brier - baseline_brier)

    rows = []
    for j, name in enumerate(HOME_FEATURE_KEYS):
        rows.append({"scope": "home", "feature": name, "delta_brier": _perm_delta(xh, j, "home")})
    for j, name in enumerate(AWAY_FEATURE_KEYS):
        rows.append({"scope": "away", "feature": name, "delta_brier": _perm_delta(xa, j, "away")})
    for j, name in enumerate(SHARED_FEATURE_KEYS):
        rows.append({"scope": "shared", "feature": name, "delta_brier": _perm_delta(xs, j, "shared")})

    imp = pd.DataFrame(rows).sort_values("delta_brier", ascending=False)
    imp.insert(0, "baseline_brier", baseline_brier)
    imp.to_csv(reports_dir / "feature_importance_permutation.csv", index=False)

    print("Saved:", str(reports_dir / "feature_importance_coefficients.csv"))
    print("Saved:", str(reports_dir / "feature_importance_permutation.csv"))


if __name__ == "__main__":
    main()


