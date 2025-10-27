from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp


def _poisson_pmf(rate: float, size: int) -> np.ndarray:
    rate = max(rate, 1e-9)
    n = np.arange(size)
    log_pmf = n * np.log(rate) - rate - gammaln(n + 1)
    pmf = np.exp(log_pmf)
    total = pmf.sum()
    if total <= 0 or not np.isfinite(total):
        pmf = np.full(size, 1.0 / size)
    else:
        pmf /= total
    return pmf


def bivariate_poisson_pmf(
    lambda_home: float,
    lambda_away: float,
    kappa: float,
    max_points: int = 170,
) -> np.ndarray:
    """Return joint PMF matrix for scores 0..max_points using stable log domain."""
    lam_h = max(lambda_home, 1e-9)
    lam_a = max(lambda_away, 1e-9)
    lam_k = max(kappa, 1e-9)
    lam_h, lam_a, lam_k = stabilize_rates(lam_h, lam_a, lam_k, max_points)
    size = max_points + 1
    pmf = np.zeros((size, size), dtype=np.float64)
    base = -(lam_h + lam_a + lam_k)
    log_lam_h = math.log(lam_h)
    log_lam_a = math.log(lam_a)
    log_lam_k = math.log(lam_k)

    for i in range(size):
        for j in range(size):
            upper = min(i, j)
            if upper == -1:
                continue
            ks = np.arange(upper + 1)
            log_terms = (
                (i - ks) * log_lam_h
                + (j - ks) * log_lam_a
                + ks * log_lam_k
                - gammaln(i - ks + 1)
                - gammaln(j - ks + 1)
                - gammaln(ks + 1)
            )
            pmf[i, j] = math.exp(base + logsumexp(log_terms))
    total = pmf.sum()
    if not np.isfinite(total) or total <= 0:
        home_mean = lam_h + lam_k
        away_mean = lam_a + lam_k
        pmf = np.outer(_poisson_pmf(home_mean, size), _poisson_pmf(away_mean, size))
    else:
        pmf /= total
    return pmf


def bivariate_poisson_log_normal_pmf(
    lambda_home: float,
    lambda_away: float,
    kappa: float,
    sigma_home: float,
    sigma_away: float,
    max_points: int = 170,
    n_samples: int = 75,
) -> np.ndarray:
    """Approximate joint PMF under Poisson-lognormal overdispersion using Gauss-Hermite quadrature."""
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_samples)
    gh_x = gh_x.astype(np.float64)
    gh_w = gh_w.astype(np.float64)

    size = max_points + 1
    pmf = np.zeros((size, size), dtype=np.float64)

    sigma_home = max(sigma_home, 1e-6)
    sigma_away = max(sigma_away, 1e-6)

    for i in range(n_samples):
        for j in range(n_samples):
            adj_lambda_home = lambda_home * np.exp(np.sqrt(2) * sigma_home * gh_x[i] - sigma_home ** 2)
            adj_lambda_away = lambda_away * np.exp(np.sqrt(2) * sigma_away * gh_x[j] - sigma_away ** 2)
            adj_lambda_home, adj_lambda_away, adj_kappa = stabilize_rates(adj_lambda_home, adj_lambda_away, kappa, max_points)
            base_pmf = bivariate_poisson_pmf(adj_lambda_home, adj_lambda_away, adj_kappa, max_points)
            pmf += gh_w[i] * gh_w[j] * base_pmf

    pmf /= np.pi
    pmf /= pmf.sum()
    return pmf


def margin_pmf(joint_pmf: np.ndarray, margin_range: Tuple[int, int] = (-60, 60)) -> np.ndarray:
    """Convolve joint PMF into margin distribution within range."""
    low, high = margin_range
    width = high - low + 1
    margin = np.zeros(width, dtype=np.float64)
    size_h, size_a = joint_pmf.shape
    for i in range(size_h):
        for j in range(size_a):
            diff = i - j
            if low <= diff <= high:
                margin[diff - low] += joint_pmf[i, j]
    residual = 1.0 - margin.sum()
    if residual > 0:
        margin[np.argmax(margin)] += residual
    return margin


def single_team_cdf(joint_pmf: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute cumulative distribution for a single team score."""
    pmf = joint_pmf.sum(axis=1 - axis)
    return np.cumsum(pmf)


def compute_win_probabilities(joint_pmf: np.ndarray) -> Tuple[float, float, float]:
    """Return (home_win, away_win, tie) probabilities from joint PMF."""
    home_win = float(np.tril(joint_pmf, -1).sum())
    away_win = float(np.triu(joint_pmf, 1).sum())
    tie = float(np.trace(joint_pmf))
    return home_win, away_win, tie


def expected_scores(joint_pmf: np.ndarray) -> Tuple[float, float]:
    """Compute expected home and away scores from the joint distribution."""
    home_indices = np.arange(joint_pmf.shape[0])[:, None]
    away_indices = np.arange(joint_pmf.shape[1])[None, :]
    expected_home = float((home_indices * joint_pmf).sum())
    expected_away = float((away_indices * joint_pmf).sum())
    return expected_home, expected_away


def expected_margin(margin_pmf: np.ndarray, margin_range: Tuple[int, int]) -> float:
    low, high = margin_range
    idx = np.arange(low, high + 1)
    return float(margin_pmf @ idx)


def stabilize_rates(lambda_home: float, lambda_away: float, kappa: float, max_points: int, buffer: int = 5) -> Tuple[float, float, float]:
    cap = max(max_points - buffer, 1)
    lam_h = float(np.clip(lambda_home, 1e-6, cap))
    lam_a = float(np.clip(lambda_away, 1e-6, cap))
    kap = float(np.clip(kappa, 0.0, cap))
    mean_home = lam_h + kap
    mean_away = lam_a + kap
    max_mean = max(mean_home, mean_away)
    if max_mean > cap:
        scale = cap / max_mean
        lam_h *= scale
        lam_a *= scale
        kap *= scale
    return lam_h, lam_a, kap

__all__ = [
    "bivariate_poisson_pmf",
    "bivariate_poisson_log_normal_pmf",
    "margin_pmf",
    "single_team_cdf",
    "compute_win_probabilities",
    "expected_scores",
    "expected_margin",
    "stabilize_rates",
]
