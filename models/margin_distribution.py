"""Distribution utilities for direct margin prediction.

This module provides simple probability calculations using the Normal distribution,
replacing the complex bivariate Poisson PMF convolution logic.

Key simplification:
- Old: Compute joint PMF, convolve into margin PMF, sum for probabilities
- New: Direct probability calculations using scipy.stats.norm
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def margin_cover_probability(
    mu: np.ndarray,
    sigma: np.ndarray,
    spread_home: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate probability of covering the spread.
    
    Args:
        mu: Predicted margins (N,) - positive favors home
        sigma: Predicted standard deviations (N,)
        spread_home: Market spreads (N,) - negative when home favored
    
    Returns:
        prob_home_covers: P(actual_margin > -spread_home)
        prob_away_covers: P(actual_margin < -spread_home) = 1 - prob_home_covers
    
    Example:
        If spread_home = -3 (home favored by 3):
        - Home covers if actual_margin > 3 (win by more than 3)
        - Away covers if actual_margin < 3 (lose by less than 3, or win)
    """
    # Home covers if: actual_margin > -spread_home
    # For spread_home = -3, home covers if margin > 3
    threshold = -spread_home
    
    # P(margin > threshold) = 1 - Φ((threshold - μ) / σ)
    z = (threshold - mu) / sigma
    prob_home_covers = 1 - norm.cdf(z)
    prob_away_covers = 1 - prob_home_covers
    
    return prob_home_covers, prob_away_covers


def margin_win_probability(
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate moneyline win probabilities.
    
    Args:
        mu: Predicted margins (N,)
        sigma: Predicted standard deviations (N,)
    
    Returns:
        prob_home_win: P(margin > 0)
        prob_away_win: P(margin < 0)
    
    Note: This ignores ties (margin = 0), which are extremely rare in NBA.
    """
    # Home wins if margin > 0
    z = (0 - mu) / sigma
    prob_home_win = 1 - norm.cdf(z)
    prob_away_win = 1 - prob_home_win
    
    return prob_home_win, prob_away_win


def margin_pmf(
    mu: float,
    sigma: float,
    margin_range: Tuple[int, int] = (-60, 60),
) -> np.ndarray:
    """
    Discretize Normal distribution into PMF over integer margins.
    
    Useful for compatibility with existing evaluation code that expects
    discrete probability mass functions.
    
    Args:
        mu: Predicted margin mean
        sigma: Predicted margin std dev
        margin_range: (low, high) integer range
    
    Returns:
        pmf: Probability mass function over margins (high - low + 1,)
    """
    low, high = margin_range
    margins = np.arange(low, high + 1)
    
    # Use midpoint rule for discretization
    # P(margin = k) ≈ P(k - 0.5 < margin < k + 0.5)
    lower_bounds = margins - 0.5
    upper_bounds = margins + 0.5
    
    pmf = norm.cdf(upper_bounds, loc=mu, scale=sigma) - norm.cdf(lower_bounds, loc=mu, scale=sigma)
    
    # Normalize (should already be close to 1, but ensure exact)
    pmf /= pmf.sum()
    
    return pmf


def margin_quantiles(
    mu: np.ndarray,
    sigma: np.ndarray,
    quantiles: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate margin quantiles for uncertainty visualization.
    
    Useful for showing prediction intervals like "90% confidence interval".
    
    Args:
        mu: Predicted margins (N,)
        sigma: Predicted standard deviations (N,)
        quantiles: Quantile levels (Q,). Defaults to [0.05, 0.25, 0.5, 0.75, 0.95]
    
    Returns:
        Array of shape (N, Q) with quantile values
    
    Example:
        For 90% confidence interval, use quantiles=[0.05, 0.95]
        Result[:, 0] = lower bound, Result[:, 1] = upper bound
    """
    if quantiles is None:
        quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    
    z_scores = norm.ppf(quantiles)
    # Broadcasting: (N, 1) + (N, 1) * (1, Q)
    return mu[:, None] + sigma[:, None] * z_scores[None, :]


def expected_margin(mu: np.ndarray) -> np.ndarray:
    """
    Return expected margin (trivial for Normal distribution).
    
    Args:
        mu: Predicted margin means (N,)
    
    Returns:
        Expected margins (N,) - same as mu for Normal distribution
    """
    return mu


__all__ = [
    "margin_cover_probability",
    "margin_win_probability",
    "margin_pmf",
    "margin_quantiles",
    "expected_margin",
]

