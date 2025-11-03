"""Distribution utilities for spread coverage prediction.

This module provides simple probability pass-through functions for the logistic
regression approach, maintaining API compatibility with previous distribution-based
predictions.

Key simplification:
- Old: Compute margin distribution, derive probabilities via CDF integration
- New: Direct probability output from logistic regression model
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def margin_cover_probability(
    prob_home_covers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pass-through coverage probabilities from logistic regression model.
    
    Args:
        prob_home_covers: Predicted probabilities that home team covers (N,)
    
    Returns:
        prob_home_covers: P(home covers) - same as input
        prob_away_covers: P(away covers) = 1 - P(home covers)
    
    Note: This function now simply returns the model's direct predictions,
    maintaining API compatibility with previous distribution-based approach.
    """
    prob_away_covers = 1 - prob_home_covers
    
    return prob_home_covers, prob_away_covers


def margin_win_probability(
    prob_home_covers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate moneyline win probabilities (deprecated).
    
    Note: Without margin distribution, we can't accurately predict win probabilities.
    This function is kept for API compatibility but returns simplified estimates.
    For accurate moneyline predictions, use market odds or a separate model.
    
    Args:
        prob_home_covers: Predicted probabilities that home team covers (N,)
    
    Returns:
        prob_home_win: Estimated P(home wins) - uses 0.5 as placeholder
        prob_away_win: Estimated P(away wins) - uses 0.5 as placeholder
    """
    # Without margin distribution, we can't accurately derive win probabilities
    # Return 0.5 (coin flip) as placeholder to maintain API compatibility
    n = len(prob_home_covers)
    prob_home_win = np.full(n, 0.5)
    prob_away_win = np.full(n, 0.5)
    
    return prob_home_win, prob_away_win


__all__ = [
    "margin_cover_probability",
    "margin_win_probability",
]
