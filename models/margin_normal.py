"""Direct margin prediction using Normal distribution.

This module provides a simpler alternative to bivariate Poisson modeling by
directly predicting the margin distribution as margin ~ N(μ, σ²).

Instead of:
  P(home_score, away_score) → derive P(margin)
  
We predict:
  P(margin) directly using Normal distribution

This is faster, simpler, and directly optimizes what we care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV


@dataclass
class MarginNormalConfig:
    """Configuration for direct margin prediction model."""
    
    use_cv: bool = True
    alphas_mean: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    )
    alphas_variance: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    cv_folds: int = 5
    alpha_mean: float = 0.5  # Fallback if use_cv=False
    alpha_variance: float = 2.0  # Fallback
    min_variance: float = 1.0  # Minimum predicted variance (σ²)


class MarginNormalModel:
    """
    Direct margin prediction using Normal distribution.
    
    Predicts:
    - μ (margin mean): Expected home - away score difference
    - σ² (margin variance): Uncertainty in the prediction
    
    Model: margin ~ N(μ, σ²)
    
    This is a heteroskedastic model - both mean and variance are predicted
    from features, allowing uncertainty to vary by game characteristics.
    """
    
    def __init__(self, config: MarginNormalConfig | None = None) -> None:
        self.config = config or MarginNormalConfig()
        
        # Model for mean prediction
        if self.config.use_cv:
            self.model_mean = RidgeCV(
                alphas=self.config.alphas_mean,
                cv=self.config.cv_folds,
                scoring='neg_mean_squared_error'
            )
        else:
            self.model_mean = Ridge(alpha=self.config.alpha_mean)
        
        # Model for variance prediction
        if self.config.use_cv:
            self.model_variance = RidgeCV(
                alphas=self.config.alphas_variance,
                cv=self.config.cv_folds,
                scoring='neg_mean_squared_error'
            )
        else:
            self.model_variance = Ridge(alpha=self.config.alpha_variance)
    
    def fit(
        self,
        x: np.ndarray,
        y_margin: np.ndarray,
        baseline_margin: np.ndarray | None = None,
    ) -> None:
        """
        Fit both mean and variance models.
        
        Args:
            x: Feature matrix (N, D)
            y_margin: Actual margins (N,) = home_score - away_score
            baseline_margin: Optional baseline (from market spread)
        """
        # Fit mean model
        if baseline_margin is not None:
            # Predict residual from baseline (similar to current bivariate approach)
            y_mean_target = y_margin - baseline_margin
        else:
            y_mean_target = y_margin
        
        self.model_mean.fit(x, y_mean_target)
        
        # Fit variance model (heteroskedastic)
        # Predict squared residuals
        y_mean_pred = self.model_mean.predict(x)
        if baseline_margin is not None:
            residuals = y_margin - (y_mean_pred + baseline_margin)
        else:
            residuals = y_margin - y_mean_pred
        
        squared_residuals = residuals ** 2
        
        # Predict log(variance) for numerical stability
        log_variance_target = np.log(np.maximum(squared_residuals, self.config.min_variance))
        self.model_variance.fit(x, log_variance_target)
    
    def predict(
        self,
        x: np.ndarray,
        baseline_margin: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict margin distribution parameters.
        
        Args:
            x: Feature matrix (N, D)
            baseline_margin: Optional baseline margins
        
        Returns:
            mu: Predicted margin means (N,)
            sigma: Predicted margin standard deviations (N,)
        """
        # Predict mean
        mean_adjustment = self.model_mean.predict(x)
        if baseline_margin is not None:
            mu = baseline_margin + mean_adjustment
        else:
            mu = mean_adjustment
        
        # Predict variance
        log_variance = self.model_variance.predict(x)
        variance = np.exp(log_variance)
        variance = np.maximum(variance, self.config.min_variance)
        sigma = np.sqrt(variance)
        
        return mu, sigma


__all__ = [
    "MarginNormalConfig",
    "MarginNormalModel",
]

