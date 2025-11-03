"""Spread coverage prediction using ElasticNet Logistic Regression.

This module provides a classification approach to predicting whether the home
team will cover the spread, replacing the previous regression-based margin prediction.

Instead of:
  P(margin) → derive P(home covers)
  
We predict:
  P(home covers) directly using logistic regression

This directly optimizes for the binary betting decision we care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


@dataclass
class MarginNormalConfig:
    """Configuration for spread coverage prediction model."""
    
    use_cv: bool = True
    Cs: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    )
    l1_ratios: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    )
    cv_folds: int = 5
    C: float = 1.0  # Fallback if use_cv=False
    l1_ratio: float = 0.5  # Fallback
    max_iter: int = 1000
    solver: str = 'saga'  # Required for elasticnet penalty


class MarginNormalModel:
    """
    Spread coverage prediction using ElasticNet Logistic Regression.
    
    Predicts:
    - P(home covers): Probability that home team covers the spread
    
    Model: binary outcome ~ logistic(features)
    
    This is a direct classification approach where the model outputs probabilities
    that can be used for betting decisions.
    """
    
    def __init__(self, config: MarginNormalConfig | None = None) -> None:
        self.config = config or MarginNormalConfig()
        self.scaler = None
        
        # Model for coverage prediction
        if self.config.use_cv:
            # Check if l1_ratios includes 0.0 (pure L2)
            # If yes, we need to handle it separately since saga+elasticnet doesn't support l1_ratio=0
            if 0.0 in self.config.l1_ratios:
                # Use liblinear solver which supports both L1 and L2
                # Note: liblinear doesn't support l1_ratios parameter
                # We'll test both 'l1' and 'l2' penalties separately
                self.model = LogisticRegressionCV(
                    Cs=self.config.Cs,
                    cv=self.config.cv_folds,
                    penalty='l2',  # Start with L2, then test elasticnet separately
                    solver='lbfgs',
                    max_iter=self.config.max_iter,
                    scoring='neg_log_loss',
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = LogisticRegressionCV(
                    Cs=self.config.Cs,
                    cv=self.config.cv_folds,
                    penalty='elasticnet',
                    solver=self.config.solver,
                    l1_ratios=self.config.l1_ratios,
                    max_iter=self.config.max_iter,
                    scoring='neg_log_loss',
                    random_state=42,
                    n_jobs=-1
                )
        else:
            if self.config.l1_ratio == 0.0:
                self.model = LogisticRegression(
                    C=self.config.C,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=self.config.max_iter,
                    random_state=42
                )
            else:
                self.model = LogisticRegression(
                    C=self.config.C,
                    penalty='elasticnet',
                    solver=self.config.solver,
                    l1_ratio=self.config.l1_ratio,
                    max_iter=self.config.max_iter,
                    random_state=42
                )
    
    def fit(
        self,
        x: np.ndarray,
        y_home_covers: np.ndarray,
    ) -> None:
        """
        Fit logistic regression model for coverage prediction.
        
        Args:
            x: Feature matrix (N, D)
            y_home_covers: Binary outcomes (N,) where 1 = home covers, 0 = away covers
        """
        # Standardize features for better convergence
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        
        self.model.fit(x_scaled, y_home_covers)
    
    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Predict coverage probabilities.
        
        Args:
            x: Feature matrix (N, D)
        
        Returns:
            prob_home_covers: Predicted probabilities that home team covers (N,)
        """
        # Scale features using same scaler from training
        if self.scaler is not None:
            x_scaled = self.scaler.transform(x)
        else:
            x_scaled = x
        
        # predict_proba returns [P(class=0), P(class=1)]
        # We want P(class=1) which is home team covering
        prob_home_covers = self.model.predict_proba(x_scaled)[:, 1]
        
        return prob_home_covers
    
    def get_coefficients(self) -> np.ndarray:
        """
        Get model coefficients for interpretation.
        
        Returns:
            Coefficient array (D,)
        """
        return self.model.coef_[0]
    
    def get_intercept(self) -> float:
        """
        Get model intercept.
        
        Returns:
            Intercept value
        """
        return float(self.model.intercept_[0])


__all__ = [
    "MarginNormalConfig",
    "MarginNormalModel",
]
