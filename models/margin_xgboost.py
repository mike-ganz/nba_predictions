"""Direct margin prediction using XGBoost.

This module provides an alternative to Ridge regression using gradient boosted trees.
Like the Ridge model, it predicts margin residuals from market baseline, but uses
tree-based learning instead of linear regression.

Key differences from Ridge:
- Automatically captures non-linear relationships and feature interactions
- No need for manual interaction features
- Provides native feature importance metrics
- Only predicts mean (μ) for now, not variance (σ)
- Can use custom objective functions (e.g., ATS-focused loss)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable

import numpy as np
import xgboost as xgb


@dataclass
class MarginXGBoostConfig:
    """Configuration for XGBoost margin prediction model."""
    
    # Model identifier
    model_type: str = "xgboost"
    
    # XGBoost hyperparameters
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    
    # Training parameters
    random_state: int = 42
    n_jobs: int = -1  # Use all CPU cores
    
    # Early stopping (optional)
    early_stopping_rounds: Optional[int] = None
    eval_metric: str = "rmse"
    
    # Custom objective function settings
    use_ats_objective: bool = False
    ats_penalty_weight: float = 10.0  # Weight for getting ATS wrong
    margin_penalty_weight: float = 0.1  # Weight for margin error
    
    # Feature engineering options
    include_diff_features: bool = False  # Trees don't need manual interactions
    
    # Feature exclusions (for compatibility with Ridge)
    exclude_features: list[str] = field(default_factory=list)


# Global variable to store spreads for custom objective
# (XGBoost doesn't provide a clean way to pass extra data to custom objectives)
_GLOBAL_SPREADS = None


class ATSObjective:
    """
    Custom ATS-focused objective function for XGBoost.
    
    This objective heavily penalizes predictions that get the ATS decision wrong.
    Spreads must be set via set_global_spreads() before training.
    
    This is implemented as a class (rather than a factory function) to ensure
    it can be properly pickled when saving the model.
    
    Args:
        ats_penalty: Weight for getting ATS decision wrong (default: 10.0)
        margin_penalty: Weight for margin prediction error (default: 0.1)
    """
    
    def __init__(self, ats_penalty: float = 10.0, margin_penalty: float = 0.1):
        self.ats_penalty = ats_penalty
        self.margin_penalty = margin_penalty
    
    def __call__(self, labels: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom gradient and hessian for ATS-focused loss.
        
        Note: XGBRegressor uses (labels, preds) signature, not DMatrix.
        
        Loss = ats_penalty * I(pred and actual on different sides of spread) + 
               margin_penalty * (pred - actual)^2
        
        Args:
            labels: True margin values (home - away)
            preds: Predicted margin values
        
        Returns:
            Tuple of (gradient, hessian) arrays
        """
        global _GLOBAL_SPREADS
        
        # Use global spreads
        if _GLOBAL_SPREADS is None or len(_GLOBAL_SPREADS) != len(labels):
            # Fallback: use standard squared error
            grad = 2 * (preds - labels)
            hess = 2 * np.ones_like(preds)
            return grad, hess
        
        spreads = _GLOBAL_SPREADS
        
        # Determine if prediction and actual are on same side of spread
        # Note: preds and labels are residuals, so we need to add back baseline
        # But for ATS decision, we can work with residuals if spread is also relative
        # Actually, let's work with absolute margins for clarity
        
        # Simple approach: penalize based on whether sign of residual matches reality
        # If actual covered (positive residual), prediction should be positive
        # If actual didn't cover (negative residual), prediction should be negative
        
        pred_sign = np.sign(preds)
        actual_sign = np.sign(labels)
        
        # ATS wrong if signs don't match
        ats_wrong = (pred_sign != actual_sign).astype(float)
        
        # Gradient components
        # 1. ATS gradient: push prediction toward correct sign
        ats_grad = np.where(
            ats_wrong > 0,
            self.ats_penalty * (-actual_sign),  # Push toward actual sign
            0.0
        )
        
        # 2. Margin gradient: standard L2 loss
        margin_grad = 2 * self.margin_penalty * (preds - labels)
        
        # Combined gradient
        grad = ats_grad + margin_grad
        
        # Hessian (second derivative)
        # For ATS: constant
        # For margin: constant (L2)
        hess = np.where(ats_wrong > 0, self.ats_penalty * 0.1, 0.01) + 2 * self.margin_penalty
        
        return grad, hess


def set_global_spreads(spreads: np.ndarray | None):
    """Set global spreads for ATS objective function."""
    global _GLOBAL_SPREADS
    _GLOBAL_SPREADS = spreads


class MarginXGBoostModel:
    """
    Direct margin prediction using XGBoost.
    
    Predicts:
    - μ (margin mean): Expected home - away score difference
    
    Model: Uses gradient boosted trees to predict margin residual from baseline.
    
    Features:
    - Custom ATS-focused objective function (optional)
    - Automatic interaction learning (no manual difference features needed)
    - Feature importance for interpretability
    
    Note: Unlike Ridge, this model does not predict variance (σ).
    Cover probabilities must be computed separately if needed.
    """
    
    def __init__(self, config: MarginXGBoostConfig | None = None) -> None:
        self.config = config or MarginXGBoostConfig()
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: Optional[list] = None
        self.spreads_train: Optional[np.ndarray] = None  # Store for custom objective
    
    def fit(
        self,
        x: np.ndarray,
        y_margin: np.ndarray,
        baseline_margin: np.ndarray | None = None,
        market_spread: np.ndarray | None = None,
        x_val: np.ndarray | None = None,
        y_val_margin: np.ndarray | None = None,
        baseline_val_margin: np.ndarray | None = None,
        market_spread_val: np.ndarray | None = None,
    ) -> None:
        """
        Fit XGBoost model for margin prediction.
        
        Args:
            x: Feature matrix (N, D)
            y_margin: Actual margins (N,) = home_score - away_score
            baseline_margin: Optional baseline (from market spread)
            market_spread: Market spread for each game (for ATS objective)
            x_val: Validation features (for early stopping)
            y_val_margin: Validation margins
            baseline_val_margin: Validation baseline margins
            market_spread_val: Validation market spreads
        """
        # Predict residual from baseline (like Ridge does)
        if baseline_margin is not None:
            y_target = y_margin - baseline_margin
        else:
            y_target = y_margin
        
        # Choose objective function
        if self.config.use_ats_objective and market_spread is not None:
            # Set global spreads for custom objective
            set_global_spreads(market_spread)
            
            # Use custom ATS-focused objective (class-based for picklability)
            objective = ATSObjective(
                ats_penalty=self.config.ats_penalty_weight,
                margin_penalty=self.config.margin_penalty_weight
            )
            
            # Store spreads for reference
            self.spreads_train = market_spread
        else:
            # Use standard squared error
            objective = 'reg:squarederror'
            self.spreads_train = None
            set_global_spreads(None)
        
        # Initialize XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective=objective,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            eval_metric=self.config.eval_metric,
        )
        
        # Prepare evaluation set for early stopping (if provided)
        eval_set = None
        if x_val is not None and y_val_margin is not None:
            if baseline_val_margin is not None:
                y_val_target = y_val_margin - baseline_val_margin
            else:
                y_val_target = y_val_margin
            eval_set = [(x_val, y_val_target)]
        
        # Fit model
        fit_params = {}
        if eval_set is not None and self.config.early_stopping_rounds is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = self.config.early_stopping_rounds
            fit_params['verbose'] = False
        
        self.model.fit(x, y_target, **fit_params)
        
        # Store feature names for later inspection
        self.feature_names = [f"feature_{i}" for i in range(x.shape[1])]
    
    def predict(
        self,
        x: np.ndarray,
        baseline_margin: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict margin means.
        
        Args:
            x: Feature matrix (N, D)
            baseline_margin: Optional baseline margins
        
        Returns:
            mu: Predicted margin means (N,)
            
        Note: Unlike Ridge model, this only returns mu (not mu and sigma).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Predict residual
        residual = self.model.predict(x)
        
        # Add baseline to get final margin
        if baseline_margin is not None:
            mu = baseline_margin + residual
        else:
            mu = residual
        
        return mu
    
    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', or 'cover')
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        return importance
    
    def get_feature_importance_array(self, importance_type: str = "gain") -> np.ndarray:
        """
        Get feature importance as array (aligned with feature order).
        
        Args:
            importance_type: Type of importance ('weight', 'gain', or 'cover')
        
        Returns:
            Array of importance scores (D,)
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get importance dict
        importance_dict = self.get_feature_importance(importance_type)
        
        # Convert to array (features not in dict get 0)
        importance_array = np.zeros(len(self.feature_names))
        for feat_name, score in importance_dict.items():
            # XGBoost uses f0, f1, f2, etc.
            if feat_name.startswith('f'):
                idx = int(feat_name[1:])
                if idx < len(importance_array):
                    importance_array[idx] = score
        
        return importance_array


__all__ = [
    "MarginXGBoostConfig",
    "MarginXGBoostModel",
    "ATSObjective",
]

