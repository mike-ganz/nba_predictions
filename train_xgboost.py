"""Training script for XGBoost spread coverage prediction model.

Usage:
    python train_xgboost.py --data data/games_train_with_players_90_norm.jsonl --output artifacts/xgboost_coverage
"""

import argparse
import yaml
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost coverage prediction model")
    parser.add_argument("--data", type=str, required=True, help="Training JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model artifacts")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (slower)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading training data from {args.data}")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"Loaded {len(records)} games")
    
    # Build dataset - use ALL features (no exclusions)
    print("\nBuilding training dataset with ALL features...")
    dataset = MarginTrainingDataset(records, exclude_features=[])
    batch = dataset.build()
    print(f"  Feature dimensionality: {batch.x.shape[1]}")
    
    # Filter to valid games
    valid_mask = ~np.isnan(batch.y_home_covers)
    X = batch.x[valid_mask]
    y = batch.y_home_covers[valid_mask].astype(int)
    spread = batch.market_spread_home[valid_mask]
    
    n_valid = valid_mask.sum()
    print(f"  Games with outcomes: {n_valid}")
    print(f"  Coverage rate: {y.mean() * 100:.2f}% (home team covers)")
    
    # Calculate class imbalance
    n_home_covers = y.sum()
    n_away_covers = len(y) - n_home_covers
    scale_pos_weight = n_away_covers / n_home_covers
    print(f"  Class balance: {n_home_covers} home / {n_away_covers} away")
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")
    
    # Train model
    print("\n" + "="*70)
    print("Training XGBoost Model")
    print("="*70)
    
    if args.tune:
        print("\nRunning hyperparameter tuning (this will take a while)...")
        
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        model = grid_search.best_estimator_
        
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV score (neg log loss): {grid_search.best_score_:.4f}")
        
    else:
        print("\nUsing conservative hyperparameters for small dataset (3.5k samples, 29 features)...")
        
        # Conservative parameters to prevent overfitting
        # With only 3,560 samples and 29 features, we need to be very careful
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=2,              # Very shallow trees (was 4)
            learning_rate=0.01,       # Slow learning (was 0.05)
            n_estimators=50,          # Fewer trees (was 200)
            min_child_weight=50,      # Require ~50 samples per leaf (was 3)
            subsample=0.7,            # Use 70% of samples per tree (was 0.8)
            colsample_bytree=0.7,     # Use 70% of features per tree (was 0.8)
            reg_alpha=1.0,            # L1 regularization (was 0)
            reg_lambda=5.0,           # L2 regularization (was 1.0)
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        print("\nHyperparameters:")
        print(f"  max_depth: 2 (shallow trees to prevent memorization)")
        print(f"  learning_rate: 0.01 (slow learning)")
        print(f"  n_estimators: 50 (fewer trees)")
        print(f"  min_child_weight: 50 (~1.4% of training data per leaf)")
        print(f"  subsample: 0.7 (use 70% of data per tree)")
        print(f"  colsample_bytree: 0.7 (use 70% of features per tree)")
        print(f"  reg_alpha: 1.0 (L1 regularization)")
        print(f"  reg_lambda: 5.0 (L2 regularization)")
        
        # Quick CV to check performance
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        model.fit(X, y)
    
    # Evaluate on training set
    print("\n" + "="*70)
    print("Training Set Evaluation")
    print("="*70)
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_pred_proba)
    brier = brier_score_loss(y, y_pred_proba)
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    print(f"  Log Loss:    {logloss:.4f} (lower is better)")
    print(f"  Brier Score: {brier:.4f} (lower is better)")
    
    # ATS Performance
    ats_roi = (accuracy * 0.91 + (1 - accuracy) * -1.0) * 100
    
    print(f"\nATS Performance:")
    print(f"  ATS Accuracy: {accuracy * 100:.2f}%")
    print(f"  Breakeven:    52.38%")
    print(f"  Edge:         {(accuracy - 0.5238) * 100:+.2f}%")
    print(f"  ROI per bet:  {ats_roi:+.2f}%")
    print(f"  Status:       {'[+] PROFITABLE' if ats_roi > 0 else '[-] UNPROFITABLE'}")
    
    # Probability distribution
    print(f"\nProbability Distribution:")
    print(f"  Min:    {y_pred_proba.min():.3f}")
    print(f"  25%:    {np.percentile(y_pred_proba, 25):.3f}")
    print(f"  Median: {np.median(y_pred_proba):.3f}")
    print(f"  75%:    {np.percentile(y_pred_proba, 75):.3f}")
    print(f"  Max:    {y_pred_proba.max():.3f}")
    
    # Check home/away distribution
    home_picks = (y_pred_proba > 0.5).sum()
    away_picks = (y_pred_proba < 0.5).sum()
    print(f"\nPrediction Distribution:")
    print(f"  Predicts HOME covers: {home_picks} games ({home_picks/len(y)*100:.1f}%)")
    print(f"  Predicts AWAY covers: {away_picks} games ({away_picks/len(y)*100:.1f}%)")
    
    # Calibration by confidence level
    print(f"\nCalibration by Confidence Level:")
    confidence = np.maximum(y_pred_proba, 1 - y_pred_proba)
    buckets = [(0.5, 0.55, "50-55%"), (0.55, 0.60, "55-60%"), (0.60, 0.65, "60-65%"), 
               (0.65, 0.70, "65-70%"), (0.70, 1.00, "70%+")]
    
    for low, high, label in buckets:
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() > 0:
            acc = (y_pred[mask] == y[mask]).mean()
            roi = (acc * 0.91 + (1 - acc) * -1.0) * 100
            print(f"  {label}: {mask.sum():4d} games, {acc*100:.1f}% accuracy, {roi:+.2f}% ROI")
    
    # Feature importance
    print(f"\n" + "="*70)
    print("Feature Importance (Top 15)")
    print("="*70)
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    from features.builder import HOME_FEATURE_KEYS, AWAY_FEATURE_KEYS, SHARED_FEATURE_KEYS
    
    feature_names = []
    for k in HOME_FEATURE_KEYS:
        feature_names.append(f"home_{k}")
    for k in AWAY_FEATURE_KEYS:
        feature_names.append(f"away_{k}")
    for k in SHARED_FEATURE_KEYS:
        feature_names.append(f"shared_{k}")
    for k in HOME_FEATURE_KEYS:
        if k in AWAY_FEATURE_KEYS:
            feature_names.append(f"diff_{k}")
    
    print("\nBy Gain (split improvement):")
    for i in indices[:15]:
        print(f"  {feature_names[i]:<35} {importance[i]:.4f}")
    
    # Save feature importance
    importance_df = {
        'feature': [feature_names[i] for i in indices],
        'importance': [importance[i] for i in indices]
    }
    import pandas as pd
    pd.DataFrame(importance_df).to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Save model
    print("\n" + "="*70)
    model_path = output_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save model info
    info = {
        'model_type': 'xgboost',
        'n_features': X.shape[1],
        'n_training_games': len(y),
        'training_accuracy': float(accuracy),
        'training_logloss': float(logloss),
        'scale_pos_weight': float(scale_pos_weight),
    }
    
    if args.tune:
        info['hyperparameters'] = grid_search.best_params_
    else:
        info['hyperparameters'] = {
            'max_depth': 2,
            'learning_rate': 0.01,
            'n_estimators': 50,
            'min_child_weight': 50,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
        }
    
    with open(output_dir / "model_info.yaml", 'w') as f:
        yaml.safe_dump(info, f)
    print(f"Model info saved to {output_dir / 'model_info.yaml'}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

