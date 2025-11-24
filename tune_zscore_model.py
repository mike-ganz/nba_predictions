"""
Hyperparameter Tuning for Z-Score Normalized Model.

Performs a Grid Search over XGBoost parameters using TimeSeriesSplit cross-validation.
Goal: Determine if tuning can recover performance for the Z-Score model.
"""
import yaml
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Import internal modules
from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from models.margin_xgboost import MarginXGBoostModel, MarginXGBoostConfig

def main():
    print("="*80)
    print("HYPERPARAMETER TUNING: Z-Score Model")
    print("="*80)
    
    # 1. Load Data
    train_path = Path("data/experiments/zscore/games_train_2124_zscore.jsonl")
    print(f"Loading training data: {train_path}")
    
    loader = GameDataLoader(train_path.parent)
    collection = loader.read_games(train_path.name)
    records = collection.games
    
    # Sort by date just to be safe for TimeSeriesSplit
    records.sort(key=lambda x: x.date)
    print(f"Loaded {len(records)} games (sorted by date)")
    
    # 2. Build Dataset
    # Using Baseline feature set (no FTR) for direct comparison
    exclude_features = [
        'away_tov_edge', 'shared_team_weighted_ts_away', 'away_usage_share_top2', 
        'shared_implied_home_winprob', 'away_orb_edge', 'away_tpar', 
        'shared_pace_mean', 'shared_pace_diff', 'home_ftr', 'away_ftr', 
        'home_b2b', 'away_b2b', 'home_three_in_four', 'away_three_in_four'
    ]
    
    dataset = MarginTrainingDataset(
        records,
        exclude_features=exclude_features,
        include_diff_features=False # XGBoost handles this
    )
    batch = dataset.build()
    X = batch.x
    y = batch.y_margin
    baseline = batch.baseline_margin
    
    print(f"Feature shape: {X.shape}")
    
    # 3. Define Grid
    # Expanding search space slightly
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.02, 0.05],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'reg_lambda': [1.0, 5.0] # L2 regularization
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Testing {len(combinations)} hyperparameter combinations...")
    
    # 4. Cross-Validation Loop
    tscv = TimeSeriesSplit(n_splits=4)
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for i, params in enumerate(combinations):
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            base_train, base_val = baseline[train_idx], baseline[val_idx]
            
            # Configure Model
            cfg = MarginXGBoostConfig(**params)
            model = MarginXGBoostModel(cfg)
            
            # Train
            model.fit(X_train, y_train, base_train)
            
            # Predict
            preds = model.predict(X_val, base_val)
            mae = mean_absolute_error(y_val, preds)
            cv_scores.append(mae)
        
        avg_mae = np.mean(cv_scores)
        results.append({**params, 'mae': avg_mae})
        
        if avg_mae < best_score:
            best_score = avg_mae
            best_params = params
            
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(combinations)}... Best MAE: {best_score:.4f}")

    print("\n" + "-"*80)
    print(f"BEST PARAMS FOUND (MAE: {best_score:.4f}):")
    print(json.dumps(best_params, indent=2))
    print("-"*80)
    
    # 5. Train Final Model with Best Params
    print("\nTraining final model on full training set...")
    final_cfg = MarginXGBoostConfig(**best_params)
    final_model = MarginXGBoostModel(final_cfg)
    final_model.fit(X, y, baseline)
    
    # Save Model
    output_dir = Path("artifacts/experiments/exp5_zscore_tuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(final_model, output_dir / "margin_model.joblib")
    
    # Save Config
    full_config = {
        'model': {
            'model_type': 'xgboost',
            'exclude_features': exclude_features,
            **best_params
        }
    }
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.safe_dump(full_config, f)
        
    # 6. Evaluate on 2025 Test Set
    print("\nEvaluating on 2025 Test Set...")
    test_path = Path("data/experiments/zscore/games_test_zscore.jsonl")
    
    # Invoke prediction script
    import subprocess
    pred_out = "predictions/experiments/exp5_zscore_tuned_predictions.csv"
    
    cmd = [
        sys.executable, "predict_margin.py",
        "--model", str(output_dir),
        "--data", str(test_path),
        "--output", pred_out
    ]
    subprocess.check_call(cmd)
    
    # Calculate ATS
    df = pd.read_csv(pred_out)
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    
    ats_correct = (df['actual_home_covers'] == df['pred_home_covers']).sum()
    ats_pct = (ats_correct / len(df)) * 100
    mae = (df['pred_margin_mu'] - df['actual_margin']).abs().mean()
    
    print("\n" + "="*80)
    print("TUNED Z-SCORE MODEL RESULTS (2025-26):")
    print(f"  MAE: {mae:.2f}")
    print(f"  ATS: {ats_pct:.2f}% ({ats_correct}/{len(df)})")
    print("="*80)
    
    # Comparison
    print("Champion (Untuned, Centered) ATS: 54.22%")
    diff = ats_pct - 54.22
    print(f"Difference: {diff:+.2f}%")

if __name__ == "__main__":
    import json
    main()

