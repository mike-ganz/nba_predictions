"""Hyperparameter tuning script for XGBoost margin model.

This script performs cross-validation to find optimal XGBoost hyperparameters.

Usage:
    python tune_xgboost_hyperparameters.py --data data/games_train_with_players_90_norm.jsonl
    python tune_xgboost_hyperparameters.py --data data/games_train_with_players_90_norm.jsonl --quick
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument("--data", type=str, required=True, help="Training JSONL path")
    parser.add_argument("--config", type=str, default="configs/margin_xgboost.yaml", 
                        help="Base config file")
    parser.add_argument("--output", type=str, default="configs/margin_xgboost_tuned.yaml",
                        help="Output config file with best hyperparameters")
    parser.add_argument("--method", type=str, choices=["grid", "random"], default="random",
                        help="Search method (grid or random)")
    parser.add_argument("--n-iter", type=int, default=50,
                        help="Number of iterations for random search")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller search space for faster tuning")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 for all cores)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading base config from {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Load training data
    print(f"\nLoading training data from {args.data}")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"Loaded {len(records)} games")
    
    # Build dataset
    print("\nBuilding training dataset...")
    exclude_features = cfg.get('model', {}).get('exclude_features', [])
    if exclude_features:
        print(f"  Excluding {len(exclude_features)} features: {exclude_features}")

    # Honor include_diff_features flag from config so that tuning uses
    # the same feature space as final training (important for Champion).
    default_include_diff = True
    include_diff_features = cfg.get('model', {}).get('include_diff_features', default_include_diff)
    if not include_diff_features:
        print("  Excluding difference features for tuning (include_diff_features = False)")

    dataset = MarginTrainingDataset(
        records,
        exclude_features=exclude_features,
        include_diff_features=include_diff_features,
    )
    batch = dataset.build()
    print(f"  Feature dimensionality: {batch.x.shape[1]}")
    
    # Prepare target (residual from baseline)
    y_target = batch.y_margin - batch.baseline_margin
    
    # Define hyperparameter search space
    if args.quick:
        print("\nUsing QUICK search space (faster, less thorough)")
        param_space = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'reg_alpha': [0.1, 1.0],
            'reg_lambda': [1.0, 5.0],
        }
    else:
        print("\nUsing FULL search space (slower, more thorough)")
        param_space = {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [2, 3, 4, 5, 6],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0, 2.0],
            'reg_lambda': [0.5, 1.0, 2.0, 5.0, 10.0],
        }
    
    # Base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=1,  # Each CV fold will use 1 job, parallelism is across folds
    )
    
    # Setup search
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING")
    print(f"{'='*70}")
    print(f"Method: {args.method.upper()}")
    print(f"CV Folds: {args.cv_folds}")
    
    if args.method == 'grid':
        n_combinations = np.prod([len(v) for v in param_space.values()])
        print(f"Grid size: {n_combinations} combinations")
        print(f"Total fits: {n_combinations * args.cv_folds}")
        
        search = GridSearchCV(
            base_model,
            param_space,
            cv=args.cv_folds,
            scoring=scorer,
            n_jobs=args.n_jobs,
            verbose=2,
            refit=True,
        )
    else:  # random
        print(f"Random iterations: {args.n_iter}")
        print(f"Total fits: {args.n_iter * args.cv_folds}")
        
        search = RandomizedSearchCV(
            base_model,
            param_space,
            n_iter=args.n_iter,
            cv=args.cv_folds,
            scoring=scorer,
            n_jobs=args.n_jobs,
            verbose=2,
            refit=True,
            random_state=42,
        )
    
    # Run search
    print(f"\nStarting search... (this may take a while)")
    search.fit(batch.x, y_target)
    
    # Results
    print(f"\n{'='*70}")
    print("TUNING RESULTS")
    print(f"{'='*70}")
    
    print(f"\nBest CV Score (negative MSE): {search.best_score_:.4f}")
    print(f"Best RMSE: {np.sqrt(-search.best_score_):.4f}")
    
    print(f"\nBest Hyperparameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Show top 5 configurations
    print(f"\nTop 5 Configurations:")
    results = search.cv_results_
    indices = np.argsort(results['mean_test_score'])[-5:][::-1]
    
    for i, idx in enumerate(indices, 1):
        score = results['mean_test_score'][idx]
        params = results['params'][idx]
        print(f"\n  Rank {i}: RMSE = {np.sqrt(-score):.4f}")
        print(f"    {params}")
    
    # Save best config
    print(f"\n{'='*70}")
    print("SAVING TUNED CONFIG")
    print(f"{'='*70}")
    
    # Update config with best params
    cfg['model'].update(search.best_params_)
    cfg['model']['model_type'] = 'xgboost'
    
    # Add tuning metadata
    cfg['tuning'] = {
        'method': args.method,
        'cv_folds': args.cv_folds,
        'best_cv_rmse': float(np.sqrt(-search.best_score_)),
        'search_space_size': args.n_iter if args.method == 'random' else n_combinations,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    
    print(f"\nTuned config saved to: {output_path}")
    print(f"\nTo train with tuned hyperparameters:")
    print(f"  python train_margin.py \\")
    print(f"    --data {args.data} \\")
    print(f"    --config {output_path} \\")
    print(f"    --model-type xgboost \\")
    print(f"    --output artifacts/margin_xgboost_tuned")
    
    print(f"\n{'='*70}")
    print("TUNING COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

