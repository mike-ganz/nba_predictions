"""Training script for margin prediction with extended experimental features.

This extends train_margin.py to support:
- home_ftr inclusion/exclusion
- favorite/underdog × home/away role indicators

Usage:
    python train_margin_extended.py --data data/games_train_with_players_90.jsonl \
        --output artifacts/margin_experiment1 \
        --config configs/margin_xgboost_experiment1_home_ftr.yaml
"""

import argparse
import yaml
from pathlib import Path
import joblib
import numpy as np

from data.loaders import GameDataLoader
from training.margin_dataset_extended import MarginTrainingDatasetExtended
from models.margin_xgboost import MarginXGBoostModel, MarginXGBoostConfig
from models.margin_distribution import margin_cover_probability


def parse_args():
    parser = argparse.ArgumentParser(description="Train margin prediction model with extended features")
    parser.add_argument("--data", type=str, required=True, help="Training JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model artifacts")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    print(f"\n{'='*70}")
    print(f"EXTENDED TRAINING: {Path(args.config).stem}")
    print(f"{'='*70}")
    
    print(f"\nLoading training data from {args.data}")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"Loaded {len(records)} games")
    
    # Build dataset with extended features
    print("\nBuilding training dataset with extended features...")
    exclude_features = cfg.get('model', {}).get('exclude_features', [])
    if exclude_features:
        print(f"  Excluding {len(exclude_features)} features: {exclude_features}")
    
    # Check experimental flags
    include_diff_features = cfg.get('model', {}).get('include_diff_features', False)
    include_fav_underdog_features = cfg.get('model', {}).get('include_fav_underdog_features', False)
    
    print(f"  Include difference features: {include_diff_features}")
    print(f"  Include favorite/underdog indicators: {include_fav_underdog_features}")
    
    # Key feature check
    if 'home_ftr' not in exclude_features:
        print(f"  ✓ home_ftr is INCLUDED in this experiment")
    else:
        print(f"  ✗ home_ftr is EXCLUDED in this experiment")
    
    dataset = MarginTrainingDatasetExtended(
        records, 
        exclude_features=exclude_features,
        include_diff_features=include_diff_features,
        include_fav_underdog_features=include_fav_underdog_features,
    )
    batch = dataset.build()
    print(f"  Feature dimensionality: {batch.x.shape[1]}")
    print(f"  Actual margin range: {batch.y_margin.min():.1f} to {batch.y_margin.max():.1f} (mean: {batch.y_margin.mean():.2f})")
    print(f"  Baseline margin range: {batch.baseline_margin.min():.1f} to {batch.baseline_margin.max():.1f} (mean: {batch.baseline_margin.mean():.2f})")
    
    # Train XGBoost model
    print("\n" + "="*70)
    print(f"Training XGBoost Margin Model")
    print("="*70)
    
    # Remove dataset-specific keys from model config
    model_cfg = {k: v for k, v in cfg.get('model', {}).items() 
                 if k not in ['exclude_features', 'model_type', 'include_diff_features', 'include_fav_underdog_features']}
    
    model_config = MarginXGBoostConfig(**model_cfg)
    model = MarginXGBoostModel(model_config)
    model.fit(batch.x, batch.y_margin, batch.baseline_margin, batch.market_spread_home)
    
    print(f"  Trees built: {model.config.n_estimators}")
    print(f"  Max depth: {model.config.max_depth}")
    print(f"  Learning rate: {model.config.learning_rate}")
    if model.config.use_ats_objective:
        print(f"  Using ATS-focused objective (penalty weight: {model.config.ats_penalty_weight})")
    
    # Evaluate on training set
    print("\n" + "="*70)
    print("Training Set Evaluation")
    print("="*70)
    
    mu_pred = model.predict(batch.x, batch.baseline_margin)
    
    # Margin error
    margin_errors = np.abs(mu_pred - batch.y_margin)
    margin_mae = margin_errors.mean()
    margin_rmse = np.sqrt(((mu_pred - batch.y_margin) ** 2).mean())
    print(f"\nMargin Prediction Accuracy:")
    print(f"  MAE:  {margin_mae:.2f} points")
    print(f"  RMSE: {margin_rmse:.2f} points")
    
    # ATS and ML accuracy
    actual_home_covers = (batch.y_margin > -batch.market_spread_home).astype(float)
    pred_home_covers = (mu_pred > -batch.market_spread_home).astype(float)
    ats_accuracy = (pred_home_covers == actual_home_covers).mean()
    
    actual_home_win = (batch.y_margin > 0).astype(float)
    pred_home_win = (mu_pred > 0).astype(float)
    ml_accuracy = (pred_home_win == actual_home_win).mean()
    
    print(f"\nATS Performance:")
    print(f"  Accuracy:    {ats_accuracy * 100:.2f}%")
    print(f"  ML Accuracy: {ml_accuracy * 100:.2f}%")
    
    # Feature importance
    print(f"\nTop 15 Most Important Features (by gain):")
    importance_dict = model.get_feature_importance(importance_type='gain')
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_importance[:15], 1):
        # Try to decode feature index
        if feat.startswith('f'):
            feat_idx = int(feat[1:])
            print(f"  {i:2d}. Feature {feat_idx:2d}: {score:.2f}")
        else:
            print(f"  {i:2d}. {feat}: {score:.2f}")
    
    # Save model
    print("\n" + "="*70)
    model_path = output_dir / "margin_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save config (update with actual settings used)
    cfg['model']['model_type'] = 'xgboost'
    cfg['model']['include_fav_underdog_features'] = include_fav_underdog_features
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f"Config saved to {config_path}")
    
    # Save metadata for prediction script
    metadata = {
        'model_type': 'xgboost',
        'training_data': str(args.data),
        'feature_count': batch.x.shape[1],
        'game_count': len(records),
        'exclude_features': exclude_features,
        'include_fav_underdog_features': include_fav_underdog_features,
        'includes_home_ftr': 'home_ftr' not in exclude_features,
        'training_ats_accuracy': float(ats_accuracy),
        'training_margin_mae': float(margin_mae),
    }
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nExperiment Summary:")
    print(f"  home_ftr included: {'YES' if metadata['includes_home_ftr'] else 'NO'}")
    print(f"  Fav/underdog features: {'YES' if include_fav_underdog_features else 'NO'}")
    print(f"  Training ATS: {ats_accuracy * 100:.2f}%")
    print(f"  Training MAE: {margin_mae:.2f} points")


if __name__ == "__main__":
    main()

