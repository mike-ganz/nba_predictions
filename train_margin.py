"""Training script for direct margin prediction model.

Usage:
    python train_margin.py --data data/games_train_with_players_90.jsonl --output artifacts/margin_test
    python train_margin.py --data data/games_train_with_players_90.jsonl --output artifacts/margin_xgboost --model-type xgboost
"""

import argparse
import yaml
from pathlib import Path
import joblib
import numpy as np

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from models.margin_normal import MarginNormalModel, MarginNormalConfig
from models.margin_xgboost import MarginXGBoostModel, MarginXGBoostConfig
from models.margin_distribution import margin_cover_probability, margin_win_probability


def parse_args():
    parser = argparse.ArgumentParser(description="Train direct margin prediction model")
    parser.add_argument("--data", type=str, required=True, help="Training JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model artifacts")
    parser.add_argument("--config", type=str, default="configs/margin_default.yaml", help="Config file")
    parser.add_argument("--model-type", type=str, choices=["ridge", "xgboost"], default=None,
                        help="Model type (overrides config file if specified)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Determine model type (CLI overrides config)
    model_type = args.model_type
    if model_type is None:
        model_type = cfg.get('model', {}).get('model_type', 'ridge')
    
    print(f"Model type: {model_type}")
    
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
    
    # Optionally exclude difference features
    # Default: True for Ridge (linear model needs interactions), False for XGBoost (trees learn them)
    default_include_diff = (model_type == 'ridge')
    include_diff_features = cfg.get('model', {}).get('include_diff_features', default_include_diff)
    
    if not include_diff_features:
        if model_type == 'xgboost':
            print(f"  Excluding difference features (XGBoost learns interactions automatically)")
        else:
            print(f"  Excluding difference features (as specified in config)")
    
    # Optionally include league context features (volatility metrics)
    include_context = cfg.get('model', {}).get('include_context', False)
    if include_context:
        print(f"  Including league context features (ctx_std_oeff, ctx_std_deff, ctx_std_pace, ctx_std_orb)")
    
    dataset = MarginTrainingDataset(
        records,
        exclude_features=exclude_features,
        include_diff_features=include_diff_features,
        include_context=include_context,
    )
    batch = dataset.build()
    print(f"  Feature dimensionality: {batch.x.shape[1]}")
    print(f"  Actual margin range: {batch.y_margin.min():.1f} to {batch.y_margin.max():.1f} (mean: {batch.y_margin.mean():.2f})")
    print(f"  Baseline margin range: {batch.baseline_margin.min():.1f} to {batch.baseline_margin.max():.1f} (mean: {batch.baseline_margin.mean():.2f})")
    
    # Optional: sample weighting around the spread (for XGBoost only)
    use_spread_weighting = bool(cfg.get('model', {}).get('use_spread_weighting', False))
    spread_weight_sigma = float(cfg.get('model', {}).get('spread_weight_sigma', 6.0))
    sample_weight = None
    if use_spread_weighting and model_type == 'xgboost':
        # Distance from the ATS decision boundary (home margin vs spread)
        # delta = |actual_margin + spread_home|
        delta = np.abs(batch.y_margin + batch.market_spread_home)
        # Higher weight for games closer to the number. Gaussian-shaped weighting:
        # w = exp(-(delta / sigma)^2), then min-floor at 0.2 to avoid zeroing games.
        raw_w = np.exp(-(delta / spread_weight_sigma) ** 2)
        sample_weight = 0.2 + 0.8 * raw_w

        print("\nUsing spread-based sample weighting:")
        print(f"  spread_weight_sigma = {spread_weight_sigma:.2f}")
        print(f"  weight range: {sample_weight.min():.3f} to {sample_weight.max():.3f}")
    
    # Train model based on type
    print("\n" + "="*70)
    print(f"Training {model_type.upper()} Margin Model")
    print("="*70)
    
    # Remove dataset- and training-specific keys from model config
    model_cfg = {
        k: v
        for k, v in cfg.get('model', {}).items()
        if k not in [
            'exclude_features',
            'model_type',
            'include_diff_features',
            'include_context',
            'use_spread_weighting',
            'spread_weight_sigma',
        ]
    }
    
    if model_type == 'ridge':
        model_config = MarginNormalConfig(**model_cfg)
        model = MarginNormalModel(model_config)
        model.fit(batch.x, batch.y_margin, batch.baseline_margin)
        
        if model.config.use_cv:
            print(f"  Mean model best alpha: {model.model_mean.alpha_:.4f}")
            print(f"  Variance model best alpha: {model.model_variance.alpha_:.4f}")
    
    elif model_type == 'xgboost':
        model_config = MarginXGBoostConfig(**model_cfg)
        model = MarginXGBoostModel(model_config)
        model.fit(
            batch.x,
            batch.y_margin,
            batch.baseline_margin,
            batch.market_spread_home,
            sample_weight=sample_weight,
        )
        
        print(f"  Trees built: {model.config.n_estimators}")
        print(f"  Max depth: {model.config.max_depth}")
        print(f"  Learning rate: {model.config.learning_rate}")
        if model.config.use_ats_objective:
            print(f"  Using ATS-focused objective (penalty weight: {model.config.ats_penalty_weight})")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate on training set
    print("\n" + "="*70)
    print("Training Set Evaluation")
    print("="*70)
    
    # Get predictions (handle different return types)
    if model_type == 'ridge':
        mu_pred, sigma_pred = model.predict(batch.x, batch.baseline_margin)
    else:  # xgboost
        mu_pred = model.predict(batch.x, batch.baseline_margin)
        sigma_pred = None
    
    # Margin error
    margin_errors = np.abs(mu_pred - batch.y_margin)
    margin_mae = margin_errors.mean()
    margin_rmse = np.sqrt(((mu_pred - batch.y_margin) ** 2).mean())
    print(f"\nMargin Prediction Accuracy:")
    print(f"  MAE:  {margin_mae:.2f} points")
    print(f"  RMSE: {margin_rmse:.2f} points")
    
    # ATS and ML accuracy (model-agnostic)
    actual_home_covers = (batch.y_margin > -batch.market_spread_home).astype(float)
    pred_home_covers = (mu_pred > -batch.market_spread_home).astype(float)
    ats_accuracy = (pred_home_covers == actual_home_covers).mean()
    
    actual_home_win = (batch.y_margin > 0).astype(float)
    pred_home_win = (mu_pred > 0).astype(float)
    ml_accuracy = (pred_home_win == actual_home_win).mean()
    
    print(f"\nATS Performance:")
    print(f"  Accuracy:    {ats_accuracy * 100:.2f}%")
    print(f"  ML Accuracy: {ml_accuracy * 100:.2f}%")
    
    # Ridge-specific metrics (cover probabilities, uncertainty)
    if model_type == 'ridge' and sigma_pred is not None:
        prob_home_covers, _ = margin_cover_probability(mu_pred, sigma_pred, batch.market_spread_home)
        brier_score = ((prob_home_covers - actual_home_covers) ** 2).mean()
        print(f"  Brier Score: {brier_score:.4f}")
        
        # Sigma distribution
        print(f"\nUncertainty (sigma) Statistics:")
        print(f"  Min:    {sigma_pred.min():.2f}")
        print(f"  25%:    {np.percentile(sigma_pred, 25):.2f}")
        print(f"  Median: {np.median(sigma_pred):.2f}")
        print(f"  75%:    {np.percentile(sigma_pred, 75):.2f}")
        print(f"  Max:    {sigma_pred.max():.2f}")
        
        # Confidence buckets
        print(f"\nCalibration by Confidence Level:")
        confidence = np.maximum(prob_home_covers, 1 - prob_home_covers)
        buckets = [(0.5, 0.55, "50-55%"), (0.55, 0.60, "55-60%"), (0.60, 0.65, "60-65%"), 
                   (0.65, 0.70, "65-70%"), (0.70, 1.00, "70%+")]
        
        for low, high, label in buckets:
            mask = (confidence >= low) & (confidence < high)
            if mask.sum() > 0:
                acc = ((prob_home_covers[mask] > 0.5) == actual_home_covers[mask]).mean()
                print(f"  {label}: {mask.sum():4d} games, {acc*100:.1f}% accuracy")
    
    # XGBoost-specific metrics (feature importance)
    if model_type == 'xgboost':
        print(f"\nTop 10 Most Important Features (by gain):")
        importance_dict = model.get_feature_importance(importance_type='gain')
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, score) in enumerate(sorted_importance[:10], 1):
            print(f"  {i}. {feat}: {score:.2f}")
    
    # Save model
    print("\n" + "="*70)
    model_path = output_dir / "margin_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save config (update with actual model type used)
    cfg['model']['model_type'] = model_type
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f"Config saved to {config_path}")
    
    # Save metadata for prediction script
    metadata = {
        'model_type': model_type,
        'training_data': str(args.data),
        'feature_count': batch.x.shape[1],
        'game_count': len(records),
        'exclude_features': exclude_features,
        'include_diff_features': include_diff_features,
        'include_context': include_context,
    }
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

