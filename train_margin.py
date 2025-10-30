"""Training script for direct margin prediction model.

Usage:
    python train_margin.py --data data/games_train_with_players_90.jsonl --output artifacts/margin_test
"""

import argparse
import yaml
from pathlib import Path
import joblib
import numpy as np

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from models.margin_normal import MarginNormalModel, MarginNormalConfig
from models.margin_distribution import margin_cover_probability, margin_win_probability


def parse_args():
    parser = argparse.ArgumentParser(description="Train direct margin prediction model")
    parser.add_argument("--data", type=str, required=True, help="Training JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model artifacts")
    parser.add_argument("--config", type=str, default="configs/margin_default.yaml", help="Config file")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    print(f"\nLoading training data from {args.data}")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"Loaded {len(records)} games")
    
    # Build dataset
    print("\nBuilding training dataset...")
    dataset = MarginTrainingDataset(records)
    batch = dataset.build()
    print(f"  Feature dimensionality: {batch.x.shape[1]}")
    print(f"  Actual margin range: {batch.y_margin.min():.1f} to {batch.y_margin.max():.1f} (mean: {batch.y_margin.mean():.2f})")
    print(f"  Baseline margin range: {batch.baseline_margin.min():.1f} to {batch.baseline_margin.max():.1f} (mean: {batch.baseline_margin.mean():.2f})")
    
    # Train model
    print("\n" + "="*70)
    print("Training Margin Model")
    print("="*70)
    model_config = MarginNormalConfig(**cfg.get('model', {}))
    model = MarginNormalModel(model_config)
    model.fit(batch.x, batch.y_margin, batch.baseline_margin)
    
    if model.config.use_cv:
        print(f"  Mean model best alpha: {model.model_mean.alpha_:.4f}")
        print(f"  Variance model best alpha: {model.model_variance.alpha_:.4f}")
    
    # Evaluate on training set
    print("\n" + "="*70)
    print("Training Set Evaluation")
    print("="*70)
    mu_pred, sigma_pred = model.predict(batch.x, batch.baseline_margin)
    
    # Margin error
    margin_errors = np.abs(mu_pred - batch.y_margin)
    margin_mae = margin_errors.mean()
    margin_rmse = np.sqrt(((mu_pred - batch.y_margin) ** 2).mean())
    print(f"\nMargin Prediction Accuracy:")
    print(f"  MAE:  {margin_mae:.2f} points")
    print(f"  RMSE: {margin_rmse:.2f} points")
    
    # Check calibration
    prob_home_covers, _ = margin_cover_probability(mu_pred, sigma_pred, batch.market_spread_home)
    actual_home_covers = (batch.y_margin > -batch.market_spread_home).astype(float)
    brier_score = ((prob_home_covers - actual_home_covers) ** 2).mean()
    ats_accuracy = ((prob_home_covers > 0.5) == actual_home_covers).mean()
    
    print(f"\nATS Performance:")
    print(f"  Accuracy:    {ats_accuracy * 100:.2f}%")
    print(f"  Brier Score: {brier_score:.4f}")
    
    # Moneyline
    prob_home_win, _ = margin_win_probability(mu_pred, sigma_pred)
    actual_home_win = (batch.y_margin > 0).astype(float)
    ml_accuracy = ((prob_home_win > 0.5) == actual_home_win).mean()
    print(f"  ML Accuracy: {ml_accuracy * 100:.2f}%")
    
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
    
    # Save model
    print("\n" + "="*70)
    model_path = output_dir / "margin_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f"Config saved to {config_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

