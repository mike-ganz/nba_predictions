"""Training script for spread coverage prediction model.

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
from models.margin_distribution import margin_cover_probability


def parse_args():
    parser = argparse.ArgumentParser(description="Train spread coverage prediction model")
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
    exclude_features = cfg.get('model', {}).get('exclude_features', [])
    if exclude_features:
        print(f"  Excluding {len(exclude_features)} features: {exclude_features}")
    dataset = MarginTrainingDataset(records, exclude_features=exclude_features)
    batch = dataset.build()
    print(f"  Feature dimensionality: {batch.x.shape[1]}")
    
    # Check for missing outcomes
    valid_mask = ~np.isnan(batch.y_home_covers)
    n_valid = valid_mask.sum()
    print(f"  Games with outcomes: {n_valid} / {len(batch.y_home_covers)}")
    
    if n_valid == 0:
        print("\nERROR: No games with valid outcomes to train on!")
        return
    
    # Filter to valid games
    x_train = batch.x[valid_mask]
    y_train = batch.y_home_covers[valid_mask].astype(int)
    spread_train = batch.market_spread_home[valid_mask]
    margin_train = batch.actual_margin[valid_mask]
    
    print(f"  Coverage rate: {y_train.mean() * 100:.2f}% (home team covers)")
    print(f"  Spread range: {spread_train.min():.1f} to {spread_train.max():.1f}")
    
    # Train model
    print("\n" + "="*70)
    print("Training Coverage Prediction Model")
    print("="*70)
    # Remove exclude_features from model config (it's for dataset, not model)
    model_cfg = {k: v for k, v in cfg.get('model', {}).items() if k != 'exclude_features'}
    model_config = MarginNormalConfig(**model_cfg)
    model = MarginNormalModel(model_config)
    model.fit(x_train, y_train)
    
    if model.config.use_cv:
        print(f"  Best C: {model.model.C_[0]:.4f}")
        # l1_ratio_ only exists and is populated for elasticnet penalty
        try:
            if model.model.l1_ratio_ is not None and len(model.model.l1_ratio_) > 0:
                print(f"  Best l1_ratio: {model.model.l1_ratio_[0]:.4f}")
            else:
                print(f"  Penalty: L2 (Ridge)")
        except (AttributeError, TypeError):
            print(f"  Penalty: L2 (Ridge)")
        print(f"  Best score (neg log loss): {model.model.scores_[1].mean(axis=0).max():.4f}")
    
    # Evaluate on training set
    print("\n" + "="*70)
    print("Training Set Evaluation")
    print("="*70)
    prob_home_covers = model.predict(x_train)
    
    # Classification metrics
    pred_home_covers = (prob_home_covers > 0.5).astype(int)
    accuracy = (pred_home_covers == y_train).mean()
    
    # Log loss (cross-entropy)
    from sklearn.metrics import log_loss, brier_score_loss
    logloss = log_loss(y_train, prob_home_covers)
    brier = brier_score_loss(y_train, prob_home_covers)
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    print(f"  Log Loss:    {logloss:.4f} (lower is better)")
    print(f"  Brier Score: {brier:.4f} (lower is better)")
    
    # ATS Performance
    ats_accuracy = accuracy  # Same as classification accuracy
    ats_roi = (accuracy * 0.91 + (1 - accuracy) * -1.0) * 100
    
    print(f"\nATS Performance:")
    print(f"  ATS Accuracy: {ats_accuracy * 100:.2f}%")
    print(f"  Breakeven:    52.38%")
    print(f"  Edge:         {(ats_accuracy - 0.5238) * 100:+.2f}%")
    print(f"  ROI per bet:  {ats_roi:+.2f}%")
    print(f"  Status:       {'[+] PROFITABLE' if ats_roi > 0 else '[-] UNPROFITABLE'}")
    
    # Calibration by confidence level
    print(f"\nCalibration by Confidence Level:")
    confidence = np.maximum(prob_home_covers, 1 - prob_home_covers)
    buckets = [(0.5, 0.55, "50-55%"), (0.55, 0.60, "55-60%"), (0.60, 0.65, "60-65%"), 
               (0.65, 0.70, "65-70%"), (0.70, 1.00, "70%+")]
    
    for low, high, label in buckets:
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() > 0:
            acc = (pred_home_covers[mask] == y_train[mask]).mean()
            roi = (acc * 0.91 + (1 - acc) * -1.0) * 100
            print(f"  {label}: {mask.sum():4d} games, {acc*100:.1f}% accuracy, {roi:+.2f}% ROI")
    
    # Probability distribution
    print(f"\nProbability Distribution:")
    print(f"  Min:    {prob_home_covers.min():.3f}")
    print(f"  25%:    {np.percentile(prob_home_covers, 25):.3f}")
    print(f"  Median: {np.median(prob_home_covers):.3f}")
    print(f"  75%:    {np.percentile(prob_home_covers, 75):.3f}")
    print(f"  Max:    {prob_home_covers.max():.3f}")
    
    # Feature importance (top coefficients)
    print(f"\nTop 10 Features by Coefficient Magnitude:")
    coeffs = model.get_coefficients()
    coeff_abs = np.abs(coeffs)
    top_indices = np.argsort(coeff_abs)[-10:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {coeffs[idx]:+.4f}")
    
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
