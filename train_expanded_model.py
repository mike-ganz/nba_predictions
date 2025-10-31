"""
Train a new model on expanded dataset (2021-2025) and save to separate directory.
This does NOT overwrite the existing model.
"""
import argparse
import json
from pathlib import Path
import yaml
import joblib

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from models.margin_normal import MarginNormalModel, MarginNormalConfig


def main():
    print("="*70)
    print("TRAINING EXPANDED MODEL (2021-2025)")
    print("="*70)
    print()
    
    # Load config from existing model
    config_path = Path("artifacts/margin_normalized/config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    print("Loading expanded training data...")
    print("  - games_train_with_players_90_norm.jsonl (21-24 train)")
    print("  - games_val_with_players_norm.jsonl (21-24 val)")
    print("  - games_predict_2024_2025_with_players_norm.jsonl (24-25)")
    print()
    
    # Load all three datasets and combine
    loader = GameDataLoader(Path("data"))
    
    train_records = loader.read_games("games_train_with_players_90_norm.jsonl").games
    print(f"Loaded {len(train_records)} games from training set")
    
    val_records = loader.read_games("games_val_with_players_norm.jsonl").games
    print(f"Loaded {len(val_records)} games from validation set")
    
    pred_2425_records = loader.read_games("games_predict_2024_2025_with_players_norm.jsonl").games
    print(f"Loaded {len(pred_2425_records)} games from 24-25 season")
    
    # Combine all records
    all_records = train_records + val_records + pred_2425_records
    print(f"\nTotal training games: {len(all_records)}")
    print()
    
    # Build dataset
    print("Building features...")
    exclude_features = cfg.get('model', {}).get('exclude_features', [])
    if exclude_features:
        print(f"  Excluding {len(exclude_features)} features: {exclude_features}")
    
    dataset = MarginTrainingDataset(all_records, exclude_features=exclude_features)
    batch = dataset.build()
    
    print(f"  Feature matrix shape: {batch.x.shape}")
    print(f"  Number of games: {len(batch.y_margin)}")
    print()
    
    # Train model
    print("Training model...")
    model_cfg = {k: v for k, v in cfg.get('model', {}).items() if k != 'exclude_features'}
    model_config = MarginNormalConfig(**model_cfg)
    model = MarginNormalModel(model_config)
    model.fit(batch.x, batch.y_margin, batch.baseline_margin)
    print("  ✅ Training complete!")
    print()
    
    # Save to new directory
    output_dir = Path("artifacts/margin_normalized_21_25")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "margin_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save config
    config_out_path = output_dir / "config.yaml"
    with open(config_out_path, 'w') as f:
        yaml.dump(cfg, f)
    print(f"✅ Config saved to: {config_out_path}")
    
    # Save metadata
    metadata = {
        'training_data': [
            'games_train_with_players_90_norm.jsonl',
            'games_val_with_players_norm.jsonl',
            'games_predict_2024_2025_with_players_norm.jsonl'
        ],
        'training_seasons': ['2021-2022', '2022-2023', '2023-2024', '2024-2025'],
        'total_games': len(all_records),
        'num_features': batch.x.shape[1],
        'excluded_features': exclude_features,
    }
    
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print(f"New model saved to: {output_dir}")
    print("Original model preserved at: artifacts/margin_normalized")
    print("="*70)


if __name__ == "__main__":
    main()

