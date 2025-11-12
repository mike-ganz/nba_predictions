"""Prediction script for direct margin model.

Usage:
    python predict_margin.py --data data/games_val_with_players.jsonl --model artifacts/margin_test --output predictions.csv
    python predict_margin.py --data data/games_val_with_players.jsonl --model artifacts/margin_xgboost --output predictions.csv
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import yaml

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from training.margin_dataset_extended import MarginTrainingDatasetExtended
from models.margin_distribution import margin_cover_probability, margin_win_probability


def parse_args():
    parser = argparse.ArgumentParser(description="Predict using margin model")
    parser.add_argument("--data", type=str, required=True, help="Games JSONL to predict")
    parser.add_argument("--model", type=str, required=True, help="Model artifact directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    return parser.parse_args()


def detect_model_type(model_dir: Path) -> str:
    """Detect model type from artifact directory."""
    # Try metadata first (preferred)
    metadata_path = model_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
            if 'model_type' in metadata:
                return metadata['model_type']
    
    # Fallback to config.yaml
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            model_type = cfg.get('model', {}).get('model_type', 'ridge')
            return model_type
    
    # Default to ridge if no metadata found
    return 'ridge'


def main():
    args = parse_args()
    model_dir = Path(args.model)
    
    # Detect model type
    model_type = detect_model_type(model_dir)
    print(f"Detected model type: {model_type}")
    
    # Load model
    model_path = model_dir / "margin_model.joblib"
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load config to get exclude_features and include_diff_features
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        exclude_features = cfg.get('model', {}).get('exclude_features', [])
        include_diff_features = cfg.get('model', {}).get('include_diff_features', True)
        include_fav_underdog_features = cfg.get('model', {}).get('include_fav_underdog_features', False)
    else:
        exclude_features = []
        include_diff_features = True
        include_fav_underdog_features = False
    
    if exclude_features:
        print(f"Excluding {len(exclude_features)} features: {exclude_features}")
    
    if not include_diff_features:
        print(f"Excluding difference features (tree-based model)")
    
    if include_fav_underdog_features:
        print(f"Including favorite/underdog indicators (extended features)")
    
    # Load data
    print(f"Loading games from {args.data}")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"Loaded {len(records)} games")
    
    # Build features (use extended dataset if needed)
    print("Building features...")
    if include_fav_underdog_features or 'home_ftr' not in exclude_features:
        # Use extended dataset for experimental models
        dataset = MarginTrainingDatasetExtended(
            records, 
            exclude_features=exclude_features,
            include_diff_features=include_diff_features,
            include_fav_underdog_features=include_fav_underdog_features
        )
    else:
        # Use standard dataset for baseline models
        dataset = MarginTrainingDataset(
            records, 
            exclude_features=exclude_features,
            include_diff_features=include_diff_features
        )
    batch = dataset.build()
    
    # Predict (handle different model types)
    print("Generating predictions...")
    if model_type == 'ridge':
        mu, sigma = model.predict(batch.x, batch.baseline_margin)
        prob_home_covers, prob_away_covers = margin_cover_probability(mu, sigma, batch.market_spread_home)
        prob_home_win, prob_away_win = margin_win_probability(mu, sigma)
    else:  # xgboost
        mu = model.predict(batch.x, batch.baseline_margin)
        sigma = None
        prob_home_covers = None
        prob_away_covers = None
        prob_home_win = None
        prob_away_win = None
    
    # Build output dataframe
    results = []
    for i, record in enumerate(records):
        row = {
            'game_id': record.game_id,
            'date': record.date,
            'game_time': record.market.game_time if record.market.game_time else None,
            'away_team': record.teams.A.team_id,
            'home_team': record.teams.H.team_id,
            'market_spread_home': batch.market_spread_home[i],
            'baseline_margin': batch.baseline_margin[i],
            'pred_margin_mu': mu[i],
        }
        
        # Add model-specific columns
        if model_type == 'ridge' and sigma is not None:
            row['pred_margin_sigma'] = sigma[i]
            row['cover_prob_home'] = prob_home_covers[i]
            row['cover_prob_away'] = prob_away_covers[i]
            row['win_prob_home'] = prob_home_win[i]
            row['win_prob_away'] = prob_away_win[i]
        
        # Add actuals if available
        if record.outcome:
            row['actual_home'] = record.outcome.home_final
            row['actual_away'] = record.outcome.away_final
            row['actual_margin'] = batch.y_margin[i]
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    
    print(f"\nSummary Statistics:")
    print(f"  Mean predicted margin: {mu.mean():.2f} (std: {mu.std():.2f})")
    
    if model_type == 'ridge' and sigma is not None:
        print(f"  Mean predicted sigma:  {sigma.mean():.2f} (range: {sigma.min():.2f} to {sigma.max():.2f})")
        print(f"  Mean home cover prob:  {prob_home_covers.mean():.3f}")
        print(f"  Mean home win prob:    {prob_home_win.mean():.3f}")
    
    # If we have actuals, show quick accuracy
    if 'actual_margin' in df.columns:
        df_with_actuals = df[df['actual_margin'].notna()]
        if len(df_with_actuals) > 0:
            mae = (df_with_actuals['pred_margin_mu'] - df_with_actuals['actual_margin']).abs().mean()
            print(f"\nQuick Accuracy Check ({len(df_with_actuals)} games with actuals):")
            print(f"  Margin MAE: {mae:.2f} points")


if __name__ == "__main__":
    main()

