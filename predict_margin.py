"""Prediction script for direct margin model.

Usage:
    python predict_margin.py --data data/games_val_with_players.jsonl --model artifacts/margin_test --output predictions.csv
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from models.margin_distribution import margin_cover_probability, margin_win_probability


def parse_args():
    parser = argparse.ArgumentParser(description="Predict using margin model")
    parser.add_argument("--data", type=str, required=True, help="Games JSONL to predict")
    parser.add_argument("--model", type=str, required=True, help="Model artifact directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model_path = Path(args.model) / "margin_model.joblib"
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load config to get exclude_features
    config_path = Path(args.model) / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        exclude_features = cfg.get('model', {}).get('exclude_features', [])
    else:
        exclude_features = []
    
    if exclude_features:
        print(f"Excluding {len(exclude_features)} features: {exclude_features}")
    
    # Load data
    print(f"Loading games from {args.data}")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"Loaded {len(records)} games")
    
    # Build features
    print("Building features...")
    dataset = MarginTrainingDataset(records, exclude_features=exclude_features)
    batch = dataset.build()
    
    # Predict
    print("Generating predictions...")
    mu, sigma = model.predict(batch.x, batch.baseline_margin)
    prob_home_covers, prob_away_covers = margin_cover_probability(mu, sigma, batch.market_spread_home)
    prob_home_win, prob_away_win = margin_win_probability(mu, sigma)
    
    # Build output dataframe
    results = []
    for i, record in enumerate(records):
        row = {
            'game_id': record.game_id,
            'date': record.date,
            'away_team': record.teams.A.team_id,
            'home_team': record.teams.H.team_id,
            'market_spread_home': batch.market_spread_home[i],
            'baseline_margin': batch.baseline_margin[i],
            'pred_margin_mu': mu[i],
            'pred_margin_sigma': sigma[i],
            'cover_prob_home': prob_home_covers[i],
            'cover_prob_away': prob_away_covers[i],
            'win_prob_home': prob_home_win[i],
            'win_prob_away': prob_away_win[i],
        }
        
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

