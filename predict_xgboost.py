"""Prediction script for XGBoost spread coverage model.

Usage:
    python predict_xgboost.py --data data/games_val.jsonl --model artifacts/xgboost_coverage --output predictions.csv
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Predict using XGBoost model")
    parser.add_argument("--data", type=str, required=True, help="Games JSONL to predict")
    parser.add_argument("--model", type=str, required=True, help="Model artifact directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model_path = Path(args.model) / "xgboost_model.joblib"
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load data
    print(f"Loading games from {args.data}")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"Loaded {len(records)} games")
    
    # Build features (no exclusions - XGBoost uses all features)
    print("Building features...")
    dataset = MarginTrainingDataset(records, exclude_features=[])
    batch = dataset.build()
    
    # Predict
    print("Generating predictions...")
    prob_home_covers = model.predict_proba(batch.x)[:, 1]
    prob_away_covers = 1 - prob_home_covers
    
    # Build output dataframe
    results = []
    for i, record in enumerate(records):
        row = {
            'game_id': record.game_id,
            'date': record.date,
            'away_team': record.teams.A.team_id,
            'home_team': record.teams.H.team_id,
            'market_spread_home': batch.market_spread_home[i],
            'cover_prob_home': prob_home_covers[i],
            'cover_prob_away': prob_away_covers[i],
        }
        
        # Add actuals if available
        if record.outcome:
            row['actual_home'] = record.outcome.home_final
            row['actual_away'] = record.outcome.away_final
            row['actual_margin'] = batch.actual_margin[i]
            row['actual_home_covers'] = int(batch.y_home_covers[i]) if not np.isnan(batch.y_home_covers[i]) else None
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    
    print(f"\nSummary Statistics:")
    print(f"  Mean home cover prob:  {prob_home_covers.mean():.3f}")
    print(f"  Median home cover prob: {np.median(prob_home_covers):.3f}")
    print(f"  Probability range:     {prob_home_covers.min():.3f} to {prob_home_covers.max():.3f}")
    
    # Home/Away distribution
    home_picks = (prob_home_covers > 0.5).sum()
    away_picks = (prob_home_covers < 0.5).sum()
    print(f"\nPrediction Distribution:")
    print(f"  Predicts HOME covers: {home_picks} games ({home_picks/len(prob_home_covers)*100:.1f}%)")
    print(f"  Predicts AWAY covers: {away_picks} games ({away_picks/len(prob_home_covers)*100:.1f}%)")
    
    # Confidence distribution
    confidence = np.maximum(prob_home_covers, 1 - prob_home_covers)
    print(f"\nConfidence Distribution:")
    print(f"  50-55%: {((confidence >= 0.50) & (confidence < 0.55)).sum()} games")
    print(f"  55-60%: {((confidence >= 0.55) & (confidence < 0.60)).sum()} games")
    print(f"  60-65%: {((confidence >= 0.60) & (confidence < 0.65)).sum()} games")
    print(f"  65-70%: {((confidence >= 0.65) & (confidence < 0.70)).sum()} games")
    print(f"  70%+:   {(confidence >= 0.70).sum()} games")
    
    # If we have actuals, show quick accuracy
    if 'actual_home_covers' in df.columns:
        df_with_actuals = df[df['actual_home_covers'].notna()]
        if len(df_with_actuals) > 0:
            pred_covers = (df_with_actuals['cover_prob_home'] > 0.5).astype(int)
            actual_covers = df_with_actuals['actual_home_covers'].astype(int)
            accuracy = (pred_covers == actual_covers).mean()
            print(f"\nQuick Accuracy Check ({len(df_with_actuals)} games with actuals):")
            print(f"  ATS Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

