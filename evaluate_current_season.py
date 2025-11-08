"""
Evaluate model performance on current season (2025-2026) completed games.

This script:
1. Loads predictions for the current season
2. Filters to only completed games (those with actual results)
3. Calculates ATS accuracy and ROI for overall and key subsets
4. Generates a detailed performance report
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


def load_predictions_with_actuals(predictions_csv: Path, games_jsonl: Path) -> pd.DataFrame:
    """
    Load predictions and merge with actual results from the JSONL file.
    
    Args:
        predictions_csv: Path to predictions CSV
        games_jsonl: Path to games JSONL file with actual scores
        
    Returns:
        DataFrame with predictions and actual results
    """
    # Load predictions
    preds = pd.read_csv(predictions_csv)
    
    # Load games to get actual scores
    games = []
    with open(games_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            # Check if outcome exists
            outcome = game.get('outcome', {})
            games.append({
                'game_id': game.get('game_id'),
                'date': game.get('date'),
                'home_team': game['teams']['H'].get('team_id'),
                'away_team': game['teams']['A'].get('team_id'),
                'actual_home_score': outcome.get('home_final'),
                'actual_away_score': outcome.get('away_final'),
            })
    
    games_df = pd.DataFrame(games)
    
    # Filter to only completed games (those with actual scores)
    games_df = games_df[
        games_df['actual_home_score'].notna() & 
        games_df['actual_away_score'].notna()
    ].copy()
    
    # Calculate actual margin
    games_df['actual_margin'] = games_df['actual_home_score'] - games_df['actual_away_score']
    
    # Merge with predictions (only merge on game_id to avoid date conflicts)
    merged = preds.merge(
        games_df[['game_id', 'actual_margin', 'actual_home_score', 'actual_away_score']], 
        on='game_id', 
        how='inner',
        suffixes=('', '_actual')
    )
    
    return merged


def calculate_ats_metrics(df: pd.DataFrame, segment_name: str = "Overall") -> Dict:
    """
    Calculate ATS accuracy and related metrics for a segment.
    
    Args:
        df: DataFrame with predictions and actuals
        segment_name: Name of the segment for reporting
        
    Returns:
        Dictionary with metrics
    """
    if len(df) == 0:
        return {
            'segment': segment_name,
            'n_games': 0,
            'ats_accuracy': 0.0,
            'roi': 0.0,
            'avg_error': 0.0,
            'median_error': 0.0,
        }
    
    # Calculate home covers (home beats the spread)
    # Home covers if: actual_margin > -market_spread_home
    df = df.copy()
    
    # Use the correct column name (market_spread_home from predictions)
    spread_col = 'market_spread_home' if 'market_spread_home' in df.columns else 'market_spread'
    
    # Rename predicted_margin if it's named differently
    if 'pred_margin_mu' in df.columns and 'predicted_margin' not in df.columns:
        df['predicted_margin'] = df['pred_margin_mu']
    
    df['actual_home_covers'] = (df['actual_margin'] > -df[spread_col]).astype(int)
    df['predicted_home_covers'] = (df['predicted_margin'] > -df[spread_col]).astype(int)
    
    # ATS accuracy
    ats_accuracy = (df['actual_home_covers'] == df['predicted_home_covers']).mean() * 100
    
    # Calculate error metrics
    prediction_error = df['predicted_margin'] - df['actual_margin']
    avg_error = prediction_error.abs().mean()
    median_error = prediction_error.abs().median()
    
    # Calculate ROI (assuming -110 odds on all bets)
    # Win: +0.909 units (bet 1.1 to win 1)
    # Loss: -1.1 units
    correct_predictions = (df['actual_home_covers'] == df['predicted_home_covers']).sum()
    incorrect_predictions = len(df) - correct_predictions
    
    profit = correct_predictions * 0.909 - incorrect_predictions * 1.1
    total_wagered = len(df) * 1.1
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0.0
    
    return {
        'segment': segment_name,
        'n_games': len(df),
        'ats_accuracy': ats_accuracy,
        'roi': roi,
        'avg_error': avg_error,
        'median_error': median_error,
        'profit': profit,
        'total_wagered': total_wagered,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate current season predictions")
    parser.add_argument(
        '--predictions',
        type=str,
        default='predictions/current_season_2025_2026_predictions.csv',
        help='Path to predictions CSV'
    )
    parser.add_argument(
        '--games',
        type=str,
        default='data/games_2025_2026_current_norm.jsonl',
        help='Path to games JSONL file with actual results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions/current_season_evaluation.txt',
        help='Output report file'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("CURRENT SEASON (2025-2026) MODEL EVALUATION")
    print("=" * 70)
    print()
    
    # Load predictions with actuals
    print(f"Loading predictions from: {args.predictions}")
    print(f"Loading games from: {args.games}")
    
    try:
        df = load_predictions_with_actuals(Path(args.predictions), Path(args.games))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure predictions have been generated first.")
        return
    
    print(f"OK - Loaded {len(df)} completed games with predictions\n")
    
    if len(df) == 0:
        print("⚠️  No completed games found. Check back after some games are played!")
        return
    
    # Overall metrics
    print("=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)
    
    overall = calculate_ats_metrics(df, "Overall")
    print(f"Games Evaluated: {overall['n_games']}")
    print(f"ATS Accuracy: {overall['ats_accuracy']:.2f}%")
    print(f"ROI: {overall['roi']:.2f}%")
    print(f"Profit/Loss: ${overall['profit']:.2f} (on ${overall['total_wagered']:.2f} wagered)")
    print(f"Avg Prediction Error: {overall['avg_error']:.2f} points")
    print(f"Median Prediction Error: {overall['median_error']:.2f} points")
    print()
    
    # Key subsets based on our identified strategies
    print("=" * 70)
    print("KEY STRATEGY SEGMENTS")
    print("=" * 70)
    print()
    
    # Use the correct column name
    spread_col = 'market_spread_home' if 'market_spread_home' in df.columns else 'market_spread'
    
    # Determine favorite type
    df['home_favored'] = df[spread_col] < 0
    df['away_favored'] = df[spread_col] > 0
    
    # Spread size buckets
    df['spread_abs'] = df[spread_col].abs()
    df['spread_small'] = df['spread_abs'] <= 3.5
    df['spread_medium'] = (df['spread_abs'] > 3.5) & (df['spread_abs'] <= 7.5)
    df['spread_large'] = df['spread_abs'] > 7.5
    
    # Evaluate segments
    segments = [
        ("Home Favored", df[df['home_favored']]),
        ("Away Favored", df[df['away_favored']]),
        ("Away Fav + Small Spread", df[df['away_favored'] & df['spread_small']]),
        ("Away Fav + Medium Spread", df[df['away_favored'] & df['spread_medium']]),
        ("Away Fav + Large Spread", df[df['away_favored'] & df['spread_large']]),
    ]
    
    results = []
    for seg_name, seg_df in segments:
        metrics = calculate_ats_metrics(seg_df, seg_name)
        results.append(metrics)
        
        print(f"{seg_name}:")
        print(f"  Games: {metrics['n_games']}")
        print(f"  ATS Accuracy: {metrics['ats_accuracy']:.2f}%")
        print(f"  ROI: {metrics['roi']:.2f}%")
        print(f"  Profit/Loss: ${metrics['profit']:.2f}")
        print()
    
    # Save detailed report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CURRENT SEASON (2025-2026) MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: models/margin_normalized/ (29 features)\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Games Evaluated: {overall['n_games']}\n")
        f.write(f"ATS Accuracy: {overall['ats_accuracy']:.2f}%\n")
        f.write(f"ROI: {overall['roi']:.2f}%\n")
        f.write(f"Profit/Loss: ${overall['profit']:.2f} (on ${overall['total_wagered']:.2f} wagered)\n")
        f.write(f"Avg Prediction Error: {overall['avg_error']:.2f} points\n")
        f.write(f"Median Prediction Error: {overall['median_error']:.2f} points\n\n")
        
        f.write("KEY STRATEGY SEGMENTS\n")
        f.write("-" * 70 + "\n")
        for metrics in results:
            f.write(f"\n{metrics['segment']}:\n")
            f.write(f"  Games: {metrics['n_games']}\n")
            f.write(f"  ATS Accuracy: {metrics['ats_accuracy']:.2f}%\n")
            f.write(f"  ROI: {metrics['roi']:.2f}%\n")
            f.write(f"  Profit/Loss: ${metrics['profit']:.2f}\n")
    
    print("=" * 70)
    print(f"OK - Detailed report saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

