"""Quick script to check what seasons are in training vs test data."""

import json
from datetime import datetime

def analyze_data(filepath):
    """Analyze seasons in a JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    
    dates = [g['date'] for g in data]
    seasons = sorted(set(g['season'] for g in data))
    years = sorted(set(d[:4] for d in dates))
    
    return {
        'filepath': filepath,
        'total_games': len(data),
        'date_range': f"{min(dates)} to {max(dates)}",
        'calendar_years': years,
        'nba_seasons': seasons
    }

print("="*70)
print("DATA SPLIT ANALYSIS")
print("="*70)

train_info = analyze_data('data/games_train_with_players_90_norm.jsonl')
print(f"\nTRAINING DATA:")
print(f"  File: {train_info['filepath']}")
print(f"  Total games: {train_info['total_games']}")
print(f"  Date range: {train_info['date_range']}")
print(f"  NBA Seasons: {', '.join(train_info['nba_seasons'])}")

test_info = analyze_data('data/games_predict_2024_2025_with_players_norm.jsonl')
print(f"\nTEST/PREDICTION DATA:")
print(f"  File: {test_info['filepath']}")
print(f"  Total games: {test_info['total_games']}")
print(f"  Date range: {test_info['date_range']}")
print(f"  NBA Seasons: {', '.join(test_info['nba_seasons'])}")

print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Model trained on: {', '.join(train_info['nba_seasons'])}")
print(f"Model tested on: {', '.join(test_info['nba_seasons'])}")
print(f"Out-of-sample test: {'Yes ✓' if not set(test_info['nba_seasons']).intersection(set(train_info['nba_seasons'])) else 'No (overlap detected)'}")

