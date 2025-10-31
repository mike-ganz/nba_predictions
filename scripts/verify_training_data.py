"""
Verify what data was actually used to train each model.
Check game dates, seasons, and any anomalies.
"""
import json
from collections import Counter
from pathlib import Path
import pandas as pd

print("="*100)
print("VERIFICATION: What data did each model actually train on?")
print("="*100)
print()

# Load all datasets
print("Loading datasets...")
print()

datasets = {
    'games_train_with_players_90_norm.jsonl': [],
    'games_val_with_players_norm.jsonl': [],
    'games_predict_2024_2025_with_players_norm.jsonl': []
}

for filename in datasets.keys():
    filepath = Path('data') / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            datasets[filename].append({
                'game_id': game['game_id'],
                'date': game.get('date', 'UNKNOWN'),
                'season': game.get('season', 'UNKNOWN')
            })
    print(f"{filename}: {len(datasets[filename])} games")

print()

# Analyze each dataset
print("="*100)
print("DATASET ANALYSIS")
print("="*100)
print()

for name, games in datasets.items():
    df = pd.DataFrame(games)
    
    print(f"\n{name}:")
    print("-"*100)
    
    # Season breakdown
    season_counts = df['season'].value_counts().sort_index()
    print(f"\nSeasons:")
    for season, count in season_counts.items():
        print(f"  {season}: {count} games")
    
    # Date range
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    valid_dates = df[df['date'].notna()]
    if len(valid_dates) > 0:
        print(f"\nDate range: {valid_dates['date'].min()} to {valid_dates['date'].max()}")
    
    print(f"Total games: {len(df)}")

print()
print("="*100)
print("OLD MODEL (21-24) - TRAINING DATA")
print("="*100)

old_model_games = datasets['games_train_with_players_90_norm.jsonl'] + datasets['games_val_with_players_norm.jsonl']
old_df = pd.DataFrame(old_model_games)
old_df['date'] = pd.to_datetime(old_df['date'], errors='coerce')

print(f"\nTotal games: {len(old_df)}")
print(f"\nSeasons:")
for season, count in old_df['season'].value_counts().sort_index().items():
    print(f"  {season}: {count} games ({count/len(old_df)*100:.1f}%)")

valid_dates = old_df[old_df['date'].notna()]
if len(valid_dates) > 0:
    print(f"\nDate range: {valid_dates['date'].min().date()} to {valid_dates['date'].max().date()}")

print()
print("="*100)
print("NEW MODEL (21-25) - TRAINING DATA")
print("="*100)

new_model_games = (datasets['games_train_with_players_90_norm.jsonl'] + 
                   datasets['games_val_with_players_norm.jsonl'] + 
                   datasets['games_predict_2024_2025_with_players_norm.jsonl'])
new_df = pd.DataFrame(new_model_games)
new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')

print(f"\nTotal games: {len(new_df)}")
print(f"\nSeasons:")
for season, count in new_df['season'].value_counts().sort_index().items():
    print(f"  {season}: {count} games ({count/len(new_df)*100:.1f}%)")

valid_dates = new_df[new_df['date'].notna()]
if len(valid_dates) > 0:
    print(f"\nDate range: {valid_dates['date'].min().date()} to {valid_dates['date'].max().date()}")

print()
print("="*100)
print("KEY COMPARISON")
print("="*100)
print()

print(f"Old Model: {len(old_df)} games")
print(f"New Model: {len(new_df)} games")
print(f"Difference: +{len(new_df) - len(old_df)} games")
print()

# Check for any overlap issues
old_game_ids = set(old_df['game_id'])
new_game_ids = set(new_df['game_id'])
test_2425_ids = set([g['game_id'] for g in datasets['games_predict_2024_2025_with_players_norm.jsonl']])

print("Checking for data leakage...")
print(f"Old model games: {len(old_game_ids)}")
print(f"New model games: {len(new_game_ids)}")
print(f"24-25 season games: {len(test_2425_ids)}")
print()

# The new model should include all old games plus the 24-25 games
expected_new = len(old_game_ids) + len(test_2425_ids)
actual_new = len(new_game_ids)
print(f"Expected new model games: {expected_new}")
print(f"Actual new model games: {actual_new}")

if expected_new != actual_new:
    print(f"⚠️  MISMATCH! Expected {expected_new} but got {actual_new}")
    overlap = old_game_ids & test_2425_ids
    if len(overlap) > 0:
        print(f"⚠️  Found {len(overlap)} overlapping games between old data and 24-25!")
else:
    print("✅ Game counts match - no unexpected overlap")

print()
print("="*100)
print("SEASON DISTRIBUTION ANALYSIS")
print("="*100)
print()

print("OLD MODEL:")
old_season_pcts = old_df['season'].value_counts(normalize=True).sort_index() * 100
for season, pct in old_season_pcts.items():
    print(f"  {season}: {pct:.1f}%")

print()
print("NEW MODEL:")
new_season_pcts = new_df['season'].value_counts(normalize=True).sort_index() * 100
for season, pct in new_season_pcts.items():
    print(f"  {season}: {pct:.1f}%")

print()
print("Change in season weights when adding 24-25:")
print("-"*100)
for season in old_season_pcts.index:
    old_pct = old_season_pcts[season]
    new_pct = new_season_pcts.get(season, 0)
    change = new_pct - old_pct
    print(f"  {season}: {old_pct:.1f}% → {new_pct:.1f}% ({change:+.1f} percentage points)")

if '2024-2025' in new_season_pcts:
    print(f"  2024-2025: 0.0% → {new_season_pcts['2024-2025']:.1f}% (+{new_season_pcts['2024-2025']:.1f} percentage points)")

print()
print("="*100)

