"""
Analyze if 24-25 season had unusual characteristics compared to prior seasons.
Look at distributions, patterns, and anomalies.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

print("="*100)
print("SEASON CHARACTERISTICS ANALYSIS")
print("Comparing 2021-2024 vs 2024-2025")
print("="*100)
print()

# Load all data
print("Loading game data...")

def load_games(filepath):
    games = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            games.append({
                'game_id': game['game_id'],
                'date': game.get('date'),
                'season': game.get('season'),
                'market_spread': game['market'].get('spread_home', 0),
                'actual_home': game.get('outcome', {}).get('home_final'),
                'actual_away': game.get('outcome', {}).get('away_final'),
            })
    return pd.DataFrame(games)

# Load historical (21-24) and current (24-25)
historical = []
for name in ['games_train_with_players_90_norm.jsonl', 'games_val_with_players_norm.jsonl']:
    historical.append(load_games(Path('data') / name))
historical_df = pd.concat(historical, ignore_index=True)

current_df = load_games(Path('data') / 'games_predict_2024_2025_with_players_norm.jsonl')

# Calculate actual margins
historical_df['actual_margin'] = historical_df['actual_home'] - historical_df['actual_away']
current_df['actual_margin'] = current_df['actual_home'] - current_df['actual_away']

# Calculate home covers
historical_df['home_covers'] = (historical_df['actual_margin'] > -historical_df['market_spread']).astype(int)
current_df['home_covers'] = (current_df['actual_margin'] > -current_df['market_spread']).astype(int)

# Calculate home favored
historical_df['home_favored'] = historical_df['market_spread'] < 0
current_df['home_favored'] = current_df['market_spread'] < 0

historical_df['away_favored'] = historical_df['market_spread'] > 0
current_df['away_favored'] = current_df['market_spread'] > 0

print(f"Historical (21-24): {len(historical_df)} games")
print(f"Current (24-25):    {len(current_df)} games")
print()

print("="*100)
print("1. BASIC STATISTICS")
print("="*100)
print()

print(f"{'Metric':<40} {'Historical (21-24)':<20} {'Current (24-25)':<20} {'Difference':<15}")
print("-"*100)

# Average margin
hist_avg_margin = historical_df['actual_margin'].mean()
curr_avg_margin = current_df['actual_margin'].mean()
print(f"{'Average margin':<40} {hist_avg_margin:>18.2f} {curr_avg_margin:>18.2f} {curr_avg_margin - hist_avg_margin:>14.2f}")

# Std dev of margin
hist_std_margin = historical_df['actual_margin'].std()
curr_std_margin = current_df['actual_margin'].std()
print(f"{'Std dev of margin':<40} {hist_std_margin:>18.2f} {curr_std_margin:>18.2f} {curr_std_margin - hist_std_margin:>14.2f}")

# Average spread
hist_avg_spread = historical_df['market_spread'].abs().mean()
curr_avg_spread = current_df['market_spread'].abs().mean()
print(f"{'Average spread size':<40} {hist_avg_spread:>18.2f} {curr_avg_spread:>18.2f} {curr_avg_spread - hist_avg_spread:>14.2f}")

# Home cover rate (baseline)
hist_home_cover_rate = historical_df['home_covers'].mean() * 100
curr_home_cover_rate = current_df['home_covers'].mean() * 100
print(f"{'Home cover rate (%)':<40} {hist_home_cover_rate:>18.2f} {curr_home_cover_rate:>18.2f} {curr_home_cover_rate - hist_home_cover_rate:>14.2f}")

# Home win rate
hist_home_win_rate = (historical_df['actual_margin'] > 0).mean() * 100
curr_home_win_rate = (current_df['actual_margin'] > 0).mean() * 100
print(f"{'Home win rate (%)':<40} {hist_home_win_rate:>18.2f} {curr_home_win_rate:>18.2f} {curr_home_win_rate - hist_home_win_rate:>14.2f}")

# Home favored rate
hist_home_fav_rate = historical_df['home_favored'].mean() * 100
curr_home_fav_rate = current_df['home_favored'].mean() * 100
print(f"{'Home favored rate (%)':<40} {hist_home_fav_rate:>18.2f} {curr_home_fav_rate:>18.2f} {curr_home_fav_rate - hist_home_fav_rate:>14.2f}")

print()
print("="*100)
print("2. HOME COURT ADVANTAGE")
print("="*100)
print()

# Calculate effective HCA (actual home margin vs market expectation)
historical_df['market_expectation'] = -historical_df['market_spread']
current_df['market_expectation'] = -current_df['market_spread']

historical_df['margin_vs_market'] = historical_df['actual_margin'] - historical_df['market_expectation']
current_df['margin_vs_market'] = current_df['actual_margin'] - current_df['market_expectation']

hist_hca = historical_df['margin_vs_market'].mean()
curr_hca = current_df['margin_vs_market'].mean()

print(f"Average actual margin vs market expectation:")
print(f"  Historical (21-24): {hist_hca:+.2f} points")
print(f"  Current (24-25):    {curr_hca:+.2f} points")
print(f"  Difference:         {curr_hca - hist_hca:+.2f} points")
print()

if abs(curr_hca - hist_hca) > 0.5:
    if curr_hca > hist_hca:
        print(f"⚠️  24-25 season: Home teams OUTPERFORMED market by {curr_hca - hist_hca:+.2f} points more than historical!")
    else:
        print(f"⚠️  24-25 season: Home teams UNDERPERFORMED market by {hist_hca - curr_hca:+.2f} points more than historical!")
else:
    print("✅ No significant difference in home court advantage")

print()
print("="*100)
print("3. FAVORITE PERFORMANCE")
print("="*100)
print()

# Home favored games
hist_home_fav = historical_df[historical_df['home_favored']]
curr_home_fav = current_df[current_df['home_favored']]

hist_home_fav_cover = hist_home_fav['home_covers'].mean() * 100
curr_home_fav_cover = curr_home_fav['home_covers'].mean() * 100

print(f"Home favorites cover rate:")
print(f"  Historical (21-24): {hist_home_fav_cover:.2f}%")
print(f"  Current (24-25):    {curr_home_fav_cover:.2f}%")
print(f"  Difference:         {curr_home_fav_cover - hist_home_fav_cover:+.2f} percentage points")
print()

# Away favored games
hist_away_fav = historical_df[historical_df['away_favored']]
curr_away_fav = current_df[current_df['away_favored']]

hist_away_fav_cover = (1 - hist_away_fav['home_covers']).mean() * 100
curr_away_fav_cover = (1 - curr_away_fav['home_covers']).mean() * 100

print(f"Away favorites cover rate:")
print(f"  Historical (21-24): {hist_away_fav_cover:.2f}%")
print(f"  Current (24-25):    {curr_away_fav_cover:.2f}%")
print(f"  Difference:         {curr_away_fav_cover - hist_away_fav_cover:+.2f} percentage points")
print()

if abs(curr_home_fav_cover - hist_home_fav_cover) > 3:
    print(f"⚠️  HOME FAVORITE cover rate shifted by {curr_home_fav_cover - hist_home_fav_cover:+.2f} pp!")
    print(f"    This is a significant market pattern change.")
    print()

if abs(curr_away_fav_cover - hist_away_fav_cover) > 3:
    print(f"⚠️  AWAY FAVORITE cover rate shifted by {curr_away_fav_cover - hist_away_fav_cover:+.2f} pp!")
    print(f"    This is a significant market pattern change.")
    print()

print("="*100)
print("4. SPREAD SIZE DISTRIBUTION")
print("="*100)
print()

for df_name, df in [("Historical (21-24)", historical_df), ("Current (24-25)", current_df)]:
    df['spread_abs'] = df['market_spread'].abs()
    small = (df['spread_abs'] <= 3.5).sum()
    medium = ((df['spread_abs'] > 3.5) & (df['spread_abs'] <= 7.5)).sum()
    large = (df['spread_abs'] > 7.5).sum()
    
    print(f"{df_name}:")
    print(f"  Small (≤3.5):  {small:4d} ({small/len(df)*100:5.1f}%)")
    print(f"  Medium (3.5-7.5): {medium:4d} ({medium/len(df)*100:5.1f}%)")
    print(f"  Large (>7.5):  {large:4d} ({large/len(df)*100:5.1f}%)")
    print()

print("="*100)
print("5. BY SEASON BREAKDOWN (Historical)")
print("="*100)
print()

for season in sorted(historical_df['season'].unique()):
    season_df = historical_df[historical_df['season'] == season]
    cover_rate = season_df['home_covers'].mean() * 100
    home_fav_rate = season_df['home_favored'].mean() * 100
    home_fav_cover = season_df[season_df['home_favored']]['home_covers'].mean() * 100
    
    print(f"{season}:")
    print(f"  Games: {len(season_df)}")
    print(f"  Home cover rate: {cover_rate:.2f}%")
    print(f"  Home favored: {home_fav_rate:.1f}%")
    print(f"  Home fav cover rate: {home_fav_cover:.2f}%")
    print()

print("="*100)
print("SUMMARY: Is 24-25 season an anomaly?")
print("="*100)
print()

anomalies = []

# Check each metric
if abs(curr_home_fav_cover - hist_home_fav_cover) > 3:
    anomalies.append(f"Home favorite cover rate: {curr_home_fav_cover - hist_home_fav_cover:+.2f} pp change")

if abs(curr_away_fav_cover - hist_away_fav_cover) > 3:
    anomalies.append(f"Away favorite cover rate: {curr_away_fav_cover - hist_away_fav_cover:+.2f} pp change")

if abs(curr_hca - hist_hca) > 0.5:
    anomalies.append(f"Home court advantage: {curr_hca - hist_hca:+.2f} points change")

if abs(curr_home_cover_rate - hist_home_cover_rate) > 3:
    anomalies.append(f"Overall home cover rate: {curr_home_cover_rate - hist_home_cover_rate:+.2f} pp change")

if len(anomalies) > 0:
    print("⚠️  ANOMALIES DETECTED IN 24-25 SEASON:")
    print()
    for i, anomaly in enumerate(anomalies, 1):
        print(f"  {i}. {anomaly}")
    print()
    print("The 24-25 season shows different patterns from 21-24.")
    print("This could explain why adding it to training hurts generalization!")
else:
    print("✅ No major anomalies detected.")
    print("24-25 season looks similar to historical patterns.")

print()
print("="*100)

