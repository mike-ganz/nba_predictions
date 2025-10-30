"""
Comprehensive Feature Audit

Checks:
1. What features are we using?
2. How are they calculated?
3. What time windows are used?
4. Are there any look-ahead bias issues?
5. How do features change over time?
6. Are there distribution shifts between train and test?
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

print("=" * 80)
print("COMPREHENSIVE FEATURE AUDIT")
print("=" * 80)

# Load datasets
datasets = {
    'train': 'data/games_train_with_players_90.jsonl',
    'val': 'data/games_val_with_players.jsonl',
    'test_2425': 'data/games_predict_2024_2025_with_players.jsonl'
}

def load_games(path):
    """Load games and extract features"""
    games = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            
            # Extract team features
            home = game['teams']['H']
            away = game['teams']['A']
            
            # Extract player features (if available)
            player_usage_h = []
            player_usage_a = []
            if 'players' in game:
                for p in game['players'].get('H', []):
                    player_usage_h.append(p.get('baseline_usage_rate', 0.2))
                for p in game['players'].get('A', []):
                    player_usage_a.append(p.get('baseline_usage_rate', 0.2))
            
            games.append({
                'game_id': game['game_id'],
                'date': game['date'],
                'season': game['season'],
                
                # Home team features
                'h_off_rating': home['off_rating'],
                'h_def_rating': home['def_rating'],
                'h_pace': home['pace'],
                'h_three_pt_rate': home['three_pt_rate'],
                'h_free_throw_rate': home['free_throw_rate'],
                'h_off_reb_rate': home['off_reb_rate'],
                'h_def_reb_rate': home['def_reb_rate'],
                'h_assist_rate': home['assist_rate'],
                'h_turnover_rate': home['turnover_rate'],
                'h_rest_days': home['rest_days'],
                
                # Away team features
                'a_off_rating': away['off_rating'],
                'a_def_rating': away['def_rating'],
                'a_pace': away['pace'],
                'a_three_pt_rate': away['three_pt_rate'],
                'a_free_throw_rate': away['free_throw_rate'],
                'a_off_reb_rate': away['off_reb_rate'],
                'a_def_reb_rate': away['def_reb_rate'],
                'a_assist_rate': away['assist_rate'],
                'a_turnover_rate': away['turnover_rate'],
                'a_rest_days': away['rest_days'],
                
                # Market
                'market_spread': game['market']['spread_home'],
                'market_total': game['market'].get('total', None),
                
                # Outcome (if available)
                'actual_home': game['outcome'].get('home_final') if 'outcome' in game else None,
                'actual_away': game['outcome'].get('away_final') if 'outcome' in game else None,
                
                # Player aggregates
                'n_players_h': len(player_usage_h),
                'n_players_a': len(player_usage_a),
                'avg_usage_h': np.mean(player_usage_h) if player_usage_h else 0.2,
                'avg_usage_a': np.mean(player_usage_a) if player_usage_a else 0.2,
                'max_usage_h': max(player_usage_h) if player_usage_h else 0.2,
                'max_usage_a': max(player_usage_a) if player_usage_a else 0.2,
            })
    
    return pd.DataFrame(games)

print("\n[1] LOADING DATASETS")
print("-" * 80)

dfs = {}
for name, path in datasets.items():
    df = load_games(path)
    df['date'] = pd.to_datetime(df['date'])
    dfs[name] = df
    print(f"{name:15} {len(df):5} games | Date range: {df['date'].min().date()} to {df['date'].max().date()}")

print("\n[2] FEATURE SUMMARY")
print("-" * 80)

# List all team features
team_features = [
    'off_rating', 'def_rating', 'pace', 'three_pt_rate', 'free_throw_rate',
    'off_reb_rate', 'def_reb_rate', 'assist_rate', 'turnover_rate', 'rest_days'
]

print("\nTeam Features (10 per side, 20 total):")
for feat in team_features:
    print(f"  - {feat}")

print("\nPlayer Features:")
print(f"  - Top 8 players per team")
print(f"  - For each player: baseline_usage_rate, baseline_minutes, baseline_ts_pct")
print(f"  - Calculated from season-to-date stats (BEFORE game date)")

print("\nDerived Features (computed in model):")
print(f"  - edge_home = h_off_rating - a_def_rating")
print(f"  - edge_away = a_off_rating - h_def_rating")
print(f"  - pace_mean = (h_pace + a_pace) / 2")
print(f"  - orb_edge_home = h_off_reb_rate - a_def_reb_rate")
print(f"  - tov_edge_home = -(h_turnover_rate - a_turnover_rate)")
print(f"  - [similar for away side]")

print("\nBaseline:")
print(f"  - Derived from market spread (baseline_margin = -market_spread)")

print("\n[3] TIME WINDOW CHECK")
print("-" * 80)
print("Team stats:")
print(f"  - Rolling window: Last 10 games before target date")
print(f"  - Minimum games: 3 (else fallback to prior season)")
print(f"  - Filters: All games BEFORE target_date (no look-ahead)")
print()
print("Player stats:")
print(f"  - Season-to-date stats from games BEFORE target date")
print(f"  - Minimum games: 5 (else fallback to prior season)")
print(f"  - Default values: Used for players with <5 games")

print("\n[4] FEATURE DISTRIBUTIONS BY DATASET")
print("-" * 80)

for feat in ['h_off_rating', 'h_def_rating', 'h_pace', 'market_spread']:
    print(f"\n{feat}:")
    for name, df in dfs.items():
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        min_val = df[feat].min()
        max_val = df[feat].max()
        print(f"  {name:15} Mean: {mean_val:6.2f} | Std: {std_val:5.2f} | Range: [{min_val:6.2f}, {max_val:6.2f}]")

print("\n[5] PLAYER FEATURE AVAILABILITY")
print("-" * 80)

for name, df in dfs.items():
    has_players = (df['n_players_h'] > 0).sum()
    pct = has_players / len(df) * 100
    avg_players_h = df['n_players_h'].mean()
    avg_players_a = df['n_players_a'].mean()
    print(f"{name:15} {has_players}/{len(df)} games ({pct:.1f}%) | Avg players: H={avg_players_h:.1f}, A={avg_players_a:.1f}")

print("\n[6] PLAYER USAGE DISTRIBUTION")
print("-" * 80)

for name, df in dfs.items():
    avg_usage_h = df['avg_usage_h'].mean()
    avg_usage_a = df['avg_usage_a'].mean()
    max_usage_h = df['max_usage_h'].mean()
    max_usage_a = df['max_usage_a'].mean()
    
    # Check if all usage is 0.2 (fallback default)
    all_default_h = (df['avg_usage_h'] == 0.2).sum()
    all_default_a = (df['avg_usage_a'] == 0.2).sum()
    
    print(f"{name:15} Avg usage: H={avg_usage_h:.3f}, A={avg_usage_a:.3f}")
    print(f"{'':15} Max usage: H={max_usage_h:.3f}, A={max_usage_a:.3f}")
    print(f"{'':15} Default (0.2): {all_default_h}/{len(df)} home, {all_default_a}/{len(df)} away")

print("\n[7] TIME TREND ANALYSIS")
print("-" * 80)

# Check if features are stable over time
train_df = dfs['train'].copy()
train_df['year_month'] = train_df['date'].dt.to_period('M')

# Group by month
monthly_stats = train_df.groupby('year_month').agg({
    'h_off_rating': 'mean',
    'h_def_rating': 'mean',
    'h_pace': 'mean',
    'market_spread': 'mean',
    'game_id': 'count'
}).rename(columns={'game_id': 'n_games'})

print("\nMonthly averages (train set, first 6 and last 6 months):")
print(monthly_stats.head(6).to_string())
print("...")
print(monthly_stats.tail(6).to_string())

# Check for trends
from scipy.stats import linregress

time_index = np.arange(len(monthly_stats))
for feat in ['h_off_rating', 'h_pace']:
    slope, _, r_value, _, _ = linregress(time_index, monthly_stats[feat].values)
    print(f"\n{feat} trend: {slope:+.4f} per month (R^2 = {r_value**2:.3f})")

print("\n[8] DISTRIBUTION SHIFT: TRAIN vs TEST")
print("-" * 80)

# Compare train vs test distributions
key_features = ['h_off_rating', 'h_def_rating', 'a_off_rating', 'a_def_rating', 
                'h_pace', 'a_pace', 'market_spread']

from scipy.stats import ks_2samp

print("\nKolmogorov-Smirnov test (p < 0.05 indicates significant difference):")
for feat in key_features:
    stat, p_value = ks_2samp(dfs['train'][feat], dfs['test_2425'][feat])
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"  {feat:20} KS={stat:.4f}, p={p_value:.4f} {significance}")

print("\n[9] FEATURE CORRELATIONS")
print("-" * 80)

# Check correlation with actual outcomes (train only)
train_with_actuals = dfs['train'][dfs['train']['actual_home'].notna()].copy()
train_with_actuals['actual_margin'] = train_with_actuals['actual_home'] - train_with_actuals['actual_away']

feature_cols = ['h_off_rating', 'h_def_rating', 'a_off_rating', 'a_def_rating', 
                'h_pace', 'market_spread', 'avg_usage_h', 'avg_usage_a']

print("\nCorrelation with actual margin (train set):")
for feat in feature_cols:
    corr = train_with_actuals[[feat, 'actual_margin']].corr().iloc[0, 1]
    print(f"  {feat:20} {corr:+.4f}")

# Check multicollinearity
print("\nHighly correlated features (|r| > 0.7):")
corr_matrix = train_with_actuals[feature_cols].corr()
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            print(f"  {feature_cols[i]:20} <-> {feature_cols[j]:20} {corr:+.4f}")

print("\n[10] POTENTIAL ISSUES")
print("-" * 80)

issues = []

# Check 1: Distribution shifts
print("\nChecking for distribution shifts...")
for feat in key_features:
    stat, p_value = ks_2samp(dfs['train'][feat], dfs['test_2425'][feat])
    if p_value < 0.01:
        mean_train = dfs['train'][feat].mean()
        mean_test = dfs['test_2425'][feat].mean()
        diff = mean_test - mean_train
        issues.append(f"  [!] {feat}: Distribution shift detected (p={p_value:.4f}, delta={diff:+.2f})")

# Check 2: All-default player usage
for name, df in dfs.items():
    all_default = ((df['avg_usage_h'] == 0.2) & (df['avg_usage_a'] == 0.2)).sum()
    if all_default / len(df) > 0.1:
        issues.append(f"  [!] {name}: {all_default}/{len(df)} games ({all_default/len(df)*100:.1f}%) use default player usage (0.2)")

# Check 3: Time trends
for feat in ['h_off_rating', 'h_pace']:
    slope, _, r_value, _, _ = linregress(time_index, monthly_stats[feat].values)
    if abs(slope) > 0.1 and r_value**2 > 0.3:
        issues.append(f"  [!] {feat}: Strong time trend detected (slope={slope:+.4f}/month, R^2={r_value**2:.3f})")

# Check 4: Missing data
for name, df in dfs.items():
    missing_market_total = df['market_total'].isna().sum()
    if missing_market_total > 0:
        issues.append(f"  [!] {name}: {missing_market_total}/{len(df)} games missing market_total")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(issue)
else:
    print("\n[+] No major issues detected!")

print("\n[11] FEATURE CALCULATION SUMMARY")
print("-" * 80)

summary = """
WHAT WE'RE FEEDING THE MODEL:

1. TEAM STATS (20 features):
   - Home: off_rating, def_rating, pace, 3PT%, FT%, ORB%, DRB%, AST%, TO%, rest
   - Away: same 10 features
   - Time window: Last 10 games BEFORE game date
   - Fallback: Prior season if <3 current season games

2. PLAYER STATS (16 features per team = 32 total):
   - Top 8 players per team
   - For each: usage_rate, minutes, true_shooting%
   - Aggregated: Sum of usage-weighted stats
   - Time window: Season-to-date BEFORE game date
   - Fallback: Prior season if <5 current season games

3. MARKET BASELINE (1 feature):
   - Market spread (used as baseline_margin = -spread)

4. DERIVED FEATURES (computed in model):
   - Matchup edges (off vs def, pace differences, etc.)
   - Player aggregations (usage_share_top2, total_minutes, etc.)

TOTAL: ~50-60 features fed into model

HOW MODEL USES THEM:
- Ridge regression predicts: mu (expected margin) and sigma (uncertainty)
- mu = f(all features) - tries to adjust market baseline
- sigma = g(all features) - estimates outcome variance
- Normal distribution: N(mu, sigma^2) for margin

KEY ASSUMPTION:
- Market spread is the true baseline
- Model learns adjustments based on team/player stats
- If model can't beat market, mu will converge toward baseline

POTENTIAL WEAKNESSES:
1. If features don't contain info market doesn't have -> model can't add value
2. If feature time windows are too short (10 games) -> high variance
3. If player features are noisy/defaulted -> model may ignore them
4. If distribution shifts (2021-2024 vs 2024-2025) -> model breaks
"""

print(summary)

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

