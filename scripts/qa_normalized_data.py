"""
QA Script for Normalized Data

Checks:
1. Normalized fields exist and have correct structure
2. Normalized values are centered around 0
3. Raw features are still intact
4. Distribution shifts are reduced
5. No crazy outliers or bugs
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 80)
print("NORMALIZED DATA QA")
print("=" * 80)

def load_games(path):
    """Load games and extract both raw and normalized features"""
    games = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            
            home = game['teams']['H']
            away = game['teams']['A']
            
            games.append({
                'game_id': game['game_id'],
                'date': game['date'],
                'season': game['season'],
                
                # Home RAW features
                'h_off_rating': home.get('off_rating'),
                'h_def_rating': home.get('def_rating'),
                'h_pace': home.get('pace'),
                'h_three_pt_rate': home.get('three_pt_rate'),
                
                # Home NORMALIZED features
                'h_off_rating_norm': home.get('off_rating_norm'),
                'h_def_rating_norm': home.get('def_rating_norm'),
                'h_pace_norm': home.get('pace_norm'),
                'h_three_pt_rate_norm': home.get('three_pt_rate_norm'),
                
                # Away RAW features
                'a_off_rating': away.get('off_rating'),
                'a_def_rating': away.get('def_rating'),
                'a_pace': away.get('pace'),
                
                # Away NORMALIZED features
                'a_off_rating_norm': away.get('off_rating_norm'),
                'a_def_rating_norm': away.get('def_rating_norm'),
                'a_pace_norm': away.get('pace_norm'),
                
                # Market
                'market_spread': game['market']['spread_home'],
            })
    
    return pd.DataFrame(games)

# Load datasets
datasets = {
    'train': 'data/games_train_with_players_90_norm.jsonl',
    'val': 'data/games_val_with_players_norm.jsonl',
    'test_2425': 'data/games_predict_2024_2025_with_players_norm.jsonl'
}

print("\n[1] LOADING NORMALIZED DATASETS")
print("-" * 80)

dfs = {}
for name, path in datasets.items():
    df = load_games(path)
    df['date'] = pd.to_datetime(df['date'])
    dfs[name] = df
    print(f"{name:15} {len(df):5} games")

print("\n[2] CHECKING NORMALIZED FIELDS EXIST")
print("-" * 80)

normalized_fields = [
    'h_off_rating_norm', 'h_def_rating_norm', 'h_pace_norm', 'h_three_pt_rate_norm',
    'a_off_rating_norm', 'a_def_rating_norm', 'a_pace_norm'
]

for name, df in dfs.items():
    missing = []
    for field in normalized_fields:
        if field not in df.columns or df[field].isna().all():
            missing.append(field)
    
    if missing:
        print(f"{name:15} [!] MISSING: {', '.join(missing)}")
    else:
        print(f"{name:15} [+] All normalized fields present")

print("\n[3] CHECKING RAW FEATURES STILL INTACT")
print("-" * 80)

raw_fields = ['h_off_rating', 'h_def_rating', 'a_off_rating', 'a_def_rating']

for name, df in dfs.items():
    intact = True
    for field in raw_fields:
        if field not in df.columns or df[field].isna().any():
            intact = False
            break
    
    if intact:
        # Check if raw values look reasonable
        h_off_mean = df['h_off_rating'].mean()
        h_off_std = df['h_off_rating'].std()
        
        if 100 < h_off_mean < 125 and 3 < h_off_std < 10:
            print(f"{name:15} [+] Raw features intact (h_off_rating: {h_off_mean:.2f} +/- {h_off_std:.2f})")
        else:
            print(f"{name:15} [!] Raw features look suspicious (h_off_rating: {h_off_mean:.2f} +/- {h_off_std:.2f})")
    else:
        print(f"{name:15} [!] Raw features missing or have NaN values")

print("\n[4] NORMALIZED FEATURE DISTRIBUTIONS")
print("-" * 80)

features_to_check = [
    ('h_off_rating_norm', 'Home Off Rating (Normalized)'),
    ('h_def_rating_norm', 'Home Def Rating (Normalized)'),
    ('h_pace_norm', 'Home Pace (Normalized)'),
    ('a_off_rating_norm', 'Away Off Rating (Normalized)'),
]

for feat, label in features_to_check:
    print(f"\n{label}:")
    for name, df in dfs.items():
        if feat in df.columns and not df[feat].isna().all():
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            min_val = df[feat].min()
            max_val = df[feat].max()
            
            # Check if centered near 0
            if abs(mean_val) < 2.0:
                status = "[+]"
            elif abs(mean_val) < 5.0:
                status = "[~]"
            else:
                status = "[!]"
            
            print(f"  {name:15} {status} Mean: {mean_val:+6.2f} | Std: {std_val:5.2f} | Range: [{min_val:+6.2f}, {max_val:+6.2f}]")
        else:
            print(f"  {name:15} [!] MISSING or all NaN")

print("\n[5] COMPARING RAW vs NORMALIZED SPREADS")
print("-" * 80)

print("\nStandard deviation of features (lower = less spread, more consistent):")
print()
print(f"{'Feature':<25} {'Raw (Train)':<12} {'Raw (Test)':<12} {'Norm (Train)':<12} {'Norm (Test)':<12} {'Improvement':<12}")
print("-" * 90)

features = [
    ('off_rating', 'h_off_rating', 'h_off_rating_norm'),
    ('def_rating', 'h_def_rating', 'h_def_rating_norm'),
    ('pace', 'h_pace', 'h_pace_norm'),
]

for feat_name, raw_col, norm_col in features:
    raw_train_std = dfs['train'][raw_col].std()
    raw_test_std = dfs['test_2425'][raw_col].std()
    
    if norm_col in dfs['train'].columns and not dfs['train'][norm_col].isna().all():
        norm_train_std = dfs['train'][norm_col].std()
        norm_test_std = dfs['test_2425'][norm_col].std()
        
        # Check if normalization reduced the difference between train and test
        raw_diff = abs(raw_train_std - raw_test_std)
        norm_diff = abs(norm_train_std - norm_test_std)
        improvement = raw_diff - norm_diff
        
        print(f"{feat_name:<25} {raw_train_std:<12.3f} {raw_test_std:<12.3f} {norm_train_std:<12.3f} {norm_test_std:<12.3f} {improvement:+.3f}")
    else:
        print(f"{feat_name:<25} {raw_train_std:<12.3f} {raw_test_std:<12.3f} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("\n[6] DISTRIBUTION SHIFT REDUCTION (KS Test)")
print("-" * 80)

from scipy.stats import ks_2samp

print("\nKolmogorov-Smirnov p-value (HIGHER is better = distributions more similar):")
print()
print(f"{'Feature':<25} {'Raw p-value':<15} {'Normalized p-value':<15} {'Improvement':<15}")
print("-" * 70)

for feat_name, raw_col, norm_col in features:
    raw_stat, raw_p = ks_2samp(dfs['train'][raw_col], dfs['test_2425'][raw_col])
    
    if norm_col in dfs['train'].columns and not dfs['train'][norm_col].isna().all():
        norm_stat, norm_p = ks_2samp(dfs['train'][norm_col], dfs['test_2425'][norm_col])
        
        if norm_p > raw_p:
            status = "[+] BETTER"
        elif norm_p > raw_p * 0.8:
            status = "[~] SIMILAR"
        else:
            status = "[!] WORSE"
        
        print(f"{feat_name:<25} {raw_p:<15.4f} {norm_p:<15.4f} {status}")
    else:
        print(f"{feat_name:<25} {raw_p:<15.4f} {'N/A':<15} {'N/A':<15}")

print("\n[7] SANITY CHECK: RECONSTRUCTION")
print("-" * 80)

# Check if raw = normalized + league_avg (spot check a few games)
print("\nSpot-checking that: raw_feature = normalized_feature + league_avg")
print("(This verifies the math is correct)")
print()

train_df = dfs['train']
sample_games = train_df.sample(min(5, len(train_df)))

for idx, row in sample_games.iterrows():
    game_id = row['game_id']
    
    # Check h_off_rating
    raw = row['h_off_rating']
    norm = row['h_off_rating_norm']
    
    if pd.notna(raw) and pd.notna(norm):
        implied_league_avg = raw - norm
        
        # Check if league avg is reasonable (should be ~110-115)
        if 105 < implied_league_avg < 120:
            print(f"[+] {game_id}: raw={raw:.2f}, norm={norm:+.2f}, league_avg={implied_league_avg:.2f}")
        else:
            print(f"[!] {game_id}: raw={raw:.2f}, norm={norm:+.2f}, league_avg={implied_league_avg:.2f} (SUSPICIOUS!)")
    else:
        print(f"[!] {game_id}: Missing data (raw={raw}, norm={norm})")

print("\n[8] OUTLIER CHECK")
print("-" * 80)

print("\nChecking for extreme outliers (>4 std devs from mean):")

for feat, label in features_to_check:
    all_values = pd.concat([dfs['train'][feat], dfs['val'][feat], dfs['test_2425'][feat]])
    
    if all_values.isna().all():
        continue
    
    mean = all_values.mean()
    std = all_values.std()
    
    outliers = all_values[(all_values < mean - 4*std) | (all_values > mean + 4*std)]
    
    if len(outliers) > 0:
        print(f"  {label}: {len(outliers)} outliers found")
        print(f"    Range: [{outliers.min():.2f}, {outliers.max():.2f}]")
    else:
        print(f"  {label}: [+] No extreme outliers")

print("\n[9] FINAL VERDICT")
print("-" * 80)

issues = []
warnings = []

# Check 1: All normalized fields exist
for name, df in dfs.items():
    for field in normalized_fields:
        if field not in df.columns or df[field].isna().all():
            issues.append(f"[!] {name}: Missing normalized field '{field}'")

# Check 2: Normalized fields are centered near 0
for name, df in dfs.items():
    for feat, label in features_to_check[:2]:  # Check first 2
        if feat in df.columns and not df[feat].isna().all():
            mean_val = df[feat].mean()
            if abs(mean_val) > 5.0:
                warnings.append(f"[~] {name}: {feat} mean is {mean_val:+.2f} (should be near 0)")

# Check 3: Distribution shift improved
improved = 0
total = 0
for feat_name, raw_col, norm_col in features:
    if norm_col in dfs['train'].columns and not dfs['train'][norm_col].isna().all():
        raw_stat, raw_p = ks_2samp(dfs['train'][raw_col], dfs['test_2425'][raw_col])
        norm_stat, norm_p = ks_2samp(dfs['train'][norm_col], dfs['test_2425'][norm_col])
        
        total += 1
        if norm_p > raw_p:
            improved += 1

if improved < total / 2:
    warnings.append(f"[~] Normalization only improved {improved}/{total} features' distribution shift")
else:
    print(f"\n[+] Distribution shift improved for {improved}/{total} features")

# Check 4: Raw features still exist
for name, df in dfs.items():
    for field in raw_fields:
        if field not in df.columns or df[field].isna().any():
            issues.append(f"[!] {name}: Raw field '{field}' missing or has NaN")

print()
if issues:
    print("CRITICAL ISSUES FOUND:")
    for issue in issues:
        print(issue)
    print("\n[!] DO NOT PROCEED - Fix issues first")
elif warnings:
    print("WARNINGS:")
    for warning in warnings:
        print(warning)
    print("\n[~] Proceed with caution - review warnings")
else:
    print("[+] ALL CHECKS PASSED!")
    print("[+] Data looks good - safe to proceed with training")

print("\n" + "=" * 80)
print("QA COMPLETE")
print("=" * 80)

