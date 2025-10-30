"""Phase 1: Data Quality Audit - Verify data integrity before investigating model."""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json

print("\n" + "="*70)
print("PHASE 1: DATA QUALITY AUDIT")
print("="*70 + "\n")

# Load predictions
val_df = pd.read_csv('reports/val_with_players/per_game_predictions.csv')
test_df = pd.read_csv('reports/2425_with_players/per_game_predictions.csv')

# =============================================================================
# TEST 1.1: Actual Outcomes Sanity Check
# =============================================================================
print("TEST 1.1: Actual Outcomes Sanity Check")
print("-" * 70)

for name, df in [("Validation", val_df), ("2024-2025", test_df)]:
    print(f"\n{name}:")
    print(f"  Games: {len(df)}")
    print(f"  Home scores: {df['actual_home'].describe()[['min', 'mean', 'max']].to_dict()}")
    print(f"  Away scores: {df['actual_away'].describe()[['min', 'mean', 'max']].to_dict()}")
    
    # Flag outliers
    suspicious = df[
        (df['actual_home'] < 70) | (df['actual_home'] > 160) |
        (df['actual_away'] < 70) | (df['actual_away'] > 160)
    ]
    if len(suspicious) > 0:
        print(f"  ⚠️  {len(suspicious)} suspicious scores found:")
        print(suspicious[['game_id', 'actual_home', 'actual_away']].head())
    else:
        print(f"  ✓ No suspicious scores")
    
    # Check for duplicates
    dupes = df.duplicated(subset=['date', 'away_team', 'home_team'])
    if dupes.sum() > 0:
        print(f"  ⚠️  {dupes.sum()} duplicate games found!")
    else:
        print(f"  ✓ No duplicate games")

# =============================================================================
# TEST 1.2: Market Lines Integrity Check
# =============================================================================
print("\n\nTEST 1.2: Market Lines Integrity Check")
print("-" * 70)

for name, df in [("Validation", val_df), ("2024-2025", test_df)]:
    print(f"\n{name}:")
    print(f"  Spreads: min={df['market_spread_home'].min():.1f}, "
          f"mean={df['market_spread_home'].mean():.1f}, "
          f"max={df['market_spread_home'].max():.1f}")
    print(f"  Totals: min={df['market_total'].min():.1f}, "
          f"mean={df['market_total'].mean():.1f}, "
          f"max={df['market_total'].max():.1f}")
    
    # Check for unusual patterns
    zero_spreads = (df['market_spread_home'] == 0).sum()
    print(f"  Pick'em games (spread=0): {zero_spreads}")
    
    # Market efficiency check
    abs_spread = abs(df['market_spread_home']).mean()
    print(f"  Avg absolute spread: {abs_spread:.2f}")

print(f"\n📊 Market Comparison:")
val_spread = abs(val_df['market_spread_home']).mean()
test_spread = abs(test_df['market_spread_home']).mean()
print(f"  Validation avg spread: {val_spread:.2f}")
print(f"  2024-2025 avg spread: {test_spread:.2f}")
print(f"  Difference: {test_spread - val_spread:+.2f} ({(test_spread/val_spread - 1)*100:+.1f}%)")

if abs(test_spread - val_spread) > 0.5:
    print(f"  ⚠️  Significant market shift detected!")
else:
    print(f"  ✓ Market lines appear consistent")

# =============================================================================
# TEST 1.3: Feature Distribution Drift
# =============================================================================
print("\n\nTEST 1.3: Feature Distribution Drift")
print("-" * 70)

def load_features(path, max_games=500):
    """Load features from JSONL game data."""
    features = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_games:
                break
            game = json.loads(line)
            features.append({
                'home_off_rating': game['teams']['H']['off_rating'],
                'home_def_rating': game['teams']['H']['def_rating'],
                'home_pace': game['teams']['H']['pace'],
                'home_three_pt_rate': game['teams']['H']['three_pt_rate'],
                'away_off_rating': game['teams']['A']['off_rating'],
                'away_def_rating': game['teams']['A']['def_rating'],
                'away_pace': game['teams']['A']['pace'],
                'away_three_pt_rate': game['teams']['A']['three_pt_rate'],
            })
    return pd.DataFrame(features)

print("Loading raw game features...")
train_features = load_features('data/games_train_with_players_90.jsonl')
test_features = load_features('data/games_predict_2024_2025_with_players.jsonl')

print(f"Train: {len(train_features)} games, Test: {len(test_features)} games\n")

print("Distribution shift analysis (Kolmogorov-Smirnov test):")
print(f"{'Feature':<25} {'Train Mean':<12} {'Test Mean':<12} {'Shift':<10} {'p-value':<10} {'Status'}")
print("-" * 90)

significant_shifts = []
for col in train_features.columns:
    train_mean = train_features[col].mean()
    test_mean = test_features[col].mean()
    shift_pct = (test_mean / train_mean - 1) * 100
    
    stat, pval = ks_2samp(train_features[col], test_features[col])
    
    status = "⚠️ SHIFT" if pval < 0.01 else "✓ OK"
    if pval < 0.01:
        significant_shifts.append(col)
    
    print(f"{col:<25} {train_mean:>11.2f} {test_mean:>11.2f} {shift_pct:>+9.1f}% {pval:>9.4f} {status}")

print(f"\n📊 Summary:")
if len(significant_shifts) > 0:
    print(f"  ⚠️  {len(significant_shifts)} features show significant distribution shift:")
    for feat in significant_shifts:
        print(f"     - {feat}")
    print(f"  This suggests NBA regime change between train and test periods.")
else:
    print(f"  ✓ No significant feature shifts detected")
    print(f"  Distribution shift is NOT the primary issue.")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "="*70)
print("PHASE 1 SUMMARY")
print("="*70 + "\n")

issues_found = []

# Check for data quality issues
if (val_df['actual_home'] < 70).any() or (test_df['actual_home'] < 70).any():
    issues_found.append("Suspicious game scores detected")

if val_df.duplicated(subset=['date', 'away_team', 'home_team']).any():
    issues_found.append("Duplicate games in data")

if abs(test_spread - val_spread) > 0.5:
    issues_found.append("Market lines shifted significantly")

if len(significant_shifts) >= 3:
    issues_found.append(f"{len(significant_shifts)} features show distribution shift")

if len(issues_found) == 0:
    print("✅ DATA QUALITY: GOOD")
    print("   No major data quality issues detected.")
    print("   The model failure is likely due to:")
    print("   - Model architecture issues")
    print("   - Overfitting/calibration problems")
    print("   - Feature engineering issues")
    print("\n   → Proceed to Phase 2: Error Decomposition")
else:
    print("⚠️  DATA QUALITY ISSUES DETECTED:")
    for issue in issues_found:
        print(f"   - {issue}")
    print("\n   → Address these data issues before investigating model")

print("\n" + "="*70)

