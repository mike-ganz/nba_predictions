"""
Validate the "away favorites" strategy on the 2021-2024 validation set

This tests whether the pattern we discovered on 2024-2025 test set
also existed in the historical holdout data.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("VALIDATING 'AWAY FAVORITES' STRATEGY ON HISTORICAL HOLDOUT")
print("=" * 80)

# Load validation set predictions (2021-2024 holdout)
print("\n[1] Loading 2021-2024 validation set predictions...")
df_val = pd.read_csv('predictions/val_normalized_predictions.csv')

# Load test set predictions (2024-2025)
print("[2] Loading 2024-2025 test set predictions...")
df_test = pd.read_csv('predictions/test_2425_normalized_predictions.csv')

def analyze_dataset(df, name):
    print(f"\n" + "=" * 80)
    print(f"{name.upper()}")
    print("=" * 80)
    
    # Calculate outcomes
    df['actual_margin'] = df['actual_home'] - df['actual_away']
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['pred_home_covers'] = (df['cover_prob_home'] > 0.5).astype(int)
    df['ats_correct'] = (df['pred_home_covers'] == df['actual_home_covers']).astype(int)
    
    # Create filters
    df['home_favored'] = df['market_spread_home'] > 0
    df['away_favored'] = df['market_spread_home'] < 0
    df['spread_abs'] = df['market_spread_home'].abs()
    df['spread_small'] = (df['spread_abs'] >= 2.5) & (df['spread_abs'] < 5.5)
    df['spread_medium'] = (df['spread_abs'] >= 5.5) & (df['spread_abs'] < 8.5)
    df['spread_large'] = df['spread_abs'] >= 8.5
    df['model_picks_home'] = df['pred_home_covers'] == 1
    df['model_picks_away'] = df['pred_home_covers'] == 0
    
    print(f"\nTotal games: {len(df)}")
    print(f"Overall ATS: {df['ats_correct'].mean()*100:.2f}%")
    
    # Favorite type analysis
    print(f"\n--- FAVORITE TYPE ---")
    
    for fav_type, label in [('home_favored', 'Home Favored'), ('away_favored', 'Away Favored')]:
        filtered = df[df[fav_type]]
        n = len(filtered)
        ats = filtered['ats_correct'].mean() * 100
        roi = filtered.apply(lambda r: 0.91 if r['ats_correct'] else -1.0, axis=1).mean() * 100
        
        # Z-score
        n_correct = filtered['ats_correct'].sum()
        expected = n * 0.5
        z = (n_correct - expected) / np.sqrt(n * 0.5 * 0.5)
        
        print(f"\n{label}:")
        print(f"  Games: {n} ({n/len(df)*100:.1f}%)")
        print(f"  ATS: {ats:.2f}%")
        print(f"  ROI: {roi:+.2f}%")
        print(f"  Z-score: {z:+.2f}")
        
        # Statistical significance
        if abs(z) > 2.58:
            sig = "***" 
        elif abs(z) > 1.96:
            sig = "**"
        elif abs(z) > 1.65:
            sig = "*"
        else:
            sig = ""
        
        if sig:
            print(f"  Significance: {sig} (p < {0.01 if abs(z) > 2.58 else 0.05 if abs(z) > 1.96 else 0.10})")
    
    # Spread size breakdown for away favorites
    print(f"\n--- AWAY FAVORITES BY SPREAD SIZE ---")
    
    away_fav = df[df['away_favored']]
    
    for spread_type, label in [('spread_small', 'Small (2.5-5.5)'),
                                ('spread_medium', 'Medium (5.5-8.5)'),
                                ('spread_large', 'Large (8.5+)')]:
        filtered = away_fav[away_fav[spread_type]]
        if len(filtered) > 0:
            n = len(filtered)
            ats = filtered['ats_correct'].mean() * 100
            roi = filtered.apply(lambda r: 0.91 if r['ats_correct'] else -1.0, axis=1).mean() * 100
            
            print(f"\n{label}:")
            print(f"  Games: {n}")
            print(f"  ATS: {ats:.2f}%")
            print(f"  ROI: {roi:+.2f}%")
    
    # Top combined strategies
    print(f"\n--- TOP COMBINED STRATEGIES (Away Favorites) ---")
    
    strategies = [
        ('Away Fav + Pick Away', away_fav[away_fav['model_picks_away']]),
        ('Away Fav + Pick Home', away_fav[away_fav['model_picks_home']]),
        ('Away Fav + Med Spread', away_fav[away_fav['spread_medium']]),
        ('Away Fav + Large Spread', away_fav[away_fav['spread_large']]),
    ]
    
    for label, filtered in strategies:
        if len(filtered) >= 10:
            n = len(filtered)
            ats = filtered['ats_correct'].mean() * 100
            roi = filtered.apply(lambda r: 0.91 if r['ats_correct'] else -1.0, axis=1).mean() * 100
            
            print(f"\n{label}:")
            print(f"  Games: {n}")
            print(f"  ATS: {ats:.2f}%")
            print(f"  ROI: {roi:+.2f}%")
    
    return df

# Analyze both datasets
df_val = analyze_dataset(df_val, "2021-2024 Validation Set (Historical Holdout)")
df_test = analyze_dataset(df_test, "2024-2025 Test Set")

# Direct comparison
print("\n" + "=" * 80)
print("COMPARISON: DOES THE PATTERN HOLD?")
print("=" * 80)

comparison_data = []

for dataset, df, name in [('val', df_val, '2021-2024'), ('test', df_test, '2024-2025')]:
    home_fav = df[df['home_favored']]
    away_fav = df[df['away_favored']]
    
    home_ats = home_fav['ats_correct'].mean() * 100 if len(home_fav) > 0 else 0
    away_ats = away_fav['ats_correct'].mean() * 100 if len(away_fav) > 0 else 0
    
    home_roi = home_fav.apply(lambda r: 0.91 if r['ats_correct'] else -1.0, axis=1).mean() * 100 if len(home_fav) > 0 else 0
    away_roi = away_fav.apply(lambda r: 0.91 if r['ats_correct'] else -1.0, axis=1).mean() * 100 if len(away_fav) > 0 else 0
    
    comparison_data.append({
        'period': name,
        'home_ats': home_ats,
        'away_ats': away_ats,
        'home_roi': home_roi,
        'away_roi': away_roi,
        'away_advantage': away_ats - home_ats
    })

print(f"\n{'Period':<15} {'Home Fav ATS':<15} {'Away Fav ATS':<15} {'Away Advantage':<15}")
print("-" * 60)
for row in comparison_data:
    print(f"{row['period']:<15} {row['home_ats']:>7.2f}%       {row['away_ats']:>7.2f}%       {row['away_advantage']:>+7.2f}%")

print(f"\n{'Period':<15} {'Home Fav ROI':<15} {'Away Fav ROI':<15} {'Difference':<15}")
print("-" * 60)
for row in comparison_data:
    print(f"{row['period']:<15} {row['home_roi']:>+7.2f}%      {row['away_roi']:>+7.2f}%      {row['away_roi'] - row['home_roi']:>+7.2f}%")

# Final verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

val_away_better = comparison_data[0]['away_ats'] > comparison_data[0]['home_ats']
test_away_better = comparison_data[1]['away_ats'] > comparison_data[1]['home_ats']

val_away_adv = comparison_data[0]['away_advantage']
test_away_adv = comparison_data[1]['away_advantage']

if val_away_better and test_away_better:
    print("\n[+] PATTERN CONFIRMED!")
    print(f"    Away favorites outperformed in BOTH validation ({val_away_adv:+.2f}%) and test ({test_away_adv:+.2f}%)")
    print(f"    This is strong evidence of a real, persistent pattern.")
    
    # Check if magnitude is consistent
    if abs(val_away_adv - test_away_adv) < 10:
        print(f"\n[+] MAGNITUDES CONSISTENT!")
        print(f"    The advantage is similar in both periods (~{(val_away_adv + test_away_adv)/2:.1f}% average)")
        print(f"    This suggests the edge is stable over time.")
    else:
        print(f"\n[~] MAGNITUDES DIFFER")
        print(f"    2021-2024: {val_away_adv:+.2f}% advantage")
        print(f"    2024-2025: {test_away_adv:+.2f}% advantage")
        print(f"    The edge may be changing over time.")
    
    print(f"\n[RECOMMENDATION] This strategy is VALIDATED for deployment.")
    print(f"Expected performance: {comparison_data[1]['away_ats']:.1f}% ATS, {comparison_data[1]['away_roi']:+.1f}% ROI")
    
elif not val_away_better and test_away_better:
    print("\n[!] WARNING: Pattern only appears in 2024-2025")
    print(f"    2021-2024 validation: Home favored better (away {val_away_adv:+.2f}%)")
    print(f"    2024-2025 test: Away favored better (away {test_away_adv:+.2f}%)")
    print(f"\n[CAUTION] This could be:")
    print(f"    1. A new market inefficiency in 2024-2025")
    print(f"    2. Data mining luck (we searched and found it in test set)")
    print(f"    3. Sample size noise")
    print(f"\n[RECOMMENDATION] Use with caution or smaller bet sizes until more data available.")
    
else:
    print("\n[-] Pattern NOT confirmed")
    print(f"    The away favorite advantage in 2024-2025 ({test_away_adv:+.2f}%)")
    print(f"    did not exist in 2021-2024 validation ({val_away_adv:+.2f}%)")
    print(f"\n[RECOMMENDATION] Do NOT deploy this strategy - likely data mining artifact.")

print("\n" + "=" * 80)

