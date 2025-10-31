"""
Compare feature-outcome relationships across all three periods:
- 21-24 (OLD model training)
- 24-25 (added to NEW model)
- 25-26 (current season test)

Goal: Determine which historical period 25-26 most resembles.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

print("="*80)
print("THREE-PERIOD FEATURE-OUTCOME RELATIONSHIP COMPARISON")
print("="*80)
print()

# Load datasets
def load_features_and_outcomes(filepath):
    """Extract key features and actual margins from JSONL"""
    data = []
    with open(filepath) as f:
        for line in f:
            game = json.loads(line)
            h = game['teams']['H']
            a = game['teams']['A']
            
            # Get outcome
            outcome = game.get('outcome', {})
            if 'home_final' not in outcome:
                continue
            
            actual_margin = outcome['home_final'] - outcome['away_final']
            
            # Extract key normalized features
            row = {
                'actual_margin': actual_margin,
                'home_off_norm': h.get('off_rating_norm', 0),
                'home_def_norm': h.get('def_rating_norm', 0),
                'away_off_norm': a.get('off_rating_norm', 0),
                'away_def_norm': a.get('def_rating_norm', 0),
                'home_3pt_norm': h.get('three_pt_rate_norm', 0),
                'away_3pt_norm': a.get('three_pt_rate_norm', 0),
                'home_ft_norm': h.get('free_throw_rate_norm', 0),
                'away_ft_norm': a.get('free_throw_rate_norm', 0),
                'home_orb_norm': h.get('off_reb_rate_norm', 0),
                'away_orb_norm': a.get('off_reb_rate_norm', 0),
            }
            data.append(row)
    
    return pd.DataFrame(data)

# Load all three periods
print("Loading data...")
train_val = []
for file in ['games_train_with_players_90_norm.jsonl', 'games_val_with_players_norm.jsonl']:
    train_val.append(load_features_and_outcomes(f'data/{file}'))
df_2124 = pd.concat(train_val, ignore_index=True)

df_2425 = load_features_and_outcomes('data/games_predict_2024_2025_with_players_norm.jsonl')
df_2526 = load_features_and_outcomes('data/games_2025_2026_current_norm.jsonl')

print(f"21-24 seasons: {len(df_2124)} games")
print(f"24-25 season:  {len(df_2425)} games")
print(f"25-26 season:  {len(df_2526)} games")
print()

# Key features to analyze
key_features = [
    'home_off_norm', 'home_def_norm', 'away_off_norm', 'away_def_norm',
    'home_3pt_norm', 'away_3pt_norm', 'home_ft_norm', 'away_ft_norm',
    'home_orb_norm', 'away_orb_norm'
]

# =============================================================================
# CORRELATION ANALYSIS FOR ALL THREE PERIODS
# =============================================================================
print("="*80)
print("FEATURE-MARGIN CORRELATIONS ACROSS ALL THREE PERIODS")
print("="*80)
print()

print(f"{'Feature':<20} {'21-24':<12} {'24-25':<12} {'25-26':<12} {'25-26 vs':<15}")
print(f"{'':20} {'Corr':<12} {'Corr':<12} {'Corr':<12} {'Most Similar':<15}")
print("-"*80)

similarity_scores = {'21-24': 0, '24-25': 0}

for feat in key_features:
    corr_2124 = df_2124[feat].corr(df_2124['actual_margin'])
    corr_2425 = df_2425[feat].corr(df_2425['actual_margin'])
    corr_2526 = df_2526[feat].corr(df_2526['actual_margin'])
    
    # Calculate distances
    dist_to_2124 = abs(corr_2526 - corr_2124)
    dist_to_2425 = abs(corr_2526 - corr_2425)
    
    # Determine which is closer
    if dist_to_2124 < dist_to_2425:
        most_similar = "21-24 ✓"
        similarity_scores['21-24'] += 1
    elif dist_to_2425 < dist_to_2124:
        most_similar = "24-25 ✓"
        similarity_scores['24-25'] += 1
    else:
        most_similar = "Equal"
    
    print(f"{feat:<20} {corr_2124:>11.4f} {corr_2425:>11.4f} {corr_2526:>11.4f} {most_similar:<15}")

print()
print("Similarity Score:")
print(f"  25-26 is closer to 21-24: {similarity_scores['21-24']} features")
print(f"  25-26 is closer to 24-25: {similarity_scores['24-25']} features")
print()

if similarity_scores['21-24'] > similarity_scores['24-25']:
    print("📊 25-26 is MORE SIMILAR to 21-24!")
    print("   This explains why OLD model (trained on 21-24) works better.")
elif similarity_scores['24-25'] > similarity_scores['21-24']:
    print("📊 25-26 is MORE SIMILAR to 24-25!")
    print("   This suggests NEW model should work better (but it doesn't - puzzling!)")
else:
    print("📊 25-26 is EQUALLY similar to both periods")

print()

# =============================================================================
# DETAILED ANALYSIS OF KEY FEATURES
# =============================================================================
print("="*80)
print("DETAILED ANALYSIS: KEY FEATURES THAT CHANGED")
print("="*80)
print()

print("1. HOME THREE-POINT RATE (home_3pt_norm)")
print("-"*80)
corr_2124_3pt = df_2124['home_3pt_norm'].corr(df_2124['actual_margin'])
corr_2425_3pt = df_2425['home_3pt_norm'].corr(df_2425['actual_margin'])
corr_2526_3pt = df_2526['home_3pt_norm'].corr(df_2526['actual_margin'])

print(f"  21-24: {corr_2124_3pt:+.4f} (positive = high 3P helps)")
print(f"  24-25: {corr_2425_3pt:+.4f} (ANOMALY: turned negative!)")
print(f"  25-26: {corr_2526_3pt:+.4f}", end="")

if abs(corr_2526_3pt - corr_2124_3pt) < abs(corr_2526_3pt - corr_2425_3pt):
    print(" ← CLOSER to 21-24 ✓")
else:
    print(" ← CLOSER to 24-25")
print()

print("  Interpretation:")
if corr_2526_3pt > 0.05:
    print("    25-26 returned to NORMAL: high 3P rate helps win")
    print("    OLD model's positive coefficient is correct")
    print("    NEW model's negative coefficient (learned from 24-25) is WRONG")
elif corr_2526_3pt < -0.05:
    print("    25-26 continues ANOMALY: high 3P rate hurts")
    print("    NEW model's negative coefficient might be right")
else:
    print("    25-26 has neutral relationship (near zero)")
print()

print("2. AWAY THREE-POINT RATE (away_3pt_norm)")
print("-"*80)
corr_2124_away3pt = df_2124['away_3pt_norm'].corr(df_2124['actual_margin'])
corr_2425_away3pt = df_2425['away_3pt_norm'].corr(df_2425['actual_margin'])
corr_2526_away3pt = df_2526['away_3pt_norm'].corr(df_2526['actual_margin'])

print(f"  21-24: {corr_2124_away3pt:+.4f} (negative = away 3P hurts home)")
print(f"  24-25: {corr_2425_away3pt:+.4f}")
print(f"  25-26: {corr_2526_away3pt:+.4f}", end="")

if abs(corr_2526_away3pt - corr_2124_away3pt) < abs(corr_2526_away3pt - corr_2425_away3pt):
    print(" ← CLOSER to 21-24 ✓")
else:
    print(" ← CLOSER to 24-25")
print()

print("3. FREE THROW RATES")
print("-"*80)
corr_2124_homeft = df_2124['home_ft_norm'].corr(df_2124['actual_margin'])
corr_2425_homeft = df_2425['home_ft_norm'].corr(df_2425['actual_margin'])
corr_2526_homeft = df_2526['home_ft_norm'].corr(df_2526['actual_margin'])

print(f"  home_ft_norm:")
print(f"    21-24: {corr_2124_homeft:+.4f}")
print(f"    24-25: {corr_2425_homeft:+.4f}")
print(f"    25-26: {corr_2526_homeft:+.4f}", end="")

if abs(corr_2526_homeft - corr_2124_homeft) < abs(corr_2526_homeft - corr_2425_homeft):
    print(" ← CLOSER to 21-24 ✓")
else:
    print(" ← CLOSER to 24-25")
print()

# =============================================================================
# LINEAR REGRESSION COEFFICIENTS
# =============================================================================
print()
print("="*80)
print("REGRESSION COEFFICIENTS COMPARISON")
print("="*80)
print()

from sklearn.linear_model import LinearRegression

# Create edge features for all three periods
for df in [df_2124, df_2425, df_2526]:
    df['off_edge'] = df['home_off_norm'] - df['away_def_norm']
    df['3pt_diff'] = df['home_3pt_norm'] - df['away_3pt_norm']
    df['ft_diff'] = df['home_ft_norm'] - df['away_ft_norm']
    df['orb_diff'] = df['home_orb_norm'] - df['away_orb_norm']

simple_features = ['off_edge', '3pt_diff', 'ft_diff', 'orb_diff']

# Train models on each period
models = {}
for name, df in [('21-24', df_2124), ('24-25', df_2425), ('25-26', df_2526)]:
    X = df[simple_features].values
    y = df['actual_margin'].values
    model = LinearRegression()
    model.fit(X, y)
    models[name] = model

print(f"{'Feature':<20} {'21-24':<12} {'24-25':<12} {'25-26':<12} {'25-26 vs':<15}")
print(f"{'':20} {'Coef':<12} {'Coef':<12} {'Coef':<12} {'Most Similar':<15}")
print("-"*80)

coef_similarity = {'21-24': 0, '24-25': 0}

for i, feat in enumerate(simple_features):
    coef_2124 = models['21-24'].coef_[i]
    coef_2425 = models['24-25'].coef_[i]
    coef_2526 = models['25-26'].coef_[i]
    
    # Calculate distances
    dist_to_2124 = abs(coef_2526 - coef_2124)
    dist_to_2425 = abs(coef_2526 - coef_2425)
    
    if dist_to_2124 < dist_to_2425:
        most_similar = "21-24 ✓"
        coef_similarity['21-24'] += 1
    elif dist_to_2425 < dist_to_2124:
        most_similar = "24-25 ✓"
        coef_similarity['24-25'] += 1
    else:
        most_similar = "Equal"
    
    print(f"{feat:<20} {coef_2124:>11.4f} {coef_2425:>11.4f} {coef_2526:>11.4f} {most_similar:<15}")

print()
print("Coefficient Similarity Score:")
print(f"  25-26 coefficients closer to 21-24: {coef_similarity['21-24']} features")
print(f"  25-26 coefficients closer to 24-25: {coef_similarity['24-25']} features")
print()

# =============================================================================
# FINAL VERDICT
# =============================================================================
print("="*80)
print("FINAL VERDICT: WHICH PERIOD IS 25-26 MOST SIMILAR TO?")
print("="*80)
print()

total_2124 = similarity_scores['21-24'] + coef_similarity['21-24']
total_2425 = similarity_scores['24-25'] + coef_similarity['24-25']

print(f"Total similarity metrics:")
print(f"  25-26 similar to 21-24: {total_2124} / {len(key_features) + len(simple_features)}")
print(f"  25-26 similar to 24-25: {total_2425} / {len(key_features) + len(simple_features)}")
print()

if total_2124 > total_2425 + 2:
    print("🎯 STRONG EVIDENCE: 25-26 is most similar to 21-24")
    print()
    print("This perfectly explains the performance results:")
    print(f"  ✅ OLD model (trained on 21-24): 61.11% ATS")
    print(f"     → Coefficients match 25-26 patterns")
    print()
    print(f"  ❌ NEW model (trained on 21-25 including anomalous 24-25): 48.61% ATS")
    print(f"     → Coefficients DON'T match 25-26 patterns")
    print()
    print("Conclusion:")
    print("  24-25 was an ANOMALY.")
    print("  25-26 returned to normal (21-24-like) patterns.")
    print("  OLD model wins because it reflects normal patterns.")
    
elif total_2425 > total_2124 + 2:
    print("🤔 UNEXPECTED: 25-26 is most similar to 24-25")
    print()
    print("This is puzzling because NEW model should work better!")
    print("Need to investigate why NEW model fails despite similar patterns.")
    
else:
    print("📊 MIXED: 25-26 shows elements of both periods")
    print()
    print("The season has characteristics of both 21-24 and 24-25.")

print()
print("="*80)

