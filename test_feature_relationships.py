"""
Test if feature-outcome relationships differ between 21-24 and 24-25 seasons.
Multiple approaches to detect fundamental pattern shifts.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

print("="*80)
print("FEATURE-OUTCOME RELATIONSHIP ANALYSIS")
print("Testing if 24-25 is fundamentally different from 21-24")
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
                'home_pace_norm': h.get('pace_norm', 0),
                'away_pace_norm': a.get('pace_norm', 0),
                'home_3pt_norm': h.get('three_pt_rate_norm', 0),
                'away_3pt_norm': a.get('three_pt_rate_norm', 0),
                'home_ft_norm': h.get('free_throw_rate_norm', 0),
                'away_ft_norm': a.get('free_throw_rate_norm', 0),
                'home_orb_norm': h.get('off_reb_rate_norm', 0),
                'away_orb_norm': a.get('off_reb_rate_norm', 0),
                'season': game.get('season', ''),
            }
            data.append(row)
    
    return pd.DataFrame(data)

# Load all data
print("Loading data...")
train_val = []
for file in ['games_train_with_players_90_norm.jsonl', 'games_val_with_players_norm.jsonl']:
    train_val.append(load_features_and_outcomes(f'data/{file}'))
df_2124 = pd.concat(train_val, ignore_index=True)

df_2425 = load_features_and_outcomes('data/games_predict_2024_2025_with_players_norm.jsonl')

print(f"21-24 seasons: {len(df_2124)} games")
print(f"24-25 season:  {len(df_2425)} games")
print()

# =============================================================================
# TEST 1: Correlation Analysis
# =============================================================================
print("="*80)
print("TEST 1: FEATURE-MARGIN CORRELATIONS")
print("="*80)
print()

key_features = [
    'home_off_norm', 'home_def_norm', 'away_off_norm', 'away_def_norm',
    'home_3pt_norm', 'away_3pt_norm', 'home_ft_norm', 'away_ft_norm',
    'home_orb_norm', 'away_orb_norm'
]

print(f"{'Feature':<20} {'21-24 Corr':<15} {'24-25 Corr':<15} {'Difference':<15} {'P-value':<15}")
print("-"*80)

significant_diffs = []

for feat in key_features:
    corr_2124 = df_2124[feat].corr(df_2124['actual_margin'])
    corr_2425 = df_2425[feat].corr(df_2425['actual_margin'])
    diff = corr_2425 - corr_2124
    
    # Fisher z-transformation to test if correlations differ significantly
    z1 = np.arctanh(corr_2124)
    z2 = np.arctanh(corr_2425)
    se_diff = np.sqrt(1/(len(df_2124)-3) + 1/(len(df_2425)-3))
    z_stat = (z2 - z1) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    marker = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else ("*" if p_value < 0.1 else ""))
    
    print(f"{feat:<20} {corr_2124:>14.4f} {corr_2425:>14.4f} {diff:>14.4f} {p_value:>14.4f} {marker}")
    
    if p_value < 0.05:
        significant_diffs.append({
            'feature': feat,
            'corr_2124': corr_2124,
            'corr_2425': corr_2425,
            'diff': diff,
            'p_value': p_value
        })

print()
if significant_diffs:
    print(f"⚠️  Found {len(significant_diffs)} features with SIGNIFICANTLY different correlations!")
    print()
    for item in significant_diffs:
        print(f"  {item['feature']}:")
        print(f"    21-24: {item['corr_2124']:+.4f} → 24-25: {item['corr_2425']:+.4f}")
        print(f"    Change: {item['diff']:+.4f} (p={item['p_value']:.4f})")
        print()
else:
    print("✓ No significant correlation differences found")
    print()

# =============================================================================
# TEST 2: Simple Linear Regression Coefficients
# =============================================================================
print("="*80)
print("TEST 2: LINEAR REGRESSION COEFFICIENTS")
print("="*80)
print()

from sklearn.linear_model import LinearRegression

# Create simple features (offensive/defensive edges)
df_2124['off_edge'] = df_2124['home_off_norm'] - df_2124['away_def_norm']
df_2124['def_edge'] = df_2124['away_off_norm'] - df_2124['home_def_norm']
df_2124['3pt_diff'] = df_2124['home_3pt_norm'] - df_2124['away_3pt_norm']
df_2124['ft_diff'] = df_2124['home_ft_norm'] - df_2124['away_ft_norm']
df_2124['orb_diff'] = df_2124['home_orb_norm'] - df_2124['away_orb_norm']

df_2425['off_edge'] = df_2425['home_off_norm'] - df_2425['away_def_norm']
df_2425['def_edge'] = df_2425['away_off_norm'] - df_2425['home_def_norm']
df_2425['3pt_diff'] = df_2425['home_3pt_norm'] - df_2425['away_3pt_norm']
df_2425['ft_diff'] = df_2425['home_ft_norm'] - df_2425['away_ft_norm']
df_2425['orb_diff'] = df_2425['home_orb_norm'] - df_2425['away_orb_norm']

simple_features = ['off_edge', 'def_edge', '3pt_diff', 'ft_diff', 'orb_diff']

X_2124 = df_2124[simple_features].values
y_2124 = df_2124['actual_margin'].values

X_2425 = df_2425[simple_features].values
y_2425 = df_2425['actual_margin'].values

# Train on each dataset
model_2124 = LinearRegression()
model_2124.fit(X_2124, y_2124)

model_2425 = LinearRegression()
model_2425.fit(X_2425, y_2425)

print(f"{'Feature':<20} {'21-24 Coef':<15} {'24-25 Coef':<15} {'Difference':<15} {'% Change':<15}")
print("-"*80)

large_changes = []
for feat, coef_2124, coef_2425 in zip(simple_features, model_2124.coef_, model_2425.coef_):
    diff = coef_2425 - coef_2124
    pct_change = (diff / abs(coef_2124) * 100) if coef_2124 != 0 else 0
    
    print(f"{feat:<20} {coef_2124:>14.4f} {coef_2425:>14.4f} {diff:>14.4f} {pct_change:>14.1f}%")
    
    if abs(pct_change) > 30:
        large_changes.append({
            'feature': feat,
            'coef_2124': coef_2124,
            'coef_2425': coef_2425,
            'pct_change': pct_change
        })

print()
if large_changes:
    print(f"⚠️  Found {len(large_changes)} features with >30% coefficient change!")
    print()
    for item in large_changes:
        print(f"  {item['feature']}: {item['pct_change']:+.1f}% change")
else:
    print("✓ All coefficients relatively stable (<30% change)")

print()

# =============================================================================
# TEST 3: Cross-Period Prediction Test
# =============================================================================
print("="*80)
print("TEST 3: CROSS-PERIOD PREDICTION TEST")
print("="*80)
print()

# Train on 21-24, predict on 24-25
pred_2425_from_2124 = model_2124.predict(X_2425)
mae_cross = np.mean(np.abs(pred_2425_from_2124 - y_2425))
r2_cross = 1 - np.sum((y_2425 - pred_2425_from_2124)**2) / np.sum((y_2425 - np.mean(y_2425))**2)

# Train on 24-25, test on itself
pred_2425_from_2425 = model_2425.predict(X_2425)
mae_self = np.mean(np.abs(pred_2425_from_2425 - y_2425))
r2_self = 1 - np.sum((y_2425 - pred_2425_from_2425)**2) / np.sum((y_2425 - np.mean(y_2425))**2)

print("When predicting 24-25 margins:")
print(f"  Model trained on 21-24: MAE={mae_cross:.2f}, R²={r2_cross:.4f}")
print(f"  Model trained on 24-25: MAE={mae_self:.2f}, R²={r2_self:.4f}")
print()

if mae_self < mae_cross * 0.9:
    print(f"⚠️  24-25-trained model is {((mae_cross/mae_self - 1)*100):.1f}% better!")
    print("   This suggests 24-25 has different patterns from 21-24")
elif mae_self > mae_cross * 1.1:
    print(f"✓ 21-24-trained model is {((mae_self/mae_cross - 1)*100):.1f}% better!")
    print("   24-25 patterns are consistent with 21-24")
else:
    print("≈ Similar performance - patterns are similar")

print()

# =============================================================================
# OVERALL VERDICT
# =============================================================================
print("="*80)
print("OVERALL VERDICT")
print("="*80)
print()

evidence_count = len(significant_diffs) + len(large_changes)

if evidence_count >= 3:
    print("🚨 STRONG EVIDENCE: 24-25 is fundamentally different from 21-24")
    print()
    print("Multiple tests show:")
    print(f"  • {len(significant_diffs)} features have significantly different correlations")
    print(f"  • {len(large_changes)} features have >30% coefficient changes")
    print()
    print("Conclusion:")
    print("  24-25 season had DIFFERENT feature-outcome relationships.")
    print("  Adding it to training caused Ridge to fit these anomalous patterns.")
    print("  Those patterns don't apply to 25-26 → poor performance.")
    print()
    print("Recommendation: Use OLD model (21-24) which learned stable patterns.")
    
elif evidence_count >= 1:
    print("⚠️  MODERATE EVIDENCE: Some differences between 21-24 and 24-25")
    print()
    print(f"  • {len(significant_diffs)} significant correlation differences")
    print(f"  • {len(large_changes)} large coefficient changes")
    print()
    print("24-25 may have some unique patterns but not drastically different.")
    
else:
    print("✓ WEAK EVIDENCE: 21-24 and 24-25 appear similar")
    print()
    print("Feature-outcome relationships are relatively stable.")
    print("The NEW model's poor performance may be due to other factors.")

print()
print("="*80)

