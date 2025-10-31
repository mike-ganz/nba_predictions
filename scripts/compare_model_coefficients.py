"""
Compare model coefficients between old and new models.
See if the weights learned changed significantly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

print("="*100)
print("MODEL COEFFICIENTS COMPARISON")
print("="*100)
print()

# Load both models
print("Loading models...")
old_model = joblib.load('artifacts/margin_normalized/margin_model.joblib')
new_model = joblib.load('artifacts/margin_normalized_21_25/margin_model.joblib')

print("✅ Models loaded")
print()

# Get coefficients for mean prediction
old_coef = old_model.model_mean.coef_
new_coef = new_model.model_mean.coef_

print(f"Old model coefficients shape: {old_coef.shape}")
print(f"New model coefficients shape: {new_coef.shape}")
print()

# Feature names (we know there are 29 features)
feature_names = [
    'home_edge', 'home_orb_edge', 'home_tov_edge', 'home_tpar', 'home_ftr', 'home_rest_days',
    'home_minutes_missing_top2', 'home_star_out', 'home_usage_share_top2',
    'away_edge', 'away_orb_edge', 'away_tov_edge', 'away_tpar', 'away_ftr', 'away_rest_days',
    'away_minutes_missing_top2', 'away_star_out', 'away_usage_share_top2',
    'shared_pace_mean', 'shared_implied_home_winprob', 'shared_implied_away_winprob',
    'shared_team_weighted_ts_home',  # Note: shared_team_weighted_ts_away excluded
    'diff_edge', 'diff_orb_edge', 'diff_tov_edge', 'diff_tpar', 'diff_ftr', 'diff_rest_days',
    'diff_minutes_missing_top2'
]

# Create comparison dataframe
comparison = pd.DataFrame({
    'feature': feature_names,
    'old_coef': old_coef,
    'new_coef': new_coef,
    'diff': new_coef - old_coef,
    'pct_change': ((new_coef - old_coef) / np.abs(old_coef) * 100)
})

comparison['abs_diff'] = np.abs(comparison['diff'])
comparison = comparison.sort_values('abs_diff', ascending=False)

print("="*100)
print("TOP 15 LARGEST COEFFICIENT CHANGES")
print("="*100)
print()

print(f"{'Feature':<35} {'Old Coef':<12} {'New Coef':<12} {'Difference':<12} {'% Change':<12}")
print("-"*100)

for _, row in comparison.head(15).iterrows():
    pct_str = f"{row['pct_change']:+.1f}%" if not np.isnan(row['pct_change']) and not np.isinf(row['pct_change']) else "N/A"
    print(f"{row['feature']:<35} {row['old_coef']:>11.4f} {row['new_coef']:>11.4f} {row['diff']:>11.4f} {pct_str:>11}")

print()

# Check if any coefficients flipped sign
sign_flips = comparison[(comparison['old_coef'] * comparison['new_coef']) < 0]
if len(sign_flips) > 0:
    print("="*100)
    print("⚠️  COEFFICIENTS THAT FLIPPED SIGN")
    print("="*100)
    print()
    print(f"{'Feature':<35} {'Old Coef':<12} {'New Coef':<12}")
    print("-"*100)
    for _, row in sign_flips.iterrows():
        print(f"{row['feature']:<35} {row['old_coef']:>11.4f} {row['new_coef']:>11.4f}")
    print()

# Summarize changes
print("="*100)
print("SUMMARY STATISTICS")
print("="*100)
print()

avg_abs_diff = comparison['abs_diff'].mean()
max_abs_diff = comparison['abs_diff'].max()
num_large_changes = (comparison['abs_diff'] > 0.1).sum()

print(f"Average absolute coefficient change: {avg_abs_diff:.4f}")
print(f"Maximum absolute coefficient change: {max_abs_diff:.4f}")
print(f"Number of features with |change| > 0.1: {num_large_changes}")
print(f"Number of sign flips: {len(sign_flips)}")
print()

# Check if this is significant
if max_abs_diff > 0.5:
    print(f"⚠️  SIGNIFICANT COEFFICIENT CHANGES DETECTED!")
    print(f"    Max change of {max_abs_diff:.4f} is substantial.")
    print()
elif num_large_changes > 5:
    print(f"⚠️  MANY COEFFICIENTS CHANGED SUBSTANTIALLY!")
    print(f"    {num_large_changes} features changed by more than 0.1.")
    print()
elif len(sign_flips) > 0:
    print(f"⚠️  SIGN FLIPS DETECTED!")
    print(f"    {len(sign_flips)} coefficients changed sign - model learned opposite relationships!")
    print()
else:
    print("✅ Coefficient changes are relatively modest.")
    print()

# Check variance model too
print("="*100)
print("VARIANCE MODEL COEFFICIENTS")
print("="*100)
print()

old_var_coef = old_model.model_variance.coef_
new_var_coef = new_model.model_variance.coef_

var_comparison = pd.DataFrame({
    'feature': feature_names,
    'old_coef': old_var_coef,
    'new_coef': new_var_coef,
    'diff': new_var_coef - old_var_coef,
})

var_comparison['abs_diff'] = np.abs(var_comparison['diff'])
var_comparison = var_comparison.sort_values('abs_diff', ascending=False)

print("TOP 10 LARGEST VARIANCE COEFFICIENT CHANGES:")
print()
print(f"{'Feature':<35} {'Old Coef':<12} {'New Coef':<12} {'Difference':<12}")
print("-"*100)

for _, row in var_comparison.head(10).iterrows():
    print(f"{row['feature']:<35} {row['old_coef']:>11.4f} {row['new_coef']:>11.4f} {row['diff']:>11.4f}")

print()

# Overall assessment
print("="*100)
print("OVERALL ASSESSMENT")
print("="*100)
print()

print("Comparing the models' learned weights:")
print()

if max_abs_diff > 1.0 or num_large_changes > 10 or len(sign_flips) > 2:
    print("🔴 MAJOR DIFFERENCES DETECTED")
    print()
    print("The new model learned SIGNIFICANTLY DIFFERENT relationships.")
    print("Key observations:")
    print(f"  - Largest coefficient change: {max_abs_diff:.4f}")
    print(f"  - Features with large changes: {num_large_changes}")
    print(f"  - Sign flips: {len(sign_flips)}")
    print()
    print("This explains the performance difference!")
    print("The new model's weights are tuned to 24-25 patterns that don't generalize.")
    
elif max_abs_diff > 0.3 or num_large_changes > 5:
    print("🟡 MODERATE DIFFERENCES DETECTED")
    print()
    print("The new model learned somewhat different relationships.")
    print("These changes could explain the performance gap on 25-26.")
    
else:
    print("🟢 MINOR DIFFERENCES")
    print()
    print("The models learned similar relationships.")
    print("The performance difference may be due to other factors.")

print()
print("="*100)

# Save comparison
comparison.to_csv('predictions/coefficient_comparison.csv', index=False)
print("\n✅ Full comparison saved to: predictions/coefficient_comparison.csv")
print("="*100)

