"""
Test Advanced Correction Approaches

This script implements and tests:
1. Time-varying HCA adjustment
2. Meta-model error correction
3. Combined approach

Compares all approaches on 2024-2025 data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" ADVANCED CORRECTION APPROACHES TEST")
print("="*80)

# Load data
df_val = pd.read_csv('artifacts/margin_test/val_predictions.csv')
df_test = pd.read_csv('artifacts/margin_test/2425_predictions.csv')

# Add derived columns
for df in [df_val, df_test]:
    df['actual_margin'] = df['actual_home'] - df['actual_away']
    df['pred_error'] = df['actual_margin'] - df['pred_margin_mu']
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['ats_correct'] = (df['pred_home_covers'] == df['actual_home_covers']).astype(int)
    df['favorite'] = np.where(df['market_spread_home'] < 0, 'home_fav', 'away_fav')
    df['spread_abs'] = np.abs(df['market_spread_home'])
    df['date'] = pd.to_datetime(df['date'])

# Sort by date
df_val = df_val.sort_values('date').reset_index(drop=True)
df_test = df_test.sort_values('date').reset_index(drop=True)

# ============================================================================
# APPROACH 1: TIME-VARYING HCA
# ============================================================================

print("\n" + "="*80)
print(" APPROACH 1: TIME-VARYING HOME COURT ADVANTAGE")
print("="*80)

# 1.1: Analyze HCA trend in validation data
print("\n[1.1] Analyzing HCA Trend in Validation Data (2021-2024)")
print("-"*80)

# Add time index (months since start)
val_start = df_val['date'].min()
df_val['months_since_start'] = (df_val['date'] - val_start).dt.days / 30.0

# Estimate HCA over time using rolling window
window_months = 3  # 3-month rolling window
df_val['rolling_hca'] = df_val['actual_margin'].rolling(window=30, min_periods=10).mean()

# Also fit linear trend
months = df_val['months_since_start'].values
margins = df_val['actual_margin'].values
hca_trend_slope, hca_trend_intercept = np.polyfit(months, margins, 1)

print(f"\nLinear Trend Analysis:")
print(f"  Slope:     {hca_trend_slope:+.4f} points per month")
print(f"  Start HCA: {hca_trend_intercept:+.3f} points")
print(f"  End HCA:   {hca_trend_intercept + hca_trend_slope * months.max():+.3f} points")
print(f"  Total change over period: {hca_trend_slope * months.max():+.3f} points")

# Test statistical significance
correlation, p_value = stats.spearmanr(months, margins)
print(f"\nSpearman correlation: {correlation:+.3f} (p={p_value:.4f})")

if p_value < 0.05:
    if hca_trend_slope < 0:
        print("  [!] HCA significantly DECLINING over time")
    else:
        print("  [!] HCA significantly INCREASING over time")
else:
    print("  [+] No significant trend (random walk)")

# Show HCA by 6-month periods
print(f"\nHCA by 6-Month Periods:")
df_val['period'] = (df_val['months_since_start'] / 6).astype(int)
period_hca = df_val.groupby('period').agg({
    'actual_margin': ['mean', 'count'],
    'date': ['min', 'max']
})
for period in sorted(df_val['period'].unique()):
    period_data = df_val[df_val['period'] == period]
    mean_margin = period_data['actual_margin'].mean()
    count = len(period_data)
    date_range = f"{period_data['date'].min().strftime('%Y-%m')} to {period_data['date'].max().strftime('%Y-%m')}"
    print(f"  Period {period} ({date_range}): {mean_margin:+.3f} (n={count})")

# 1.2: Apply time-varying HCA to test data
print("\n\n[1.2] Applying Time-Varying HCA to 2024-2025")
print("-"*80)

# Set test data time index (continuation from validation)
test_start_offset = (df_test['date'].min() - val_start).days / 30.0
df_test['months_since_start'] = (df_test['date'] - val_start).dt.days / 30.0

# Historical HCA (average from validation)
historical_hca = df_val['actual_margin'].mean()
print(f"\nHistorical HCA (2021-2024 avg): {historical_hca:+.3f} points")

# Method 1: Linear trend extrapolation
df_test['predicted_hca_linear'] = hca_trend_intercept + hca_trend_slope * df_test['months_since_start']
df_test['hca_adjustment_linear'] = df_test['predicted_hca_linear'] - historical_hca
df_test['pred_margin_hca_linear'] = df_test['pred_margin_mu'] + df_test['hca_adjustment_linear']

# Method 2: Use last known HCA from validation (simple)
last_val_hca = df_val['actual_margin'].tail(60).mean()  # Last ~2 months
df_test['hca_adjustment_simple'] = last_val_hca - historical_hca
df_test['pred_margin_hca_simple'] = df_test['pred_margin_mu'] + df_test['hca_adjustment_simple']

print(f"\nMethod 1 - Linear Trend:")
print(f"  Predicted HCA at test start: {df_test['predicted_hca_linear'].iloc[0]:+.3f}")
print(f"  Predicted HCA at test end:   {df_test['predicted_hca_linear'].iloc[-1]:+.3f}")
print(f"  Average adjustment:          {df_test['hca_adjustment_linear'].mean():+.3f}")

print(f"\nMethod 2 - Last Known HCA:")
print(f"  Last validation HCA:  {last_val_hca:+.3f}")
print(f"  Adjustment:           {df_test['hca_adjustment_simple'].mean():+.3f}")

# Calculate actual HCA in test data (for comparison)
actual_test_hca = df_test['actual_margin'].mean()
print(f"\nActual 2024-2025 HCA: {actual_test_hca:+.3f} points")
print(f"  Prediction accuracy (linear):  error = {abs(df_test['predicted_hca_linear'].mean() - actual_test_hca):.3f}")
print(f"  Prediction accuracy (simple):  error = {abs(last_val_hca - actual_test_hca):.3f}")

# Evaluate ATS performance
def eval_ats(df, pred_col, label):
    df[f'{pred_col}_covers'] = (df[pred_col] > -df['market_spread_home']).astype(int)
    ats_acc = (df[f'{pred_col}_covers'] == df['actual_home_covers']).mean()
    mae = np.abs(df[pred_col] - df['actual_margin']).mean()
    
    print(f"\n{label}:")
    print(f"  ATS Accuracy: {ats_acc*100:.2f}%")
    print(f"  Margin MAE:   {mae:.2f} points")
    return ats_acc, mae

print("\n\n[1.3] Time-Varying HCA Results on 2024-2025")
print("-"*80)

baseline_ats, baseline_mae = eval_ats(df_test, 'pred_margin_mu', 'Baseline (no adjustment)')
linear_ats, linear_mae = eval_ats(df_test, 'pred_margin_hca_linear', 'HCA Linear Trend')
simple_ats, simple_mae = eval_ats(df_test, 'pred_margin_hca_simple', 'HCA Simple (last known)')

print(f"\nImprovement over baseline:")
print(f"  Linear: {(linear_ats - baseline_ats)*100:+.2f}% ATS, {baseline_mae - linear_mae:+.2f} MAE")
print(f"  Simple: {(simple_ats - baseline_ats)*100:+.2f}% ATS, {baseline_mae - simple_mae:+.2f} MAE")

# ============================================================================
# APPROACH 2: META-MODEL ERROR CORRECTION
# ============================================================================

print("\n\n" + "="*80)
print(" APPROACH 2: META-MODEL ERROR CORRECTION")
print("="*80)

# 2.1: Build meta-features
print("\n[2.1] Building Meta-Features")
print("-"*80)

def build_meta_features(df):
    """Build features for meta-model."""
    meta_df = pd.DataFrame()
    
    # Original predictions
    meta_df['pred_margin_mu'] = df['pred_margin_mu']
    meta_df['pred_margin_sigma'] = df['pred_margin_sigma']
    
    # Market context
    meta_df['market_spread_home'] = df['market_spread_home']
    meta_df['baseline_margin'] = df['baseline_margin']
    meta_df['spread_abs'] = df['spread_abs']
    
    # Derived features
    meta_df['is_home_fav'] = (df['market_spread_home'] < 0).astype(int)
    meta_df['is_away_fav'] = (df['market_spread_home'] > 0).astype(int)
    meta_df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    meta_df['distance_from_spread'] = df['pred_margin_mu'] + df['market_spread_home']
    meta_df['model_market_diff'] = df['pred_margin_mu'] - df['baseline_margin']
    
    # Interaction features
    meta_df['home_fav_pred_home'] = meta_df['is_home_fav'] * meta_df['pred_home_covers']
    meta_df['home_fav_pred_away'] = meta_df['is_home_fav'] * (1 - meta_df['pred_home_covers'])
    meta_df['away_fav_pred_home'] = meta_df['is_away_fav'] * meta_df['pred_home_covers']
    meta_df['away_fav_pred_away'] = meta_df['is_away_fav'] * (1 - meta_df['pred_home_covers'])
    
    # Spread buckets
    meta_df['spread_0_3'] = (df['spread_abs'] < 3).astype(int)
    meta_df['spread_3_6'] = ((df['spread_abs'] >= 3) & (df['spread_abs'] < 6)).astype(int)
    meta_df['spread_6_10'] = ((df['spread_abs'] >= 6) & (df['spread_abs'] < 10)).astype(int)
    
    # Confidence indicators
    meta_df['high_distance'] = (np.abs(meta_df['distance_from_spread']) > 5).astype(int)
    meta_df['low_distance'] = (np.abs(meta_df['distance_from_spread']) < 2).astype(int)
    
    # Time features (if available)
    if 'months_since_start' in df.columns:
        meta_df['months_since_start'] = df['months_since_start']
        meta_df['season_progress'] = df['months_since_start'] / df['months_since_start'].max()
    
    return meta_df

val_meta_features = build_meta_features(df_val)
test_meta_features = build_meta_features(df_test)

print(f"Meta-features created: {len(val_meta_features.columns)} features")
print(f"Features: {', '.join(val_meta_features.columns[:10])}...")

# 2.2: Train meta-model with time-series cross-validation
print("\n\n[2.2] Training Meta-Model with Time-Series CV")
print("-"*80)

# Use first 70% for meta-model training, last 30% for meta-model validation
split_idx = int(len(df_val) * 0.7)
val_meta_train = val_meta_features.iloc[:split_idx]
val_meta_valid = val_meta_features.iloc[split_idx:]
val_error_train = df_val['pred_error'].iloc[:split_idx]
val_error_valid = df_val['pred_error'].iloc[split_idx:]

print(f"\nMeta-model training set: {len(val_meta_train)} games")
print(f"Meta-model validation set: {len(val_meta_valid)} games")

# Train meta-model
meta_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

print(f"\nTraining Gradient Boosting meta-model...")
meta_model.fit(val_meta_train, val_error_train)

# Validate on held-out validation set
val_meta_pred_error = meta_model.predict(val_meta_valid)
val_meta_corrected = df_val['pred_margin_mu'].iloc[split_idx:] + val_meta_pred_error

print(f"\nMeta-model performance on validation holdout:")
baseline_val_mae = np.abs(df_val['pred_margin_mu'].iloc[split_idx:] - df_val['actual_margin'].iloc[split_idx:]).mean()
meta_val_mae = np.abs(val_meta_corrected - df_val['actual_margin'].iloc[split_idx:]).mean()
print(f"  Baseline MAE: {baseline_val_mae:.2f}")
print(f"  Meta MAE:     {meta_val_mae:.2f}")
print(f"  Improvement:  {baseline_val_mae - meta_val_mae:+.2f}")

# Feature importance
print(f"\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': val_meta_features.columns,
    'importance': meta_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")

# 2.3: Apply meta-model to test data
print("\n\n[2.3] Applying Meta-Model to 2024-2025")
print("-"*80)

test_meta_pred_error = meta_model.predict(test_meta_features)
df_test['pred_margin_meta'] = df_test['pred_margin_mu'] + test_meta_pred_error

meta_ats, meta_mae = eval_ats(df_test, 'pred_margin_meta', 'Meta-Model Corrected')

print(f"\nImprovement over baseline:")
print(f"  ATS: {(meta_ats - baseline_ats)*100:+.2f}%")
print(f"  MAE: {baseline_mae - meta_mae:+.2f} points")

# Analyze what the meta-model learned
print(f"\nMeta-Model Error Predictions by Subgroup:")
print("-"*60)

# By favorite type
for fav in ['home_fav', 'away_fav']:
    fav_mask = df_test['favorite'] == fav
    avg_correction = test_meta_pred_error[fav_mask].mean()
    fav_label = 'Home Favorites' if fav == 'home_fav' else 'Away Favorites'
    print(f"  {fav_label}: {avg_correction:+.3f} points")

# By favorite + prediction direction
scenarios = [
    ('Home fav, pred home', (df_test['favorite']=='home_fav') & (df_test['pred_home_covers']==1)),
    ('Home fav, pred away', (df_test['favorite']=='home_fav') & (df_test['pred_home_covers']==0)),
    ('Away fav, pred home', (df_test['favorite']=='away_fav') & (df_test['pred_home_covers']==1)),
    ('Away fav, pred away', (df_test['favorite']=='away_fav') & (df_test['pred_home_covers']==0)),
]

for label, mask in scenarios:
    if mask.sum() > 0:
        avg_correction = test_meta_pred_error[mask].mean()
        print(f"  {label}: {avg_correction:+.3f} points")

# ============================================================================
# APPROACH 3: COMBINED (HCA + META-MODEL)
# ============================================================================

print("\n\n" + "="*80)
print(" APPROACH 3: COMBINED (HCA + META-MODEL)")
print("="*80)

print("\n[3.1] Building Combined Approach")
print("-"*80)

# Apply HCA adjustment first, then use meta-model on the adjusted predictions
df_test['pred_margin_hca_adj'] = df_test['pred_margin_hca_simple']  # Use simple HCA adjustment

# Build meta-features from HCA-adjusted predictions
test_combined_features = build_meta_features(df_test.assign(
    pred_margin_mu=df_test['pred_margin_hca_adj']
))

# Predict error on HCA-adjusted predictions
test_combined_error = meta_model.predict(test_combined_features)
df_test['pred_margin_combined'] = df_test['pred_margin_hca_adj'] + test_combined_error

combined_ats, combined_mae = eval_ats(df_test, 'pred_margin_combined', 'Combined (HCA + Meta)')

print(f"\nImprovement over baseline:")
print(f"  ATS: {(combined_ats - baseline_ats)*100:+.2f}%")
print(f"  MAE: {baseline_mae - combined_mae:+.2f} points")

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("\n\n" + "="*80)
print(" FINAL COMPARISON - ALL APPROACHES")
print("="*80)

results = []

# Baseline
results.append({
    'Approach': 'Baseline (Original)',
    'ATS': baseline_ats,
    'MAE': baseline_mae,
    'ATS_vs_baseline': 0.0,
    'MAE_vs_baseline': 0.0
})

# HCA Linear
results.append({
    'Approach': 'HCA Linear Trend',
    'ATS': linear_ats,
    'MAE': linear_mae,
    'ATS_vs_baseline': (linear_ats - baseline_ats) * 100,
    'MAE_vs_baseline': baseline_mae - linear_mae
})

# HCA Simple
results.append({
    'Approach': 'HCA Simple (last known)',
    'ATS': simple_ats,
    'MAE': simple_mae,
    'ATS_vs_baseline': (simple_ats - baseline_ats) * 100,
    'MAE_vs_baseline': baseline_mae - simple_mae
})

# Meta-model
results.append({
    'Approach': 'Meta-Model Only',
    'ATS': meta_ats,
    'MAE': meta_mae,
    'ATS_vs_baseline': (meta_ats - baseline_ats) * 100,
    'MAE_vs_baseline': baseline_mae - meta_mae
})

# Combined
results.append({
    'Approach': 'Combined (HCA + Meta)',
    'ATS': combined_ats,
    'MAE': combined_mae,
    'ATS_vs_baseline': (combined_ats - baseline_ats) * 100,
    'MAE_vs_baseline': baseline_mae - combined_mae
})

results_df = pd.DataFrame(results)

print(f"\n{'Approach':<30} {'ATS':<12} {'MAE':<12} {'ATS Improve':<12} {'MAE Improve':<12} {'Status'}")
print("-" * 90)

for _, row in results_df.iterrows():
    status = ""
    if row['ATS_vs_baseline'] > 1.5:
        status = "[+] Good"
    elif row['ATS_vs_baseline'] > 0.5:
        status = "[~] Slight"
    
    print(f"{row['Approach']:<30} {row['ATS']*100:>6.2f}%    {row['MAE']:>6.2f} pts   {row['ATS_vs_baseline']:>+6.2f}%      {row['MAE_vs_baseline']:>+6.2f} pts   {status}")

# Determine best approach
best_ats = results_df.loc[results_df['ATS'].idxmax()]
best_mae = results_df.loc[results_df['MAE'].idxmin()]

print(f"\n{'='*80}")
print(" RECOMMENDATION")
print('='*80)

print(f"\nBest ATS Accuracy: {best_ats['Approach']} ({best_ats['ATS']*100:.2f}%)")
print(f"Best MAE:          {best_mae['Approach']} ({best_mae['MAE']:.2f} points)")

if best_ats['Approach'] == best_mae['Approach']:
    print(f"\n[CLEAR WINNER] {best_ats['Approach']}")
    print(f"  - Improves ATS by {best_ats['ATS_vs_baseline']:+.2f}%")
    print(f"  - Improves MAE by {best_ats['MAE_vs_baseline']:+.2f} points")
else:
    print(f"\n[SPLIT DECISION]")
    print(f"  - For betting (ATS): Use {best_ats['Approach']}")
    print(f"  - For accuracy (MAE): Use {best_mae['Approach']}")

# Check if profitable
breakeven = 0.5238
if best_ats['ATS'] > breakeven:
    roi = (best_ats['ATS'] * 0.91 + (1 - best_ats['ATS']) * -1) * 100
    print(f"\n[!] PROFITABILITY CHECK:")
    print(f"    Best ATS: {best_ats['ATS']*100:.2f}% > {breakeven*100:.2f}% breakeven")
    print(f"    Estimated ROI: {roi:+.2f}%")
    if roi > 0:
        print(f"    Status: PROFITABLE")
    else:
        print(f"    Status: Not quite profitable yet")
else:
    print(f"\n[!] Still below breakeven ({best_ats['ATS']*100:.2f}% < {breakeven*100:.2f}%)")

print(f"\n{'='*80}")
print(" ANALYSIS COMPLETE")
print('='*80)

# Save corrected predictions for further analysis
output_df = df_test[['game_id', 'date', 'home_team', 'away_team', 'market_spread_home',
                      'pred_margin_mu', 'pred_margin_hca_simple', 'pred_margin_meta', 
                      'pred_margin_combined', 'actual_margin']].copy()
output_df.to_csv('artifacts/margin_test/corrected_predictions.csv', index=False)
print(f"\nCorrected predictions saved to: artifacts/margin_test/corrected_predictions.csv")

