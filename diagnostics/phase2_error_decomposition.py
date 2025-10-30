"""Phase 2: Error Decomposition - Where and how does the model fail?"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("PHASE 2: ERROR DECOMPOSITION")
print("="*70 + "\n")

# Load data
val_df = pd.read_csv('reports/val_with_players/per_game_predictions.csv')
test_df = pd.read_csv('reports/2425_with_players/per_game_predictions.csv')

# Add derived columns
for df in [val_df, test_df]:
    df['pred_margin'] = df['pred_home'] - df['pred_away']
    df['actual_margin'] = df['actual_home'] - df['actual_away']
    df['margin_error'] = df['pred_margin'] - df['actual_margin']
    df['abs_margin_error'] = abs(df['margin_error'])
    df['correct_direction'] = np.sign(df['pred_margin']) == np.sign(df['actual_margin'])
    
    # Market analysis
    df['market_implied_margin'] = -df['market_spread_home']
    df['deviation_from_market'] = df['pred_margin'] - df['market_implied_margin']

# =============================================================================
# TEST 2.1: Directional Bias Analysis
# =============================================================================
print("TEST 2.1: Directional Bias Analysis")
print("-" * 70)

for name, df in [("Validation", val_df), ("2024-2025", test_df)]:
    print(f"\n{name}:")
    
    # Direction prediction
    dir_acc = df['correct_direction'].mean()
    print(f"  Directional accuracy: {dir_acc:.1%}")
    print(f"  Moneyline accuracy: {df['ml_correct'].mean():.1%}")
    
    # Bias check
    pred_margin_avg = df['pred_margin'].mean()
    actual_margin_avg = df['actual_margin'].mean()
    bias = df['margin_error'].mean()
    
    print(f"\n  Predicted margin (avg): {pred_margin_avg:+.2f}")
    print(f"  Actual margin (avg): {actual_margin_avg:+.2f}")
    print(f"  Systematic bias: {bias:+.2f} points")
    
    if abs(bias) > 1:
        if bias > 0:
            print(f"  ⚠️  Model over-predicts home team margin")
        else:
            print(f"  ⚠️  Model over-predicts away team margin")
    else:
        print(f"  ✓ No significant systematic bias")
    
    # Market deviation
    avg_dev = df['deviation_from_market'].mean()
    abs_dev = abs(df['deviation_from_market']).mean()
    print(f"\n  Avg deviation from market: {avg_dev:+.2f}")
    print(f"  Avg absolute deviation: {abs_dev:.2f}")
    
    if abs_dev < 2:
        print(f"  ⚠️  Model is very close to market (possible redundancy)")

# Check if directional accuracy ≠ moneyline accuracy
test_dir = test_df['correct_direction'].mean()
test_ml = test_df['ml_correct'].mean()
if abs(test_dir - test_ml) > 0.05:
    print(f"\n🚨 CRITICAL ISSUE:")
    print(f"   Directional accuracy ({test_dir:.1%}) differs from moneyline ({test_ml:.1%})")
    print(f"   This suggests win probability calculation is broken!")

# =============================================================================
# TEST 2.2: Error Magnitude Analysis
# =============================================================================
print("\n\nTEST 2.2: Error Magnitude Analysis")
print("-" * 70)

for name, df in [("Validation", val_df), ("2024-2025", test_df)]:
    print(f"\n{name}:")
    
    # Error distribution
    print(f"  Mean absolute error: {df['abs_margin_error'].mean():.2f} points")
    print(f"  Median absolute error: {df['abs_margin_error'].median():.2f} points")
    print(f"  Std dev of errors: {df['margin_error'].std():.2f} points")
    
    # Error buckets
    df['error_bucket'] = pd.cut(df['abs_margin_error'], 
                                 bins=[0, 5, 10, 15, 20, 100],
                                 labels=['0-5', '5-10', '10-15', '15-20', '20+'])
    
    print(f"\n  Error distribution:")
    error_dist = df['error_bucket'].value_counts().sort_index()
    for bucket, count in error_dist.items():
        pct = count / len(df) * 100
        print(f"    {bucket} pts: {count:4d} ({pct:5.1f}%)")

# Compare error distributions
print(f"\n📊 Error Comparison:")
val_mae = val_df['abs_margin_error'].mean()
test_mae = test_df['abs_margin_error'].mean()
print(f"  Validation MAE: {val_mae:.2f}")
print(f"  2024-2025 MAE: {test_mae:.2f}")
print(f"  Increase: {test_mae - val_mae:+.2f} points ({(test_mae/val_mae - 1)*100:+.1f}%)")

if test_mae > val_mae * 1.1:
    print(f"  ⚠️  Errors increased significantly on test set")

# =============================================================================
# TEST 2.3: Performance by Game Type
# =============================================================================
print("\n\nTEST 2.3: Performance by Game Type")
print("-" * 70)

# Analyze by spread magnitude
test_df['spread_magnitude'] = abs(test_df['market_spread_home'])
test_df['game_type'] = pd.cut(test_df['spread_magnitude'],
                               bins=[0, 3, 7, 12, 50],
                               labels=['Close (<3)', 'Moderate (3-7)', 
                                      'Large (7-12)', 'Blowout (12+)'])

print("\n2024-2025 Performance by Spread Magnitude:")
print(f"{'Game Type':<20} {'Games':<8} {'ATS%':<8} {'ML%':<8} {'MAE':<8}")
print("-" * 60)

for game_type in ['Close (<3)', 'Moderate (3-7)', 'Large (7-12)', 'Blowout (12+)']:
    subset = test_df[test_df['game_type'] == game_type]
    if len(subset) > 0:
        ats_acc = subset['ats_correct'].mean()
        ml_acc = subset['ml_correct'].mean()
        mae = subset['abs_margin_error'].mean()
        print(f"{game_type:<20} {len(subset):<8} {ats_acc:<7.1%} {ml_acc:<7.1%} {mae:<8.2f}")

# Analyze by team
print("\n\nWorst Teams (2024-2025):")
team_perf = test_df.groupby('home_team').agg({
    'ats_correct': 'mean',
    'game_id': 'count'
}).rename(columns={'game_id': 'games'})
team_perf = team_perf[team_perf['games'] >= 10]
worst_teams = team_perf.nsmallest(5, 'ats_correct')
print(worst_teams)

print("\n\nBest Teams (2024-2025):")
best_teams = team_perf.nlargest(5, 'ats_correct')
print(best_teams)

# Temporal analysis
test_df['date_dt'] = pd.to_datetime(test_df['date'])
test_df['month'] = test_df['date_dt'].dt.month

print("\n\nPerformance by Month (2024-2025):")
print(f"{'Month':<12} {'Games':<8} {'ATS%':<8} {'ML%':<8}")
print("-" * 40)

monthly = test_df.groupby('month').agg({
    'ats_correct': 'mean',
    'ml_correct': 'mean',
    'game_id': 'count'
})

month_names = {10: 'October', 11: 'November', 12: 'December', 
               1: 'January', 2: 'February', 3: 'March',
               4: 'April', 5: 'May', 6: 'June'}

for month, row in monthly.iterrows():
    month_name = month_names.get(month, f'Month {month}')
    print(f"{month_name:<12} {row['game_id']:<8.0f} {row['ats_correct']:<7.1%} {row['ml_correct']:<7.1%}")

# Check for trend
if len(test_df) > 100:
    test_df = test_df.sort_values('date_dt')
    test_df['game_number'] = range(len(test_df))
    
    # Simple linear regression
    from scipy.stats import linregress
    slope, intercept, r, p, se = linregress(test_df['game_number'], 
                                            test_df['ats_correct'].astype(int))
    
    print(f"\n📊 Temporal Trend:")
    print(f"  Slope: {slope:.6f} per game")
    print(f"  P-value: {p:.4f}")
    
    if p < 0.05:
        if slope < 0:
            print(f"  ⚠️  Performance DECLINING over time")
        else:
            print(f"  Performance IMPROVING over time")
    else:
        print(f"  No significant temporal trend")

# =============================================================================
# TEST 2.4: Confidence vs Performance
# =============================================================================
print("\n\nTEST 2.4: Confidence vs Performance")
print("-" * 70)

test_df['max_cover_prob'] = test_df[['cover_prob_home', 'cover_prob_away']].max(axis=1)
test_df['confidence_bucket'] = pd.cut(test_df['max_cover_prob'],
                                       bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.0],
                                       labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+'])

print("\nAccuracy by Confidence Level (2024-2025):")
print(f"{'Confidence':<12} {'Games':<8} {'ATS%':<8} {'Expected%':<12}")
print("-" * 45)

for conf in ['50-55%', '55-60%', '60-65%', '65-70%', '70%+']:
    subset = test_df[test_df['confidence_bucket'] == conf]
    if len(subset) > 0:
        ats_acc = subset['ats_correct'].mean()
        expected = subset['max_cover_prob'].mean()
        print(f"{conf:<12} {len(subset):<8} {ats_acc:<7.1%} {expected:<11.1%}")

# Correlation check
corr = test_df['max_cover_prob'].corr(test_df['ats_correct'])
print(f"\nCorrelation (confidence vs correctness): {corr:+.3f}")

if corr < 0:
    print(f"🚨 CRITICAL: Negative correlation!")
    print(f"   Model is LESS accurate when MORE confident")
    print(f"   This is a severe calibration failure")
elif corr < 0.1:
    print(f"⚠️  Very weak correlation")
    print(f"   Confidence estimates are poorly calibrated")
else:
    print(f"✓ Positive correlation (good)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "="*70)
print("PHASE 2 SUMMARY: KEY FINDINGS")
print("="*70 + "\n")

findings = []

# Check for issues
if abs(test_df['margin_error'].mean()) > 1:
    findings.append(f"Systematic bias: {test_df['margin_error'].mean():+.2f} points")

if abs(test_dir - test_ml) > 0.05:
    findings.append("Directional accuracy ≠ moneyline accuracy (probability issue)")

if test_mae > val_mae * 1.1:
    findings.append(f"Error increased {(test_mae/val_mae - 1)*100:+.1f}% on test set")

if corr < 0:
    findings.append("CRITICAL: Inverted calibration (high confidence = low accuracy)")
elif corr < 0.1:
    findings.append("Poor calibration (confidence uninformative)")

if abs(test_df['deviation_from_market'].mean()) < 2:
    findings.append("Model very close to market (< 2 pts deviation)")

# Temporal trend
if 'slope' in locals() and p < 0.05:
    if slope < 0:
        findings.append("Performance declining over 2024-2025 season")
    else:
        findings.append("Performance improving over 2024-2025 season")

if len(findings) > 0:
    print("🔍 Issues Identified:")
    for i, finding in enumerate(findings, 1):
        print(f"   {i}. {finding}")
else:
    print("✓ No obvious error patterns detected")

print("\n📋 Recommended Next Steps:")
if corr < 0:
    print("   → PRIORITY: Fix calibration (Phase 4)")
if abs(test_df['deviation_from_market'].mean()) < 2:
    print("   → Reduce market blend weight")
if 'slope' in locals() and slope < 0 and p < 0.05:
    print("   → Investigate temporal drift (Phase 5)")

print("\n   → Proceed to Phase 3: Ablation Study")

print("\n" + "="*70)

