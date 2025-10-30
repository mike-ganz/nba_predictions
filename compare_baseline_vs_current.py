"""Compare baseline model (no players) vs current model (with players)"""
import pandas as pd
import numpy as np

print("\n" + "="*80)
print("BASELINE vs CURRENT MODEL COMPARISON (2024-2025 Season)")
print("="*80 + "\n")

# Load predictions from both models
baseline_df = pd.read_csv('reports/baseline_2425/per_game_predictions.csv')
current_df = pd.read_csv('reports/2425_with_players/per_game_predictions.csv')

print(f"Games evaluated: {len(baseline_df)} (baseline) vs {len(current_df)} (current)\n")

# Calculate ATS performance for both
for name, df in [("BASELINE (No Players)", baseline_df), ("CURRENT (With Players)", current_df)]:
    # Calculate ATS correctness
    df['home_margin'] = df['actual_home'] - df['actual_away']
    df['home_covers'] = df['home_margin'] + df['market_spread_home'] > 0
    df['away_covers'] = df['home_margin'] + df['market_spread_home'] < 0
    df['model_pick_home'] = df['cover_prob_home'] > df['cover_prob_away']
    df['ats_correct'] = np.where(
        df['model_pick_home'],
        df['home_covers'],
        df['away_covers']
    )
    
    # Moneyline
    df['pred_winner'] = (df['pred_home'] > df['pred_away']).astype(int)
    df['actual_winner'] = (df['actual_home'] > df['actual_away']).astype(int)
    df['ml_correct'] = (df['pred_winner'] == df['actual_winner'])
    
    # Totals
    df['predicted_total'] = df['pred_home'] + df['pred_away']
    df['actual_total'] = df['actual_home'] + df['actual_away']
    df['model_pick_over'] = df['predicted_total'] > df['market_total']
    df['actual_over'] = df['actual_total'] > df['market_total']
    df['total_correct'] = df['model_pick_over'] == df['actual_over']
    
    # Errors
    df['pred_margin'] = df['pred_home'] - df['pred_away']
    df['actual_margin'] = df['actual_home'] - df['actual_away']
    df['margin_error'] = abs(df['pred_margin'] - df['actual_margin'])
    df['total_error'] = abs(df['predicted_total'] - df['actual_total'])
    
    # ROI calculation
    df['ats_profit'] = np.where(df['ats_correct'], 1.0, -1.1)
    
    # Confidence
    df['max_cover_prob'] = df[['cover_prob_home', 'cover_prob_away']].max(axis=1)

# Print comparison
print("="*80)
print("PERFORMANCE COMPARISON")
print("="*80 + "\n")

metrics = [
    ('ATS Accuracy', 'ats_correct', 'mean', '%'),
    ('Moneyline Accuracy', 'ml_correct', 'mean', '%'),
    ('Total Accuracy', 'total_correct', 'mean', '%'),
    ('Mean Margin Error', 'margin_error', 'mean', 'pts'),
    ('Mean Total Error', 'total_error', 'mean', 'pts'),
]

print(f"{'Metric':<25} {'Baseline':<15} {'Current':<15} {'Difference':<15}")
print("-" * 80)

for metric_name, col, agg, unit in metrics:
    baseline_val = getattr(baseline_df[col], agg)()
    current_val = getattr(current_df[col], agg)()
    diff = baseline_val - current_val
    
    if unit == '%':
        print(f"{metric_name:<25} {baseline_val:>14.1%} {current_val:>14.1%} {diff:>+14.1%}")
    else:
        print(f"{metric_name:<25} {baseline_val:>14.2f} {current_val:>14.2f} {diff:>+14.2f}")

# ROI
baseline_roi = (baseline_df['ats_profit'].sum() / (len(baseline_df) * 1.1)) * 100
current_roi = (current_df['ats_profit'].sum() / (len(current_df) * 1.1)) * 100

print(f"{'ATS ROI':<25} {baseline_roi:>13.2f}% {current_roi:>13.2f}% {baseline_roi - current_roi:>+13.2f}%")

# High confidence picks
print("\n" + "="*80)
print("HIGH CONFIDENCE PICKS (>60% cover probability)")
print("="*80 + "\n")

for name, df in [("Baseline", baseline_df), ("Current", current_df)]:
    high_conf = df[df['max_cover_prob'] > 0.6]
    if len(high_conf) > 0:
        hc_acc = high_conf['ats_correct'].mean()
        hc_profit = high_conf['ats_profit'].sum()
        hc_roi = (hc_profit / (len(high_conf) * 1.1)) * 100
        print(f"{name:>10}: {len(high_conf):4d} games, {hc_acc:.1%} accuracy, {hc_roi:+.2f}% ROI")

# Calibration check
print("\n" + "="*80)
print("CALIBRATION (Confidence vs Accuracy)")
print("="*80 + "\n")

baseline_df['confidence_bucket'] = pd.cut(baseline_df['max_cover_prob'],
                                           bins=[0.5, 0.6, 0.7, 1.0],
                                           labels=['50-60%', '60-70%', '70%+'])
current_df['confidence_bucket'] = pd.cut(current_df['max_cover_prob'],
                                          bins=[0.5, 0.6, 0.7, 1.0],
                                          labels=['50-60%', '60-70%', '70%+'])

print(f"{'Confidence':<15} {'Baseline Acc':<15} {'Current Acc':<15} {'Difference'}")
print("-" * 60)

for bucket in ['50-60%', '60-70%', '70%+']:
    baseline_subset = baseline_df[baseline_df['confidence_bucket'] == bucket]
    current_subset = current_df[current_df['confidence_bucket'] == bucket]
    
    if len(baseline_subset) > 0 and len(current_subset) > 0:
        baseline_acc = baseline_subset['ats_correct'].mean()
        current_acc = current_subset['ats_correct'].mean()
        diff = baseline_acc - current_acc
        
        print(f"{bucket:<15} {baseline_acc:>14.1%} {current_acc:>14.1%} {diff:>+14.1%}")

# Correlation check
baseline_corr = baseline_df['max_cover_prob'].corr(baseline_df['ats_correct'])
current_corr = current_df['max_cover_prob'].corr(current_df['ats_correct'])

print(f"\nCalibration correlation:")
print(f"  Baseline: {baseline_corr:+.3f}")
print(f"  Current:  {current_corr:+.3f}")

# Final verdict
print("\n" + "="*80)
print("VERDICT")
print("="*80 + "\n")

baseline_ats = baseline_df['ats_correct'].mean()
current_ats = current_df['ats_correct'].mean()
ats_diff = baseline_ats - current_ats

if baseline_ats > current_ats + 0.01:  # At least 1% better
    print("BASELINE MODEL PERFORMS BETTER")
    print(f"  ATS Improvement: {ats_diff:+.1%} ({ats_diff*100:+.1f} percentage points)")
    print(f"  ROI Improvement: {baseline_roi - current_roi:+.2f}%")
    print("\nCONCLUSION: Player features are HURTING the model.")
    print("RECOMMENDATION: Remove player features from production model.")
    
    if baseline_ats > 0.524:
        print(f"\n✓ Baseline beats breakeven (52.4%)!")
        print(f"  Edge: {(baseline_ats - 0.524)*100:+.1f} percentage points")
    else:
        print(f"\n✗ Baseline still below breakeven (52.4%)")
        print(f"  Shortfall: {(baseline_ats - 0.524)*100:.1f} percentage points")
        print("  Further improvements needed.")
        
elif current_ats > baseline_ats + 0.01:
    print("CURRENT MODEL PERFORMS BETTER")
    print(f"  ATS Advantage: {-ats_diff:+.1%}")
    print("\nCONCLUSION: Player features ARE helping (or baseline is flawed).")
    print("RECOMMENDATION: Keep player features but investigate calibration issues.")
    
else:
    print("BOTH MODELS PERFORM SIMILARLY")
    print(f"  ATS Difference: {ats_diff:+.1%} (negligible)")
    print("\nCONCLUSION: Player features neither help nor hurt significantly.")
    print("RECOMMENDATION: Focus on calibration improvements rather than features.")

print("\n" + "="*80)

