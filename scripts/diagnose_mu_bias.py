"""Comprehensive mu bias and threshold optimization analysis.

This script runs:
- Threshold optimization to find optimal decision boundary
- mu bias analysis to understand systematic prediction errors
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print(" MU BIAS & THRESHOLD OPTIMIZATION ANALYSIS")
print("="*70)

# Load predictions
df_val = pd.read_csv('artifacts/margin_test/val_predictions.csv')
df_2425 = pd.read_csv('artifacts/margin_test/2425_predictions.csv')

def analyze_threshold_and_bias(df, label, is_validation=False):
    """Complete threshold and bias analysis."""
    print(f"\n{'='*70}")
    print(f" {label}")
    print('='*70)
    
    # Calculate actual covers
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    
    # =================================================================
    # PART 1: THRESHOLD OPTIMIZATION
    # =================================================================
    print(f"\n{'='*70}")
    print(" PART 1: THRESHOLD OPTIMIZATION")
    print('='*70)
    
    print(f"\nTesting decision thresholds from 0.40 to 0.60...")
    print(f"(If optimal != 0.50, probabilities are systematically miscalibrated)")
    
    thresholds = np.linspace(0.40, 0.60, 41)
    results = []
    
    for thresh in thresholds:
        pred_covers = (df['cover_prob_home'] > thresh).astype(int)
        accuracy = (pred_covers == df['actual_home_covers']).mean()
        roi = (accuracy * 0.91 + (1 - accuracy) * -1.0) * 100
        
        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'roi': roi
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    best_acc = results_df.loc[results_df['accuracy'].idxmax()]
    best_roi = results_df.loc[results_df['roi'].idxmax()]
    baseline_50 = results_df[results_df['threshold'] == 0.50].iloc[0]
    
    print(f"\nResults at key thresholds:")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'ROI':<12}")
    print("-" * 40)
    for t in [0.45, 0.48, 0.50, 0.52, 0.55]:
        row = results_df[results_df['threshold'] == t].iloc[0]
        marker = " <-- BEST" if abs(row['threshold'] - best_acc['threshold']) < 0.001 else ""
        print(f"{row['threshold']:<12.2f} {row['accuracy']*100:>8.2f}%   {row['roi']:>+8.2f}%{marker}")
    
    print(f"\nOptimal threshold: {best_acc['threshold']:.3f}")
    print(f"  Accuracy: {best_acc['accuracy']*100:.2f}% (vs {baseline_50['accuracy']*100:.2f}% at 0.50)")
    print(f"  ROI:      {best_acc['roi']:+.2f}% (vs {baseline_50['roi']:+.2f}% at 0.50)")
    print(f"  Improvement: {(best_acc['accuracy'] - baseline_50['accuracy'])*100:+.2f}%")
    
    if abs(best_acc['threshold'] - 0.50) > 0.02:
        if best_acc['threshold'] > 0.50:
            print(f"\n  [!] FINDING: Optimal threshold > 0.50")
            print(f"      -> Probabilities are systematically TOO HIGH")
            print(f"      -> Model is predicting home covers too often")
        else:
            print(f"\n  [!] FINDING: Optimal threshold < 0.50")
            print(f"      -> Probabilities are systematically TOO LOW")
            print(f"      -> Model is predicting away covers too often")
    else:
        print(f"\n  [+] Threshold is well-calibrated (close to 0.50)")
    
    # =================================================================
    # PART 2: MU BIAS ANALYSIS
    # =================================================================
    print(f"\n{'='*70}")
    print(" PART 2: MU BIAS ANALYSIS")
    print('='*70)
    
    # Calculate bias (prediction error)
    df['mu_bias'] = df['actual_margin'] - df['pred_margin_mu']
    df['abs_mu_bias'] = np.abs(df['mu_bias'])
    
    print(f"\nOverall mu Bias:")
    print(f"  Mean bias:     {df['mu_bias'].mean():+.3f} (should be ~0)")
    print(f"  Median bias:   {df['mu_bias'].median():+.3f}")
    print(f"  Std of bias:   {df['mu_bias'].std():.3f}")
    
    if abs(df['mu_bias'].mean()) > 0.5:
        if df['mu_bias'].mean() > 0:
            print(f"  [!] SYSTEMATIC UNDER-PREDICTION of margins by {df['mu_bias'].mean():.2f} points")
        else:
            print(f"  [!] SYSTEMATIC OVER-PREDICTION of margins by {-df['mu_bias'].mean():.2f} points")
    else:
        print(f"  [+] No significant overall bias")
    
    # Bias by favorite direction
    print(f"\n{'='*70}")
    print(" Bias by Favorite (Critical for ATS)")
    print('='*70)
    
    df['favorite'] = np.where(df['market_spread_home'] < 0, 'home_fav', 'away_fav')
    
    print(f"\n{'Favorite':<12} {'Count':<8} {'Mean Bias':<12} {'Mean |Bias|':<12} {'ATS Acc':<12}")
    print("-" * 60)
    
    for fav in ['home_fav', 'away_fav']:
        fav_df = df[df['favorite'] == fav]
        mean_bias = fav_df['mu_bias'].mean()
        mean_abs_bias = fav_df['abs_mu_bias'].mean()
        ats_acc = (fav_df['pred_margin_mu'] > -fav_df['market_spread_home']).eq(
            fav_df['actual_margin'] > -fav_df['market_spread_home']
        ).mean()
        
        print(f"{fav:<12} {len(fav_df):<8} {mean_bias:>+10.3f}   {mean_abs_bias:>10.3f}   {ats_acc*100:>8.2f}%")
    
    # Check if bias differs by favorite
    home_fav_bias = df[df['favorite'] == 'home_fav']['mu_bias'].mean()
    away_fav_bias = df[df['favorite'] == 'away_fav']['mu_bias'].mean()
    
    if abs(home_fav_bias - away_fav_bias) > 1.0:
        print(f"\n  [!] ASYMMETRIC BIAS DETECTED")
        print(f"      Difference: {home_fav_bias - away_fav_bias:+.2f} points")
        if home_fav_bias > away_fav_bias:
            print(f"      -> Under-predicting home favorites more than away favorites")
        else:
            print(f"      -> Under-predicting away favorites more than home favorites")
    
    # Bias by spread size
    print(f"\n{'='*70}")
    print(" Bias by Market Spread Size")
    print('='*70)
    
    df['spread_abs'] = np.abs(df['market_spread_home'])
    df['spread_bin'] = pd.cut(df['spread_abs'], 
                               bins=[0, 3, 6, 10, 30],
                               labels=['0-3 (close)', '3-6 (medium)', '6-10 (large)', '>10 (blowout)'])
    
    print(f"\n{'Spread Range':<16} {'Count':<8} {'Mean Bias':<12} {'Mean |Bias|':<12} {'ATS Acc':<12}")
    print("-" * 64)
    
    for spread_range in ['0-3 (close)', '3-6 (medium)', '6-10 (large)', '>10 (blowout)']:
        spread_df = df[df['spread_bin'] == spread_range]
        if len(spread_df) > 0:
            mean_bias = spread_df['mu_bias'].mean()
            mean_abs_bias = spread_df['abs_mu_bias'].mean()
            ats_acc = (spread_df['pred_margin_mu'] > -spread_df['market_spread_home']).eq(
                spread_df['actual_margin'] > -spread_df['market_spread_home']
            ).mean()
            
            print(f"{spread_range:<16} {len(spread_df):<8} {mean_bias:>+10.3f}   {mean_abs_bias:>10.3f}   {ats_acc*100:>8.2f}%")
    
    # Critical: Bias near the decision boundary
    print(f"\n{'='*70}")
    print(" Bias Near Decision Boundary (Most Important for ATS)")
    print('='*70)
    
    # Distance from decision threshold
    df['distance_from_threshold'] = df['pred_margin_mu'] + df['market_spread_home']
    df['near_boundary'] = np.abs(df['distance_from_threshold']) < 3
    
    print(f"\nGames near decision boundary (|mu + spread| < 3):")
    near_df = df[df['near_boundary']]
    far_df = df[~df['near_boundary']]
    
    print(f"\n{'Location':<16} {'Count':<8} {'Mean Bias':<12} {'ATS Acc':<12}")
    print("-" * 50)
    
    for label_str, subset in [('Near boundary', near_df), ('Far from boundary', far_df)]:
        if len(subset) > 0:
            mean_bias = subset['mu_bias'].mean()
            ats_acc = (subset['pred_margin_mu'] > -subset['market_spread_home']).eq(
                subset['actual_margin'] > -subset['market_spread_home']
            ).mean()
            print(f"{label_str:<16} {len(subset):<8} {mean_bias:>+10.3f}   {ats_acc*100:>8.2f}%")
    
    if len(near_df) > 0 and len(far_df) > 0:
        near_ats = (near_df['pred_margin_mu'] > -near_df['market_spread_home']).eq(
            near_df['actual_margin'] > -near_df['market_spread_home']
        ).mean()
        far_ats = (far_df['pred_margin_mu'] > -far_df['market_spread_home']).eq(
            far_df['actual_margin'] > -far_df['market_spread_home']
        ).mean()
        
        if near_ats < far_ats - 0.05:
            print(f"\n  [!] PERFORMANCE DEGRADES NEAR DECISION BOUNDARY")
            print(f"      This suggests systematic bias exactly where it matters most")
    
    # Signed distance analysis (which direction is bias?)
    print(f"\n{'='*70}")
    print(" Bias by Prediction Confidence Direction")
    print('='*70)
    
    df['pred_home_covers'] = df['pred_margin_mu'] > -df['market_spread_home']
    
    print(f"\n{'Prediction':<20} {'Count':<8} {'Mean Bias':<12} {'ATS Acc':<12} {'Status'}")
    print("-" * 64)
    
    for pred_result in [True, False]:
        pred_label = 'Home covers' if pred_result else 'Away covers'
        pred_df = df[df['pred_home_covers'] == pred_result]
        
        if len(pred_df) > 0:
            mean_bias = pred_df['mu_bias'].mean()
            ats_acc = (pred_df['actual_home_covers'] == pred_result).mean()
            
            # Determine status
            if pred_result and mean_bias > 0:
                status = "[!] Under-pred"
            elif pred_result and mean_bias < 0:
                status = "[!] Over-pred"
            elif not pred_result and mean_bias < 0:
                status = "[!] Under-pred"
            elif not pred_result and mean_bias > 0:
                status = "[!] Over-pred"
            else:
                status = "[+] OK"
            
            print(f"{pred_label:<20} {len(pred_df):<8} {mean_bias:>+10.3f}   {ats_acc*100:>8.2f}%   {status}")
    
    # When we predict home covers, are we systematically wrong?
    home_cover_pred_df = df[df['pred_home_covers'] == True]
    away_cover_pred_df = df[df['pred_home_covers'] == False]
    
    if len(home_cover_pred_df) > 0 and len(away_cover_pred_df) > 0:
        home_bias = home_cover_pred_df['mu_bias'].mean()
        away_bias = away_cover_pred_df['mu_bias'].mean()
        
        if home_bias > 0.5:
            print(f"\n  [!] When predicting HOME covers, we under-predict margin by {home_bias:.2f}")
            print(f"      -> We're not confident enough in home covers")
        if away_bias < -0.5:
            print(f"\n  [!] When predicting AWAY covers, we over-predict margin by {-away_bias:.2f}")
            print(f"      -> We're not confident enough in away covers")
    
    # Correlation between mu and spread
    print(f"\n{'='*70}")
    print(" Relationship Between mu and Market Spread")
    print('='*70)
    
    corr_mu_spread = df['pred_margin_mu'].corr(df['baseline_margin'])
    print(f"\nCorrelation(mu, baseline_margin): {corr_mu_spread:.3f}")
    print(f"  (Baseline margin = -spread, what market expects)")
    
    if corr_mu_spread > 0.95:
        print(f"  [!] Very high correlation - model is mostly just copying the market")
        print(f"      -> Not adding much independent information")
    elif corr_mu_spread > 0.7:
        print(f"  [+] Good correlation - model uses market but adds value")
    else:
        print(f"  [!] Low correlation - model diverges significantly from market")
    
    # Return summary statistics for comparison
    return {
        'best_threshold': best_acc['threshold'],
        'best_accuracy': best_acc['accuracy'],
        'baseline_accuracy': baseline_50['accuracy'],
        'mean_bias': df['mu_bias'].mean(),
        'home_fav_bias': home_fav_bias,
        'away_fav_bias': away_fav_bias,
        'near_boundary_ats': (near_df['pred_margin_mu'] > -near_df['market_spread_home']).eq(
            near_df['actual_margin'] > -near_df['market_spread_home']
        ).mean() if len(near_df) > 0 else None,
        'far_boundary_ats': (far_df['pred_margin_mu'] > -far_df['market_spread_home']).eq(
            far_df['actual_margin'] > -far_df['market_spread_home']
        ).mean() if len(far_df) > 0 else None,
    }

# Run analysis on both datasets
val_results = analyze_threshold_and_bias(df_val, "VALIDATION SET (2021-2024)", is_validation=True)
test_results = analyze_threshold_and_bias(df_2425, "2024-2025 SEASON (Out-of-Sample)")

# Final summary and recommendations
print(f"\n\n{'='*70}")
print(" FINAL DIAGNOSIS & RECOMMENDATIONS")
print('='*70)

print(f"\n1. THRESHOLD CALIBRATION:")
print(f"   Validation optimal:   {val_results['best_threshold']:.3f}")
print(f"   2024-2025 optimal:    {test_results['best_threshold']:.3f}")

if abs(val_results['best_threshold'] - 0.50) < 0.02 and abs(test_results['best_threshold'] - 0.50) < 0.02:
    print(f"   [+] Probabilities are well-calibrated")
else:
    print(f"   [!] Probabilities are miscalibrated")
    if val_results['best_threshold'] > 0.50 and test_results['best_threshold'] > 0.50:
        print(f"       -> Systematically over-predicting home cover probability")
    elif val_results['best_threshold'] < 0.50 and test_results['best_threshold'] < 0.50:
        print(f"       -> Systematically under-predicting home cover probability")

print(f"\n2. MU BIAS:")
print(f"   Validation mean bias: {val_results['mean_bias']:+.3f}")
print(f"   2024-2025 mean bias:  {test_results['mean_bias']:+.3f}")

if abs(test_results['mean_bias']) > 0.5:
    print(f"   [!] Significant overall bias on 2024-2025")

print(f"\n3. ASYMMETRIC BIAS BY FAVORITE:")
print(f"   Validation:")
print(f"     Home favorites: {val_results['home_fav_bias']:+.3f}")
print(f"     Away favorites: {val_results['away_fav_bias']:+.3f}")
print(f"   2024-2025:")
print(f"     Home favorites: {test_results['home_fav_bias']:+.3f}")
print(f"     Away favorites: {test_results['away_fav_bias']:+.3f}")

home_asymmetry = abs(test_results['home_fav_bias'] - test_results['away_fav_bias'])
if home_asymmetry > 1.0:
    print(f"   [!] Large asymmetry: {home_asymmetry:.2f} points")

print(f"\n4. BOUNDARY PERFORMANCE:")
if test_results['near_boundary_ats'] and test_results['far_boundary_ats']:
    print(f"   Near boundary:  {test_results['near_boundary_ats']*100:.2f}%")
    print(f"   Far boundary:   {test_results['far_boundary_ats']*100:.2f}%")
    if test_results['near_boundary_ats'] < test_results['far_boundary_ats'] - 0.05:
        print(f"   [!] Significant degradation near decision boundary")

print(f"\n{'='*70}")
print(" RECOMMENDED ACTIONS")
print('='*70)

# Determine primary issue
issues = []

if abs(test_results['best_threshold'] - 0.50) > 0.02:
    issues.append(('probability_calibration', f"Optimal threshold is {test_results['best_threshold']:.3f}, not 0.50"))

if abs(test_results['mean_bias']) > 0.5:
    issues.append(('overall_bias', f"Mean bias of {test_results['mean_bias']:+.2f} points"))

if home_asymmetry > 1.0:
    issues.append(('asymmetric_bias', f"Asymmetric bias between favorites ({home_asymmetry:.2f} pts)"))

if test_results['near_boundary_ats'] and test_results['far_boundary_ats']:
    if test_results['near_boundary_ats'] < test_results['far_boundary_ats'] - 0.05:
        issues.append(('boundary_degradation', "Poor performance near decision boundary"))

if not issues:
    print("\n[+] No major systematic issues detected")
    print("    The 0.76% ATS difference may be statistical noise")
    print("\nRECOMMENDATION: Accept the margin model as-is")
else:
    print(f"\n{len(issues)} systematic issue(s) detected:\n")
    for i, (issue_type, description) in enumerate(issues, 1):
        print(f"{i}. {description}")
    
    print(f"\nRECOMMENDED FIXES (in priority order):")
    
    for i, (issue_type, _) in enumerate(issues, 1):
        if issue_type == 'overall_bias':
            print(f"\n{i}. Add bias correction:")
            print(f"   adjusted_mu = pred_mu + {-test_results['mean_bias']:.3f}")
        
        elif issue_type == 'asymmetric_bias':
            print(f"\n{i}. Add favorite-specific bias correction:")
            print(f"   if home_favorite: adjusted_mu = pred_mu + correction_home")
            print(f"   if away_favorite: adjusted_mu = pred_mu + correction_away")
        
        elif issue_type == 'probability_calibration':
            print(f"\n{i}. Recalibrate probabilities:")
            print(f"   Use isotonic regression or Platt scaling")
            print(f"   Or simply adjust decision threshold to {test_results['best_threshold']:.3f}")
        
        elif issue_type == 'boundary_degradation':
            print(f"\n{i}. Add confidence-based filtering:")
            print(f"   Only bet when |mu + spread| > threshold (e.g., 3 points)")
            print(f"   Or use ensemble with market probabilities for close games")

print(f"\n{'='*70}")

