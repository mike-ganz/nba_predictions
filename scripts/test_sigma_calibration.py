"""Test sigma calibration hypothesis.

This script runs:
- Test 2: Constant sigma baseline comparison
- Test 1: Sigma calibration plot
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print(" SIGMA CALIBRATION ANALYSIS")
print("="*70)

# Load predictions
df_val = pd.read_csv('artifacts/margin_test/val_predictions.csv')
df_2425 = pd.read_csv('artifacts/margin_test/2425_predictions.csv')

def analyze_sigma(df, label):
    """Analyze sigma calibration for a dataset."""
    print(f"\n{'='*70}")
    print(f" {label}")
    print('='*70)
    
    # Calculate actual covers
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    
    # Current ATS accuracy (with learned sigma)
    df['pred_covers_learned'] = (df['cover_prob_home'] > 0.5).astype(int)
    ats_learned = (df['pred_covers_learned'] == df['actual_home_covers']).mean()
    
    print(f"\n[BASELINE] Current Performance (Learned Sigma):")
    print(f"  Mean sigma:    {df['pred_margin_sigma'].mean():.2f}")
    print(f"  Sigma range:   {df['pred_margin_sigma'].min():.2f} - {df['pred_margin_sigma'].max():.2f}")
    print(f"  ATS Accuracy:  {ats_learned*100:.2f}%")
    
    # TEST 2: Try constant sigma values
    print(f"\n{'='*70}")
    print(" TEST 2: CONSTANT SIGMA BASELINE")
    print('='*70)
    
    constant_sigmas = [8, 10, 12, 14, 16]
    results = []
    
    for const_sigma in constant_sigmas:
        # Recalculate cover probability with constant sigma
        threshold = -df['market_spread_home'].values
        z = (threshold - df['pred_margin_mu'].values) / const_sigma
        prob_cover_const = 1 - norm.cdf(z)
        
        # Calculate ATS accuracy
        pred_covers_const = (prob_cover_const > 0.5).astype(int)
        ats_const = (pred_covers_const == df['actual_home_covers']).mean()
        
        # Calculate ROI
        roi_const = (ats_const * 0.91 + (1 - ats_const) * -1.0) * 100
        
        results.append({
            'sigma': const_sigma,
            'ats_accuracy': ats_const,
            'roi': roi_const,
            'improvement': (ats_const - ats_learned) * 100
        })
        
        status = "[+]" if ats_const > ats_learned else "[-]"
        print(f"  sigma={const_sigma:2d}:  ATS={ats_const*100:5.2f}%  ROI={roi_const:+6.2f}%  {status} {(ats_const - ats_learned)*100:+.2f}%")
    
    # Find best constant sigma
    best = max(results, key=lambda x: x['ats_accuracy'])
    print(f"\n  Best constant sigma: {best['sigma']} ({best['ats_accuracy']*100:.2f}% ATS, {best['improvement']:+.2f}% improvement)")
    
    if best['improvement'] > 0:
        print(f"  [!] CONSTANT SIGMA OUTPERFORMS LEARNED SIGMA")
        print(f"      This suggests learned sigma has systematic error")
    else:
        print(f"  [+] Learned sigma is competitive")
    
    # TEST 1: Sigma calibration plot
    print(f"\n{'='*70}")
    print(" TEST 1: SIGMA CALIBRATION ANALYSIS")
    print('='*70)
    
    # Calculate actual errors
    df['abs_error'] = np.abs(df['actual_margin'] - df['pred_margin_mu'])
    
    # Overall: predicted sigma vs actual error
    mean_predicted_sigma = df['pred_margin_sigma'].mean()
    mean_actual_error = df['abs_error'].mean()
    
    print(f"\nOverall Calibration:")
    print(f"  Mean predicted sigma: {mean_predicted_sigma:.2f}")
    print(f"  Mean actual |error|:  {mean_actual_error:.2f}")
    print(f"  Ratio (sigma/error):  {mean_predicted_sigma/mean_actual_error:.2f}")
    
    if mean_predicted_sigma / mean_actual_error > 1.15:
        print(f"  [!] UNDERCONFIDENT: Predicted uncertainty is {((mean_predicted_sigma/mean_actual_error - 1)*100):.1f}% too high")
    elif mean_predicted_sigma / mean_actual_error < 0.85:
        print(f"  [!] OVERCONFIDENT: Predicted uncertainty is {((1 - mean_predicted_sigma/mean_actual_error)*100):.1f}% too low")
    else:
        print(f"  [+] Well calibrated overall")
    
    # Bin by predicted sigma
    df['sigma_bin'] = pd.cut(df['pred_margin_sigma'], bins=[0, 6, 8, 10, 15, 100], 
                              labels=['< 6', '6-8', '8-10', '10-15', '> 15'])
    
    print(f"\nCalibration by Predicted Sigma:")
    print(f"{'Sigma Bin':<12} {'Count':<8} {'Mean Sigma':<12} {'Mean |Error|':<14} {'Ratio':<8} {'Status'}")
    print("-" * 70)
    
    for bin_name in ['< 6', '6-8', '8-10', '10-15', '> 15']:
        bin_df = df[df['sigma_bin'] == bin_name]
        if len(bin_df) > 0:
            mean_sigma_bin = bin_df['pred_margin_sigma'].mean()
            mean_error_bin = bin_df['abs_error'].mean()
            ratio = mean_sigma_bin / mean_error_bin if mean_error_bin > 0 else 0
            
            if ratio > 1.15:
                status = "[!] Under"
            elif ratio < 0.85:
                status = "[!] Over"
            else:
                status = "[+] Good"
            
            print(f"{bin_name:<12} {len(bin_df):<8} {mean_sigma_bin:>10.2f}   {mean_error_bin:>12.2f}   {ratio:>6.2f}  {status}")
    
    # Bin by market spread (maybe sigma calibration depends on game type)
    df['spread_bin'] = pd.cut(np.abs(df['market_spread_home']), 
                               bins=[0, 3, 6, 10, 30], 
                               labels=['0-3', '3-6', '6-10', '> 10'])
    
    print(f"\nCalibration by Market Spread:")
    print(f"{'Spread':<12} {'Count':<8} {'Mean Sigma':<12} {'Mean |Error|':<14} {'Ratio':<8} {'Status'}")
    print("-" * 70)
    
    for bin_name in ['0-3', '3-6', '6-10', '> 10']:
        bin_df = df[df['spread_bin'] == bin_name]
        if len(bin_df) > 0:
            mean_sigma_bin = bin_df['pred_margin_sigma'].mean()
            mean_error_bin = bin_df['abs_error'].mean()
            ratio = mean_sigma_bin / mean_error_bin if mean_error_bin > 0 else 0
            
            if ratio > 1.15:
                status = "[!] Under"
            elif ratio < 0.85:
                status = "[!] Over"
            else:
                status = "[+] Good"
            
            print(f"{bin_name:<12} {len(bin_df):<8} {mean_sigma_bin:>10.2f}   {mean_error_bin:>12.2f}   {ratio:>6.2f}  {status}")
    
    # Standardized residuals check (should be ~N(0,1))
    df['standardized_residual'] = (df['actual_margin'] - df['pred_margin_mu']) / df['pred_margin_sigma']
    std_resid_mean = df['standardized_residual'].mean()
    std_resid_std = df['standardized_residual'].std()
    
    print(f"\nStandardized Residuals (should be N(0,1) if calibrated):")
    print(f"  Mean: {std_resid_mean:+.3f} (should be ~0)")
    print(f"  Std:  {std_resid_std:.3f} (should be ~1)")
    
    if abs(std_resid_mean) > 0.1:
        print(f"  [!] Mean is off - systematic bias in mu")
    else:
        print(f"  [+] Mean is good")
    
    if std_resid_std > 1.15:
        print(f"  [!] Std > 1 - sigma is too small (overconfident)")
    elif std_resid_std < 0.85:
        print(f"  [!] Std < 1 - sigma is too large (underconfident)")
    else:
        print(f"  [+] Std is good")
    
    return {
        'ats_learned': ats_learned,
        'best_constant_sigma': best['sigma'],
        'best_constant_ats': best['ats_accuracy'],
        'sigma_ratio': mean_predicted_sigma / mean_actual_error,
        'std_resid_std': std_resid_std,
    }

# Analyze both datasets
val_results = analyze_sigma(df_val, "VALIDATION SET (2021-2024)")
test_results = analyze_sigma(df_2425, "2024-2025 SEASON (Out-of-Sample)")

# Summary
print(f"\n\n{'='*70}")
print(" SUMMARY & DIAGNOSIS")
print('='*70)

print(f"\nValidation Set:")
print(f"  Learned sigma ATS:      {val_results['ats_learned']*100:.2f}%")
print(f"  Best constant sigma:    {val_results['best_constant_sigma']} ({val_results['best_constant_ats']*100:.2f}%)")
print(f"  Sigma/Error ratio:      {val_results['sigma_ratio']:.2f}")
print(f"  Std. residual std:      {val_results['std_resid_std']:.2f}")

print(f"\n2024-2025 Season:")
print(f"  Learned sigma ATS:      {test_results['ats_learned']*100:.2f}%")
print(f"  Best constant sigma:    {test_results['best_constant_sigma']} ({test_results['best_constant_ats']*100:.2f}%)")
print(f"  Sigma/Error ratio:      {test_results['sigma_ratio']:.2f}")
print(f"  Std. residual std:      {test_results['std_resid_std']:.2f}")

# Diagnosis
print(f"\n{'='*70}")
print(" DIAGNOSIS")
print('='*70)

if test_results['best_constant_ats'] > test_results['ats_learned']:
    improvement = (test_results['best_constant_ats'] - test_results['ats_learned']) * 100
    print(f"\n[FINDING 1] Constant sigma outperforms learned sigma by {improvement:.2f}%")
    print(f"  -> Learned sigma has systematic error and should be replaced or recalibrated")

if test_results['sigma_ratio'] > 1.15:
    print(f"\n[FINDING 2] Predicted uncertainty is {((test_results['sigma_ratio'] - 1)*100):.1f}% too high")
    print(f"  -> Model is UNDERCONFIDENT")
    print(f"  -> Fix: Rescale sigma by {1/test_results['sigma_ratio']:.3f}")
elif test_results['sigma_ratio'] < 0.85:
    print(f"\n[FINDING 2] Predicted uncertainty is {((1 - test_results['sigma_ratio'])*100):.1f}% too low")
    print(f"  -> Model is OVERCONFIDENT")
    print(f"  -> Fix: Rescale sigma by {1/test_results['sigma_ratio']:.3f}")

if test_results['std_resid_std'] > 1.15:
    print(f"\n[FINDING 3] Standardized residual std = {test_results['std_resid_std']:.2f} > 1")
    print(f"  -> Sigma predictions are too small (overconfident)")
    print(f"  -> This conflicts with Finding 2 - investigate distribution assumptions")
elif test_results['std_resid_std'] < 0.85:
    print(f"\n[FINDING 3] Standardized residual std = {test_results['std_resid_std']:.2f} < 1")
    print(f"  -> Sigma predictions are too large (underconfident)")
    print(f"  -> This aligns with Finding 2")

print(f"\n{'='*70}")

