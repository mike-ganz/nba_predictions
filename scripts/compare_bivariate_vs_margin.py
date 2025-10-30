"""Head-to-head comparison: Bivariate Poisson vs Direct Margin models.

This script compares the performance of both approaches on validation and 2024-2025 data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_dataset(df_biv, df_margin, label):
    """Analyze and compare one dataset (validation or 2024-2025)."""
    print(f"\n{'='*70}")
    print(f" {label}")
    print('='*70)
    
    # === Margin Error ===
    # Bivariate: derive margin from predicted home - away
    biv_pred_margin = df_biv['pred_home'] - df_biv['pred_away']
    biv_actual_margin = df_biv['actual_home'] - df_biv['actual_away']
    biv_mae = (biv_pred_margin - biv_actual_margin).abs().mean()
    biv_rmse = np.sqrt(((biv_pred_margin - biv_actual_margin) ** 2).mean())
    
    # Direct margin
    margin_mae = (df_margin['pred_margin_mu'] - df_margin['actual_margin']).abs().mean()
    margin_rmse = np.sqrt(((df_margin['pred_margin_mu'] - df_margin['actual_margin']) ** 2).mean())
    
    print(f"\n{'METRIC':<25} {'Bivariate':<15} {'Direct Margin':<15} {'Winner'}")
    print("-" * 70)
    print(f"{'Margin MAE':<25} {biv_mae:>8.2f} pts    {margin_mae:>8.2f} pts    {'[+] Margin' if margin_mae < biv_mae else '[-] Bivariate'}")
    print(f"{'Margin RMSE':<25} {biv_rmse:>8.2f} pts    {margin_rmse:>8.2f} pts    {'[+] Margin' if margin_rmse < biv_rmse else '[-] Bivariate'}")
    improvement_mae = biv_mae - margin_mae
    print(f"{'Improvement':<25} {improvement_mae:>+8.2f} pts")
    
    # === ATS Accuracy ===
    # Bivariate
    df_biv['actual_home_covers'] = ((df_biv['actual_home'] - df_biv['actual_away']) > -df_biv['market_spread_home']).astype(int)
    df_biv['pred_covers'] = (df_biv['cover_prob_home'] > 0.5).astype(int)
    biv_ats = (df_biv['pred_covers'] == df_biv['actual_home_covers']).mean()
    
    # Direct margin
    df_margin['actual_home_covers'] = (df_margin['actual_margin'] > -df_margin['market_spread_home']).astype(int)
    df_margin['pred_covers'] = (df_margin['cover_prob_home'] > 0.5).astype(int)
    margin_ats = (df_margin['pred_covers'] == df_margin['actual_home_covers']).mean()
    
    print(f"\n{'Metric':<25} {'Bivariate':<15} {'Direct Margin':<15} {'Winner'}")
    print("-" * 70)
    print(f"{'ATS Accuracy':<25} {biv_ats*100:>9.2f}%     {margin_ats*100:>9.2f}%     {'[+] Margin' if margin_ats > biv_ats else '[-] Bivariate'}")
    
    biv_roi = (biv_ats * 0.91 + (1 - biv_ats) * -1.0) * 100
    margin_roi = (margin_ats * 0.91 + (1 - margin_ats) * -1.0) * 100
    print(f"{'ROI per bet':<25} {biv_roi:>+9.2f}%     {margin_roi:>+9.2f}%     {'[+] Margin' if margin_roi > biv_roi else '[-] Bivariate'}")
    improvement_ats = (margin_ats - biv_ats) * 100
    print(f"{'ATS Improvement':<25} {improvement_ats:>+9.2f}%")
    
    # === Brier Score ===
    brier_biv = ((df_biv['cover_prob_home'] - df_biv['actual_home_covers']) ** 2).mean()
    brier_margin = ((df_margin['cover_prob_home'] - df_margin['actual_home_covers']) ** 2).mean()
    
    print(f"\n{'Metric':<25} {'Bivariate':<15} {'Direct Margin':<15} {'Winner'}")
    print("-" * 70)
    print(f"{'Brier Score':<25} {brier_biv:>11.4f}     {brier_margin:>11.4f}     {'[+] Margin' if brier_margin < brier_biv else '[-] Bivariate'}")
    
    # === Calibration ===
    print(f"\n{'CALIBRATION BY CONFIDENCE':<70}")
    print("-" * 70)
    
    df_margin['confidence'] = df_margin['cover_prob_home'].apply(lambda p: max(p, 1-p))
    df_margin['conf_bucket'] = pd.cut(df_margin['confidence'], 
                                       bins=[0.5, 0.6, 0.7, 1.0], 
                                       labels=['Low (50-60%)', 'Med (60-70%)', 'High (70%+)'])
    
    print(f"\n{'Confidence':<20} {'Count':<10} {'Accuracy':<15}")
    print("-" * 45)
    cal_proper = True
    prev_acc = 0
    for bucket in ['Low (50-60%)', 'Med (60-70%)', 'High (70%+)']:
        bucket_df = df_margin[df_margin['conf_bucket'] == bucket]
        if len(bucket_df) > 0:
            acc = ((bucket_df['pred_covers'] == bucket_df['actual_home_covers']).mean() * 100)
            print(f"{bucket:<20} {len(bucket_df):<10} {acc:>6.1f}%")
            if acc < prev_acc:
                cal_proper = False
            prev_acc = acc
    
    print(f"\nCalibration Status: {'[+] PROPER (increasing)' if cal_proper else '[!] INVERTED or FLAT'}")
    
    # === Summary for this dataset ===
    margin_wins = sum([
        margin_mae < biv_mae,
        margin_rmse < biv_rmse,
        margin_ats > biv_ats,
        brier_margin < brier_biv,
    ])
    
    print(f"\n{'='*70}")
    print(f" SCORE: Direct Margin {margin_wins}/4 metrics")
    print('='*70)
    
    return {
        'biv_mae': biv_mae,
        'margin_mae': margin_mae,
        'biv_ats': biv_ats,
        'margin_ats': margin_ats,
        'biv_brier': brier_biv,
        'margin_brier': brier_margin,
        'margin_wins': margin_wins,
        'cal_proper': cal_proper,
    }


def main():
    print("="*70)
    print(" BIVARIATE vs. DIRECT MARGIN: Head-to-Head Comparison")
    print("="*70)
    
    # Check if files exist
    biv_val_path = 'reports/val_with_players/per_game_predictions.csv'
    biv_2425_path = 'reports/2425_with_players/per_game_predictions.csv'
    margin_val_path = 'artifacts/margin_test/val_predictions.csv'
    margin_2425_path = 'artifacts/margin_test/2425_predictions.csv'
    
    missing_files = []
    for path in [biv_val_path, biv_2425_path, margin_val_path, margin_2425_path]:
        if not Path(path).exists():
            missing_files.append(path)
    
    if missing_files:
        print("\nERROR: Missing required files:")
        for path in missing_files:
            print(f"  - {path}")
        print("\nPlease run:")
        print("  1. Bivariate model evaluation (if needed)")
        print("  2. scripts/test_margin_approach.ps1")
        return
    
    # Load data
    print("\nLoading results...")
    df_biv_val = pd.read_csv(biv_val_path)
    df_biv_2425 = pd.read_csv(biv_2425_path)
    df_margin_val = pd.read_csv(margin_val_path)
    df_margin_2425 = pd.read_csv(margin_2425_path)
    
    # Analyze validation set
    val_results = analyze_dataset(df_biv_val, df_margin_val, "VALIDATION SET (2021-2024)")
    
    # Analyze 2024-2025 set
    test_results = analyze_dataset(df_biv_2425, df_margin_2425, "2024-2025 SEASON (Out-of-Sample)")
    
    # === Final Verdict ===
    print("\n\n")
    print("="*70)
    print(" FINAL VERDICT")
    print("="*70)
    
    total_margin_wins = val_results['margin_wins'] + test_results['margin_wins']
    
    print(f"\nOverall Score: Direct Margin {total_margin_wins}/8 metrics")
    print(f"  Validation:  {val_results['margin_wins']}/4")
    print(f"  2024-2025:   {test_results['margin_wins']}/4")
    
    print("\nKey Improvements on 2024-2025 (Out-of-Sample):")
    print(f"  Margin MAE:  {test_results['biv_mae']:.2f} -> {test_results['margin_mae']:.2f} ({test_results['biv_mae'] - test_results['margin_mae']:+.2f})")
    print(f"  ATS Acc:     {test_results['biv_ats']*100:.2f}% -> {test_results['margin_ats']*100:.2f}% ({(test_results['margin_ats'] - test_results['biv_ats'])*100:+.2f}%)")
    print(f"  Calibration: {'[+] PROPER' if test_results['cal_proper'] else '[-] INVERTED'}")
    
    # Recommendation
    print("\n" + "="*70)
    if total_margin_wins >= 6:  # Win at least 6/8 metrics
        print(" RECOMMENDATION: [+] SWITCH TO DIRECT MARGIN MODEL")
        print("="*70)
        print("\nThe direct margin approach shows clear improvements:")
        print("  • Lower prediction error")
        print("  • Better ATS accuracy")
        print("  • Simpler implementation")
        print("  • Faster predictions")
        print("\nNext steps:")
        print("  1. Review artifacts/margin_test/*/evaluation_summary.txt")
        print("  2. Update train_and_evaluate.ps1 to use margin model")
        print("  3. Update README.md with new architecture")
    else:
        print(" RECOMMENDATION: [!] INVESTIGATE FURTHER")
        print("="*70)
        print("\nDirect margin shows mixed results. Consider:")
        print("  1. Checking if feature engineering needs adjustment")
        print("  2. Tuning regularization parameters")
        print("  3. Reviewing calibration on high-confidence picks")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

