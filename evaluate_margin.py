"""Evaluation script for margin model predictions.

Usage:
    python evaluate_margin.py --predictions predictions.csv --output reports/margin_eval
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate margin model predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions CSV")
    parser.add_argument("--output", type=str, required=True, help="Output report directory")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df)} predictions")
    
    # Detect if this is Ridge or XGBoost based on columns
    has_variance = 'pred_margin_sigma' in df.columns
    has_probs = 'cover_prob_home' in df.columns
    model_type = 'ridge' if (has_variance and has_probs) else 'xgboost'
    print(f"Detected model type: {model_type}")
    
    # Filter to games with actuals
    df = df[df['actual_margin'].notna()].copy()
    print(f"{len(df)} games with actual outcomes")
    
    if len(df) == 0:
        print("\nNo games with actual outcomes to evaluate!")
        return
    
    # === Margin Accuracy ===
    margin_errors = (df['pred_margin_mu'] - df['actual_margin']).abs()
    margin_error = margin_errors.mean()
    margin_rmse = np.sqrt(((df['pred_margin_mu'] - df['actual_margin']) ** 2).mean())
    margin_r2 = 1 - ((df['pred_margin_mu'] - df['actual_margin']) ** 2).sum() / ((df['actual_margin'] - df['actual_margin'].mean()) ** 2).sum()
    
    print("\n" + "="*70)
    print("MARGIN PREDICTION ACCURACY")
    print("="*70)
    print(f"Mean Absolute Error:  {margin_error:.2f} points")
    print(f"Root Mean Squared:    {margin_rmse:.2f} points")
    print(f"R² Score:             {margin_r2:.3f}")
    print(f"Median Abs Error:     {margin_errors.median():.2f} points")
    
    # === ATS Performance ===
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    
    # Determine predicted cover based on model type
    if has_probs:
        # Ridge model: use cover probabilities
        df['pred_home_covers'] = (df['cover_prob_home'] > 0.5).astype(int)
    else:
        # XGBoost model: use predicted margin vs spread
        df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    
    df['ats_correct'] = (df['pred_home_covers'] == df['actual_home_covers']).astype(int)
    
    ats_accuracy = df['ats_correct'].mean()
    ats_profit = df.apply(lambda row: 
        (0.91 if row['ats_correct'] else -1.0), axis=1
    ).mean()
    
    print("\n" + "="*70)
    print("ATS PERFORMANCE")
    print("="*70)
    print(f"ATS Accuracy:   {ats_accuracy * 100:.2f}%")
    print(f"Breakeven:      52.38%")
    print(f"Edge:           {(ats_accuracy - 0.5238) * 100:+.2f}%")
    print(f"ROI per bet:    {ats_profit * 100:+.2f}%")
    print(f"Status:         {'[+] PROFITABLE' if ats_profit > 0 else '[-] UNPROFITABLE'}")
    
    # === Calibration Analysis (Ridge only) ===
    if has_probs:
        df['confidence'] = df['cover_prob_home'].apply(lambda p: max(p, 1-p))
        df['confidence_bucket'] = pd.cut(df['confidence'], 
                                          bins=[0.5, 0.55, 0.60, 0.65, 0.70, 1.0],
                                          labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+'])
        
        print("\n" + "="*70)
        print("CALIBRATION BY CONFIDENCE LEVEL")
        print("="*70)
        print(f"{'Confidence':<12} {'Count':<8} {'Accuracy':<10} {'ROI':<10}")
        print("-" * 70)
        
        for bucket in ['50-55%', '55-60%', '60-65%', '65-70%', '70%+']:
            bucket_df = df[df['confidence_bucket'] == bucket]
            if len(bucket_df) > 0:
                acc = bucket_df['ats_correct'].mean()
                roi = (acc * 0.91 + (1 - acc) * -1.0) * 100
                print(f"{bucket:<12} {len(bucket_df):<8} {acc*100:>6.1f}%     {roi:>+6.2f}%")
        
        # Check if calibration is correct (higher confidence = higher accuracy)
        cal_by_bucket = df.groupby('confidence_bucket', observed=True)['ats_correct'].mean()
        cal_increasing = all(cal_by_bucket.iloc[i] <= cal_by_bucket.iloc[i+1] for i in range(len(cal_by_bucket)-1) if pd.notna(cal_by_bucket.iloc[i]) and pd.notna(cal_by_bucket.iloc[i+1]))
        
        print(f"\nCalibration Check: {'[+] PROPER (increasing)' if cal_increasing else '[!] INVERTED or FLAT'}")
        
        # === Brier Score ===
        brier = ((df['cover_prob_home'] - df['actual_home_covers']) ** 2).mean()
        # Reference: random guessing = 0.25, perfect = 0.0
        print(f"\n" + "="*70)
        print("PROBABILITY CALIBRATION")
        print("="*70)
        print(f"Brier Score: {brier:.4f} (lower is better, random=0.25, perfect=0.0)")
    else:
        # XGBoost doesn't have calibration metrics
        cal_increasing = None
        brier = None
    
    # === Moneyline Performance ===
    df['actual_home_wins'] = (df['actual_margin'] > 0).astype(int)
    
    if has_probs:
        # Ridge model: use win probabilities
        df['pred_home_wins'] = (df['win_prob_home'] > 0.5).astype(int)
    else:
        # XGBoost model: use predicted margin
        df['pred_home_wins'] = (df['pred_margin_mu'] > 0).astype(int)
    
    df['ml_correct'] = (df['pred_home_wins'] == df['actual_home_wins']).astype(int)
    
    ml_accuracy = df['ml_correct'].mean()
    
    print("\n" + "="*70)
    print("MONEYLINE PERFORMANCE")
    print("="*70)
    print(f"Moneyline Accuracy: {ml_accuracy * 100:.2f}%")
    print(f"Expected (50/50):   50.00%")
    print(f"Edge:               {(ml_accuracy - 0.5) * 100:+.2f}%")
    
    # === Uncertainty Analysis (Ridge only) ===
    if has_variance:
        print("\n" + "="*70)
        print("UNCERTAINTY (sigma) ANALYSIS")
        print("="*70)
        print(f"Predicted sigma range: {df['pred_margin_sigma'].min():.2f} to {df['pred_margin_sigma'].max():.2f}")
        print(f"Mean sigma:            {df['pred_margin_sigma'].mean():.2f}")
        print(f"Median sigma:          {df['pred_margin_sigma'].median():.2f}")
        
        # Check if higher sigma correlates with larger errors (should be!)
        df['abs_error'] = margin_errors
        corr = df[['pred_margin_sigma', 'abs_error']].corr().iloc[0, 1]
        print(f"\nCorrelation (sigma vs |error|): {corr:.3f}")
        if corr > 0.1:
            print("[+] Higher uncertainty correlates with larger errors (as expected)")
        else:
            print("[!] Uncertainty not correlated with errors")
    else:
        corr = None
        df['abs_error'] = margin_errors
    
    # === Save Reports ===
    print("\n" + "="*70)
    print("SAVING REPORTS")
    print("="*70)
    
    # Summary report
    report_path = output_dir / "evaluation_summary.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"MARGIN MODEL EVALUATION SUMMARY ({model_type.upper()})\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {len(df)} games\n\n")
        f.write(f"Margin MAE:      {margin_error:.2f} points\n")
        f.write(f"Margin RMSE:     {margin_rmse:.2f} points\n")
        f.write(f"Margin R²:       {margin_r2:.3f}\n\n")
        f.write(f"ATS Accuracy:    {ats_accuracy * 100:.2f}%\n")
        f.write(f"ATS ROI:         {ats_profit * 100:+.2f}%\n")
        
        if brier is not None:
            f.write(f"Brier Score:     {brier:.4f}\n")
        
        f.write(f"ML Accuracy:     {ml_accuracy * 100:.2f}%\n\n")
        
        if cal_increasing is not None:
            f.write(f"Calibration:     {'PROPER' if cal_increasing else 'INVERTED/FLAT'}\n")
        
        if corr is not None:
            f.write(f"Sigma-Error Corr: {corr:.3f}\n")
    
    print(f"Summary saved to {report_path}")
    
    # Per-game details
    df.to_csv(output_dir / "per_game_analysis.csv", index=False)
    print(f"Per-game analysis saved to {output_dir / 'per_game_analysis.csv'}")
    
    # Calibration breakdown (Ridge only)
    if has_probs:
        cal_report = df.groupby('confidence_bucket', observed=True).agg({
            'ats_correct': ['count', 'mean'],
            'abs_error': 'mean'
        }).round(3)
        cal_report.to_csv(output_dir / "calibration_by_confidence.csv")
        print(f"Calibration report saved to {output_dir / 'calibration_by_confidence.csv'}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

