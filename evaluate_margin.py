"""Evaluation script for spread coverage model predictions.

Usage:
    python evaluate_margin.py --predictions predictions.csv --output reports/margin_eval
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate spread coverage model predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions CSV")
    parser.add_argument("--output", type=str, required=True, help="Output report directory")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df)} predictions")
    
    # Filter to games with actuals
    df = df[df['actual_home_covers'].notna()].copy()
    print(f"{len(df)} games with actual outcomes")
    
    if len(df) == 0:
        print("\nNo games with actual outcomes to evaluate!")
        return
    
    # Ensure correct types
    df['actual_home_covers'] = df['actual_home_covers'].astype(int)
    df['pred_home_covers'] = (df['cover_prob_home'] > 0.5).astype(int)
    df['ats_correct'] = (df['pred_home_covers'] == df['actual_home_covers']).astype(int)
    
    # === Classification Metrics ===
    from sklearn.metrics import log_loss, brier_score_loss
    
    accuracy = df['ats_correct'].mean()
    logloss = log_loss(df['actual_home_covers'], df['cover_prob_home'])
    brier = brier_score_loss(df['actual_home_covers'], df['cover_prob_home'])
    
    print("\n" + "="*70)
    print("CLASSIFICATION METRICS")
    print("="*70)
    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Log Loss:    {logloss:.4f} (lower is better)")
    print(f"Brier Score: {brier:.4f} (lower is better, random=0.25, perfect=0.0)")
    
    # === ATS Performance ===
    ats_accuracy = accuracy  # Same as classification accuracy
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
    
    # === Calibration Analysis ===
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
    
    # Check if calibration is proper (higher confidence = higher accuracy)
    cal_by_bucket = df.groupby('confidence_bucket', observed=True)['ats_correct'].mean()
    cal_increasing = all(
        cal_by_bucket.iloc[i] <= cal_by_bucket.iloc[i+1] 
        for i in range(len(cal_by_bucket)-1) 
        if pd.notna(cal_by_bucket.iloc[i]) and pd.notna(cal_by_bucket.iloc[i+1])
    )
    
    print(f"\nCalibration Check: {'[+] PROPER (increasing)' if cal_increasing else '[!] INVERTED or FLAT'}")
    
    # === Probability Distribution ===
    print(f"\n" + "="*70)
    print("PROBABILITY DISTRIBUTION")
    print("="*70)
    print(f"Mean:   {df['cover_prob_home'].mean():.3f}")
    print(f"Median: {df['cover_prob_home'].median():.3f}")
    print(f"Min:    {df['cover_prob_home'].min():.3f}")
    print(f"Max:    {df['cover_prob_home'].max():.3f}")
    print(f"Std:    {df['cover_prob_home'].std():.3f}")
    
    # === Performance by Spread ===
    print(f"\n" + "="*70)
    print("PERFORMANCE BY SPREAD SIZE")
    print("="*70)
    df['spread_bucket'] = pd.cut(df['market_spread_home'], 
                                   bins=[-40, -10, -5, -2, 2, 5, 10, 40],
                                   labels=['Heavy Fav (<-10)', 'Fav (-10 to -5)', 'Slight Fav (-5 to -2)',
                                          'Pick Em (-2 to 2)', 'Slight Dog (2 to 5)', 'Dog (5 to 10)', 'Heavy Dog (>10)'])
    
    print(f"{'Spread Range':<20} {'Count':<8} {'Accuracy':<10} {'ROI':<10}")
    print("-" * 70)
    
    for bucket in df['spread_bucket'].cat.categories:
        bucket_df = df[df['spread_bucket'] == bucket]
        if len(bucket_df) > 0:
            acc = bucket_df['ats_correct'].mean()
            roi = (acc * 0.91 + (1 - acc) * -1.0) * 100
            print(f"{bucket:<20} {len(bucket_df):<8} {acc*100:>6.1f}%     {roi:>+6.2f}%")
    
    # === Save Reports ===
    print("\n" + "="*70)
    print("SAVING REPORTS")
    print("="*70)
    
    # Summary report
    report_path = output_dir / "evaluation_summary.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SPREAD COVERAGE MODEL EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {len(df)} games\n\n")
        f.write(f"Accuracy:        {accuracy * 100:.2f}%\n")
        f.write(f"Log Loss:        {logloss:.4f}\n")
        f.write(f"Brier Score:     {brier:.4f}\n\n")
        f.write(f"ATS Accuracy:    {ats_accuracy * 100:.2f}%\n")
        f.write(f"ATS ROI:         {ats_profit * 100:+.2f}%\n\n")
        f.write(f"Calibration:     {'PROPER' if cal_increasing else 'INVERTED/FLAT'}\n")
    
    print(f"Summary saved to {report_path}")
    
    # Per-game details
    df.to_csv(output_dir / "per_game_analysis.csv", index=False)
    print(f"Per-game analysis saved to {output_dir / 'per_game_analysis.csv'}")
    
    # Calibration breakdown
    cal_report = df.groupby('confidence_bucket', observed=True).agg({
        'ats_correct': ['count', 'mean'],
        'cover_prob_home': 'mean'
    }).round(3)
    cal_report.to_csv(output_dir / "calibration_by_confidence.csv")
    print(f"Calibration report saved to {output_dir / 'calibration_by_confidence.csv'}")
    
    # Spread analysis
    spread_report = df.groupby('spread_bucket', observed=True).agg({
        'ats_correct': ['count', 'mean'],
        'cover_prob_home': 'mean'
    }).round(3)
    spread_report.to_csv(output_dir / "performance_by_spread.csv")
    print(f"Spread analysis saved to {output_dir / 'performance_by_spread.csv'}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
