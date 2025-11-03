"""Analyze ATS accuracy by probability bucket.

Shows whether the model is well-calibrated by checking if predicted probabilities
match actual outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_calibration(predictions_file, output_dir):
    """Analyze calibration by 1% probability buckets."""
    
    df = pd.read_csv(predictions_file)
    
    # Filter to games with actual outcomes
    df = df[df['actual_home_covers'].notna()].copy()
    df['actual_home_covers'] = df['actual_home_covers'].astype(int)
    
    print(f"Analyzing {len(df)} games")
    print(f"Probability range: {df['cover_prob_home'].min():.3f} to {df['cover_prob_home'].max():.3f}")
    
    # For each prediction, determine which side was predicted and if it covered
    # If prob_home > 0.5, we predict home covers
    # Otherwise we predict away covers
    
    # Create adjusted probability (always >= 0.5 for the predicted side)
    df['predicted_side'] = np.where(df['cover_prob_home'] > 0.5, 'home', 'away')
    df['predicted_prob'] = np.where(
        df['cover_prob_home'] > 0.5,
        df['cover_prob_home'],
        1 - df['cover_prob_home']
    )
    
    # Did the predicted side cover?
    df['predicted_covered'] = np.where(
        df['predicted_side'] == 'home',
        df['actual_home_covers'] == 1,
        df['actual_home_covers'] == 0
    ).astype(int)
    
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS: ATS% BY PREDICTED PROBABILITY")
    print("="*80)
    print("\nNote: Probabilities shown are for the PREDICTED side to cover")
    print("(If model predicts away at 54%, we show 54% not 46%)\n")
    
    # Create 1% buckets from 50% to 55% (and beyond if data exists)
    min_prob = 0.50
    max_prob = min(0.70, df['predicted_prob'].max() + 0.01)  # Cap at 70% or max observed
    
    buckets = []
    bucket_size = 0.01
    
    for prob_start in np.arange(min_prob, max_prob, bucket_size):
        prob_end = prob_start + bucket_size
        
        mask = (df['predicted_prob'] >= prob_start) & (df['predicted_prob'] < prob_end)
        bucket_games = df[mask]
        
        if len(bucket_games) > 0:
            actual_rate = bucket_games['predicted_covered'].mean()
            count = len(bucket_games)
            
            # Calculate ROI
            roi = (actual_rate * 0.91 + (1 - actual_rate) * -1.0) * 100
            
            buckets.append({
                'prob_start': prob_start,
                'prob_end': prob_end,
                'prob_midpoint': (prob_start + prob_end) / 2,
                'count': count,
                'actual_ats_pct': actual_rate,
                'roi': roi
            })
    
    results_df = pd.DataFrame(buckets)
    
    if len(results_df) > 0:
        # Print table
        print(f"{'Prob Range':<15} {'Count':>7} {'Actual ATS%':>12} {'Expected':>10} {'Diff':>8} {'ROI':>10}")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            prob_label = f"{row['prob_start']*100:.0f}-{row['prob_end']*100:.0f}%"
            expected = row['prob_midpoint'] * 100
            actual = row['actual_ats_pct'] * 100
            diff = actual - expected
            
            # Color code the diff
            if abs(diff) < 2:
                status = "[OK]"
            elif diff > 0:
                status = "[+] "
            else:
                status = "[-] "
            
            print(f"{prob_label:<15} {row['count']:>7} {actual:>11.1f}% {expected:>9.1f}% {status} {diff:>+5.1f}% {row['roi']:>9.1f}%")
        
        # Calculate overall calibration metrics
        print("\n" + "="*80)
        print("CALIBRATION METRICS")
        print("="*80)
        
        # Mean absolute calibration error
        results_df['cal_error'] = abs(results_df['actual_ats_pct'] - results_df['prob_midpoint'])
        weighted_cal_error = (results_df['cal_error'] * results_df['count']).sum() / results_df['count'].sum()
        
        print(f"\nMean Absolute Calibration Error: {weighted_cal_error * 100:.2f}%")
        print(f"  (Difference between predicted probability and actual outcome rate)")
        print(f"  Perfect calibration = 0%, Random = varies")
        
        # Check if well-calibrated (should be close to diagonal)
        if weighted_cal_error < 0.03:
            print(f"  Status: [WELL CALIBRATED]")
        elif weighted_cal_error < 0.05:
            print(f"  Status: [ACCEPTABLE]")
        else:
            print(f"  Status: [POORLY CALIBRATED]")
        
        # Overall stats
        total_games = results_df['count'].sum()
        weighted_actual = (results_df['actual_ats_pct'] * results_df['count']).sum() / total_games
        weighted_roi = (results_df['roi'] * results_df['count']).sum() / total_games
        
        print(f"\nOverall Performance:")
        print(f"  Total Games: {total_games}")
        print(f"  Overall ATS%: {weighted_actual * 100:.2f}%")
        print(f"  Overall ROI: {weighted_roi:+.2f}%")
        
        # Analyze by confidence level
        print("\n" + "="*80)
        print("SUMMARY BY CONFIDENCE LEVEL")
        print("="*80)
        
        confidence_levels = [
            (0.50, 0.51, "50-51%"),
            (0.51, 0.52, "51-52%"),
            (0.52, 0.53, "52-53%"),
            (0.53, 0.55, "53-55%"),
            (0.55, 1.00, "55%+"),
        ]
        
        for low, high, label in confidence_levels:
            mask = (results_df['prob_midpoint'] >= low) & (results_df['prob_midpoint'] < high)
            if mask.any():
                subset = results_df[mask]
                count = subset['count'].sum()
                if count > 0:
                    avg_actual = (subset['actual_ats_pct'] * subset['count']).sum() / count
                    avg_roi = (subset['roi'] * subset['count']).sum() / count
                    print(f"  {label:<10} {count:>5} games  {avg_actual*100:>5.1f}% ATS  {avg_roi:>+6.2f}% ROI")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / 'calibration_by_probability.csv', index=False)
        print(f"\nDetailed results saved to {output_dir / 'calibration_by_probability.csv'}")
        
        # Create calibration plot
        plt.figure(figsize=(10, 8))
        
        # Main calibration curve
        plt.subplot(2, 1, 1)
        plt.plot([0.5, 0.7], [0.5, 0.7], 'k--', alpha=0.3, label='Perfect Calibration')
        plt.scatter(results_df['prob_midpoint'], results_df['actual_ats_pct'], 
                   s=results_df['count']*2, alpha=0.6, c=results_df['roi'], 
                   cmap='RdYlGn', vmin=-10, vmax=10)
        plt.colorbar(label='ROI (%)')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Cover Rate')
        plt.title('Calibration Curve (size = game count)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0.49, min(0.70, results_df['prob_midpoint'].max() + 0.02))
        plt.ylim(0.3, 0.8)
        plt.legend()
        
        # Distribution
        plt.subplot(2, 1, 2)
        plt.bar(results_df['prob_midpoint'], results_df['count'], 
               width=0.008, alpha=0.6, edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Number of Games')
        plt.title('Distribution of Predictions')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xlim(0.49, min(0.70, results_df['prob_midpoint'].max() + 0.02))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png', dpi=150, bbox_inches='tight')
        print(f"Calibration plot saved to {output_dir / 'calibration_plot.png'}")
        plt.close()
        
    else:
        print("No data in specified probability range!")
    
    return results_df


if __name__ == "__main__":
    predictions_file = "predictions/logistic_2425_test.csv"
    output_dir = Path("analysis/calibration")
    
    print("Analyzing logistic regression model calibration...")
    print("="*80 + "\n")
    
    results = analyze_calibration(predictions_file, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

