"""Compare Ridge and XGBoost model predictions side-by-side.

This script loads predictions from both models on the same test set and
performs detailed comparison of their performance.

Usage:
    python compare_ridge_vs_xgboost.py \
        --ridge-predictions predictions/ridge_2425_predictions.csv \
        --xgboost-predictions predictions/xgboost_2425_predictions.csv \
        --output reports/ridge_vs_xgboost_comparison
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Ridge vs XGBoost predictions")
    parser.add_argument("--ridge-predictions", type=str, required=True,
                        help="Ridge model predictions CSV")
    parser.add_argument("--xgboost-predictions", type=str, required=True,
                        help="XGBoost model predictions CSV")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for comparison report")
    return parser.parse_args()


def compute_metrics(df):
    """Compute prediction metrics from dataframe."""
    # Filter to games with actuals
    df = df[df['actual_margin'].notna()].copy()
    
    if len(df) == 0:
        return None
    
    # Margin accuracy
    margin_mae = (df['pred_margin_mu'] - df['actual_margin']).abs().mean()
    margin_rmse = np.sqrt(((df['pred_margin_mu'] - df['actual_margin']) ** 2).mean())
    margin_r2 = 1 - ((df['pred_margin_mu'] - df['actual_margin']) ** 2).sum() / \
                    ((df['actual_margin'] - df['actual_margin'].mean()) ** 2).sum()
    
    # ATS performance
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    ats_accuracy = (df['pred_home_covers'] == df['actual_home_covers']).mean()
    ats_roi = (ats_accuracy * 0.91 + (1 - ats_accuracy) * -1.0)
    
    # ML performance
    df['actual_home_wins'] = (df['actual_margin'] > 0).astype(int)
    df['pred_home_wins'] = (df['pred_margin_mu'] > 0).astype(int)
    ml_accuracy = (df['pred_home_wins'] == df['actual_home_wins']).mean()
    
    return {
        'n_games': len(df),
        'margin_mae': margin_mae,
        'margin_rmse': margin_rmse,
        'margin_r2': margin_r2,
        'ats_accuracy': ats_accuracy,
        'ats_roi': ats_roi,
        'ml_accuracy': ml_accuracy,
        'predictions': df,
    }


def paired_test(ridge_errors, xgb_errors):
    """Perform paired t-test on prediction errors."""
    # Test if mean errors are significantly different
    t_stat, p_value = stats.ttest_rel(ridge_errors, xgb_errors)
    return t_stat, p_value


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("Loading predictions...")
    ridge_df = pd.read_csv(args.ridge_predictions)
    xgb_df = pd.read_csv(args.xgboost_predictions)
    
    print(f"Ridge predictions: {len(ridge_df)} games")
    print(f"XGBoost predictions: {len(xgb_df)} games")
    
    # Verify same games
    ridge_games = set(ridge_df['game_id'])
    xgb_games = set(xgb_df['game_id'])
    
    if ridge_games != xgb_games:
        print("\nWARNING: Models predicted different sets of games!")
        print(f"  Ridge only: {len(ridge_games - xgb_games)} games")
        print(f"  XGBoost only: {len(xgb_games - ridge_games)} games")
        print(f"  Common: {len(ridge_games & xgb_games)} games")
        print("\nUsing only common games for comparison...")
        
        common_games = ridge_games & xgb_games
        ridge_df = ridge_df[ridge_df['game_id'].isin(common_games)]
        xgb_df = xgb_df[xgb_df['game_id'].isin(common_games)]
    
    # Compute metrics for each model
    print("\nComputing metrics...")
    ridge_metrics = compute_metrics(ridge_df)
    xgb_metrics = compute_metrics(xgb_df)
    
    if ridge_metrics is None or xgb_metrics is None:
        print("ERROR: No games with actual outcomes to compare!")
        return
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\nDataset: {ridge_metrics['n_games']} games with actual outcomes\n")
    
    print(f"{'Metric':<25} {'Ridge':<15} {'XGBoost':<15} {'Difference'}")
    print("-" * 70)
    
    # Margin metrics
    print(f"{'Margin MAE':<25} {ridge_metrics['margin_mae']:>8.2f}      "
          f"{xgb_metrics['margin_mae']:>8.2f}      "
          f"{xgb_metrics['margin_mae'] - ridge_metrics['margin_mae']:>+8.2f}")
    
    print(f"{'Margin RMSE':<25} {ridge_metrics['margin_rmse']:>8.2f}      "
          f"{xgb_metrics['margin_rmse']:>8.2f}      "
          f"{xgb_metrics['margin_rmse'] - ridge_metrics['margin_rmse']:>+8.2f}")
    
    print(f"{'Margin R²':<25} {ridge_metrics['margin_r2']:>8.3f}      "
          f"{xgb_metrics['margin_r2']:>8.3f}      "
          f"{xgb_metrics['margin_r2'] - ridge_metrics['margin_r2']:>+8.3f}")
    
    print()
    
    # ATS metrics
    print(f"{'ATS Accuracy':<25} {ridge_metrics['ats_accuracy']*100:>7.2f}%      "
          f"{xgb_metrics['ats_accuracy']*100:>7.2f}%      "
          f"{(xgb_metrics['ats_accuracy'] - ridge_metrics['ats_accuracy'])*100:>+7.2f}%")
    
    print(f"{'ATS ROI':<25} {ridge_metrics['ats_roi']*100:>7.2f}%      "
          f"{xgb_metrics['ats_roi']*100:>7.2f}%      "
          f"{(xgb_metrics['ats_roi'] - ridge_metrics['ats_roi'])*100:>+7.2f}%")
    
    print()
    
    # ML metrics
    print(f"{'ML Accuracy':<25} {ridge_metrics['ml_accuracy']*100:>7.2f}%      "
          f"{xgb_metrics['ml_accuracy']*100:>7.2f}%      "
          f"{(xgb_metrics['ml_accuracy'] - ridge_metrics['ml_accuracy'])*100:>+7.2f}%")
    
    # Statistical significance tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE")
    print("="*70)
    
    # Merge dataframes for paired comparisons
    ridge_preds = ridge_metrics['predictions'].sort_values('game_id')
    xgb_preds = xgb_metrics['predictions'].sort_values('game_id')
    
    ridge_errors = (ridge_preds['pred_margin_mu'] - ridge_preds['actual_margin']).abs().values
    xgb_errors = (xgb_preds['pred_margin_mu'] - xgb_preds['actual_margin']).abs().values
    
    t_stat, p_value = paired_test(ridge_errors, xgb_errors)
    
    print(f"\nPaired t-test on absolute margin errors:")
    print(f"  t-statistic: {t_stat:+.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        winner = "XGBoost" if xgb_metrics['margin_mae'] < ridge_metrics['margin_mae'] else "Ridge"
        print(f"  Result: {winner} is SIGNIFICANTLY better (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    # Game-by-game comparison
    print("\n" + "="*70)
    print("GAME-BY-GAME ANALYSIS")
    print("="*70)
    
    merged = pd.merge(
        ridge_preds[['game_id', 'date', 'home_team', 'away_team', 'actual_margin', 
                     'pred_margin_mu', 'market_spread_home']],
        xgb_preds[['game_id', 'pred_margin_mu']],
        on='game_id',
        suffixes=('_ridge', '_xgb')
    )
    
    merged['error_ridge'] = (merged['pred_margin_mu_ridge'] - merged['actual_margin']).abs()
    merged['error_xgb'] = (merged['pred_margin_mu_xgb'] - merged['actual_margin']).abs()
    merged['error_diff'] = merged['error_xgb'] - merged['error_ridge']
    
    ridge_better = (merged['error_ridge'] < merged['error_xgb']).sum()
    xgb_better = (merged['error_xgb'] < merged['error_ridge']).sum()
    tied = (merged['error_ridge'] == merged['error_xgb']).sum()
    
    print(f"\nGames where Ridge had lower error: {ridge_better} ({ridge_better/len(merged)*100:.1f}%)")
    print(f"Games where XGBoost had lower error: {xgb_better} ({xgb_better/len(merged)*100:.1f}%)")
    print(f"Tied: {tied}")
    
    # Save detailed comparison
    print("\n" + "="*70)
    print("SAVING REPORTS")
    print("="*70)
    
    # Summary report
    report_path = output_dir / "comparison_summary.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RIDGE vs XGBOOST COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {ridge_metrics['n_games']} games\n\n")
        
        f.write("MARGIN ACCURACY:\n")
        f.write(f"  Ridge MAE:     {ridge_metrics['margin_mae']:.2f}\n")
        f.write(f"  XGBoost MAE:   {xgb_metrics['margin_mae']:.2f}\n")
        f.write(f"  Difference:    {xgb_metrics['margin_mae'] - ridge_metrics['margin_mae']:+.2f}\n\n")
        
        f.write(f"  Ridge RMSE:    {ridge_metrics['margin_rmse']:.2f}\n")
        f.write(f"  XGBoost RMSE:  {xgb_metrics['margin_rmse']:.2f}\n")
        f.write(f"  Difference:    {xgb_metrics['margin_rmse'] - ridge_metrics['margin_rmse']:+.2f}\n\n")
        
        f.write("ATS PERFORMANCE:\n")
        f.write(f"  Ridge ATS:     {ridge_metrics['ats_accuracy']*100:.2f}%\n")
        f.write(f"  XGBoost ATS:   {xgb_metrics['ats_accuracy']*100:.2f}%\n")
        f.write(f"  Difference:    {(xgb_metrics['ats_accuracy'] - ridge_metrics['ats_accuracy'])*100:+.2f}%\n\n")
        
        f.write(f"  Ridge ROI:     {ridge_metrics['ats_roi']*100:+.2f}%\n")
        f.write(f"  XGBoost ROI:   {xgb_metrics['ats_roi']*100:+.2f}%\n")
        f.write(f"  Difference:    {(xgb_metrics['ats_roi'] - ridge_metrics['ats_roi'])*100:+.2f}%\n\n")
        
        f.write("STATISTICAL TEST:\n")
        f.write(f"  t-statistic:   {t_stat:+.4f}\n")
        f.write(f"  p-value:       {p_value:.4f}\n")
        
        if p_value < 0.05:
            winner = "XGBoost" if xgb_metrics['margin_mae'] < ridge_metrics['margin_mae'] else "Ridge"
            f.write(f"  Significant:   YES ({winner} better)\n")
        else:
            f.write(f"  Significant:   NO\n")
    
    print(f"Summary saved to {report_path}")
    
    # Game-by-game details
    merged.to_csv(output_dir / "game_by_game_comparison.csv", index=False)
    print(f"Game-by-game comparison saved to {output_dir / 'game_by_game_comparison.csv'}")
    
    # Performance by spread size
    merged['spread_bucket'] = pd.cut(
        merged['market_spread_home'].abs(),
        bins=[0, 3, 6, 9, 12, 100],
        labels=['0-3', '3-6', '6-9', '9-12', '12+']
    )
    
    spread_comparison = merged.groupby('spread_bucket').agg({
        'error_ridge': 'mean',
        'error_xgb': 'mean',
        'game_id': 'count'
    }).round(2)
    spread_comparison.columns = ['Ridge MAE', 'XGBoost MAE', 'Count']
    spread_comparison['Difference'] = spread_comparison['XGBoost MAE'] - spread_comparison['Ridge MAE']
    
    spread_comparison.to_csv(output_dir / "comparison_by_spread.csv")
    print(f"Spread comparison saved to {output_dir / 'comparison_by_spread.csv'}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    
    # Print recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if p_value < 0.05:
        if xgb_metrics['margin_mae'] < ridge_metrics['margin_mae']:
            print("\nXGBoost shows SIGNIFICANTLY better margin prediction accuracy.")
            print(f"Improvement: {ridge_metrics['margin_mae'] - xgb_metrics['margin_mae']:.2f} points MAE")
        else:
            print("\nRidge shows SIGNIFICANTLY better margin prediction accuracy.")
            print(f"Improvement: {xgb_metrics['margin_mae'] - ridge_metrics['margin_mae']:.2f} points MAE")
    else:
        print("\nNo significant difference in margin prediction accuracy.")
        print("Consider other factors:")
        print(f"  - Training time")
        print(f"  - Interpretability (Ridge has clear coefficients)")
        print(f"  - ATS performance ({xgb_metrics['ats_accuracy']*100:.2f}% vs {ridge_metrics['ats_accuracy']*100:.2f}%)")


if __name__ == "__main__":
    main()

