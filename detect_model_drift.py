#!/usr/bin/env python
"""
Model Drift Detection for Champion Model

This script tracks model performance drift by analyzing:
1. MAE (Mean Absolute Error) over time
2. ATS (Against The Spread) accuracy over time
3. Rolling performance windows
4. Performance by game segments

Usage:
    python detect_model_drift.py
    
    # Custom predictions file
    python detect_model_drift.py --predictions path/to/predictions.csv
    
    # Custom window size for rolling metrics
    python detect_model_drift.py --window 30
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Training baseline metrics (from Champion model artifacts)
TRAINING_BASELINE = {
    'mae': 10.20,  # Training MAE
    'rmse': 13.08,  # Training RMSE
    'ats_pct': 57.5,  # Training ATS accuracy %
    'training_games': 3560,  # 2021-2024 seasons
}


def load_predictions(predictions_file: Path) -> pd.DataFrame:
    """Load predictions CSV with actual results."""
    print(f"Loading predictions from {predictions_file}...")
    
    df = pd.read_csv(predictions_file)
    
    # Standardize column names
    column_map = {
        'pred_margin_mu': 'predicted_margin',
        'market_spread_home': 'spread',
        'date': 'game_date',
    }
    
    df = df.rename(columns=column_map)
    
    # Ensure required columns exist after mapping
    required_cols = ['predicted_margin', 'actual_margin', 'spread']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter to only completed games (have actual results)
    df = df[df['actual_margin'].notna()].copy()
    
    if len(df) == 0:
        raise ValueError("No completed games found in predictions file")
    
    print(f"Loaded {len(df)} completed games")
    
    return df


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall performance metrics."""
    
    # Prediction errors
    errors = df['predicted_margin'] - df['actual_margin']
    abs_errors = np.abs(errors)
    
    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    # ATS accuracy
    # Home team covers if: actual_margin > -spread
    # Model predicts cover if: predicted_margin > -spread
    actual_cover = (df['actual_margin'] > -df['spread']).astype(int)
    predicted_cover = (df['predicted_margin'] > -df['spread']).astype(int)
    ats_correct = (actual_cover == predicted_cover).sum()
    ats_pct = (ats_correct / len(df)) * 100
    
    # ML (Moneyline) accuracy - just predict winner
    actual_win = (df['actual_margin'] > 0).astype(int)
    predicted_win = (df['predicted_margin'] > 0).astype(int)
    ml_correct = (actual_win == predicted_win).sum()
    ml_pct = (ml_correct / len(df)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'ats_correct': ats_correct,
        'ats_total': len(df),
        'ats_pct': ats_pct,
        'ml_correct': ml_correct,
        'ml_total': len(df),
        'ml_pct': ml_pct,
        'mean_error': errors.mean(),  # Bias check
        'median_error': errors.median(),
    }


def calculate_rolling_metrics(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Calculate rolling performance metrics."""
    
    df = df.copy()
    
    # Ensure sorted by date/game order
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values('game_date')
    
    # Calculate per-game metrics
    df['error'] = df['predicted_margin'] - df['actual_margin']
    df['abs_error'] = np.abs(df['error'])
    
    actual_cover = (df['actual_margin'] > -df['spread']).astype(int)
    predicted_cover = (df['predicted_margin'] > -df['spread']).astype(int)
    df['ats_correct'] = (actual_cover == predicted_cover).astype(int)
    
    # Rolling calculations
    df[f'mae_{window}game'] = df['abs_error'].rolling(window, min_periods=10).mean()
    df[f'ats_{window}game'] = df['ats_correct'].rolling(window, min_periods=10).mean() * 100
    df[f'bias_{window}game'] = df['error'].rolling(window, min_periods=10).mean()
    
    return df


def detect_drift(current_metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, Any]:
    """Detect drift by comparing current metrics to training baseline."""
    
    mae_drift = ((current_metrics['mae'] - baseline['mae']) / baseline['mae']) * 100
    ats_drift = current_metrics['ats_pct'] - baseline['ats_pct']  # Percentage points
    
    # Determine drift severity
    if mae_drift > 15 or ats_drift < -5:
        severity = '🚨 CRITICAL'
    elif mae_drift > 10 or ats_drift < -3:
        severity = '⚠️  WARNING'
    elif mae_drift > 5 or ats_drift < -2:
        severity = '⚡ WATCH'
    else:
        severity = '✓ NORMAL'
    
    return {
        'severity': severity,
        'mae_drift_pct': mae_drift,
        'ats_drift_pct_points': ats_drift,
        'mae_current': current_metrics['mae'],
        'mae_baseline': baseline['mae'],
        'ats_current': current_metrics['ats_pct'],
        'ats_baseline': baseline['ats_pct'],
    }


def analyze_performance_by_segment(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze performance across different game segments."""
    
    segments = {}
    
    # By spread magnitude
    segments['close_games'] = df[np.abs(df['spread']) < 3]
    segments['medium_spreads'] = df[(np.abs(df['spread']) >= 3) & (np.abs(df['spread']) < 7)]
    segments['large_spreads'] = df[np.abs(df['spread']) >= 7]
    
    # By home/away performance (if we have the data)
    if 'home_team' in df.columns and 'away_team' in df.columns:
        # Analyze home favorites vs away favorites
        segments['home_favorites'] = df[df['spread'] < 0]  # Negative spread = home favored
        segments['away_favorites'] = df[df['spread'] > 0]
    
    results = {}
    
    for seg_name, seg_df in segments.items():
        if len(seg_df) < 5:
            continue
        
        metrics = calculate_performance_metrics(seg_df)
        
        # Test for systematic bias
        errors = seg_df['predicted_margin'] - seg_df['actual_margin']
        mean_error = errors.mean()
        std_error = errors.std()
        t_stat = mean_error / (std_error / np.sqrt(len(seg_df))) if std_error > 0 else 0
        
        results[seg_name] = {
            'n_games': len(seg_df),
            'mae': metrics['mae'],
            'ats_pct': metrics['ats_pct'],
            'mean_error': mean_error,
            'systematic_bias': abs(t_stat) > 2.0,  # Statistically significant
            't_statistic': t_stat
        }
    
    return results


def plot_drift_over_time(df_rolling: pd.DataFrame, window: int, output_dir: Path = None):
    """Create visualization of drift over time."""
    try:
        import matplotlib.pyplot as plt
        
        if output_dir:
            output_dir.mkdir(exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: MAE over time
        ax1.plot(range(len(df_rolling)), df_rolling[f'mae_{window}game'], 
                linewidth=2, color='#ef4444', label=f'{window}-game rolling MAE')
        ax1.axhline(y=TRAINING_BASELINE['mae'], color='#3b82f6', linestyle='--', 
                   linewidth=2, label=f"Training baseline ({TRAINING_BASELINE['mae']:.2f})")
        ax1.axhline(y=TRAINING_BASELINE['mae'] * 1.15, color='#f59e0b', linestyle=':', 
                   linewidth=1.5, label='Warning threshold (+15%)')
        
        ax1.set_xlabel('Game Number', fontsize=12)
        ax1.set_ylabel('Mean Absolute Error (points)', fontsize=12)
        ax1.set_title('Model Prediction Error Over Time (2025-26 Season)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ATS% over time
        ax2.plot(range(len(df_rolling)), df_rolling[f'ats_{window}game'], 
                linewidth=2, color='#10b981', label=f'{window}-game rolling ATS%')
        ax2.axhline(y=TRAINING_BASELINE['ats_pct'], color='#3b82f6', linestyle='--', 
                   linewidth=2, label=f"Training baseline ({TRAINING_BASELINE['ats_pct']:.1f}%)")
        ax2.axhline(y=52.38, color='#6b7280', linestyle=':', 
                   linewidth=1.5, label='Break-even (52.38%)')
        ax2.axhline(y=50.0, color='#ef4444', linestyle=':', 
                   linewidth=1.5, label='Warning threshold (50%)')
        
        ax2.set_xlabel('Game Number', fontsize=12)
        ax2.set_ylabel('ATS Accuracy (%)', fontsize=12)
        ax2.set_title('Model ATS Performance Over Time (2025-26 Season)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(40, 70)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = output_dir / 'model_drift_performance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved drift chart to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return True
        
    except ImportError:
        print("  Warning: matplotlib not available, skipping drift chart")
        return False


def print_header():
    """Print formatted header."""
    print()
    print("=" * 80)
    print("MODEL DRIFT DETECTION - CHAMPION MODEL")
    print("=" * 80)
    print()


def print_section(title: str):
    """Print section header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Detect model drift via performance metrics"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=r"predictions\current_season_champion_2025_2026_predictions.csv",
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window size for metrics (default: 30 games)"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )
    
    args = parser.parse_args()
    
    predictions_file = Path(args.predictions)
    
    if not predictions_file.exists():
        print(f"Error: Predictions file not found: {predictions_file}")
        return
    
    print_header()
    
    # Load predictions
    print_section("STEP 1: Loading Predictions & Actual Results")
    df = load_predictions(predictions_file)
    
    # Calculate overall metrics
    print_section("STEP 2: Overall Performance Metrics")
    
    overall_metrics = calculate_performance_metrics(df)
    
    print(f"Season Performance ({len(df)} completed games):")
    print()
    print(f"  Prediction Accuracy:")
    print(f"    MAE:         {overall_metrics['mae']:.2f} points")
    print(f"    RMSE:        {overall_metrics['rmse']:.2f} points")
    print(f"    Mean Error:  {overall_metrics['mean_error']:+.2f} points (bias check)")
    print(f"    Median Err:  {overall_metrics['median_error']:+.2f} points")
    print()
    print(f"  Betting Performance:")
    print(f"    ATS Accuracy:  {overall_metrics['ats_correct']}/{overall_metrics['ats_total']} "
          f"({overall_metrics['ats_pct']:.2f}%)")
    print(f"    ML Accuracy:   {overall_metrics['ml_correct']}/{overall_metrics['ml_total']} "
          f"({overall_metrics['ml_pct']:.2f}%)")
    print()
    
    # Drift detection
    print_section("STEP 3: Drift Detection")
    
    drift_results = detect_drift(overall_metrics, TRAINING_BASELINE)
    
    print(f"Drift Status: {drift_results['severity']}")
    print()
    print(f"Comparison to Training Baseline:")
    print(f"  MAE:")
    print(f"    Training:  {drift_results['mae_baseline']:.2f} points")
    print(f"    Current:   {drift_results['mae_current']:.2f} points")
    print(f"    Drift:     {drift_results['mae_drift_pct']:+.1f}%")
    
    if drift_results['mae_drift_pct'] > 15:
        print(f"    🚨 CRITICAL: MAE degraded >15%")
    elif drift_results['mae_drift_pct'] > 10:
        print(f"    ⚠️  WARNING: MAE degraded >10%")
    elif drift_results['mae_drift_pct'] > 5:
        print(f"    ⚡ WATCH: MAE degraded >5%")
    else:
        print(f"    ✓ NORMAL: MAE within acceptable range")
    
    print()
    print(f"  ATS Accuracy:")
    print(f"    Training:  {drift_results['ats_baseline']:.1f}%")
    print(f"    Current:   {drift_results['ats_current']:.2f}%")
    print(f"    Drift:     {drift_results['ats_drift_pct_points']:+.2f} percentage points")
    
    if drift_results['ats_drift_pct_points'] < -5:
        print(f"    🚨 CRITICAL: ATS dropped >5 percentage points")
    elif drift_results['ats_drift_pct_points'] < -3:
        print(f"    ⚠️  WARNING: ATS dropped >3 percentage points")
    elif drift_results['ats_drift_pct_points'] < -2:
        print(f"    ⚡ WATCH: ATS dropped >2 percentage points")
    else:
        print(f"    ✓ NORMAL: ATS within acceptable range")
    
    print()
    
    # Rolling metrics
    print_section(f"STEP 4: Rolling Performance ({args.window}-game windows)")
    
    df_rolling = calculate_rolling_metrics(df, window=args.window)
    
    # Get most recent rolling metrics (last complete window)
    recent_idx = df_rolling[f'mae_{args.window}game'].last_valid_index()
    if recent_idx is not None and recent_idx >= args.window:
        recent_mae = df_rolling.loc[recent_idx, f'mae_{args.window}game']
        recent_ats = df_rolling.loc[recent_idx, f'ats_{args.window}game']
        
        print(f"Most Recent {args.window}-game Performance:")
        print(f"  MAE:  {recent_mae:.2f} points")
        print(f"  ATS:  {recent_ats:.2f}%")
        print()
        
        # Check for recent deterioration
        if recent_mae > overall_metrics['mae'] * 1.1:
            print(f"  ⚠️  Recent {args.window}-game MAE is {((recent_mae/overall_metrics['mae']-1)*100):.1f}% "
                  f"worse than season average")
        
        if recent_ats < overall_metrics['ats_pct'] - 3:
            print(f"  ⚠️  Recent {args.window}-game ATS is {(overall_metrics['ats_pct']-recent_ats):.1f} "
                  f"percentage points worse than season average")
    
    # Segment analysis
    print_section("STEP 5: Performance by Game Segment")
    
    segment_results = analyze_performance_by_segment(df)
    
    print("Performance varies by game type:")
    print()
    
    for segment_name, metrics in segment_results.items():
        print(f"  {segment_name.replace('_', ' ').title()}:")
        print(f"    Games:      {metrics['n_games']}")
        print(f"    MAE:        {metrics['mae']:.2f} points")
        print(f"    ATS:        {metrics['ats_pct']:.1f}%")
        print(f"    Mean Error: {metrics['mean_error']:+.2f} points")
        
        if metrics['systematic_bias']:
            print(f"    ⚠️  Systematic bias detected (t={metrics['t_statistic']:.2f})")
        
        print()
    
    # Generate charts
    if not args.no_charts:
        print_section("STEP 6: Generating Performance Charts")
        print("Creating drift visualization...")
        output_dir = Path("analysis")
        plot_drift_over_time(df_rolling, args.window, output_dir)
    
    # Summary & Recommendations
    print_section("STEP 7: Summary & Recommendations")
    
    print("Drift Assessment:")
    print()
    
    if drift_results['severity'] == '🚨 CRITICAL':
        print("  🚨 CRITICAL DRIFT DETECTED")
        print("     Model performance has degraded significantly.")
        print()
        print("  Recommended Actions:")
        print("    1. URGENT: Review recent predictions for systematic errors")
        print("    2. Consider retraining model on 2024-2025 data immediately")
        print("    3. Reduce bet sizing until model is updated")
        print("    4. Investigate which features are causing poor performance")
        
    elif drift_results['severity'] == '⚠️  WARNING':
        print("  ⚠️  WARNING: Moderate drift detected")
        print("     Model performance has declined but is still operational.")
        print()
        print("  Recommended Actions:")
        print("    1. Monitor closely - run this script weekly")
        print("    2. Plan model retraining within 2-4 weeks")
        print("    3. Review segment analysis for specific weaknesses")
        print("    4. Check if feature distributions have changed (run monitor_nba_metrics.py)")
        
    elif drift_results['severity'] == '⚡ WATCH':
        print("  ⚡ Minor drift detected")
        print("     Model performance has slightly declined.")
        print()
        print("  Recommended Actions:")
        print("    1. Continue weekly monitoring")
        print("    2. Consider retraining if trend continues")
        print("    3. No immediate action required")
        
    else:
        print("  ✓ NO SIGNIFICANT DRIFT")
        print("     Model performing within expected parameters.")
        print()
        print("  Recommended Actions:")
        print("    1. Continue normal monitoring schedule")
        print("    2. Model retraining not urgently needed")
    
    print()
    print("=" * 80)
    print(f"Drift detection complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

