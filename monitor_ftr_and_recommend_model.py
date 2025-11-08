#!/usr/bin/env python
"""
FTR Monitoring and Model Recommendation Script

This script monitors the current season's Free Throw Rate (FTR) and recommends
which model to use based on whether we're experiencing a regime change.

Key Findings from Analysis:
- Experiment 2 excels during regime changes (+4.20 pp on 2025-26)
- Champion excels during stable periods (+5.63 pp on 2024-25)
- FTR features contribute +2.12 pp more during regime changes
- Role indicators contribute +5.31 pp more during regime changes

Strategy:
- Use Exp2 when current season FTR is significantly above historical baseline
- Use Champion when current season FTR is near historical baseline
- Threshold: If current FTR > historical baseline + 1%, use Exp2

Usage:
    python monitor_ftr_and_recommend_model.py
    
    # Custom threshold (in percentage points)
    python monitor_ftr_and_recommend_model.py --threshold 1.5
    
    # Show detailed stats
    python monitor_ftr_and_recommend_model.py --verbose
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def find_most_recent_boxscore(current_dir: Path) -> Path:
    """Find the most recent boxscore file in the current directory."""
    xlsx_files = list(current_dir.glob("*.xlsx"))
    
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {current_dir}")
    
    # Parse dates from filenames (format: MM-DD-YYYY-nba-season-team-feed.xlsx)
    files_with_dates = []
    for file in xlsx_files:
        try:
            # Extract date part (MM-DD-YYYY)
            date_str = file.stem.split('-nba-season')[0]
            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
            files_with_dates.append((file, date_obj))
        except ValueError:
            continue
    
    if not files_with_dates:
        raise ValueError("No files with valid date format found")
    
    # Sort by date and get most recent
    files_with_dates.sort(key=lambda x: x[1], reverse=True)
    most_recent = files_with_dates[0]
    
    return most_recent[0], most_recent[1]


def load_historical_ftr(historical_dir: Path) -> Dict[str, float]:
    """Load historical FTR data from all seasons."""
    historical_ftrs = {}
    
    xlsx_files = list(historical_dir.glob("*.xlsx"))
    
    for file in xlsx_files:
        # Extract season from filename (e.g., 2021-2022_NBA_Box_Score_Team-Stats.xlsx)
        season = file.stem.split('_')[0]
        
        try:
            df = pd.read_excel(file)
            
            # Calculate FTR (FTA / FGA)
            # Handle potential column name variations
            fta_col = None
            fga_col = None
            
            for col in df.columns:
                col_upper = str(col).upper()
                if 'FTA' in col_upper and 'FTA%' not in col_upper:
                    fta_col = col
                elif 'FGA' in col_upper and 'FGA%' not in col_upper:
                    fga_col = col
            
            if fta_col is None or fga_col is None:
                print(f"  Warning: Could not find FTA/FGA columns in {file.name}")
                continue
            
            # Sum all FTA and FGA for the season
            total_fta = df[fta_col].sum()
            total_fga = df[fga_col].sum()
            
            if total_fga > 0:
                season_ftr = (total_fta / total_fga) * 100  # Convert to percentage
                historical_ftrs[season] = season_ftr
            
        except Exception as e:
            print(f"  Warning: Error processing {file.name}: {e}")
            continue
    
    return historical_ftrs


def calculate_current_ftr(file_path: Path) -> Tuple[float, int, Dict[str, float]]:
    """Calculate current season FTR from most recent file."""
    df = pd.read_excel(file_path)
    
    # Find FTA and FGA columns
    fta_col = None
    fga_col = None
    
    for col in df.columns:
        col_upper = str(col).upper()
        if 'FTA' in col_upper and 'FTA%' not in col_upper:
            fta_col = col
        elif 'FGA' in col_upper and 'FGA%' not in col_upper:
            fga_col = col
    
    if fta_col is None or fga_col is None:
        raise ValueError(f"Could not find FTA/FGA columns in {file_path.name}")
    
    # Calculate overall FTR
    total_fta = df[fta_col].sum()
    total_fga = df[fga_col].sum()
    
    # Divide by 2 because each game has 2 teams (rows in the data)
    n_games = len(df) // 2
    
    current_ftr = (total_fta / total_fga) * 100 if total_fga > 0 else 0
    
    # Calculate per-game stats (per actual game, not per team)
    stats = {
        'total_fta': total_fta,
        'total_fga': total_fga,
        'avg_fta_per_game': total_fta / n_games if n_games > 0 else 0,
        'avg_fga_per_game': total_fga / n_games if n_games > 0 else 0,
    }
    
    return current_ftr, n_games, stats


def load_all_current_files(current_dir: Path) -> pd.DataFrame:
    """Load all current season files and calculate FTR progression over time."""
    xlsx_files = list(current_dir.glob("*.xlsx"))
    
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {current_dir}")
    
    # Parse dates and sort chronologically
    files_with_dates = []
    for file in xlsx_files:
        try:
            date_str = file.stem.split('-nba-season')[0]
            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
            files_with_dates.append((file, date_obj))
        except ValueError:
            continue
    
    files_with_dates.sort(key=lambda x: x[1])
    
    # Load and process each file
    all_data = []
    
    for file, file_date in files_with_dates:
        try:
            df = pd.read_excel(file)
            
            # Find FTA and FGA columns
            fta_col = None
            fga_col = None
            
            for col in df.columns:
                col_upper = str(col).upper()
                if 'FTA' in col_upper and 'FTA%' not in col_upper:
                    fta_col = col
                elif 'FGA' in col_upper and 'FGA%' not in col_upper:
                    fga_col = col
            
            if fta_col is None or fga_col is None:
                continue
            
            # Extract data for this snapshot
            total_fta = df[fta_col].sum()
            total_fga = df[fga_col].sum()
            # Divide by 2 because each game has 2 teams (rows)
            n_games = len(df) // 2
            
            if total_fga > 0:
                ftr = (total_fta / total_fga) * 100
                all_data.append({
                    'date': file_date,
                    'n_games': n_games,
                    'total_fta': total_fta,
                    'total_fga': total_fga,
                    'ftr': ftr
                })
        except Exception as e:
            print(f"  Warning: Error processing {file.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data found in current season files")
    
    return pd.DataFrame(all_data)


def calculate_rolling_ftr(df: pd.DataFrame, historical_baseline: float) -> Dict[str, float]:
    """Calculate rolling FTR averages and trends."""
    # Get most recent snapshot
    latest = df.iloc[-1]
    
    # Calculate overall stats
    overall_ftr = latest['ftr']
    overall_games = latest['n_games']
    
    # Calculate trends (change between snapshots)
    if len(df) >= 2:
        recent_ftr = latest['ftr']
        prev_ftr = df.iloc[-2]['ftr']
        ftr_change = recent_ftr - prev_ftr
        
        # Calculate velocity (rate of change)
        days_diff = (latest['date'] - df.iloc[-2]['date']).days
        games_diff = latest['n_games'] - df.iloc[-2]['n_games']
        
        if len(df) >= 3:
            # 3-point trend
            oldest_ftr = df.iloc[-3]['ftr']
            trend_direction = "increasing" if recent_ftr > prev_ftr > oldest_ftr else \
                            "decreasing" if recent_ftr < prev_ftr < oldest_ftr else \
                            "stable/mixed"
        else:
            trend_direction = "increasing" if ftr_change > 0 else \
                            "decreasing" if ftr_change < 0 else "stable"
    else:
        ftr_change = 0
        games_diff = 0
        trend_direction = "insufficient data"
    
    # Distance from baseline
    distance_from_baseline = overall_ftr - historical_baseline
    
    return {
        'overall_ftr': overall_ftr,
        'overall_games': overall_games,
        'recent_change': ftr_change,
        'recent_games_added': games_diff,
        'trend_direction': trend_direction,
        'distance_from_baseline': distance_from_baseline,
    }


def plot_ftr_progression(df: pd.DataFrame, historical_baseline: float, 
                        output_path: Path = None):
    """Create a visual chart of FTR progression over the season."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot FTR over time
        ax.plot(df['date'], df['ftr'], 'o-', linewidth=2, markersize=8, 
                color='#3b82f6', label='Current Season FTR')
        
        # Add baseline
        ax.axhline(y=historical_baseline, color='#10b981', linestyle='--', 
                  linewidth=2, label=f'Historical Baseline ({historical_baseline:.2f}%)')
        
        # Add threshold lines
        threshold = 1.0
        ax.axhline(y=historical_baseline + threshold, color='#f59e0b', 
                  linestyle=':', linewidth=1.5, alpha=0.7,
                  label=f'Exp2 Threshold ({historical_baseline + threshold:.2f}%)')
        
        # Fill area between current and baseline
        ax.fill_between(df['date'], df['ftr'], historical_baseline, 
                       where=(df['ftr'] >= historical_baseline),
                       alpha=0.3, color='#ef4444', label='Above Baseline')
        ax.fill_between(df['date'], df['ftr'], historical_baseline,
                       where=(df['ftr'] < historical_baseline),
                       alpha=0.3, color='#10b981', label='Below Baseline')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Free Throw Rate (%)', fontsize=12)
        ax.set_title('2025-26 Season FTR Progression', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved chart to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return True
    except ImportError:
        print("  Warning: matplotlib not available, skipping chart generation")
        return False


def recommend_model(current_ftr: float, historical_baseline: float, 
                   threshold: float) -> Tuple[str, str, float]:
    """
    Recommend which model to use based on FTR comparison.
    
    Returns:
        Tuple of (model_name, recommendation_reason, confidence_score)
    """
    ftr_diff = current_ftr - historical_baseline
    
    if ftr_diff >= threshold:
        # Regime change detected - use Exp2
        confidence = min(100, 50 + (ftr_diff / threshold) * 25)
        model = "Experiment 2 (FTR + Role Indicators)"
        reason = f"Current FTR is {ftr_diff:.2f} pp above baseline (threshold: {threshold:.2f} pp)"
        return model, reason, confidence
    
    elif ftr_diff >= threshold * 0.5:
        # Borderline - slight preference for Exp2
        confidence = 60
        model = "Experiment 2 (FTR + Role Indicators)"
        reason = f"Current FTR is {ftr_diff:.2f} pp above baseline (borderline regime change)"
        return model, reason, confidence
    
    else:
        # Stable period - use Champion
        confidence = min(100, 50 + abs(ftr_diff) / threshold * 25)
        model = "Champion (Optimized Standard)"
        reason = f"Current FTR is {ftr_diff:.2f} pp from baseline (stable period)"
        return model, reason, confidence


def print_header():
    """Print formatted header."""
    print()
    print("=" * 80)
    print("FTR MONITORING & MODEL RECOMMENDATION")
    print("=" * 80)
    print()


def print_section(title: str):
    """Print section header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)
    print()


def print_recommendation_box(model: str, reason: str, confidence: float):
    """Print highlighted recommendation."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + f"  RECOMMENDED MODEL: {model}".ljust(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + f"  Confidence: {confidence:.0f}%".ljust(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + f"  Reason: {reason}".ljust(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor FTR and recommend model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="FTR difference threshold in percentage points (default: 1.0)"
    )
    parser.add_argument(
        "--current-dir",
        type=str,
        default="data/team_boxscores/current",
        help="Directory containing current season boxscore files"
    )
    parser.add_argument(
        "--historical-dir",
        type=str,
        default="data/team_boxscores/historical",
        help="Directory containing historical boxscore files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed statistics"
    )
    
    args = parser.parse_args()
    
    current_dir = Path(args.current_dir)
    historical_dir = Path(args.historical_dir)
    
    print_header()
    
    # Step 1: Load all current season data and analyze progression
    print_section("STEP 1: Loading Current Season Data & Progression")
    
    try:
        # Load all files to see progression
        print("Loading all current season files...")
        progression_df = load_all_current_files(current_dir)
        
        print(f"Found {len(progression_df)} data snapshots from {progression_df['date'].min().strftime('%m/%d/%Y')} to {progression_df['date'].max().strftime('%m/%d/%Y')}")
        print()
        
        # Get most recent snapshot
        most_recent_file, file_date = find_most_recent_boxscore(current_dir)
        print(f"Most recent file: {most_recent_file.name}")
        print(f"Date: {file_date.strftime('%B %d, %Y')}")
        print()
        
        current_ftr, n_games, current_stats = calculate_current_ftr(most_recent_file)
        
        print(f"Current Season Statistics:")
        print(f"  Games analyzed: {n_games}")
        print(f"  Total FTA: {current_stats['total_fta']:.0f}")
        print(f"  Total FGA: {current_stats['total_fga']:.0f}")
        print(f"  Average FTA per game: {current_stats['avg_fta_per_game']:.2f}")
        print(f"  Average FGA per game: {current_stats['avg_fga_per_game']:.2f}")
        print(f"  Current FTR: {current_ftr:.3f}%")
        print()
        
        # Show progression snapshots
        print("FTR Progression Over Season:")
        for _, row in progression_df.iterrows():
            print(f"  {row['date'].strftime('%m/%d/%Y')}: {row['ftr']:.3f}% ({row['n_games']} games)")
        
    except Exception as e:
        print(f"Error loading current season data: {e}")
        return
    
    # Step 2: Load historical baseline
    print_section("STEP 2: Loading Historical Baseline")
    
    try:
        historical_ftrs = load_historical_ftr(historical_dir)
        
        if not historical_ftrs:
            print("Error: No historical data found")
            return
        
        print(f"Historical FTR by Season:")
        for season in sorted(historical_ftrs.keys()):
            ftr = historical_ftrs[season]
            print(f"  {season}: {ftr:.3f}%")
        
        # Calculate baseline (average of historical seasons)
        baseline_ftr = np.mean(list(historical_ftrs.values()))
        baseline_std = np.std(list(historical_ftrs.values()))
        
        print()
        print(f"Historical Baseline:")
        print(f"  Mean FTR: {baseline_ftr:.3f}%")
        print(f"  Std Dev: {baseline_std:.3f}%")
        print(f"  Range: {min(historical_ftrs.values()):.3f}% - {max(historical_ftrs.values()):.3f}%")
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return
    
    # Step 3: Analyze trends and progression
    print_section("STEP 3: FTR Trend Analysis")
    
    trend_stats = calculate_rolling_ftr(progression_df, baseline_ftr)
    
    print(f"Trend Analysis:")
    print(f"  Recent change (since last snapshot): {trend_stats['recent_change']:+.3f} pp")
    print(f"  Games added since last snapshot: {trend_stats['recent_games_added']}")
    print(f"  Trend direction: {trend_stats['trend_direction'].upper()}")
    print(f"  Distance from baseline: {trend_stats['distance_from_baseline']:+.3f} pp")
    print()
    
    # Generate visual chart
    print("Generating FTR progression chart...")
    chart_output = Path("analysis") / "ftr_progression.png"
    chart_output.parent.mkdir(exist_ok=True)
    plot_ftr_progression(progression_df, baseline_ftr, chart_output)
    print()
    
    # Step 4: Compare and analyze
    print_section("STEP 4: Regime Change Analysis")
    
    ftr_diff = current_ftr - baseline_ftr
    z_score = ftr_diff / baseline_std if baseline_std > 0 else 0
    
    print(f"Current vs Historical:")
    print(f"  Current FTR: {current_ftr:.3f}%")
    print(f"  Baseline FTR: {baseline_ftr:.3f}%")
    print(f"  Difference: {ftr_diff:+.3f} percentage points")
    print(f"  Z-Score: {z_score:.2f} standard deviations")
    print()
    
    if abs(z_score) >= 2.0:
        regime_status = "⚠️  SIGNIFICANT REGIME CHANGE DETECTED"
    elif abs(z_score) >= 1.0:
        regime_status = "⚡ Moderate deviation from baseline"
    else:
        regime_status = "✓ Within normal historical range"
    
    print(f"Status: {regime_status}")
    print()
    
    # Interpret trend
    if trend_stats['trend_direction'] == 'increasing':
        print("⬆️  FTR is INCREASING - regime change may be intensifying")
    elif trend_stats['trend_direction'] == 'decreasing':
        print("⬇️  FTR is DECREASING - possible normalization toward baseline")
        if trend_stats['distance_from_baseline'] > args.threshold:
            print("   ⚠️  Still above threshold, but monitor closely for continued decline")
    else:
        print("➡️  FTR is STABLE - regime appears consistent")
    
    # Step 5: Model recommendation
    print_section("STEP 5: Model Recommendation")
    
    model, reason, confidence = recommend_model(current_ftr, baseline_ftr, args.threshold)
    
    print(f"Decision Threshold: {args.threshold:.2f} percentage points")
    print(f"Current Difference: {ftr_diff:+.2f} percentage points")
    print()
    
    # Explain the logic
    if ftr_diff >= args.threshold:
        print("Analysis:")
        print("  → Current FTR exceeds threshold")
        print("  → Regime change likely in effect")
        print("  → Exp2 features (FTR + role indicators) add value")
        print("  → Expected advantage: +4-5 percentage points ATS")
    elif ftr_diff >= args.threshold * 0.5:
        print("Analysis:")
        print("  → Current FTR near threshold (borderline)")
        print("  → Possible regime change developing")
        print("  → Exp2 features may provide edge")
        print("  → Monitor closely for threshold breach")
    else:
        print("Analysis:")
        print("  → Current FTR within normal range")
        print("  → Stable period, no regime change")
        print("  → Champion model more reliable")
        print("  → Expected advantage: +5-6 percentage points ATS")
    
    print_recommendation_box(model, reason, confidence)
    
    # Model paths
    if "Experiment 2" in model:
        model_path = "artifacts/margin_experiment2_ftr_plus_role"
        script_path = "daily_betting_recommendations_exp2.py"
    else:
        model_path = "artifacts/margin_xgboost_optimized_with2425"
        script_path = "daily_betting_recommendations.py"
    
    print("Implementation:")
    print(f"  Model Path: {model_path}")
    print(f"  Script: {script_path}")
    print()
    
    # Verbose output
    if args.verbose:
        print_section("DETAILED PERFORMANCE EXPECTATIONS")
        
        print("Based on permutation importance analysis:")
        print()
        
        if "Experiment 2" in model:
            print("Experiment 2 Advantages (Current Regime):")
            print("  • FTR features: +2.12 pp contribution")
            print("  • Role indicators: +5.31 pp contribution")
            print("  • True shooting: +5.47 pp contribution")
            print("  • Total new signal: ~13 pp from experimental features")
            print()
            print("Risk Factors:")
            print("  • If FTR normalizes, performance will degrade")
            print("  • Historical ATS: 49.66% (below breakeven)")
            print("  • High variance across different regimes")
        else:
            print("Champion Advantages (Stable Period):")
            print("  • Consistent across historical periods")
            print("  • Proven 55%+ ATS on similar data")
            print("  • Lower variance, more predictable")
            print("  • Traditional features remain strong")
            print()
            print("Risk Factors:")
            print("  • Cannot adapt to FTR changes")
            print("  • May underperform if regime persists")
            print("  • Missing 40% of Exp2's signal sources")
        print()
    
    # Monitoring advice
    print_section("MONITORING RECOMMENDATIONS")
    
    print("Next Steps:")
    print("  1. Run this script weekly to track FTR trends")
    print("  2. If recommendation changes, retrain or switch models")
    print("  3. Track live ATS performance of recommended model")
    print("  4. Consider ensemble if near threshold")
    print()
    
    print("Alert Thresholds:")
    print(f"  • Switch to Exp2: FTR > {baseline_ftr + args.threshold:.3f}%")
    print(f"  • Switch to Champion: FTR < {baseline_ftr + args.threshold * 0.5:.3f}%")
    print(f"  • High Alert: |Z-Score| > 2.0 (currently: {abs(z_score):.2f})")
    print()
    
    print("=" * 80)
    print(f"Analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

