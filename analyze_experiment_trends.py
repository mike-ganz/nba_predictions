"""
Analyze how ATS accuracy trends over the 2025-26 season for each experiment.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_predictions(filepath: str) -> pd.DataFrame:
    """Load predictions and ensure required columns exist."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate ATS result
    # Home covers if: actual_margin > -market_spread_home
    # Prediction covers if: pred_margin > -market_spread_home
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['ats_correct'] = (df['actual_home_covers'] == df['pred_home_covers']).astype(int)
    
    return df

def calculate_cumulative_ats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative ATS accuracy over time."""
    df = df.copy()
    df['cumulative_correct'] = df['ats_correct'].cumsum()
    df['cumulative_games'] = range(1, len(df) + 1)
    df['cumulative_ats_pct'] = (df['cumulative_correct'] / df['cumulative_games']) * 100
    return df

def calculate_rolling_ats(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Calculate rolling ATS accuracy over time."""
    df = df.copy()
    df['rolling_ats_pct'] = df['ats_correct'].rolling(window=window, min_periods=10).mean() * 100
    return df

def main():
    print("="*80)
    print("EXPERIMENT ATS TREND ANALYSIS - 2025-26 Season")
    print("="*80)
    
    # Define experiments (skip exp1 since it's same as champion)
    experiments = [
        {
            'name': 'Current Champion',
            'file': 'predictions/current_season_champion_2025_2026_predictions.csv',
            'color': '#2E86AB',  # Blue
            'label': 'Champion (14 feat, 21-24)'
        },
        {
            'name': 'Exp2',
            'file': 'predictions/experiments/exp2_predictions.csv',
            'color': '#A23B72',  # Purple
            'label': 'Exp2: 14 feat, 21-25'
        },
        {
            'name': 'Exp3',
            'file': 'predictions/experiments/exp3_predictions.csv',
            'color': '#F18F01',  # Orange
            'label': 'Exp3: 16 feat+FTR, 21-24'
        },
        {
            'name': 'Exp4',
            'file': 'predictions/experiments/exp4_predictions.csv',
            'color': '#C73E1D',  # Red
            'label': 'Exp4: 16 feat+FTR, 21-25'
        }
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('ATS Accuracy Trends - 2025-26 Season', fontsize=16, fontweight='bold')
    
    stats_summary = []
    
    for exp in experiments:
        print(f"\nLoading {exp['name']}...")
        df = load_predictions(exp['file'])
        
        # Calculate both cumulative and rolling
        df = calculate_cumulative_ats(df)
        df = calculate_rolling_ats(df, window=30)
        
        # Stats for summary
        total_games = len(df)
        total_correct = df['ats_correct'].sum()
        overall_pct = (total_correct / total_games) * 100
        
        # First half vs second half
        midpoint = total_games // 2
        first_half_pct = (df['ats_correct'].iloc[:midpoint].sum() / midpoint) * 100
        second_half_pct = (df['ats_correct'].iloc[midpoint:].sum() / (total_games - midpoint)) * 100
        
        # Last 30 games
        last_30_pct = (df['ats_correct'].iloc[-30:].sum() / 30) * 100
        
        stats_summary.append({
            'Model': exp['label'],
            'Overall': f"{overall_pct:.1f}%",
            'First Half': f"{first_half_pct:.1f}%",
            'Second Half': f"{second_half_pct:.1f}%",
            'Last 30': f"{last_30_pct:.1f}%",
            'Change': f"{second_half_pct - first_half_pct:+.1f}%"
        })
        
        print(f"  Overall: {overall_pct:.1f}% ({total_correct}/{total_games})")
        print(f"  First Half: {first_half_pct:.1f}%")
        print(f"  Second Half: {second_half_pct:.1f}%")
        print(f"  Last 30 Games: {last_30_pct:.1f}%")
        
        # Plot cumulative ATS
        ax1.plot(df['cumulative_games'], df['cumulative_ats_pct'], 
                color=exp['color'], linewidth=2, label=exp['label'], alpha=0.9)
        
        # Plot rolling ATS
        ax2.plot(df['cumulative_games'], df['rolling_ats_pct'],
                color=exp['color'], linewidth=2, label=exp['label'], alpha=0.9)
    
    # Configure cumulative plot
    ax1.set_xlabel('Games Played', fontsize=11)
    ax1.set_ylabel('Cumulative ATS Accuracy (%)', fontsize=11)
    ax1.set_title('Cumulative ATS Accuracy Over Time', fontsize=13, fontweight='bold')
    ax1.axhline(y=52.4, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Break-even (~52.4%)')
    ax1.axhline(y=50, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_ylim([45, 65])
    
    # Configure rolling plot
    ax2.set_xlabel('Games Played', fontsize=11)
    ax2.set_ylabel('Rolling 30-Game ATS Accuracy (%)', fontsize=11)
    ax2.set_title('Rolling 30-Game ATS Accuracy (Recent Form)', fontsize=13, fontweight='bold')
    ax2.axhline(y=52.4, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Break-even (~52.4%)')
    ax2.axhline(y=50, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_ylim([40, 70])
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('predictions/experiments/ats_trends_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    df_summary = pd.DataFrame(stats_summary)
    print(df_summary.to_string(index=False))
    
    # Save summary to CSV
    df_summary.to_csv('predictions/experiments/ats_trends_summary.csv', index=False)
    print(f"\n✓ Summary saved to: predictions/experiments/ats_trends_summary.csv")
    
    print("\n" + "="*80)
    print("Key Observations:")
    print("="*80)
    print("1. Check if all models degrade similarly over time (parallel lines)")
    print("2. Look for models that maintain better performance in recent games")
    print("3. Rolling chart shows short-term trends (30-game window)")
    print("4. Cumulative chart shows overall trajectory from season start")
    print("="*80)

if __name__ == "__main__":
    main()

