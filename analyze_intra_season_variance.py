"""
Analyze intra-season distribution shifts (Variance/Volatility) for key NBA metrics.

Part 1: Compare Early 2025 (Pre-Nov 15) vs Recent 2025 (Post-Nov 15).
Part 2: Analyze historical volatility trends (Standard Deviation over time) for 2021-2025.
"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime

def load_data_with_date(filepath: str, label: str) -> pd.DataFrame:
    print(f"Loading {label} from {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            date_str = game['date']
            season = game['season']
            
            # Flatten features
            row = {
                'date': pd.to_datetime(date_str),
                'season': season,
                'game_id': game['game_id']
            }
            
            # Extract normalized metrics (combining Home and Away to get league distribution)
            # We treat H and A as independent observations of "Team Performance"
            for side in ['H', 'A']:
                team_data = game['teams'][side]
                for k, v in team_data.items():
                    if k.endswith('_norm'):
                        # Add side prefix to avoid overwriting? No, we want distribution of ALL teams.
                        # We'll make separate rows for H and A
                        pass
            
            # Actually, better to create two rows per game (one per team) 
            # to correctly calculate distribution across the league
            
            # Home Team Row
            row_h = row.copy()
            row_h['side'] = 'Home'
            for k, v in game['teams']['H'].items():
                if k.endswith('_norm'):
                    row_h[k] = v
            data.append(row_h)
            
            # Away Team Row
            row_a = row.copy()
            row_a['side'] = 'Away'
            for k, v in game['teams']['A'].items():
                if k.endswith('_norm'):
                    row_a[k] = v
            data.append(row_a)
            
    return pd.DataFrame(data)

def analyze_2025_split(df_2025):
    print("\n" + "="*60)
    print("PART 1: 2025 Season Split (Pre vs Post Nov 15)")
    print("="*60)
    
    split_date = pd.Timestamp("2025-11-15")
    
    df_early = df_2025[df_2025['date'] < split_date]
    df_recent = df_2025[df_2025['date'] >= split_date]
    
    print(f"Early Season Games (Pre-11/15): {len(df_early) // 2} games")
    print(f"Recent Season Games (Post-11/15): {len(df_recent) // 2} games")
    
    metrics = [
        'def_rating_norm', 'off_reb_rate_norm', 'turnover_rate_norm', 
        'pace_norm', 'three_pt_rate_norm'
    ]
    
    stats = []
    
    # Plotting
    fig, axes = plt.subplots(len(metrics), 2, figsize=(16, 4 * len(metrics)))
    fig.suptitle('2025 Intra-Season Drift: Early (Pre-11/15) vs Recent (Post-11/15)', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        # 1. Calculate Variance
        std_early = df_early[metric].std()
        std_recent = df_recent[metric].std()
        pct_change = ((std_recent - std_early) / std_early) * 100
        
        stats.append({
            'Metric': metric,
            'Early Std': std_early,
            'Recent Std': std_recent,
            'Change': f"{pct_change:+.1f}%"
        })
        
        # 2. Plot KDE
        ax_kde = axes[i, 0]
        sns.kdeplot(data=df_early, x=metric, label='Early (Pre-11/15)', fill=True, ax=ax_kde, color='blue', alpha=0.3)
        sns.kdeplot(data=df_recent, x=metric, label='Recent (Post-11/15)', fill=True, ax=ax_kde, color='orange', alpha=0.3)
        ax_kde.set_title(f"{metric}: Distribution Shape")
        ax_kde.legend()
        
        # 3. Plot Boxplot (to show outliers/spread clearer)
        ax_box = axes[i, 1]
        # Combine for boxplot
        df_plot = pd.concat([
            df_early[[metric]].assign(Period='Early'),
            df_recent[[metric]].assign(Period='Recent')
        ])
        sns.boxplot(data=df_plot, x=metric, y='Period', ax=ax_box, palette={'Early': 'lightblue', 'Recent': 'orange'})
        ax_box.set_title(f"{metric}: Variance Spread ({pct_change:+.1f}%)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('analysis/intra_season_2025_drift.png')
    print("Saved 2025 analysis to analysis/intra_season_2025_drift.png")
    
    print("\nVariance Change Summary:")
    print(pd.DataFrame(stats).to_string(index=False))

def analyze_historical_seasonality(df_hist):
    print("\n" + "="*60)
    print("PART 2: Historical Seasonality (Does Variance Normally Change?)")
    print("="*60)
    
    # Add "Month of Season" (Oct=1, Nov=2, ... Apr=7)
    # Standard NBA season starts in Oct.
    def get_season_month(date):
        m = date.month
        if m >= 10: return m - 9 # Oct=1, Nov=2, Dec=3
        return m + 3 # Jan=4, Feb=5, ...
    
    df_hist['season_month'] = df_hist['date'].apply(get_season_month)
    
    metrics = ['def_rating_norm', 'off_reb_rate_norm', 'turnover_rate_norm']
    
    # Calculate Std Dev by Month
    monthly_volatility = df_hist.groupby('season_month')[metrics].std()
    
    print("Monthly Volatility (Std Dev) - Historical Avg:")
    print(monthly_volatility)
    
    # Plot Volatility Trends
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        # Normalize to start at 1.0 for comparison
        start_val = monthly_volatility.loc[1, metric] if 1 in monthly_volatility.index else monthly_volatility.iloc[0][metric]
        plt.plot(monthly_volatility.index, monthly_volatility[metric] / start_val, marker='o', linewidth=2, label=metric)
    
    plt.title("Historical Seasonal Volatility Trend (2021-2024)\n(1.0 = October Baseline Variance)", fontsize=14)
    plt.xlabel("Month of Season (1=Oct, 4=Jan, 7=Apr)")
    plt.ylabel("Relative Variance (Std Dev)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(1.0, color='black', linestyle='--')
    
    plt.savefig('analysis/historical_seasonality_variance.png')
    print("Saved historical analysis to analysis/historical_seasonality_variance.png")

def main():
    # Load Data
    df_2025 = load_data_with_date("data/games_2025_2026_current_norm.jsonl", "2025 Current")
    df_hist = load_data_with_date("data/games_train_with_players_90_norm.jsonl", "Historical")
    
    # Run Analyses
    analyze_2025_split(df_2025)
    analyze_historical_seasonality(df_hist)

if __name__ == "__main__":
    main()

