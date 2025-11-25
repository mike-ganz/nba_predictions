"""
Analyze and plot feature distributions comparing historical seasons vs. current season.
Focuses on normalized features to show exactly what the model sees.
"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_features(filepath: str, label: str) -> pd.DataFrame:
    print(f"Loading {label} from {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            # Extract home and away team normalized features
            # We combine them to get the full distribution of "team performances"
            
            # Home
            row_h = {'season_type': label}
            for k, v in game['teams']['H'].items():
                if k.endswith('_norm') or k in ['pace', 'off_rating', 'def_rating']:
                    row_h[k] = v
            data.append(row_h)
            
            # Away
            row_a = {'season_type': label}
            for k, v in game['teams']['A'].items():
                if k.endswith('_norm') or k in ['pace', 'off_rating', 'def_rating']:
                    row_a[k] = v
            data.append(row_a)
            
    return pd.DataFrame(data)

def plot_distributions():
    # Load datasets
    df_hist = load_features("data/games_train_with_players_90_norm.jsonl", "Historical (2021-24)")
    df_curr = load_features("data/games_2025_2026_current_norm.jsonl", "Current (2025-26)")
    
    # Combine
    df_all = pd.concat([df_hist, df_curr], ignore_index=True)
    
    # Features to analyze (based on drift report)
    features_of_interest = [
        ('def_rating_norm', 'Defensive Rating (Norm)', 'Variance +66%'),
        ('off_reb_rate_norm', 'Off. Rebound Rate (Norm)', 'Variance +62%'),
        ('off_rating_norm', 'Offensive Rating (Norm)', 'Variance +40%'),
        ('turnover_rate_norm', 'Turnover Rate (Norm)', 'Variance +27%'),
        ('pace_norm', 'Pace (Norm)', 'Variance +15%'),
        ('assist_rate_norm', 'Assist Rate (Norm)', 'Variance +23%')
    ]
    
    # Setup plot grid
    n_cols = 2
    n_rows = (len(features_of_interest) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('Feature Distribution Drift: Historical vs. Current Season\n(Normalized Values seen by Model)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (col, title, note) in enumerate(features_of_interest):
        ax = axes[i]
        
        # Plot KDE (Kernel Density Estimate)
        sns.kdeplot(data=df_all, x=col, hue='season_type', fill=True, common_norm=False, ax=ax, linewidth=2)
        
        # Calculate stats for annotation
        std_hist = df_hist[col].std()
        std_curr = df_curr[col].std()
        range_hist = df_hist[col].max() - df_hist[col].min()
        range_curr = df_curr[col].max() - df_curr[col].min()
        
        # Add visual cues
        ax.set_title(f"{title}\n{note}", fontweight='bold')
        ax.set_xlabel(f"Normalized Value (Difference from League Avg)")
        ax.grid(True, alpha=0.3)
        
        # Add text stats
        stats_text = (
            f"Std Dev:\n"
            f"  Hist: {std_hist:.2f}\n"
            f"  Curr: {std_curr:.2f}\n\n"
            f"Range:\n"
            f"  Hist: {range_hist:.1f}\n"
            f"  Curr: {range_curr:.1f}"
        )
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = Path("analysis/feature_drift_distributions.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    plot_distributions()

