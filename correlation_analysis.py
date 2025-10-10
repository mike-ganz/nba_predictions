"""
Correlation Analysis of PBP Stats
Identify redundant metrics across all players with 25+ games
"""

import pandas as pd
import numpy as np
from player_stats_from_pbp import calculate_player_pbp_stats, load_pbp_data
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("PBP STATS CORRELATION ANALYSIS")
print("=" * 100)

season = "2023-2024"
max_date = "2024-04-13"  # End of regular season
min_games = 25

# Load PBP data to get all unique players
print(f"\n📊 Loading play-by-play data for {season}...")
pbp_df = load_pbp_data(season)

# Filter to games before max_date
pbp_df_filtered = pbp_df[pbp_df['date'] < pd.to_datetime(max_date)]

# Get all unique players
print("🔍 Identifying all unique players...")
all_players = set()
for col in ['player', 'h1', 'h2', 'h3', 'h4', 'h5', 'a1', 'a2', 'a3', 'a4', 'a5']:
    all_players.update(pbp_df_filtered[col].dropna().unique())

all_players = sorted(list(all_players))
print(f"✅ Found {len(all_players)} unique players")

# Calculate stats for all players
print(f"\n🔄 Calculating stats for all players (filtering for {min_games}+ games)...")
player_stats_list = []

for i, player in enumerate(all_players):
    if (i + 1) % 50 == 0:
        print(f"   Progress: {i + 1}/{len(all_players)} players processed...")
    
    try:
        stats = calculate_player_pbp_stats(player, max_date, season)
        if stats and stats['games_played'] >= min_games:
            # Add player name to stats dict
            stats['player_name'] = player
            player_stats_list.append(stats)
    except Exception as e:
        # Silently skip players with errors
        pass

print(f"\n✅ Successfully calculated stats for {len(player_stats_list)} players with {min_games}+ games")

# Convert to DataFrame
df = pd.DataFrame(player_stats_list)

# Select only numeric columns for correlation (exclude context fields)
correlation_metrics = [
    'rim_attempt_rate',
    'corner_3_rate',
    'non_corner_3_rate',
    'short_mid_rate',
    'long_mid_rate',
    'mid_range_rate',
    'pullup_3_rate',
    'catch_shoot_3_rate',
    'ftr',
    'shooting_fouls_per_100',
    'and1_rate',
    'assisted_2pt_rate',
    'assisted_3pt_rate',
    'assists_per_100',
    'turnovers_per_100',
    'ast_to_ratio',
    'steals_per_100',
    'blocks_per_100',
    'def_reb_share',
    'total_fouls_per_100'
]

# Create correlation matrix
print(f"\n📊 Calculating correlation matrix for {len(correlation_metrics)} metrics...")
corr_df = df[correlation_metrics]
correlation_matrix = corr_df.corr()

# Save correlation matrix
correlation_matrix.to_csv('pbp_stats_correlation_matrix.csv')
print("✅ Saved correlation matrix to: pbp_stats_correlation_matrix.csv")

# Print full correlation matrix
print("\n" + "=" * 100)
print("FULL CORRELATION MATRIX")
print("=" * 100)
print(correlation_matrix.to_string())

# Find highly correlated pairs
print("\n" + "=" * 100)
print("HIGHLY CORRELATED METRIC PAIRS")
print("=" * 100)

# Extract upper triangle of correlation matrix
correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        metric1 = correlation_matrix.columns[i]
        metric2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        
        correlation_pairs.append({
            'metric1': metric1,
            'metric2': metric2,
            'correlation': corr_value,
            'abs_correlation': abs(corr_value)
        })

# Sort by absolute correlation
correlation_pairs_df = pd.DataFrame(correlation_pairs)
correlation_pairs_df = correlation_pairs_df.sort_values('abs_correlation', ascending=False)

# High correlation threshold
high_corr_threshold = 0.7
very_high_corr_threshold = 0.85

print(f"\n🔴 VERY HIGH CORRELATION (|r| > {very_high_corr_threshold}):")
print("-" * 100)
very_high = correlation_pairs_df[correlation_pairs_df['abs_correlation'] > very_high_corr_threshold]
if len(very_high) > 0:
    print(f"{'Metric 1':<30} {'Metric 2':<30} {'Correlation':>12}")
    print("-" * 100)
    for _, row in very_high.iterrows():
        print(f"{row['metric1']:<30} {row['metric2']:<30} {row['correlation']:>12.4f}")
else:
    print("None found")

print(f"\n🟡 HIGH CORRELATION ({high_corr_threshold} < |r| <= {very_high_corr_threshold}):")
print("-" * 100)
high = correlation_pairs_df[
    (correlation_pairs_df['abs_correlation'] > high_corr_threshold) & 
    (correlation_pairs_df['abs_correlation'] <= very_high_corr_threshold)
]
if len(high) > 0:
    print(f"{'Metric 1':<30} {'Metric 2':<30} {'Correlation':>12}")
    print("-" * 100)
    for _, row in high.iterrows():
        print(f"{row['metric1']:<30} {row['metric2']:<30} {row['correlation']:>12.4f}")
else:
    print("None found")

print(f"\n🟢 MODERATE CORRELATION (0.5 < |r| <= {high_corr_threshold}):")
print("-" * 100)
moderate = correlation_pairs_df[
    (correlation_pairs_df['abs_correlation'] > 0.5) & 
    (correlation_pairs_df['abs_correlation'] <= high_corr_threshold)
]
if len(moderate) > 0:
    print(f"{'Metric 1':<30} {'Metric 2':<30} {'Correlation':>12}")
    print("-" * 100)
    for _, row in moderate.head(20).iterrows():  # Top 20
        print(f"{row['metric1']:<30} {row['metric2']:<30} {row['correlation']:>12.4f}")
    if len(moderate) > 20:
        print(f"... and {len(moderate) - 20} more")
else:
    print("None found")

# Analysis of specific metric groups
print("\n" + "=" * 100)
print("CORRELATION ANALYSIS BY METRIC GROUP")
print("=" * 100)

# Shot Location Metrics
print("\n📍 SHOT LOCATION METRICS:")
shot_location_metrics = ['rim_attempt_rate', 'corner_3_rate', 'non_corner_3_rate', 
                         'short_mid_rate', 'long_mid_rate', 'mid_range_rate']
shot_corr = correlation_matrix.loc[shot_location_metrics, shot_location_metrics]
print(shot_corr.to_string())

# Shot Type Metrics (3PT breakdown)
print("\n🏀 3-POINT SHOT TYPE METRICS:")
three_pt_metrics = ['corner_3_rate', 'non_corner_3_rate', 'pullup_3_rate', 'catch_shoot_3_rate']
three_corr = correlation_matrix.loc[three_pt_metrics, three_pt_metrics]
print(three_corr.to_string())

# Shot Quality Metrics
print("\n⭐ SHOT QUALITY METRICS:")
quality_metrics = ['ftr', 'shooting_fouls_per_100', 'and1_rate', 
                  'assisted_2pt_rate', 'assisted_3pt_rate']
quality_corr = correlation_matrix.loc[quality_metrics, quality_metrics]
print(quality_corr.to_string())

# Creation Metrics
print("\n🎨 CREATION METRICS:")
creation_metrics = ['assists_per_100', 'turnovers_per_100', 'ast_to_ratio']
creation_corr = correlation_matrix.loc[creation_metrics, creation_metrics]
print(creation_corr.to_string())

# Defensive Metrics
print("\n🛡️  DEFENSIVE METRICS:")
defensive_metrics = ['steals_per_100', 'blocks_per_100', 'def_reb_share', 'total_fouls_per_100']
defensive_corr = correlation_matrix.loc[defensive_metrics, defensive_metrics]
print(defensive_corr.to_string())

# Key relationships to investigate
print("\n" + "=" * 100)
print("KEY CROSS-METRIC RELATIONSHIPS")
print("=" * 100)

print("\n🔍 Shot Location vs Assisted Rates:")
print(f"{'Metric Pair':<60} {'Correlation':>12}")
print("-" * 100)
for shot_metric in ['rim_attempt_rate', 'mid_range_rate', 'non_corner_3_rate']:
    for assist_metric in ['assisted_2pt_rate', 'assisted_3pt_rate']:
        corr = correlation_matrix.loc[shot_metric, assist_metric]
        print(f"{shot_metric:<30} vs {assist_metric:<28} {corr:>12.4f}")

print("\n🔍 Playmaking vs Shot Creation:")
print(f"{'Metric Pair':<60} {'Correlation':>12}")
print("-" * 100)
for create_metric in ['assists_per_100', 'turnovers_per_100']:
    for shot_metric in ['assisted_2pt_rate', 'pullup_3_rate']:
        corr = correlation_matrix.loc[create_metric, shot_metric]
        print(f"{create_metric:<30} vs {shot_metric:<28} {corr:>12.4f}")

# Summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print(f"\nTotal metrics analyzed: {len(correlation_metrics)}")
print(f"Total unique metric pairs: {len(correlation_pairs_df)}")
print(f"Very high correlation pairs (|r| > {very_high_corr_threshold}): {len(very_high)}")
print(f"High correlation pairs (|r| > {high_corr_threshold}): {len(high) + len(very_high)}")
print(f"Moderate correlation pairs (|r| > 0.5): {len(moderate) + len(high) + len(very_high)}")

# Descriptive stats for each metric
print("\n" + "=" * 100)
print("DESCRIPTIVE STATISTICS FOR EACH METRIC")
print("=" * 100)
print(f"\nBased on {len(df)} players with {min_games}+ games\n")

desc_stats = corr_df.describe().T
desc_stats['range'] = desc_stats['max'] - desc_stats['min']
print(desc_stats.to_string())

# Create visualization (if matplotlib available)
try:
    print("\n📊 Creating correlation heatmap...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 18))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBU_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                vmin=-1,
                vmax=1,
                ax=ax)
    
    plt.title(f'PBP Stats Correlation Matrix\n({len(df)} players with {min_games}+ games)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('pbp_stats_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved correlation heatmap to: pbp_stats_correlation_heatmap.png")
    
    # Create a focused heatmap for highly correlated metrics
    if len(very_high) > 0 or len(high) > 0:
        high_corr_metrics = set()
        for _, row in pd.concat([very_high, high]).iterrows():
            high_corr_metrics.add(row['metric1'])
            high_corr_metrics.add(row['metric2'])
        
        if len(high_corr_metrics) > 1:
            high_corr_metrics = sorted(list(high_corr_metrics))
            focused_corr = correlation_matrix.loc[high_corr_metrics, high_corr_metrics]
            
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            sns.heatmap(focused_corr,
                       annot=True,
                       fmt='.3f',
                       cmap='RdBU_r',
                       center=0,
                       square=True,
                       linewidths=1,
                       cbar_kws={"shrink": 0.8},
                       vmin=-1,
                       vmax=1,
                       ax=ax2)
            
            plt.title('Highly Correlated Metrics (|r| > 0.7)', 
                     fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig('pbp_stats_high_correlation_focus.png', dpi=300, bbox_inches='tight')
            print("✅ Saved focused heatmap to: pbp_stats_high_correlation_focus.png")
    
except Exception as e:
    print(f"⚠️ Could not create visualization: {e}")

# Recommendations
print("\n" + "=" * 100)
print("RECOMMENDATIONS FOR METRIC SELECTION")
print("=" * 100)

print("\n💡 Based on the correlation analysis:\n")

recommendations = []

# Check for redundant shot location metrics
if correlation_matrix.loc['short_mid_rate', 'mid_range_rate'] > 0.9:
    recommendations.append(
        "• SHORT_MID_RATE and MID_RANGE_RATE: Very high correlation.\n"
        "  → Consider keeping only MID_RANGE_RATE (total) and optionally LONG_MID_RATE"
    )

# Check for redundant 3PT metrics
total_3pt_corr_with_pullup = abs(correlation_matrix.loc['pullup_3_rate', 'non_corner_3_rate'])
total_3pt_corr_with_catch = abs(correlation_matrix.loc['catch_shoot_3_rate', 'non_corner_3_rate'])

if total_3pt_corr_with_pullup > 0.7 or total_3pt_corr_with_catch > 0.7:
    recommendations.append(
        "• 3PT SHOT TYPES: Pullup and catch-shoot rates may be correlated with total 3PT rate.\n"
        "  → Keep all three for shot creation insights (they tell different stories)"
    )

# Check assisted rates
assisted_corr = correlation_matrix.loc['assisted_2pt_rate', 'assisted_3pt_rate']
if abs(assisted_corr) > 0.7:
    recommendations.append(
        "• ASSISTED RATES: 2PT and 3PT assisted rates are highly correlated.\n"
        "  → Keep both as they provide different context (inside vs perimeter)"
    )

# Check foul metrics
foul_corr = correlation_matrix.loc['shooting_fouls_per_100', 'total_fouls_per_100']
if foul_corr > 0.7:
    recommendations.append(
        f"• FOUL METRICS: Shooting fouls and total fouls are correlated (r={foul_corr:.3f}).\n"
        "  → Consider keeping only TOTAL_FOULS_PER_100 or separate into offensive/defensive"
    )

# Check creation metrics
if abs(correlation_matrix.loc['assists_per_100', 'ast_to_ratio']) > 0.7:
    recommendations.append(
        "• CREATION METRICS: AST/100 and AST/TO ratio may be correlated.\n"
        "  → Keep both - AST/100 = volume, AST/TO = efficiency"
    )

if recommendations:
    for rec in recommendations:
        print(rec)
else:
    print("✅ No highly redundant metrics found - all metrics provide unique information!")

print("\n🎯 GENERAL GUIDELINES:")
print("  • Metrics with |r| > 0.85 are likely redundant - choose one")
print("  • Metrics with 0.70 < |r| < 0.85 may be complementary - evaluate based on use case")
print("  • Metrics with |r| < 0.70 provide distinct information - keep all")

print("\n" + "=" * 100)
print("CORRELATION ANALYSIS COMPLETE")
print("=" * 100)

# Export player stats to CSV for further analysis
df.to_csv('all_player_pbp_stats.csv', index=False)
print(f"\n✅ Exported {len(df)} player stat profiles to: all_player_pbp_stats.csv")

