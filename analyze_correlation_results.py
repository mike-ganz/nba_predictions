"""
Analyze the correlation results from the full season analysis
"""

import pandas as pd
import numpy as np

# Load correlation matrix
corr_matrix = pd.read_csv('pbp_stats_correlation_matrix.csv', index_col=0)

# Load player stats
player_stats = pd.read_csv('all_player_pbp_stats.csv')

print("=" * 100)
print("CORRELATION ANALYSIS RESULTS SUMMARY")
print("=" * 100)

print(f"\nDataset: {len(player_stats)} players with 25+ games")
print(f"Games played range: {player_stats['games_played'].min()} - {player_stats['games_played'].max()}")
print(f"Season: {player_stats['season'].iloc[0]}")
print(f"Date range: Through {player_stats['max_date'].iloc[0]}")

# Extract correlation pairs
correlation_pairs = []
metrics = corr_matrix.columns.tolist()

for i in range(len(metrics)):
    for j in range(i + 1, len(metrics)):
        metric1 = metrics[i]
        metric2 = metrics[j]
        corr_value = corr_matrix.iloc[i, j]
        
        correlation_pairs.append({
            'metric1': metric1,
            'metric2': metric2,
            'correlation': corr_value,
            'abs_correlation': abs(corr_value)
        })

corr_df = pd.DataFrame(correlation_pairs).sort_values('abs_correlation', ascending=False)

# Thresholds
very_high = 0.85
high = 0.70
moderate = 0.50

print("\n" + "=" * 100)
print("HIGH CORRELATION PAIRS (|r| > 0.70)")
print("=" * 100)

high_corr = corr_df[corr_df['abs_correlation'] > high]

if len(high_corr) > 0:
    print(f"\nFound {len(high_corr)} pairs with |r| > {high}\n")
    print(f"{'Metric 1':<30} {'Metric 2':<30} {'Correlation':>12} {'|r|':>8}")
    print("-" * 100)
    
    for _, row in high_corr.iterrows():
        marker = "🔴" if row['abs_correlation'] > very_high else "🟡"
        print(f"{marker} {row['metric1']:<28} {row['metric2']:<28} {row['correlation']:>12.4f} {row['abs_correlation']:>8.4f}")
else:
    print("\n✅ No pairs with correlation > 0.70 found!")

print("\n" + "=" * 100)
print("MODERATE CORRELATION PAIRS (0.50 < |r| <= 0.70)")
print("=" * 100)

moderate_corr = corr_df[(corr_df['abs_correlation'] > moderate) & (corr_df['abs_correlation'] <= high)]

if len(moderate_corr) > 0:
    print(f"\nFound {len(moderate_corr)} pairs\n")
    print(f"{'Metric 1':<30} {'Metric 2':<30} {'Correlation':>12} {'|r|':>8}")
    print("-" * 100)
    
    for _, row in moderate_corr.head(15).iterrows():  # Top 15
        print(f"  {row['metric1']:<28} {row['metric2']:<28} {row['correlation']:>12.4f} {row['abs_correlation']:>8.4f}")
    
    if len(moderate_corr) > 15:
        print(f"\n  ... and {len(moderate_corr) - 15} more")

print("\n" + "=" * 100)
print("KEY FINDINGS")
print("=" * 100)

# Analyze specific relationships
print("\n📍 SHOT LOCATION METRICS:")
shot_pairs = [
    ('rim_attempt_rate', 'non_corner_3_rate'),
    ('short_mid_rate', 'mid_range_rate'),
    ('corner_3_rate', 'non_corner_3_rate'),
]

for m1, m2 in shot_pairs:
    corr = corr_matrix.loc[m1, m2]
    print(f"  {m1:<28} vs {m2:<28} r = {corr:>7.4f}")

print("\n🏀 3-POINT BREAKDOWN:")
three_pairs = [
    ('pullup_3_rate', 'catch_shoot_3_rate'),
    ('pullup_3_rate', 'non_corner_3_rate'),
    ('catch_shoot_3_rate', 'non_corner_3_rate'),
    ('catch_shoot_3_rate', 'corner_3_rate'),
]

for m1, m2 in three_pairs:
    corr = corr_matrix.loc[m1, m2]
    print(f"  {m1:<28} vs {m2:<28} r = {corr:>7.4f}")

print("\n⭐ ASSISTED RATES:")
assist_pairs = [
    ('assisted_2pt_rate', 'assisted_3pt_rate'),
    ('assisted_2pt_rate', 'assists_per_100'),
    ('assisted_3pt_rate', 'catch_shoot_3_rate'),
]

for m1, m2 in assist_pairs:
    corr = corr_matrix.loc[m1, m2]
    print(f"  {m1:<28} vs {m2:<28} r = {corr:>7.4f}")

print("\n🎨 CREATION METRICS:")
create_pairs = [
    ('assists_per_100', 'turnovers_per_100'),
    ('assists_per_100', 'ast_to_ratio'),
    ('turnovers_per_100', 'ast_to_ratio'),
    ('assists_per_100', 'assisted_2pt_rate'),
]

for m1, m2 in create_pairs:
    corr = corr_matrix.loc[m1, m2]
    print(f"  {m1:<28} vs {m2:<28} r = {corr:>7.4f}")

print("\n🛡️  DEFENSIVE METRICS:")
def_pairs = [
    ('steals_per_100', 'blocks_per_100'),
    ('blocks_per_100', 'def_reb_share'),
    ('shooting_fouls_per_100', 'total_fouls_per_100'),
    ('blocks_per_100', 'rim_attempt_rate'),
]

for m1, m2 in def_pairs:
    corr = corr_matrix.loc[m1, m2]
    print(f"  {m1:<28} vs {m2:<28} r = {corr:>7.4f}")

print("\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)

recommendations = []

# Check for very high correlations
if len(high_corr[high_corr['abs_correlation'] > very_high]) > 0:
    print("\n🔴 VERY HIGH CORRELATIONS (|r| > 0.85) - Consider keeping only one:")
    for _, row in high_corr[high_corr['abs_correlation'] > very_high].iterrows():
        print(f"   • {row['metric1']} and {row['metric2']} (r = {row['correlation']:.4f})")
        
        # Specific recommendation
        if 'shooting_fouls_per_100' in [row['metric1'], row['metric2']] and 'total_fouls_per_100' in [row['metric1'], row['metric2']]:
            print(f"     → RECOMMEND: Keep total_fouls_per_100 (more comprehensive)")
        elif 'short_mid_rate' in [row['metric1'], row['metric2']] and 'mid_range_rate' in [row['metric1'], row['metric2']]:
            print(f"     → RECOMMEND: Keep mid_range_rate (total) + optionally long_mid_rate")

# Check for high correlations that might be complementary
if len(high_corr[(high_corr['abs_correlation'] > high) & (high_corr['abs_correlation'] <= very_high)]) > 0:
    print("\n🟡 HIGH CORRELATIONS (0.70 < |r| <= 0.85) - Evaluate based on use case:")
    for _, row in high_corr[(high_corr['abs_correlation'] > high) & (high_corr['abs_correlation'] <= very_high)].iterrows():
        print(f"   • {row['metric1']} and {row['metric2']} (r = {row['correlation']:.4f})")
        
        # Context-specific recommendations
        if 'catch_shoot_3_rate' in [row['metric1'], row['metric2']] and 'non_corner_3_rate' in [row['metric1'], row['metric2']]:
            print(f"     → These tell different stories - catch-shoot is HOW, non-corner is WHERE")
        elif 'pullup_3_rate' in [row['metric1'], row['metric2']] and 'non_corner_3_rate' in [row['metric1'], row['metric2']]:
            print(f"     → Keep both - pullup shows shot creation ability")
        elif 'assists_per_100' in [row['metric1'], row['metric2']] and 'ast_to_ratio' in [row['metric1'], row['metric2']]:
            print(f"     → Keep both - assists/100 = volume, AST/TO = efficiency")

print("\n✅ METRICS WITH LOW CORRELATION:")
print("   These provide unique information and should be kept:")

# Find metrics that don't have high correlations with others
low_corr_metrics = set()
for metric in metrics:
    max_corr = corr_matrix[metric].drop(metric).abs().max()
    if max_corr < moderate:
        low_corr_metrics.add(metric)

if low_corr_metrics:
    for metric in sorted(low_corr_metrics):
        max_corr = corr_matrix[metric].drop(metric).abs().max()
        print(f"   • {metric:<30} (max |r| = {max_corr:.4f})")
else:
    print("   All metrics have at least moderate correlation with another metric")

print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print(f"\nTotal metrics: {len(metrics)}")
print(f"Total unique pairs: {len(corr_df)}")
print(f"Very high correlation (|r| > {very_high}): {len(high_corr[high_corr['abs_correlation'] > very_high])}")
print(f"High correlation (|r| > {high}): {len(high_corr)}")
print(f"Moderate correlation (|r| > {moderate}): {len(moderate_corr) + len(high_corr)}")
print(f"Low correlation (|r| <= {moderate}): {len(corr_df) - len(moderate_corr) - len(high_corr)}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

