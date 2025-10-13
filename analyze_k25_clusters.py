"""
Generate detailed cluster analysis with player samples and aggregated stats.
"""
import pandas as pd

# Load data
assignments = pd.read_csv('clustering_assignments.csv')
profiles = pd.read_csv('clustering_profiles.csv')

print("=" * 100)
print("K=25 CLUSTER ANALYSIS - PLAYER SAMPLES & PERFORMANCE STATS")
print("=" * 100)

# Get unique clusters (excluding "Other")
clusters = sorted([c for c in assignments['cluster'].unique() if c != -1])

for cluster_id in clusters:
    cluster_data = assignments[assignments['cluster'] == cluster_id]
    cluster_name = cluster_data.iloc[0]['cluster_name']
    cluster_profile = profiles[profiles['cluster'] == cluster_id].iloc[0]
    
    print(f"\n{'=' * 100}")
    print(f"CLUSTER {cluster_id}: {cluster_name}")
    print(f"{'=' * 100}")
    print(f"Size: {len(cluster_data)} players")
    
    # Key performance stats
    print(f"\n📊 Statistical Profile:")
    print(f"   Shot Location:")
    print(f"      Rim Rate:           {cluster_profile['rim_attempt_rate']*100:5.1f}%")
    print(f"      Corner 3 Rate:      {cluster_profile['corner_3_rate']*100:5.1f}%")
    print(f"      Non-Corner 3 Rate:  {cluster_profile['non_corner_3_rate']*100:5.1f}%")
    print(f"      Mid-Range Rate:     {cluster_profile['mid_range_rate']*100:5.1f}%")
    
    print(f"\n   Shot Creation:")
    print(f"      Pull-up 3 Rate:     {cluster_profile['pullup_3_rate']*100:5.1f}%")
    print(f"      Catch & Shoot 3:    {cluster_profile['catch_shoot_3_rate']*100:5.1f}%")
    print(f"      Assisted 2PT:       {cluster_profile['assisted_2pt_rate']*100:5.1f}%")
    print(f"      Assisted 3PT:       {cluster_profile['assisted_3pt_rate']*100:5.1f}%")
    
    print(f"\n   Playmaking & Efficiency:")
    print(f"      Assists/100:        {cluster_profile['assists_per_100']:5.1f}")
    print(f"      Turnovers/100:      {cluster_profile['turnovers_per_100']:5.1f}")
    print(f"      AST/TO Ratio:       {cluster_profile['ast_to_ratio']:5.2f}")
    print(f"      Free Throw Rate:    {cluster_profile['ftr']:5.3f}")
    
    print(f"\n   Defense & Rebounding:")
    print(f"      Steals/100:         {cluster_profile['steals_per_100']:5.1f}")
    print(f"      Blocks/100:         {cluster_profile['blocks_per_100']:5.1f}")
    print(f"      Def Reb Share:      {cluster_profile['def_reb_share']*100:5.1f}%")
    print(f"      Fouls/100:          {cluster_profile['total_fouls_per_100']:5.1f}")
    
    # Representative players (sorted by games played)
    sample_players = cluster_data.nlargest(8, 'games_played')
    
    print(f"\n   🏀 Representative Players ({len(sample_players)} of {len(cluster_data)}):")
    for _, player in sample_players.iterrows():
        print(f"      • {player['player_name']:<30} ({player['games_played']} games)")

# Handle "Other" cluster separately
if -1 in assignments['cluster'].values:
    other_data = assignments[assignments['cluster'] == -1]
    print(f"\n{'=' * 100}")
    print(f"CLUSTER 'Other': Other / Unique Stars")
    print(f"{'=' * 100}")
    print(f"Size: {len(other_data)} players")
    print(f"\n   🏀 Players (combined from small clusters):")
    for _, player in other_data.iterrows():
        print(f"      • {player['player_name']:<30} ({player['games_played']} games)")

print(f"\n{'=' * 100}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 100}")

