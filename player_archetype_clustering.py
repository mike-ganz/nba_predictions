"""
Clustering Analysis - Detailed Player Archetypes
Configurable k value with automatic handling of small clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
K_CLUSTERS = 25  # Number of clusters to find
MIN_CLUSTER_SIZE = 5  # Minimum players per cluster (smaller go to "Other")

print("=" * 100)
print(f"NBA PLAYER CLUSTERING ANALYSIS - k={K_CLUSTERS} CLUSTERS")
print("=" * 100)

# Load player stats
print("\n📊 Loading player stats...")
df = pd.read_csv('all_player_pbp_stats.csv')
print(f"✅ Loaded {len(df)} players with 25+ games")

# Features (redundant ones already removed)
clustering_features = [
    'rim_attempt_rate',
    'corner_3_rate',
    'non_corner_3_rate',
    'long_mid_rate',
    'mid_range_rate',
    'pullup_3_rate',
    'catch_shoot_3_rate',
    'ftr',
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

# Prepare data
X = df[clustering_features].values
player_names = df['player_name'].values

# Standardize
print("\n🔄 Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-Means with configured k
print(f"\n🎯 Running K-Means clustering with k={K_CLUSTERS}...")
k = K_CLUSTERS
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add to dataframe
df['cluster'] = cluster_labels

# Identify small clusters and group them into "Other"
cluster_counts = df['cluster'].value_counts().sort_index()
small_clusters = [c for c in range(k) if cluster_counts.get(c, 0) < MIN_CLUSTER_SIZE]

if small_clusters:
    print(f"\n⚠️  Found {len(small_clusters)} clusters with < {MIN_CLUSTER_SIZE} players")
    print(f"   Grouping into 'Other' category: {small_clusters}")
    
    # Reassign small clusters to a new "Other" cluster (use -1 as marker)
    for small_cluster in small_clusters:
        df.loc[df['cluster'] == small_cluster, 'cluster'] = -1
    
    # Get updated counts
    cluster_counts = df['cluster'].value_counts().sort_index()

# Calculate silhouette score (before regrouping)
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"✅ Silhouette Score: {sil_score:.3f}")

# Cluster sizes
print(f"\n📊 Final cluster sizes:")
if -1 in cluster_counts.index:
    other_count = cluster_counts.get(-1, 0)
    print(f"   Cluster 'Other': {other_count:3d} players ({other_count/len(df)*100:5.1f}%) [Combined from small clusters]")

for cluster_id in range(k):
    if cluster_id in small_clusters:
        continue  # Skip small clusters (already in Other)
    count = cluster_counts.get(cluster_id, 0)
    if count > 0:
        pct = (count / len(df)) * 100
        print(f"   Cluster {cluster_id:2d}: {count:3d} players ({pct:5.1f}%)")

# Calculate cluster profiles
cluster_profiles = df.groupby('cluster')[clustering_features].mean()

# Function to name clusters based on stats
def name_cluster(cluster_id, profile, cluster_players):
    """Assign descriptive name to cluster based on stats and players - GRANULAR version"""
    
    # Extract all key stats
    rim_rate = profile['rim_attempt_rate']
    corner_3 = profile['corner_3_rate']
    nc_3 = profile['non_corner_3_rate']
    three_rate = corner_3 + nc_3
    mid_rate = profile['mid_range_rate']
    pullup_3 = profile['pullup_3_rate']
    catch_shoot = profile['catch_shoot_3_rate']
    assists = profile['assists_per_100']
    turnovers = profile['turnovers_per_100']
    ast_to = profile['ast_to_ratio']
    blocks = profile['blocks_per_100']
    steals = profile['steals_per_100']
    def_reb = profile['def_reb_share']
    assisted_2pt = profile['assisted_2pt_rate']
    assisted_3pt = profile['assisted_3pt_rate']
    ftr = profile['ftr']
    and1_rate = profile['and1_rate']
    
    # Build name components
    role = ""
    volume = ""
    style = ""
    
    # === PRIMARY ROLE (based on shot location & creation) ===
    
    # Elite ball-dominant creators
    if assists > 12 and pullup_3 > 0.20 and assisted_2pt < 0.40:
        if three_rate > 0.40:
            role = "Elite Perimeter Creator"
        else:
            role = "Elite Playmaking Guard"
    
    # Secondary creators/playmakers
    elif assists > 8 and pullup_3 > 0.15:
        if three_rate > 0.45:
            role = "Shot-Creating Guard"
        elif assists > 10:
            role = "Secondary Playmaker"
        else:
            role = "Scoring Guard"
    
    # Traditional Centers (rim + defense)
    elif rim_rate > 0.70 and blocks > 1.8:
        if assisted_2pt > 0.75:
            role = "Rim-Running Center"
        else:
            role = "Post-Up Center"
    
    # Modern Bigs (rim + shooting)
    elif rim_rate > 0.40 and three_rate > 0.25:
        if blocks > 1.5:
            role = "Stretch Big"
        elif assists > 5:
            role = "Playmaking Big"
        else:
            role = "Scoring Big"
    
    # Floor-spacing forwards/centers
    elif three_rate > 0.50:
        if corner_3 > 0.10:
            role = "Corner Specialist"
        elif catch_shoot > 0.45:
            role = "Spot-Up Shooter"
        elif pullup_3 > 0.15:
            role = "Pull-Up Shooter"
        else:
            role = "High-Volume Shooter"
    
    # 3&D archetypes
    elif three_rate > 0.40 and (steals > 1.5 or blocks > 0.8):
        if catch_shoot > 0.40:
            role = "3&D Wing"
        else:
            role = "Two-Way Wing"
    
    # Mid-range specialists
    elif mid_rate > 0.35:
        if assists > 6:
            role = "Mid-Range Playmaker"
        elif rim_rate > 0.30:
            role = "Mid-Range Scorer"
        else:
            role = "Mid-Range Specialist"
    
    # Slashing/cutting wings
    elif rim_rate > 0.35 and rim_rate < 0.65:
        if ftr > 0.30:
            role = "Slashing Wing"
        elif assisted_2pt > 0.65:
            role = "Cutting Wing"
        else:
            role = "Versatile Wing"
    
    # Paint-bound players (non-centers)
    elif rim_rate > 0.50 and blocks < 1.5:
        if ftr > 0.35:
            role = "Paint Finisher"
        else:
            role = "Interior Scorer"
    
    # Defensive specialists
    elif (steals > 2.0 or blocks > 1.5) and assists < 5:
        if rim_rate > 0.60:
            role = "Defensive Anchor"
        else:
            role = "Defensive Specialist"
    
    # === MODIFIERS (usage, efficiency, style) ===
    
    # Volume modifier
    if turnovers > 3.5:
        volume = "High-Usage "
    elif assists > 8 or (three_rate > 0.45 and catch_shoot < 0.30):
        volume = "Primary "
    elif assisted_2pt > 0.70 and assists < 4:
        volume = "Low-Usage "
    
    # Style modifier based on assists + creation
    if assists > 8 and ast_to > 2.5:
        style = " (Facilitator)"
    elif pullup_3 > 0.20 and assisted_3pt < 0.80:
        style = " (Self-Creator)"
    elif catch_shoot > 0.45:
        style = " (Off-Ball)"
    elif ftr > 0.35 and and1_rate > 0.03:
        style = " (Aggressive)"
    elif steals > 2.0 or blocks > 1.5:
        style = " (Defensive)"
    
    # Fallback for uncategorized
    if not role:
        if assists > 5:
            role = "Versatile Role Player"
        elif three_rate > 0.30:
            role = "Role Player"
        else:
            role = "Utility Player"
    
    return f"{volume}{role}{style}".strip()

# Analyze and name each cluster
print("\n" + "=" * 100)
print("DETAILED CLUSTER ANALYSIS")
print("=" * 100)

cluster_names = {}

# First, handle "Other" cluster if it exists
if -1 in df['cluster'].values:
    cluster_names[-1] = "Other / Unique Stars"
    cluster_players_other = df[df['cluster'] == -1].nlargest(20, 'games_played')
    
    print(f"\n{'=' * 100}")
    print(f"CLUSTER 'Other': Other / Unique Stars")
    print(f"{'=' * 100}")
    print(f"Size: {len(cluster_players_other)} players ({len(cluster_players_other)/len(df)*100:.1f}%)")
    print("\n   🏀 Players (combined from clusters with <5 players):")
    for idx, player in cluster_players_other.iterrows():
        print(f"      • {player['player_name']:<30} ({player['games_played']} games)")

for cluster_id in range(k):
    if cluster_id in small_clusters:
        continue  # Skip small clusters
    
    if cluster_counts.get(cluster_id, 0) == 0:
        continue  # Skip empty clusters
    
    profile = cluster_profiles.loc[cluster_id]
    cluster_players = df[df['cluster'] == cluster_id].nlargest(15, 'games_played')
    
    # Name the cluster
    cluster_name = name_cluster(cluster_id, profile, cluster_players)
    cluster_names[cluster_id] = cluster_name
    
    count = cluster_counts.get(cluster_id, 0)
    
    print(f"\n{'=' * 100}")
    print(f"CLUSTER {cluster_id}: {cluster_name}")
    print(f"{'=' * 100}")
    print(f"Size: {count} players ({count/len(df)*100:.1f}%)")
    
    # Key statistics
    print(f"\n📊 Statistical Profile:")
    print(f"   Shot Location:")
    print(f"      Rim Rate:          {profile['rim_attempt_rate']:6.1%}")
    print(f"      Corner 3 Rate:     {profile['corner_3_rate']:6.1%}")
    print(f"      Non-Corner 3 Rate: {profile['non_corner_3_rate']:6.1%}")
    print(f"      Mid-Range Rate:    {profile['mid_range_rate']:6.1%}")
    
    print(f"\n   Shot Type:")
    print(f"      Pull-up 3 Rate:    {profile['pullup_3_rate']:6.1%}")
    print(f"      Catch & Shoot 3:   {profile['catch_shoot_3_rate']:6.1%}")
    
    print(f"\n   Shot Creation:")
    print(f"      Assisted 2PT:      {profile['assisted_2pt_rate']:6.1%}")
    print(f"      Assisted 3PT:      {profile['assisted_3pt_rate']:6.1%}")
    print(f"      Free Throw Rate:   {profile['ftr']:6.3f}")
    
    print(f"\n   Playmaking:")
    print(f"      Assists/100:       {profile['assists_per_100']:6.1f}")
    print(f"      Turnovers/100:     {profile['turnovers_per_100']:6.1f}")
    print(f"      AST/TO Ratio:      {profile['ast_to_ratio']:6.2f}")
    
    print(f"\n   Defense:")
    print(f"      Steals/100:        {profile['steals_per_100']:6.1f}")
    print(f"      Blocks/100:        {profile['blocks_per_100']:6.1f}")
    print(f"      Def Reb Share:     {profile['def_reb_share']:6.1%}")
    print(f"      Fouls/100:         {profile['total_fouls_per_100']:6.1f}")
    
    # Representative players
    print(f"\n   🏀 Representative Players:")
    for idx, player in cluster_players.iterrows():
        print(f"      • {player['player_name']:<30} ({player['games_played']} games)")

# Save cluster assignments with names
df['cluster_name'] = df['cluster'].map(cluster_names)
df[['player_name', 'cluster', 'cluster_name', 'games_played']].to_csv(
    'clustering_assignments.csv', index=False
)
print("\n✅ Saved: clustering_assignments.csv")

# Save cluster profiles with names (only non-small clusters)
cluster_profiles_named = cluster_profiles.copy()
valid_clusters = [i for i in range(k) if i not in small_clusters and cluster_counts.get(i, 0) > 0]
if -1 in cluster_names:
    valid_clusters.append(-1)

cluster_profiles_subset = cluster_profiles.loc[valid_clusters] if valid_clusters else cluster_profiles
cluster_profiles_subset['cluster_name'] = [cluster_names.get(i, f"Cluster {i}") for i in cluster_profiles_subset.index]
cluster_profiles_subset['player_count'] = [cluster_counts.get(i, 0) for i in cluster_profiles_subset.index]
cluster_profiles_subset.to_csv('clustering_profiles.csv')
print("✅ Saved: clustering_profiles.csv")

# Create visualization
print("\n🎨 Creating visualizations...")

# PCA for visualization
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# 2D scatter plot with names
fig, ax = plt.subplots(figsize=(20, 14))

colors = plt.cm.tab20(np.linspace(0, 1, k))

# Plot valid clusters
for cluster_id in range(k):
    if cluster_id not in cluster_names:
        continue  # Skip small clusters that were moved to "Other"
    mask = cluster_labels == cluster_id
    if np.sum(mask) > 0:
        ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                   c=[colors[cluster_id]], 
                   label=f'{cluster_names[cluster_id]} (n={np.sum(mask)})',
                   s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

# Plot "Other" cluster if it exists
if -1 in cluster_names:
    mask = df['cluster'] == -1
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
               c='gray', 
               label=f'{cluster_names[-1]} (n={np.sum(mask)})',
               s=100, alpha=0.6, edgecolors='black', linewidth=0.5, marker='s')

# Cluster centers (only for valid clusters)
valid_center_ids = [i for i in range(k) if i in cluster_names]
if valid_center_ids:
    centers_pca = pca_2d.transform(kmeans.cluster_centers_[valid_center_ids])
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='red', marker='X', s=500, edgecolors='black', linewidth=2,
               label='Cluster Centers', zorder=10)

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
ax.set_title(f'NBA Player Archetypes (k={K_CLUSTERS} Clusters)', fontsize=18, fontweight='bold')
ax.legend(loc='best', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Saved: clustering_visualization.png")
plt.close()

# Summary by cluster name
print("\n" + "=" * 100)
print("CLUSTER SUMMARY BY ARCHETYPE")
print("=" * 100)

summary_df = df.groupby('cluster_name').agg({
    'player_name': 'count',
    'rim_attempt_rate': 'mean',
    'non_corner_3_rate': 'mean',
    'assists_per_100': 'mean',
    'blocks_per_100': 'mean'
}).rename(columns={'player_name': 'count'})

summary_df = summary_df.sort_values('count', ascending=False)

print(f"\n{'Archetype':<30} {'Count':>6} {'Rim%':>7} {'3PT%':>7} {'AST/100':>8} {'BLK/100':>8}")
print("-" * 100)
for archetype, row in summary_df.iterrows():
    print(f"{archetype:<30} {row['count']:>6.0f} {row['rim_attempt_rate']:>6.1%} "
          f"{row['non_corner_3_rate']:>6.1%} {row['assists_per_100']:>8.1f} {row['blocks_per_100']:>8.1f}")

# Create cluster distribution bar chart
fig, ax = plt.subplots(figsize=(14, 8))

archetype_counts = df['cluster_name'].value_counts().sort_values(ascending=True)
colors_bar = [colors[list(cluster_names.values()).index(name)] for name in archetype_counts.index]

archetype_counts.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='black')
ax.set_xlabel('Number of Players', fontsize=13, fontweight='bold')
ax.set_ylabel('Player Archetype', fontsize=13, fontweight='bold')
ax.set_title('Distribution of NBA Player Archetypes (k=12)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add count labels
for i, (archetype, count) in enumerate(archetype_counts.items()):
    ax.text(count + 1, i, f'{count}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('clustering_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: clustering_distribution.png")
plt.close()

# Create heatmap of cluster profiles
fig, ax = plt.subplots(figsize=(16, 12))

# Select key features for heatmap
key_features = [
    'rim_attempt_rate', 'non_corner_3_rate', 'mid_range_rate',
    'pullup_3_rate', 'catch_shoot_3_rate', 
    'assisted_2pt_rate', 'assists_per_100',
    'blocks_per_100', 'def_reb_share'
]

# Only use valid clusters (not small ones moved to "Other")
valid_cluster_ids = [i for i in range(k) if i in cluster_names]
heatmap_data = cluster_profiles.loc[valid_cluster_ids, key_features].T
heatmap_data.columns = [cluster_names[i] for i in valid_cluster_ids]

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={'label': 'Value (scaled)'}, ax=ax, linewidths=0.5)

ax.set_title('Cluster Profiles Heatmap', fontsize=16, fontweight='bold')
ax.set_xlabel('Player Archetype', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('clustering_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: clustering_heatmap.png")
plt.close()

print("\n" + "=" * 100)
print(f"✅ K={K_CLUSTERS} CLUSTERING ANALYSIS COMPLETE!")
print("=" * 100)

print("\n📁 Generated files:")
print("  • clustering_assignments.csv - Player assignments with archetype names")
print("  • clustering_profiles.csv - Statistical profiles of each archetype")
print("  • clustering_visualization.png - 2D PCA visualization")
print("  • clustering_distribution.png - Archetype distribution bar chart")
print("  • clustering_heatmap.png - Feature profiles heatmap")

print("\n🏀 Discovered Archetypes:")
for cluster_id in sorted(cluster_names.keys()):
    if cluster_id == -1:
        print(f"   {'Other':>2s}. {cluster_names[cluster_id]}")
    else:
        print(f"   {cluster_id:2d}. {cluster_names[cluster_id]}")

