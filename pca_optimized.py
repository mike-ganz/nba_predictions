import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import json
import hashlib
import time
from functools import lru_cache
import transform_player_stats_optimized as optimized_stats

# Configuration
CACHE_DIR = 'data/cache/pca'

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def generate_cache_filename(player_name, max_date, current_season):
    unique_string = f"{player_name}_{max_date}_{current_season}"
    hashed = hashlib.md5(unique_string.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.json")

def save_to_cache(player_name, max_date, current_season, data):
    ensure_cache_dir()
    filename = generate_cache_filename(player_name, max_date, current_season)
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_from_cache(player_name, max_date, current_season):
    filename = generate_cache_filename(player_name, max_date, current_season)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def get_pca_metric_fields(metric_type):
    """Get the fields used for each PCA metric type."""
    metric_fields = {
        "offense": ['PPG', 'APG', 'TS%', 'TO'],
        "defense": ['BPG', 'SPG', 'FPG'], 
        "shot_selection": ['3PR', 'FTR'],
        "efficiency": ['TS%', 'eFG%', 'TO']
    }
    
    if metric_type not in metric_fields:
        raise ValueError(f"Invalid metric type: {metric_type}. Available: {list(metric_fields.keys())}")
    
    return metric_fields[metric_type]

def load_all_player_stats_for_date(max_date, current_season="2023-2024", min_games=5):
    """
    🚀 OPTIMIZED: Load stats for ALL players for a specific date efficiently.
    
    Uses the optimized player stats cache system instead of calculating individually.
    """
    print(f"📊 Loading ALL player stats for {max_date}...")
    start_time = time.time()
    
    # Get all distinct players from the optimized system
    all_players = optimized_stats.get_distinct_players()
    
    all_stats = []
    cache_hits = 0
    cache_misses = 0
    
    for player in all_players:
        # Try to get from optimized cache first
        stats = optimized_stats.calculate_player_stats(player, max_date, current_season)
        
        if stats:
            if stats.get('GP', 0) >= min_games:
                all_stats.append(stats)
                cache_hits += 1
            else:
                # Fall back to previous season if not enough games
                prev_season = str(int(current_season.split('-')[0]) - 1) + '-' + str(int(current_season.split('-')[1]) - 1)
                fallback_stats = optimized_stats.calculate_player_stats(player, None, prev_season)
                if fallback_stats:
                    all_stats.append(fallback_stats)
                    cache_misses += 1
    
    elapsed = time.time() - start_time
    print(f"✅ Loaded {len(all_stats)} player stats in {elapsed:.2f}s")
    print(f"📊 Cache performance: {cache_hits} hits, {cache_misses} misses")
    
    return all_stats

@lru_cache(maxsize=128)
def calculate_all_pca_scores_for_date(max_date, current_season="2023-2024"):
    """
    🚀 OPTIMIZED: Calculate PCA scores for ALL players for ALL metrics at once.
    
    This replaces the old system that calculated PCA for each player individually.
    """
    print(f"🚀 Calculating ALL PCA scores for {max_date}...")
    start_time = time.time()
    
    # Load ALL player stats once
    all_stats = load_all_player_stats_for_date(max_date, current_season)
    
    if len(all_stats) < 10:  # Need minimum players for meaningful PCA
        print(f"⚠️ Only {len(all_stats)} players found - insufficient for PCA")
        return {}
    
    # Convert to DataFrame once
    df = pd.DataFrame(all_stats)
    
    # Calculate PCA for all metric types
    all_pca_results = {}
    metric_types = ["offense", "defense", "shot_selection", "efficiency"]
    
    for metric_type in metric_types:
        print(f"  📈 Calculating {metric_type} PCA...")
        
        # Get fields for this metric
        selected_fields = get_pca_metric_fields(metric_type)
        
        # Extract numeric data for PCA
        numeric_df = df[selected_fields]
        
        # Remove rows with NaN values and track valid indices
        valid_indices = numeric_df.dropna().index
        valid_df = numeric_df.loc[valid_indices]
        valid_stats = [all_stats[i] for i in valid_indices]
        
        if len(valid_df) < 5:
            print(f"    ⚠️ Only {len(valid_df)} valid players for {metric_type}")
            continue
        
        # Convert to numpy array
        X = valid_df.values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run PCA
        pca = PCA(n_components=1)
        pca_scores = pca.fit_transform(X_scaled)
        
        # Create player mapping
        for i, stats in enumerate(valid_stats):
            player_name = stats['PLAYER_NAME']
            if player_name not in all_pca_results:
                all_pca_results[player_name] = {}
            all_pca_results[player_name][metric_type] = float(pca_scores[i][0])
    
    elapsed = time.time() - start_time
    print(f"✅ Calculated PCA for {len(all_pca_results)} players in {elapsed:.2f}s")
    print(f"⚡ Speed: {len(all_pca_results) * 4 / elapsed:.0f} PCA calculations/second")
    
    return all_pca_results

def get_player_pca_score(player_name, max_date, current_season="2023-2024", log_queue=None):
    """
    🚀 OPTIMIZED: Get PCA score for a single player using batch-calculated results.
    
    This now uses the batch calculation system instead of individual calculations.
    """
    # Try cache first
    cached_data = load_from_cache(player_name, max_date, current_season)
    if cached_data:
        return (
            cached_data.get('offense', 0.0),
            cached_data.get('defense', 0.0),
            cached_data.get('shot_selection', 0.0),
            cached_data.get('efficiency', 0.0)
        )
    
    # Get all PCA scores for this date (batch calculation)
    all_pca_results = calculate_all_pca_scores_for_date(max_date, current_season)
    
    # Extract scores for this specific player
    player_scores = all_pca_results.get(player_name, {})
    
    if not player_scores:
        print(f"⚠️ No PCA scores found for {player_name} on {max_date}")
        return 0.0, 0.0, 0.0, 0.0
    
    # Prepare cache data
    cache_data = {
        'offense': player_scores.get('offense', 0.0),
        'defense': player_scores.get('defense', 0.0),
        'shot_selection': player_scores.get('shot_selection', 0.0),
        'efficiency': player_scores.get('efficiency', 0.0)
    }
    
    # Save to cache
    save_to_cache(player_name, max_date, current_season, cache_data)
    
    return (
        cache_data['offense'],
        cache_data['defense'], 
        cache_data['shot_selection'],
        cache_data['efficiency']
    )

def batch_cache_pca_scores(max_date, current_season="2023-2024"):
    """
    🚀 NEW: Build PCA cache for ALL players for a specific date at once.
    
    This is much more efficient than calling get_player_pca_score individually.
    """
    print(f"🚀 Building PCA cache for ALL players on {max_date}...")
    start_time = time.time()
    
    # Calculate all PCA scores at once
    all_pca_results = calculate_all_pca_scores_for_date(max_date, current_season)
    
    # Save to cache for each player
    cached_count = 0
    for player_name, scores in all_pca_results.items():
        cache_data = {
            'offense': scores.get('offense', 0.0),
            'defense': scores.get('defense', 0.0),
            'shot_selection': scores.get('shot_selection', 0.0),
            'efficiency': scores.get('efficiency', 0.0)
        }
        
        save_to_cache(player_name, max_date, current_season, cache_data)
        cached_count += 1
    
    elapsed = time.time() - start_time
    print(f"✅ Cached PCA scores for {cached_count} players in {elapsed:.2f}s")
    print(f"⚡ Speed: {cached_count / elapsed:.0f} players/second")
    
    return cached_count

def batch_cache_pca_for_date_range(start_date, end_date, date_interval_days=7, current_season="2023-2024"):
    """
    🚀 NEW: Build PCA cache for multiple dates efficiently.
    """
    print(f"🚀 Building PCA cache from {start_date} to {end_date} (every {date_interval_days} days)")
    
    # Generate target dates
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{date_interval_days}D')
    target_dates = [date.strftime('%Y-%m-%d') for date in date_range]
    
    print(f"📅 Target dates ({len(target_dates)}): {target_dates[:3]} ... {target_dates[-3:]}")
    
    total_cached = 0
    for i, date in enumerate(target_dates):
        print(f"\n📅 Processing date {i+1}/{len(target_dates)}: {date}")
        cached_count = batch_cache_pca_scores(date, current_season)
        total_cached += cached_count
        
        # Brief pause to avoid overwhelming the system
        if i < len(target_dates) - 1:
            print("💤 Brief pause...")
            time.sleep(1)
    
    print(f"\n🎉 PCA cache building complete!")
    print(f"📊 Total PCA scores cached: {total_cached}")
    print(f"📅 Dates processed: {len(target_dates)}")
    
    return total_cached

# Backwards compatibility with old API
def run_pca(metric_type, max_date, current_season="2025", log_queue=None):
    """Backwards compatible function that uses the new optimized system."""
    all_pca_results = calculate_all_pca_scores_for_date(max_date, current_season)
    
    # Convert to old format
    player_pca_mapping = {}
    stacked_stats = []
    
    for player_name, scores in all_pca_results.items():
        if metric_type in scores:
            player_pca_mapping[player_name] = {metric_type: scores[metric_type]}
            # Create fake stacked_stats entry for compatibility
            stacked_stats.append({'PLAYER_NAME': player_name})
    
    return player_pca_mapping, stacked_stats

if __name__ == "__main__":
    # Example: Build PCA cache for early season
    print("🚀 NBA PCA Optimization Test")
    
    # Build cache for first week of season
    cached_count = batch_cache_pca_for_date_range(
        start_date="2023-10-24",
        end_date="2023-10-31", 
        date_interval_days=2
    )
    
    print(f"✅ Cached {cached_count} total PCA scores!")
