#!/usr/bin/env python3
"""
🚀 OPTIMIZED PCA Cache Builder - Vectorized batch processing for PCA scores
"""

from config import set_season_year
import pca_optimized as optimized_pca
import pandas as pd
import os
import glob
import time
from datetime import datetime, timedelta

def count_pca_cache_files():
    """Count current PCA cache files."""
    cache_dir = "data/cache/pca"
    if os.path.exists(cache_dir):
        return len(glob.glob(f"{cache_dir}/*.json"))
    return 0

def build_pca_cache_for_month_optimized(year_month, date_interval_days=3):
    """
    🚀 OPTIMIZED: Build PCA cache for ALL players in a month using vectorized processing.
    
    This builds PCA cache for all players simultaneously for multiple dates,
    rather than processing one player-date combination at a time.
    """
    print(f"\n📅 🚀 OPTIMIZED PCA Processing {year_month}...")
    
    # Parse month
    year, month = year_month.split('-')
    start_date = f"{year}-{month}-01"
    
    # Calculate end date (last day of month)
    if month == '12':
        next_month = f"{int(year)+1}-01-01"
    else:
        next_month = f"{year}-{int(month)+1:02d}-01"
    
    end_date = (pd.to_datetime(next_month) - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"📅 Date range: {start_date} to {end_date} (every {date_interval_days} days)")
    
    cache_before = count_pca_cache_files()
    start_time = time.time()
    
    # 🚀 VECTORIZED: Build PCA cache for entire month at once
    cache_entries_created = optimized_pca.batch_cache_pca_for_date_range(
        start_date=start_date,
        end_date=end_date, 
        date_interval_days=date_interval_days,
        current_season="2023-2024"
    )
    
    end_time = time.time()
    cache_after = count_pca_cache_files()
    
    print(f"✅ OPTIMIZED PCA: Created {cache_entries_created} cache entries in {end_time - start_time:.1f}s")
    print(f"💾 PCA Cache files: {cache_before} → {cache_after} (+{cache_after - cache_before})")
    print(f"⚡ Speed: {cache_entries_created/(end_time - start_time):.0f} PCA scores/second")
    
    return cache_entries_created

def build_pca_cache_traditional(year_month):
    """Traditional PCA cache building for comparison."""
    print(f"\n📅 📊 TRADITIONAL PCA Processing {year_month}...")
    
    # We'll simulate the traditional approach by calling the old API
    import pca as original_pca
    from data.loaders import data_loader
    
    # Load some games from the month
    play_by_play = data_loader.load_play_by_play_data("2023-2024")
    
    if play_by_play is None:
        print("❌ Could not load play-by-play data")
        return 0
    
    # Handle flexible date formats
    play_by_play['date'] = pd.to_datetime(play_by_play['date'], format='mixed')
    
    # Filter games for this month
    year, month = year_month.split('-')
    month_games = play_by_play[
        (play_by_play['date'].dt.year == int(year)) & 
        (play_by_play['date'].dt.month == int(month))
    ]['game_id'].unique()
    
    # Limit games for fair comparison
    selected_games = sorted(month_games)[:5]
    print(f"🏀 Processing {len(selected_games)} games for traditional comparison")
    
    if len(selected_games) == 0:
        print(f"⚠️ No games found for {year_month}")
        return 0
    
    cache_before = count_pca_cache_files()
    start_time = time.time()
    
    # Simulate traditional PCA cache building by calling PCA for some players
    test_players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo"]
    pca_scores_calculated = 0
    
    for i, game_id in enumerate(selected_games):
        # Get game date
        game_date = play_by_play[play_by_play['game_id'] == game_id]['date'].iloc[0]
        max_date = game_date.strftime('%Y-%m-%d')
        
        for player in test_players:
            try:
                scores = original_pca.get_player_pca_score(player, max_date, "2023-2024")
                pca_scores_calculated += 1
            except Exception as e:
                print(f"⚠️ Failed to get PCA for {player}: {e}")
    
    end_time = time.time()
    cache_after = count_pca_cache_files()
    
    print(f"📊 TRADITIONAL PCA: {pca_scores_calculated} scores in {end_time - start_time:.1f}s")
    print(f"💾 PCA Cache files: {cache_before} → {cache_after} (+{cache_after - cache_before})")
    
    return pca_scores_calculated

def performance_comparison():
    """Compare optimized vs traditional PCA methods."""
    print("🏁 PCA PERFORMANCE COMPARISON")
    print("=" * 50)
    
    test_month = "2023-10"
    
    # Clear cache for fair test
    cache_dir = "data/cache/pca"
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print("🧹 Cleared PCA cache for fair comparison")
    
    # Test traditional method
    print("\n1️⃣ Testing TRADITIONAL PCA method...")
    traditional_time = time.time()
    traditional_result = build_pca_cache_traditional(test_month)
    traditional_elapsed = time.time() - traditional_time
    traditional_cache_count = count_pca_cache_files()
    
    print(f"📊 Traditional: {traditional_result} PCA scores, {traditional_cache_count} cache files, {traditional_elapsed:.1f}s")
    
    # Clear cache again
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Test optimized method  
    print("\n2️⃣ Testing OPTIMIZED PCA method...")
    optimized_time = time.time()
    optimized_result = build_pca_cache_for_month_optimized(test_month, date_interval_days=4)
    optimized_elapsed = time.time() - optimized_time
    optimized_cache_count = count_pca_cache_files()
    
    print(f"🚀 Optimized: {optimized_result} PCA scores, {optimized_cache_count} cache files, {optimized_elapsed:.1f}s")
    
    # Results
    print("\n🏆 PCA PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Traditional: {traditional_elapsed:.1f}s → {traditional_cache_count} files")
    print(f"Optimized:   {optimized_elapsed:.1f}s → {optimized_cache_count} files")
    
    if optimized_elapsed > 0 and traditional_elapsed > 0:
        speedup = traditional_elapsed / optimized_elapsed
        efficiency = optimized_cache_count / optimized_elapsed
        print(f"\n⚡ Speedup: {speedup:.1f}x faster")
        print(f"🎯 Efficiency: {efficiency:.0f} PCA cache files/second")

def main():
    """Main execution - choose your approach."""
    print("🚀 OPTIMIZED PCA Cache Builder")
    set_season_year("2023-2024")
    
    choice = input("\nChoose mode:\n1. Performance comparison\n2. Build optimized PCA cache\n3. Build traditional PCA cache\nChoice (1-3): ")
    
    if choice == "1":
        performance_comparison()
    
    elif choice == "2":
        print("\n🚀 OPTIMIZED PCA CACHE BUILDING")
        months = ["2023-10", "2023-11", "2023-12"] 
        
        total_entries = 0
        for month in months:
            entries = build_pca_cache_for_month_optimized(month, date_interval_days=3)
            total_entries += entries
            print(f"💤 Brief pause...")
            time.sleep(5)
        
        print(f"\n🎉 OPTIMIZED Complete! {total_entries} total PCA scores cached")
        print(f"📊 Final PCA cache count: {count_pca_cache_files()}")
    
    elif choice == "3":
        print("\n📊 TRADITIONAL PCA CACHE BUILDING")
        months = ["2023-10"]  # Just one month for traditional (it's slow)
        
        total_scores = 0
        for month in months:
            scores = build_pca_cache_traditional(month)
            total_scores += scores
            print(f"💤 Pause between months...")
            time.sleep(10)
        
        print(f"\n🎉 TRADITIONAL Complete! {total_scores} total PCA scores")
        print(f"📊 Final PCA cache count: {count_pca_cache_files()}")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
