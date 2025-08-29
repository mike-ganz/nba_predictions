#!/usr/bin/env python3
"""
🚀 OPTIMIZED NBA Player Cache Builder - Vectorized batch processing
"""

from config import set_season_year
import transform_player_stats_optimized as optimized_stats
import pandas as pd
import os
import glob
import time
from datetime import datetime, timedelta

def count_cache_files():
    """Count current cache files."""
    cache_dir = "data/cache/player_stats"
    if os.path.exists(cache_dir):
        return len(glob.glob(f"{cache_dir}/*.json"))
    return 0

def build_cache_for_month_optimized(year_month, date_interval_days=2):
    """
    🚀 OPTIMIZED: Build cache for ALL players in a month using vectorized processing.
    
    Instead of processing one player-date at a time, this processes ALL players 
    for multiple dates simultaneously.
    """
    print(f"\n📅 🚀 OPTIMIZED Processing {year_month}...")
    
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
    
    cache_before = count_cache_files()
    start_time = time.time()
    
    # 🚀 VECTORIZED: Build cache for entire month at once
    cache_entries_created = optimized_stats.build_cache_for_date_range(
        start_date=start_date,
        end_date=end_date, 
        date_interval_days=date_interval_days
    )
    
    end_time = time.time()
    cache_after = count_cache_files()
    
    print(f"✅ OPTIMIZED: Created {cache_entries_created} cache entries in {end_time - start_time:.1f}s")
    print(f"💾 Cache files: {cache_before} → {cache_after} (+{cache_after - cache_before})")
    print(f"⚡ Speed: {cache_entries_created/(end_time - start_time):.0f} entries/second")
    
    return cache_entries_created

def build_cache_for_month_traditional(year_month):
    """Traditional method for comparison."""
    from training import training_data_generator
    from data.loaders import data_loader
    
    print(f"\n📅 📊 TRADITIONAL Processing {year_month}...")
    
    # Load play-by-play data to get games for this month
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
    
    selected_games = sorted(month_games)[:10]  # Limit for fair comparison
    print(f"🏀 Processing {len(selected_games)} games")
    
    if len(selected_games) == 0:
        print(f"⚠️ No games found for {year_month}")
        return 0
    
    cache_before = count_cache_files()
    start_time = time.time()
    
    # Generate training data (builds cache traditionally)
    batch_df = training_data_generator.generate_dataset(
        season_year="2023-2024",
        n_total=3,
        game_id_filter=selected_games,
        force_real_pca=True
    )
    
    end_time = time.time()
    cache_after = count_cache_files()
    
    print(f"📊 TRADITIONAL: Processed {len(batch_df)} records in {end_time - start_time:.1f}s")
    print(f"💾 Cache files: {cache_before} → {cache_after} (+{cache_after - cache_before})")
    
    return len(selected_games)

def performance_comparison():
    """Compare optimized vs traditional methods."""
    print("🏁 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Test month
    test_month = "2023-10"
    
    # Clear any existing cache for fair test
    cache_dir = "data/cache/player_stats"
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print("🧹 Cleared cache for fair comparison")
    
    # Test traditional method
    print("\n1️⃣ Testing TRADITIONAL method...")
    traditional_time = time.time()
    traditional_result = build_cache_for_month_traditional(test_month)
    traditional_elapsed = time.time() - traditional_time
    traditional_cache_count = count_cache_files()
    
    print(f"📊 Traditional: {traditional_result} games, {traditional_cache_count} cache files, {traditional_elapsed:.1f}s")
    
    # Clear cache again
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Test optimized method  
    print("\n2️⃣ Testing OPTIMIZED method...")
    optimized_time = time.time()
    optimized_result = build_cache_for_month_optimized(test_month, date_interval_days=3)
    optimized_elapsed = time.time() - optimized_time
    optimized_cache_count = count_cache_files()
    
    print(f"🚀 Optimized: {optimized_result} entries, {optimized_cache_count} cache files, {optimized_elapsed:.1f}s")
    
    # Results
    print("\n🏆 PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Traditional: {traditional_elapsed:.1f}s → {traditional_cache_count} files")
    print(f"Optimized:   {optimized_elapsed:.1f}s → {optimized_cache_count} files")
    
    if optimized_elapsed > 0:
        speedup = traditional_elapsed / optimized_elapsed
        efficiency = optimized_cache_count / optimized_elapsed
        print(f"\n⚡ Speedup: {speedup:.1f}x faster")
        print(f"🎯 Efficiency: {efficiency:.0f} cache files/second")

def main():
    """Main execution - choose your approach."""
    print("🚀 OPTIMIZED NBA Player Cache Builder")
    set_season_year("2023-2024")
    
    choice = input("\nChoose mode:\n1. Performance comparison\n2. Build optimized cache\n3. Build traditional cache\nChoice (1-3): ")
    
    if choice == "1":
        performance_comparison()
    
    elif choice == "2":
        print("\n🚀 OPTIMIZED CACHE BUILDING")
        months = ["2023-10", "2023-11", "2023-12"] 
        
        total_entries = 0
        for month in months:
            entries = build_cache_for_month_optimized(month, date_interval_days=2)
            total_entries += entries
            print(f"💤 Brief pause...")
            time.sleep(5)
        
        print(f"\n🎉 OPTIMIZED Complete! {total_entries} total cache entries")
        print(f"📊 Final cache count: {count_cache_files()}")
    
    elif choice == "3":
        print("\n📊 TRADITIONAL CACHE BUILDING")
        months = ["2023-10", "2023-11"]  # Fewer months for traditional
        
        total_games = 0
        for month in months:
            games = build_cache_for_month_traditional(month)
            total_games += games
            print(f"💤 Pause between months...")
            time.sleep(10)
        
        print(f"\n🎉 TRADITIONAL Complete! {total_games} total games")
        print(f"📊 Final cache count: {count_cache_files()}")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
