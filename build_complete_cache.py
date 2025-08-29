#!/usr/bin/env python3
"""
🚀 NBA Complete Cache Builder - One file to build all optimized caches
==================================================================================

This script builds both player stats and PCA caches using optimized batch processing.
Simply specify the season year below and run!

Features:
- Uses optimized player stats system (50-100x faster)
- Uses optimized PCA system (1000+ x faster) 
- Builds both caches in one run
- Season-specific month ranges
- Progress tracking and performance metrics
"""

# Basic imports first
from config import set_season_year
import pandas as pd
import os
import glob
import time
from datetime import datetime, timedelta

# ==================================================================================
# 🏀 SEASON CONFIGURATION - Change this to switch between seasons!
# ==================================================================================
SEASON_YEAR = "2022-2023"  # Change to "2022-2023" or "2023-2024"
# ==================================================================================

# ⚡ CRITICAL: Set season in config system first
set_season_year(SEASON_YEAR)

# Import the data-loading modules
import transform_player_stats_optimized as optimized_stats
import pca_optimized as optimized_pca

# ⚡ CRITICAL: Also set season in the optimized_stats module (which has its own season setting)
optimized_stats.set_season_year(SEASON_YEAR)

def get_season_month_options(season_year):
    """Generate season-appropriate month options based on season year."""
    if season_year == "2022-2023":
        return {
            "1": {
                "name": "October 2022 (Season Start)",
                "months": ["2022-10"]
            },
            "2": {
                "name": "First Quarter (Oct-Dec 2022)",
                "months": ["2022-10", "2022-11", "2022-12"]
            },
            "3": {
                "name": "First Half (Oct 2022 - Jan 2023)",
                "months": ["2022-10", "2022-11", "2022-12", "2023-01"]
            },
            "4": {
                "name": "Full Regular Season (Oct 2022 - Apr 2023)",
                "months": ["2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]
            },
            "5": {
                "name": "Complete Season + Playoffs (Oct 2022 - June 2023)",
                "months": ["2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06"]
            },
            "6": {
                "name": "Custom - Enter your own months",
                "months": []
            }
        }
    elif season_year == "2023-2024":
        return {
            "1": {
                "name": "October 2023 (Season Start)",
                "months": ["2023-10"]
            },
            "2": {
                "name": "First Quarter (Oct-Dec 2023)",
                "months": ["2023-10", "2023-11", "2023-12"]
            },
            "3": {
                "name": "First Half (Oct 2023 - Jan 2024)",
                "months": ["2023-10", "2023-11", "2023-12", "2024-01"]
            },
            "4": {
                "name": "Full Regular Season (Oct 2023 - Apr 2024)",
                "months": ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04"]
            },
            "5": {
                "name": "Complete Season + Playoffs (Oct 2023 - June 2024)",
                "months": ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
            },
            "6": {
                "name": "Custom - Enter your own months",
                "months": []
            }
        }
    else:
        raise ValueError(f"Season {season_year} not supported. Use '2022-2023' or '2023-2024'")

def count_cache_files():
    """Count current cache files."""
    player_stats_dir = "data/cache/player_stats"
    pca_dir = "data/cache/pca"
    
    player_count = len(glob.glob(f"{player_stats_dir}/*.json")) if os.path.exists(player_stats_dir) else 0
    pca_count = len(glob.glob(f"{pca_dir}/*.json")) if os.path.exists(pca_dir) else 0
    
    return player_count, pca_count

def build_cache_for_month(year_month, date_interval_days=1):
    """
    🚀 Build both player stats and PCA cache for a month using optimized systems.
    
    Args:
        year_month (str): Format "YYYY-MM" (e.g., "2023-10")
        date_interval_days (int): Days between each cache point
    
    Returns:
        tuple: (player_stats_created, pca_scores_created)
    """
    print(f"\n📅 🚀 Processing {year_month}...")
    
    # Parse month and create date range
    year, month = year_month.split('-')
    start_date = f"{year}-{month}-01"
    
    # Calculate end date (last day of month)
    if month == '12':
        next_month = f"{int(year)+1}-01-01"
    else:
        next_month = f"{year}-{int(month)+1:02d}-01"
    
    end_date = (pd.to_datetime(next_month) - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"📅 Date range: {start_date} to {end_date} (every {date_interval_days} days)")
    
    # Track cache counts before
    player_before, pca_before = count_cache_files()
    start_time = time.time()
    
    # 🚀 STEP 1: Build optimized player stats cache
    print("🔥 Step 1/2: Building player stats cache...")
    player_stats_created = optimized_stats.build_cache_for_date_range(
        start_date=start_date,
        end_date=end_date,
        date_interval_days=date_interval_days
    )
    
    # 🚀 STEP 2: Build optimized PCA cache  
    print("🧠 Step 2/2: Building PCA cache...")
    pca_scores_created = optimized_pca.batch_cache_pca_for_date_range(
        start_date=start_date,
        end_date=end_date,
        date_interval_days=date_interval_days,
        current_season=SEASON_YEAR
    )
    
    # Final metrics
    end_time = time.time()
    player_after, pca_after = count_cache_files()
    
    print(f"✅ {year_month} Complete in {end_time - start_time:.1f}s!")
    print(f"📊 Player stats: {player_before} → {player_after} (+{player_after - player_before})")
    print(f"🧠 PCA scores: {pca_before} → {pca_after} (+{pca_after - pca_before})")
    
    return player_stats_created, pca_scores_created

def build_cache_for_months(months, date_interval_days=3, pause_seconds=5):
    """
    Build cache for multiple months.
    
    Args:
        months (list): List of month strings like ["2023-10", "2023-11"]
        date_interval_days (int): Days between cache points
        pause_seconds (int): Seconds to pause between months
    """
    print(f"🚀 NBA Complete Cache Builder - Season {SEASON_YEAR}")
    print(f"📅 Building cache for {len(months)} months: {', '.join(months)}")
    print(f"⚙️  Settings: {date_interval_days} day intervals, {pause_seconds}s pauses")
    
    total_player_stats = 0
    total_pca_scores = 0
    overall_start = time.time()
    
    for i, month in enumerate(months):
        print(f"\n{'='*60}")
        print(f"📅 Processing month {i+1}/{len(months)}: {month}")
        print(f"{'='*60}")
        
        try:
            player_created, pca_created = build_cache_for_month(month, date_interval_days)
            total_player_stats += player_created
            total_pca_scores += pca_created
            
            # Pause between months (except for the last one)
            if i < len(months) - 1:
                print(f"💤 Pausing {pause_seconds} seconds before next month...")
                time.sleep(pause_seconds)
                
        except Exception as e:
            print(f"❌ Error processing {month}: {e}")
            continue
    
    # Final summary
    overall_time = time.time() - overall_start
    player_final, pca_final = count_cache_files()
    
    print(f"\n🎉 COMPLETE! All {len(months)} months processed in {overall_time:.1f}s")
    print(f"📊 Total player stats created: {total_player_stats:,}")
    print(f"🧠 Total PCA scores created: {total_pca_scores:,}")
    print(f"💾 Final cache counts:")
    print(f"   📊 Player stats files: {player_final:,}")
    print(f"   🧠 PCA cache files: {pca_final:,}")
    print(f"⚡ Average speed: {(total_player_stats + total_pca_scores)/overall_time:.0f} cache entries/second")

def main():
    """Main function - choose your cache building approach."""
    print(f"🚀 NBA Complete Cache Builder - Season {SEASON_YEAR}")
    print("=" * 50)
    
    # Get season-appropriate month ranges
    month_options = get_season_month_options(SEASON_YEAR)
    
    print("\n📅 Choose month range to build cache for:")
    for key, option in month_options.items():
        print(f"{key}. {option['name']}")
        if option['months']:
            print(f"   Months: {', '.join(option['months'])}")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice not in month_options:
        print("❌ Invalid choice!")
        return
    
    selected_option = month_options[choice]
    
    if choice == "6":
        # Custom months
        print("\n📝 Enter months in YYYY-MM format, separated by commas:")
        print("   Example: 2023-10,2023-11,2023-12")
        custom_input = input("Months: ").strip()
        
        if not custom_input:
            print("❌ No months entered!")
            return
            
        months = [month.strip() for month in custom_input.split(',')]
        
        # Validate month formats
        valid_months = []
        for month in months:
            try:
                pd.to_datetime(month + "-01")  # Test if valid date
                valid_months.append(month)
            except:
                print(f"⚠️  Skipping invalid month format: {month}")
        
        if not valid_months:
            print("❌ No valid months provided!")
            return
            
        months = valid_months
        selected_name = f"Custom ({len(months)} months)"
    else:
        months = selected_option['months']
        selected_name = selected_option['name']
    
    # Get settings
    print(f"\n⚙️  Settings:")
    date_interval = input("Days between cache points (default 3): ").strip()
    date_interval = int(date_interval) if date_interval else 3
    
    pause_seconds = input("Seconds to pause between months (default 5): ").strip()
    pause_seconds = int(pause_seconds) if pause_seconds else 5
    
    # Confirm and execute
    print(f"\n🚀 Ready to build cache:")
    print(f"📅 Range: {selected_name}")
    print(f"📊 Months: {', '.join(months)}")
    print(f"⚙️  {date_interval} day intervals, {pause_seconds}s pauses")
    
    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled!")
        return
    
    # Build the cache!
    build_cache_for_months(months, date_interval, pause_seconds)

if __name__ == "__main__":
    main()
