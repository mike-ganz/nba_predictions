#!/usr/bin/env python3
"""
Test PCA optimization performance
"""

import time
import pca as original_pca
import pca_optimized as optimized_pca
from config import set_season_year
import shutil
import os

def clear_pca_cache():
    """Clear PCA cache for fair testing."""
    cache_dir = "data/cache/pca"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("🧹 Cleared PCA cache")

def test_single_player_pca():
    """Test getting PCA scores for a single player."""
    print("\n🧪 Testing Single Player PCA...")
    
    player = "LeBron James"
    max_date = "2023-10-30"
    season = "2023-2024"
    
    clear_pca_cache()
    
    # Test original method
    print("📊 Testing ORIGINAL PCA system...")
    start_time = time.time()
    try:
        original_result = original_pca.get_player_pca_score(player, max_date, season)
        original_time = time.time() - start_time
        print(f"📊 Original: {original_time:.2f}s → {original_result}")
    except Exception as e:
        original_time = float('inf')
        original_result = None
        print(f"❌ Original failed: {e}")
    
    clear_pca_cache()
    
    # Test optimized method
    print("🚀 Testing OPTIMIZED PCA system...")
    start_time = time.time()
    try:
        optimized_result = optimized_pca.get_player_pca_score(player, max_date, season)
        optimized_time = time.time() - start_time
        print(f"🚀 Optimized: {optimized_time:.2f}s → {optimized_result}")
    except Exception as e:
        optimized_time = float('inf')
        optimized_result = None
        print(f"❌ Optimized failed: {e}")
    
    # Compare results
    if original_result and optimized_result:
        print(f"\n📈 {player} PCA comparison:")
        metrics = ['offense', 'defense', 'shot_selection', 'efficiency']
        for i, metric in enumerate(metrics):
            orig_val = original_result[i] if original_result else 'N/A'
            opt_val = optimized_result[i] if optimized_result else 'N/A'
            
            if isinstance(orig_val, float) and isinstance(opt_val, float):
                diff = abs(orig_val - opt_val)
                match = "✅" if diff < 0.001 else f"❌ (diff: {diff:.4f})"
            else:
                match = "✅" if orig_val == opt_val else "❌"
                
            print(f"  {metric}: {orig_val:.3f} vs {opt_val:.3f} {match}")
    
    # Performance comparison
    if original_time != float('inf') and optimized_time != float('inf'):
        speedup = original_time / optimized_time if optimized_time > 0 else 0
        print(f"\n⚡ Performance: {original_time:.2f}s → {optimized_time:.2f}s")
        print(f"🚀 Speedup: {speedup:.1f}x faster")
    
    return original_result, optimized_result

def test_multiple_players_pca():
    """Test getting PCA scores for multiple players."""
    print("\n🧪 Testing Multiple Players PCA...")
    
    players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo", "Nikola Jokic", "Jayson Tatum"]
    max_date = "2023-10-30"
    season = "2023-2024"
    
    clear_pca_cache()
    
    # Test original method (individual calls)
    print(f"📊 Testing ORIGINAL PCA for {len(players)} players...")
    start_time = time.time()
    original_results = {}
    try:
        for player in players:
            result = original_pca.get_player_pca_score(player, max_date, season)
            original_results[player] = result
        original_time = time.time() - start_time
        print(f"📊 Original: {original_time:.2f}s for {len(players)} players")
    except Exception as e:
        original_time = float('inf')
        print(f"❌ Original failed: {e}")
    
    clear_pca_cache()
    
    # Test optimized method (batch calculation)
    print(f"🚀 Testing OPTIMIZED PCA for {len(players)} players...")
    start_time = time.time()
    optimized_results = {}
    try:
        # Method 1: Individual calls (should be fast due to batch caching)
        for player in players:
            result = optimized_pca.get_player_pca_score(player, max_date, season)
            optimized_results[player] = result
        optimized_time = time.time() - start_time
        print(f"🚀 Optimized: {optimized_time:.2f}s for {len(players)} players")
    except Exception as e:
        optimized_time = float('inf')
        print(f"❌ Optimized failed: {e}")
    
    # Performance comparison
    if original_time != float('inf') and optimized_time != float('inf'):
        speedup = original_time / optimized_time if optimized_time > 0 else 0
        efficiency_orig = len(players) / original_time if original_time > 0 else 0
        efficiency_opt = len(players) / optimized_time if optimized_time > 0 else 0
        
        print(f"\n⚡ Performance Comparison:")
        print(f"📊 Original: {original_time:.2f}s ({efficiency_orig:.1f} players/sec)")
        print(f"🚀 Optimized: {optimized_time:.2f}s ({efficiency_opt:.1f} players/sec)")
        print(f"🎯 Speedup: {speedup:.1f}x faster")
    
    return len(players)

def test_batch_pca_cache():
    """Test the new batch PCA cache building."""
    print("\n🧪 Testing Batch PCA Cache Building...")
    
    clear_pca_cache()
    
    start_time = time.time()
    cached_count = optimized_pca.batch_cache_pca_for_date_range(
        start_date="2023-10-24",
        end_date="2023-10-28",
        date_interval_days=2
    )
    elapsed = time.time() - start_time
    
    print(f"✅ Batch cache building: {cached_count} PCA scores in {elapsed:.2f}s")
    print(f"⚡ Rate: {cached_count/elapsed:.0f} PCA scores per second")
    
    return cached_count

def main():
    """Run all PCA optimization tests."""
    print("🚀 NBA PCA Optimization Test Suite")
    set_season_year("2023-2024")
    
    # Test 1: Single player
    test_single_player_pca()
    
    # Test 2: Multiple players  
    test_multiple_players_pca()
    
    # Test 3: Batch cache building
    test_batch_pca_cache()
    
    print("\n✅ All PCA optimization tests complete!")

if __name__ == "__main__":
    main()
