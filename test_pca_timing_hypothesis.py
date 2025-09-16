#!/usr/bin/env python3
"""
Test the hypothesis that PCA differences are caused by timing-dependent data loading.
"""

import json
import pandas as pd
from generate_training_data import create_llm_training_data, load_play_by_play_data
from pca_optimized import calculate_all_pca_scores_for_date
from generate_lineup import load_all_player_boxscores

def test_pca_timing_hypothesis():
    """Test if PCA scores change depending on when they're calculated in the pipeline."""
    print("🧪 TESTING PCA TIMING HYPOTHESIS")
    print("=" * 50)
    
    # Load test data
    print("📂 Loading sample data...")
    season_df = load_play_by_play_data("2023-2024")
    
    # Use first game for focused testing
    test_game_id = sorted(season_df['game_id'].unique())[0]
    test_df = season_df[season_df['game_id'] == test_game_id].head(100)
    
    # Get the test date
    test_date = test_df.iloc[0]['date']
    season = "2023-2024"
    
    print(f"🎯 Testing with game {test_game_id} on date {test_date}")
    
    # Clear any existing caches to start fresh
    calculate_all_pca_scores_for_date.cache_clear()
    
    print("\n" + "="*50)
    print("📊 STAGE 1: PCA Calculation BEFORE main pipeline")
    print("="*50)
    
    # Calculate PCA scores BEFORE any other data loading (simulating batch method)
    early_pca_scores = calculate_all_pca_scores_for_date(test_date, season)
    
    print(f"✅ Early PCA calculation complete - found {len(early_pca_scores)} players")
    
    # Show sample early scores
    sample_players = list(early_pca_scores.keys())[:3]
    print("📋 Sample early PCA scores:")
    for player in sample_players:
        scores = early_pca_scores[player]
        print(f"  {player}: [{scores.get('offense', 0):.2f}, {scores.get('defense', 0):.2f}, {scores.get('shot_selection', 0):.2f}, {scores.get('efficiency', 0):.2f}]")
    
    print("\n" + "="*50)
    print("📦 STAGE 2: Main pipeline operations (data loading, filtering)")
    print("="*50)
    
    # Simulate the main pipeline operations that happen before individual PCA calls
    print("🔄 Loading boxscore data (as done in main pipeline)...")
    boxscore_data = load_all_player_boxscores()
    print(f"📊 Loaded boxscore data: {len(boxscore_data)} records")
    
    # Simulate the filtering that happens in create_llm_training_data
    print("🔄 Simulating main pipeline filtering operations...")
    game_play_counts = test_df.groupby('game_id').size()
    valid_games = game_play_counts[game_play_counts >= 50].index
    filtered_df = test_df[test_df['game_id'].isin(valid_games)]
    print(f"📊 Filtered data: {len(filtered_df)} plays from {len(valid_games)} games")
    
    print("\n" + "="*50)
    print("📊 STAGE 3: PCA Calculation AFTER main pipeline operations")
    print("="*50)
    
    # Clear cache to force recalculation (simulating individual method)
    calculate_all_pca_scores_for_date.cache_clear()
    
    # Calculate PCA scores AFTER main pipeline operations (simulating individual method)
    late_pca_scores = calculate_all_pca_scores_for_date(test_date, season)
    
    print(f"✅ Late PCA calculation complete - found {len(late_pca_scores)} players")
    
    # Show sample late scores
    print("📋 Sample late PCA scores:")
    for player in sample_players:
        if player in late_pca_scores:
            scores = late_pca_scores[player]
            print(f"  {player}: [{scores.get('offense', 0):.2f}, {scores.get('defense', 0):.2f}, {scores.get('shot_selection', 0):.2f}, {scores.get('efficiency', 0):.2f}]")
        else:
            print(f"  {player}: NOT FOUND in late calculation!")
    
    print("\n" + "="*50)
    print("🔍 COMPARISON ANALYSIS")
    print("="*50)
    
    # Compare the results
    early_player_count = len(early_pca_scores)
    late_player_count = len(late_pca_scores)
    
    print(f"📊 Player counts:")
    print(f"  Early calculation: {early_player_count} players")
    print(f"  Late calculation:  {late_player_count} players")
    
    if early_player_count != late_player_count:
        print("❌ DIFFERENT PLAYER COUNTS - This could explain the PCA differences!")
        
        early_players = set(early_pca_scores.keys())
        late_players = set(late_pca_scores.keys())
        
        only_early = early_players - late_players
        only_late = late_players - early_players
        
        if only_early:
            print(f"  🔍 Players only in early: {len(only_early)} - {list(only_early)[:5]}...")
        if only_late:
            print(f"  🔍 Players only in late: {len(only_late)} - {list(only_late)[:5]}...")
    
    # Compare actual scores for common players
    common_players = set(early_pca_scores.keys()) & set(late_pca_scores.keys())
    print(f"📊 Common players: {len(common_players)}")
    
    mismatches = 0
    perfect_matches = 0
    
    print(f"\n🔍 Detailed score comparison for sample players:")
    for player in list(common_players)[:5]:  # Check first 5 common players
        early = early_pca_scores[player]
        late = late_pca_scores[player]
        
        early_scores = [early.get('offense', 0), early.get('defense', 0), 
                       early.get('shot_selection', 0), early.get('efficiency', 0)]
        late_scores = [late.get('offense', 0), late.get('defense', 0), 
                      late.get('shot_selection', 0), late.get('efficiency', 0)]
        
        if early_scores != late_scores:
            print(f"  ❌ {player}:")
            print(f"     Early: {[f'{s:.3f}' for s in early_scores]}")
            print(f"     Late:  {[f'{s:.3f}' for s in late_scores]}")
            mismatches += 1
        else:
            print(f"  ✅ {player}: Perfect match")
            perfect_matches += 1
    
    print(f"\n📊 FINAL ANALYSIS:")
    print(f"  Perfect matches: {perfect_matches}")
    print(f"  Mismatches: {mismatches}")
    
    if early_player_count != late_player_count:
        print(f"\n🎯 HYPOTHESIS CONFIRMED: Different player datasets loaded at different times!")
        print(f"   This explains why batch and individual PCA methods give different results.")
        print(f"   The PCA algorithm is relative - different player sets = different PCA scores for everyone.")
        return True
    elif mismatches > 0:
        print(f"\n🎯 HYPOTHESIS PARTIALLY CONFIRMED: Same players, but different scores!")
        print(f"   This suggests the underlying data for individual players changed between calculations.")
        return True
    else:
        print(f"\n❓ HYPOTHESIS NOT CONFIRMED: PCA scores are identical at different times.")
        print(f"   The issue must be elsewhere in the pipeline.")
        return False

if __name__ == "__main__":
    hypothesis_confirmed = test_pca_timing_hypothesis()
    exit(0 if hypothesis_confirmed else 1)
