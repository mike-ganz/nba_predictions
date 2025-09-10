#!/usr/bin/env python3
"""Debug script to find why game 22400530 isn't found"""

import sys
sys.path.append(".")

try:
    from data.loaders import load_play_by_play_data
    
    print("🔍 DEBUGGING GAME 22400530 SEARCH")
    print("=" * 50)
    
    # Load the data
    data = load_play_by_play_data("2024-2025")
    print(f"📊 Total records loaded: {len(data)}")
    
    # Check unique game IDs
    unique_games = data['game_id'].unique()
    print(f"📋 Total unique games: {len(unique_games)}")
    
    # Show first few game IDs to see the pattern
    print("\n🎯 First 10 game IDs:")
    for game_id in sorted(unique_games)[:10]:
        print(f"   {game_id}")
    
    # Check if there are any games starting with 22400530
    similar_games = [g for g in unique_games if str(g).startswith('224005')]
    print(f"\n🔍 Games starting with '224005': {len(similar_games)}")
    for game_id in sorted(similar_games)[:10]:
        print(f"   {game_id}")
        
    # Check if the exact game exists
    exact_match = [g for g in unique_games if str(g) == '22400530']
    print(f"\n🎯 Exact match for '22400530': {len(exact_match)}")
    
    # Check data types
    sample_game_id = unique_games[0] if len(unique_games) > 0 else None
    if sample_game_id:
        print(f"\n📊 Sample game ID type: {type(sample_game_id)}")
        print(f"   Sample game ID value: {repr(sample_game_id)}")
    
    # Search with different approaches
    target_as_str = '22400530'
    target_as_int = 22400530
    
    str_matches = data[data['game_id'].astype(str) == target_as_str]
    int_matches = data[data['game_id'] == target_as_int]
    
    print(f"\n🔍 Matches with string search: {len(str_matches)}")
    print(f"🔍 Matches with int search: {len(int_matches)}")
    
except Exception as e:
    print(f"❌ Debug failed: {e}")
    import traceback
    traceback.print_exc()
