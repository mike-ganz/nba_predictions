#!/usr/bin/env python3
"""Test script to verify data loading modules work on GCP VM"""

import sys
import os

print("🔍 TESTING DATA LOADING SYSTEM")
print("=" * 50)

# Add current directory to Python path
sys.path.append(".")

print(f"📁 Current directory: {os.getcwd()}")
print(f"🐍 Python path: {sys.path[:3]}...")

# Test each import individually
imports_to_test = [
    ("data.loaders", "load_play_by_play_data"),
    ("analysis.player_stats", "get_player_pca_score"), 
    ("game.team_utils", "determine_home_away_teams"),
    ("config.settings", "config")
]

for module_name, import_item in imports_to_test:
    try:
        exec(f"from {module_name} import {import_item}")
        print(f"✅ {module_name}.{import_item} imported successfully")
    except Exception as e:
        print(f"❌ {module_name}.{import_item} import failed: {e}")

print("\n🏀 TESTING PLAY-BY-PLAY DATA ACCESS")
print("=" * 50)

try:
    from data.loaders import load_play_by_play_data
    
    # Test loading 2024-2025 data
    data = load_play_by_play_data("2024-2025")
    if data is not None and len(data) > 0:
        print(f"✅ Successfully loaded {len(data)} play-by-play records for 2024-2025")
        
        # Check if target game exists
        target_game_data = data[data['game_id'] == '22400530']
        if len(target_game_data) > 0:
            print(f"🎯 Found {len(target_game_data)} records for target game 22400530")
            print("✅ REAL DATA LOADING WORKS!")
        else:
            print("❌ Target game 22400530 not found in loaded data")
    else:
        print("❌ No data loaded or data is empty")
        
except Exception as e:
    print(f"❌ Failed to test play-by-play data loading: {e}")

print(f"\n📊 Test completed!")
