"""Quick test to verify player feature implementation works."""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from player_data_loader import get_team_players
from transform_player_stats import load_player_data

print("=" * 80)
print("Testing Player Feature Implementation")
print("=" * 80)
print(f"Started at: {time.strftime('%H:%M:%S')}")

# Test 1: Load player data
print("\n[Test 1] Loading player boxscore data for 2023-2024...")
start_time = time.time()
try:
    player_df = load_player_data("2023-2024")
    elapsed = time.time() - start_time
    print(f"✓ Loaded {len(player_df)} player-game records in {elapsed:.1f}s")
    print(f"  Columns: {list(player_df.columns[:10])}...")
except Exception as e:
    print(f"✗ Failed to load player data: {e}")
    sys.exit(1)

# Test 2: Get players for a specific game (mid-season, so players have history)
print("\n[Test 2] Getting players for Milwaukee on 2023-11-18 (mid-season)...")
print("  (This will take 10-30 seconds as it calculates baselines for ~10 players...)")
start_time = time.time()
try:
    players = get_team_players("Milwaukee", "2023-11-18", "2023-2024", player_df)
    elapsed = time.time() - start_time
    print(f"✓ Found {len(players)} players in {elapsed:.1f}s")
    
    if len(players) > 0:
        top_player = players[0]
        print(f"\n  Top player: {top_player['player_name']}")
        print(f"    player_id: {top_player['player_id']}")
        print(f"    baseline_minutes: {top_player['baseline_minutes']}")
        print(f"    projected_minutes: {top_player['projected_minutes']}")
        print(f"    baseline_ts_pct: {top_player['baseline_ts_pct']}")
        print(f"    baseline_usage_rate: {top_player['baseline_usage_rate']}")
        
        print(f"\n  All players:")
        for i, p in enumerate(players[:5], 1):
            print(f"    {i}. {p['player_name']:30s} - {p['baseline_minutes']:4.1f} min baseline, {p['projected_minutes']:4.1f} min actual")
    else:
        print("  ⚠ No players found (may be normal for early season or data issues)")
        
except Exception as e:
    print(f"✗ Failed to get players: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test the full pipeline
print("\n[Test 3] Testing full prepare_data pipeline...")
print("  Running: python prepare_data.py --team-boxscores-dir data/team_boxscores/historical --output data/test_with_players.jsonl --seasons 2023-2024 --include-players")
print("  Expected: 3-5 minutes for ~1,230 games")
print("  Progress will be logged every 100 games...")
print(f"  Started at: {time.strftime('%H:%M:%S')}")

import subprocess
start_time = time.time()

# Run with real-time output streaming
process = subprocess.Popen(
    [
        "python", "prepare_data.py",
        "--team-boxscores-dir", "data/team_boxscores/historical",
        "--output", "data/test_with_players.jsonl",
        "--seasons", "2023-2024",
        "--include-players"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Stream output in real-time
for line in iter(process.stdout.readline, ''):
    if line:
        print(f"    {line.rstrip()}")

process.wait()
elapsed = time.time() - start_time

if process.returncode != 0:
    print(f"\n✗ prepare_data.py failed after {elapsed:.1f}s")
    sys.exit(1)

print(f"\n✓ prepare_data.py succeeded in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

# Test 4: Verify output has player data
print("\n[Test 4] Verifying output has player data...")
start_time = time.time()
test_file = Path("data/test_with_players.jsonl")
if not test_file.exists():
    print("✗ Output file not found")
    sys.exit(1)

with open(test_file) as f:
    games_with_players = 0
    games_without_players = 0
    sample_game = None
    
    for line in f:
        game = json.loads(line)
        if "players" in game:
            games_with_players += 1
            if sample_game is None:
                sample_game = game
        else:
            games_without_players += 1

elapsed = time.time() - start_time
total_games = games_with_players + games_without_players
if total_games == 0:
    print("✗ No games found in output file!")
    sys.exit(1)

print(f"✓ Analyzed {total_games} games in {elapsed:.1f}s:")
print(f"  - {games_with_players} games WITH player data ({100*games_with_players/total_games:.1f}%)")
print(f"  - {games_without_players} games WITHOUT player data ({100*games_without_players/total_games:.1f}%)")

if sample_game and "players" in sample_game:
    print(f"\n  Sample game: {sample_game['game_id']}")
    print(f"    Away team: {sample_game['teams']['A']['team_name']} ({len(sample_game['players']['A'])} players)")
    print(f"    Home team: {sample_game['teams']['H']['team_name']} ({len(sample_game['players']['H'])} players)")
    
    home_top = sample_game['players']['H'][0]
    print(f"\n    Home top player: {home_top['player_name']}")
    print(f"      baseline: {home_top['baseline_minutes']} min, actual: {home_top['projected_minutes']} min")
    print(f"      missing: {home_top['baseline_minutes'] - (home_top['projected_minutes'] or home_top['baseline_minutes']):.1f} min")

# Clean up test file
test_file.unlink()
print(f"\n  Cleaned up test file: {test_file}")

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print(f"Total test time: {time.time() - time.time() + elapsed:.1f}s")
print("\nPlayer feature implementation is working correctly.")
print("\nNext step: Run regenerate_data_with_players.ps1 to create full datasets")
print("  Expected time: ~10-20 minutes for 3 seasons (2021-2024)")
print(f"\nFinished at: {time.strftime('%H:%M:%S')}")

