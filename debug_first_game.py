"""Debug why first game is taking so long."""
import time
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

print("Loading team boxscore...")
start = time.time()
df = pd.read_excel("data/team_boxscores/historical/2021-2022_NBA_Box_Score_Team-Stats.xlsx")
print(f"Loaded in {time.time() - start:.1f}s")

print("\nLoading player boxscore...")
start = time.time()
from transform_player_stats import load_player_data
player_df = load_player_data("2021-2022")
print(f"Loaded in {time.time() - start:.1f}s - {len(player_df)} records")

print("\nProcessing first game...")
start = time.time()

from prepare_data import normalize_columns, parse_venue_columns, build_game_record

df = normalize_columns(df)
venue_col = parse_venue_columns(df)

first_game_id = df['GAME-ID'].iloc[0]
first_game = df[df['GAME-ID'] == first_game_id]

print(f"First game ID: {first_game_id}")
print(f"Building game record with player data...")

record_start = time.time()
record = build_game_record(first_game, "2021-2022", venue_col, player_df)
record_time = time.time() - record_start

print(f"Game record built in {record_time:.1f}s")
print(f"Has players: {'players' in record}")
if 'players' in record:
    print(f"  Away players: {len(record['players']['A'])}")
    print(f"  Home players: {len(record['players']['H'])}")

total_time = time.time() - start
print(f"\nTotal time for first game: {total_time:.1f}s")
print(f"Estimated time for 1,200 games: {total_time * 1200 / 60:.1f} minutes")

