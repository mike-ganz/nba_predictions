"""Test performance on mid-season game (after cache is warm)."""
import time
import pandas as pd
from prepare_data import normalize_columns, parse_venue_columns, build_game_record
from transform_player_stats import load_player_data
from player_data_loader import _load_prior_season_cache

print("Loading data...")
df = pd.read_excel("data/team_boxscores/historical/2021-2022_NBA_Box_Score_Team-Stats.xlsx")
player_df = load_player_data("2021-2022")

# Pre-load prior season cache
_load_prior_season_cache("2021-2022")

df = normalize_columns(df)
venue_col = parse_venue_columns(df)

# Process first 20 games to warm up cache
print("Warming up cache with first 20 games...")
start = time.time()
for game_id in df['GAME-ID'].unique()[:20]:
    game = df[df['GAME-ID'] == game_id]
    build_game_record(game, "2021-2022", venue_col, player_df)
warmup_time = time.time() - start
print(f"Warmup: {warmup_time:.1f}s ({warmup_time/20:.2f}s per game)")

# Now test games 21-30 (cache should be warm)
print("\nTesting games 21-30 with warm cache...")
start = time.time()
for game_id in df['GAME-ID'].unique()[20:30]:
    game = df[df['GAME-ID'] == game_id]
    build_game_record(game, "2021-2022", venue_col, player_df)
steady_time = time.time() - start
print(f"Steady state: {steady_time:.1f}s ({steady_time/10:.2f}s per game)")

print(f"\nEstimated time for 3,700 games: {(steady_time/10 * 3700) / 60:.1f} minutes")

