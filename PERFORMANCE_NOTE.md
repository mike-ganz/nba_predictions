# Performance Note: Player Data Generation

## Expected Runtime

When running `prepare_data.py` with `--include-players`:

### Per Season:
- **Without player data**: ~30-60 seconds
- **With player data**: ~3-5 minutes

### Full Training Set (3 seasons):
- **Without player data**: ~2-3 minutes
- **With player data**: ~10-20 minutes

## Why It's Slower

For each game, the script:
1. Loads team boxscore data (fast)
2. **For each of ~20 players per game:**
   - Calls `calculate_player_stats(player_name, max_date=game_date)` 
   - This loads all player boxscore data for that season (cached after first call)
   - Filters to games before this date
   - Calculates rolling averages for MPG, TS%, usage rate
   - **This is cached to disk** (`data/cache/player_stats/`)

### Total Operations:
- ~1,230 games per season × 20 players = **~24,600 player lookups**
- First run: Most will hit disk cache (37K+ cached files)
- Subsequent runs: Even faster due to in-memory cache

## Optimization Applied

I added a **global in-memory cache** to `player_data_loader.py` that stores player baselines for the current season, so:
- First lookup for a player: Calls `calculate_player_stats()` → disk I/O
- Subsequent lookups: Returns from memory → instant

This should reduce runtime by **30-50%** compared to the original implementation.

## Progress Logging

The script now logs every 100 games:
```
[INFO] Processing data/team_boxscores/historical/2022-2023_NBA_Box_Score_Team-Stats.xlsx
[INFO]   Processed 100/1230 games...
[INFO]   Processed 200/1230 games...
...
```

## Tips for Faster Generation

1. **Use cached player stats**: The 37K+ cached files in `data/cache/player_stats/` speed things up significantly. Don't delete this cache!

2. **Generate one season at a time** if testing:
   ```bash
   python prepare_data.py \
     --team-boxscores-dir data/team_boxscores/historical \
     --output data/test_2022.jsonl \
     --seasons 2022-2023 \
     --include-players
   ```
   
3. **Skip player data for quick iterations**:
   ```bash
   # Without --include-players flag (30-60 seconds)
   python prepare_data.py \
     --team-boxscores-dir data/team_boxscores/historical \
     --output data/test_quick.jsonl \
     --seasons 2022-2023
   ```

4. **Use the test script** to verify before full run:
   ```bash
   python test_player_features.py  # Tests 1 season
   ```

## Bottlenecks

If it's taking longer than expected:

1. **Disk I/O**: Check if `data/cache/player_stats/` exists and has files
2. **Memory**: Processing uses ~1-2GB RAM with player data
3. **Excel loading**: `pd.read_excel()` is slow; consider converting to parquet if running frequently

## Future Optimization Ideas

1. **Parquet format**: Convert Excel files to parquet for 5-10x faster loading
2. **Parallel processing**: Process multiple games in parallel
3. **Pre-compute all baselines**: Build a master lookup table once per season
4. **SQLite database**: Store player baselines in a database for instant lookups

For now, **10-20 minutes for full training data** is acceptable for a one-time generation.

