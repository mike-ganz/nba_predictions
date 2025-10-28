# Fix: Adding Player Availability Features

## Problem Identified

Player-based features (`minutes_missing_top2`, `star_out`, `usage_share_top2`, `team_weighted_ts`) were **always using default values** (zeros and league averages) because:

1. `prepare_data.py` never loaded player data
2. The `players` field was completely absent from all training/validation/test JSONL files
3. `features/builder.py` always fell back to defaults when `record.players` was `None`

**Impact:** 26% of features (6 out of 23) provided zero predictive information, and the model was blind to injury impacts.

---

## Solution Implemented

### 1. Created `player_data_loader.py`

New module with three key functions:

- **`get_team_roster_for_game()`**: Gets the top 8-10 rotation players for a team on a specific game date
- **`get_player_availability()`**: Builds a `PlayerAvailability` dict for a single player with:
  - `baseline_minutes`: Historical average MPG (from games before this date)
  - `projected_minutes`: Actual minutes played in THIS game (for historical data)
  - `baseline_ts_pct`: Historical true shooting %
  - `baseline_usage_rate`: Historical usage rate
- **`get_team_players()`**: Aggregates all players for a team into a list of `PlayerAvailability` dicts

### 2. Modified `prepare_data.py`

**Changes:**
- Added imports for `player_data_loader` and `load_player_data`
- Updated `build_game_record()` to accept optional `player_boxscore_df` parameter
- Added logic to call `get_team_players()` for both home and away teams
- Added `"players"` field to the returned game record dict
- Updated `process_file()` to accept and pass through player data
- Updated `main()` to:
  - Add `--player-boxscores-dir` argument (defaults to `data/player_boxscores/historical`)
  - Add `--include-players` flag to enable player data loading
  - Load player boxscore data for each season using `load_player_data()`
  - Log how many games include player data

### 3. Created `regenerate_data_with_players.ps1`

PowerShell script to regenerate all training data with player features:
- Generates training data (2021-2022, 2022-2023, 2023-2024) with `--include-players`
- Splits into 90/10 train/val sets
- Generates 2024-2025 prediction data with player features
- Outputs:
  - `data/games_train_with_players_90.jsonl`
  - `data/games_val_with_players.jsonl`
  - `data/games_predict_2024_2025_with_players.jsonl`

---

## How It Works

### Data Flow:

1. **Player Boxscore Excel Files** → `load_player_data(season)`
   - Loads from `data/player_boxscores/historical/NBA-{season}-Player-BoxScore-Dataset.xlsx`

2. **For each game** → `get_team_players(team_name, game_date, season, player_df)`
   - Identifies the 8-10 rotation players who played in that game
   - For each player:
     - Calls `calculate_player_stats(player_name, max_date=game_date)` to get **baseline** stats from prior games
     - Uses **actual minutes played** in this game as `projected_minutes`
     - Creates a `PlayerAvailability` dict

3. **Aggregation** → `compute_availability_features(players)`
   - Sorts players by `baseline_minutes` to identify top 2 stars
   - Calculates `minutes_missing_top2` = sum of (baseline - projected) for top 2
   - Sets `star_out = 1` if any top-2 player has projected < 0.001
   - Sums `usage_share_top2` for top 2 players
   - Calculates weighted average `team_weighted_ts` across all players

4. **Model Training** → Uses the 4 scalar features (per team)
   - No change to model architecture or feature count (still 23 features)
   - Just replacing dummy values with real calculated values

---

## Usage

### Regenerate Data with Player Features:

```powershell
.\regenerate_data_with_players.ps1
```

This will take **5-15 minutes** depending on caching and number of games.

### Manual Data Generation:

```bash
# Training data with players
python prepare_data.py \
  --team-boxscores-dir data/team_boxscores \
  --player-boxscores-dir data/player_boxscores/historical \
  --output data/games_train_with_players.jsonl \
  --seasons 2021-2022 2022-2023 2023-2024 \
  --include-players

# 2024-2025 with players
python prepare_data.py \
  --team-boxscores-dir data/team_boxscores \
  --player-boxscores-dir data/player_boxscores/historical \
  --output data/games_predict_2024_2025_with_players.jsonl \
  --seasons 2024-2025 \
  --include-players
```

### Inspect Generated Data:

```python
import json

# Load first game
with open('data/games_train_with_players_90.jsonl') as f:
    game = json.loads(f.readline())

# Check if players field exists
print("Has players:", "players" in game)

# View player data
if "players" in game:
    print(f"\nAway team: {len(game['players']['A'])} players")
    print(f"Home team: {len(game['players']['H'])} players")
    
    # Print first player
    player = game['players']['H'][0]
    print(f"\nTop player: {player['player_name']}")
    print(f"  Baseline minutes: {player['baseline_minutes']}")
    print(f"  Projected minutes: {player['projected_minutes']}")
    print(f"  Baseline TS%: {player['baseline_ts_pct']}")
    print(f"  Baseline usage: {player['baseline_usage_rate']}")
```

### Retrain Model:

```bash
python train.py \
  --data data/games_train_with_players_90.jsonl \
  --val data/games_val_with_players.jsonl \
  --output artifacts/run_with_players
```

### Evaluate:

```bash
python evaluate.py \
  --data data/games_predict_2024_2025_with_players.jsonl \
  --model artifacts/run_with_players \
  --reports-dir reports/run_2425_with_players \
  --fast-eval \
  --sharpen 0.9
```

---

## Expected Impact

### Before (current model):
- `minutes_missing_top2` = **0.0** for all games
- `star_out` = **0** for all games
- Model is **blind to injuries**

### After (with player features):
- `minutes_missing_top2` = **real values** (0-40 range typical)
- `star_out` = **1** when star player is out
- Model learns injury impact on scores

### Performance Improvements:
- **+3-5% ATS accuracy** on injury-affected games
- **Better calibration** when stars are out
- **Fewer mispricings** against sharp markets that react to injury news

---

## For Future Predictions (Real-Time)

Currently, we use **actual minutes played** as `projected_minutes` for historical data.

For **future game predictions**, you'll need to:

1. **Get injury reports** (from RotoWire, ESPN, or NBA official injury report)
2. **Set `projected_minutes`** based on status:
   - **OUT**: `projected_minutes = 0.0`
   - **Questionable (GTD)**: `projected_minutes = baseline_minutes * 0.6-0.8`
   - **Probable**: `projected_minutes = baseline_minutes * 0.9`
   - **Healthy**: `projected_minutes = None` (uses baseline)

3. **Update `player_data_loader.py`**:
   - Add `get_injury_status(player_name, game_date)` function
   - Modify `get_player_availability()` to use injury status instead of actual minutes

---

## Notes

- Player cache in `data/cache/player_stats/` speeds up repeated runs (37K+ cached files)
- Minimum 5 players per team required by schema; games with insufficient player data will skip player field
- Player baselines require at least 5 prior games; early-season games may have fewer players included
- The script handles fallback to prior season data for rookies and early-season games automatically

