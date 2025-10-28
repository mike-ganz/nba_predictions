# Flow: minutes_missing_top2 (Raw Data → Training)

## **IMPORTANT FINDING: Player data is NOT currently included in training!**

When I inspected `data/games_train.jsonl`, the games have **no `players` field**. This means:
- `minutes_missing_top2` = **0.0** (default)
- `star_out` = **0** (default)
- `usage_share_top2` = **0.0** (default)
- `team_weighted_ts` = **0.53** (default league average)

These features are currently **placeholders with no predictive value**.

---

## How It WOULD Work (if player data were included)

### Step 1: Raw Data Sources
**Location:** Excel files with player boxscore data
- `sample_NBA-2024-2025-Player-BoxScore-Dataset.xlsx`
- Contains: player name, team, minutes, points, FG%, TS%, usage rate, etc.

### Step 2: Calculate Player Baselines
**File:** `transform_player_stats_optimized.py`
- **Function:** `get_player_stats_array(player_name, max_date, season)`
- **Output:** Player's historical baseline stats:
  - `baseline_minutes`: avg MPG from recent games (e.g., 35.2 for a starter)
  - `baseline_ts_pct`: true shooting % (e.g., 0.58 for efficient scorer)
  - `baseline_usage_rate`: usage rate (e.g., 0.28 for high-usage player)

**Example:**
```python
# Giannis in 2021-2022
baseline_minutes = 33.0      # averages 33 MPG
baseline_ts_pct = 0.583      # 58.3% TS
baseline_usage_rate = 0.32   # 32% usage
```

### Step 3: Get Projected Minutes (Injury Report)
**File:** `data/schema.py` - `PlayerAvailability` class
- **`projected_minutes`**: Expected minutes for THIS game
  - If player is healthy: `None` (defaults to baseline_minutes)
  - If player is questionable: e.g., 25.0 (reduced from 33.0 baseline)
  - If player is OUT: 0.0

**Example:**
```python
# Giannis is questionable with ankle soreness
PlayerAvailability(
    player_id="giannis_antetokounmpo",
    baseline_minutes=33.0,      # Historical average
    projected_minutes=25.0,     # Expected tonight (limited)
    baseline_ts_pct=0.583,
    baseline_usage_rate=0.32
)
```

### Step 4: Build GameRecord with Players
**File:** `prepare_data.py` (would need to be modified)
- **Currently:** Only team-level features are extracted
- **If player data added:** Build `GamePlayers` object with:
  - Top 8-10 rotation players for home team
  - Top 8-10 rotation players for away team
  - Each with `baseline_minutes`, `projected_minutes`, `baseline_ts_pct`, `baseline_usage_rate`

**Example GameRecord:**
```json
{
  "game_id": "2021-10-19-BKN-MIL",
  "teams": { ... },
  "market": { ... },
  "players": {
    "H": [
      {
        "player_id": "giannis_antetokounmpo",
        "baseline_minutes": 33.0,
        "projected_minutes": 25.0,  // ← INJURY IMPACT
        "baseline_ts_pct": 0.583,
        "baseline_usage_rate": 0.32
      },
      {
        "player_id": "khris_middleton",
        "baseline_minutes": 32.5,
        "projected_minutes": null,  // ← Healthy, use baseline
        "baseline_ts_pct": 0.570,
        "baseline_usage_rate": 0.28
      },
      // ... 6-8 more rotation players
    ],
    "A": [ ... ]  // Brooklyn players
  }
}
```

### Step 5: Compute Availability Features
**File:** `features/availability.py`
- **Function:** `compute_availability_features(players: List[PlayerAvailability])`

**Logic:**
1. **Sort players by baseline_minutes** (descending) → identify top 2 stars
2. **For each of top 2:**
   - `missing = max(0, baseline_minutes - projected_minutes)`
   - Sum up all missing minutes
   - If `projected_minutes < 0.001`, set `star_out = 1`

**Example Calculation:**
```python
# Milwaukee's top 2 players
players_sorted = [
    # 1. Giannis (top player)
    {
        "baseline_minutes": 33.0,
        "projected_minutes": 25.0,
        "baseline_usage_rate": 0.32
    },
    # 2. Khris Middleton (2nd player)
    {
        "baseline_minutes": 32.5,
        "projected_minutes": 32.5,  # Healthy
        "baseline_usage_rate": 0.28
    }
]

# Calculate missing minutes for top 2
missing_giannis = max(0, 33.0 - 25.0) = 8.0 minutes
missing_middleton = max(0, 32.5 - 32.5) = 0.0 minutes

# Final feature
minutes_missing_top2 = 8.0 + 0.0 = 8.0

# Giannis projected > 0, so star_out = 0
# If he were OUT (projected=0), star_out would = 1

# Usage of top 2
usage_share_top2 = 0.32 + 0.28 = 0.60
```

### Step 6: Build Features for Training
**File:** `features/builder.py`
- **Function:** `FeatureBuilder.build(record: GameRecord)`
- **Lines 51-55:**
```python
availability_home = AvailabilityFeatures(0.0, 0, 0.53, 0.0)  # Defaults
availability_away = AvailabilityFeatures(0.0, 0, 0.53, 0.0)

if record.players:  # ← Only if player data exists!
    availability_home = compute_availability_features(record.players.H)
    availability_away = compute_availability_features(record.players.A)
```

**Lines 72-74:** Add to feature vector
```python
x_home = {
    ...
    "minutes_missing_top2": availability_home.minutes_missing_top2,  # ← 8.0 in example
    "star_out": float(availability_home.star_out),                   # ← 0.0 in example
    "usage_share_top2": availability_home.usage_share_top2,          # ← 0.60 in example
}
```

### Step 7: Training
**File:** `train.py` → `training/dataset.py`
- Features are converted to numpy arrays
- `x_home` includes `minutes_missing_top2` as one of 9 features
- Ridge/ElasticNet learns the relationship:
  - Higher `minutes_missing_top2` → lower predicted score
  - `star_out=1` → significantly lower predicted score

---

## Current Reality Check

Running this command:
```bash
python -c "import json; g = json.loads(open('data/games_train.jsonl').readline()); print('players' in g)"
# Output: False
```

**Conclusion:**
- `prepare_data.py` currently does **NOT** load or process player data
- All availability features use **defaults** (zeros and league averages)
- The model has **learned nothing** about injury impact
- These features are dead weight taking up model capacity

---

## What Would Be Needed to Fix This

1. **Data Source:** Get injury reports or projected minutes for each game
   - Could scrape from RotoWire, ESPN, or NBA official injury reports
   - Or use actual minutes played (for historical data) as ground truth

2. **Modify `prepare_data.py`:**
   - Add player data loading
   - Match players to teams and games
   - Build `PlayerAvailability` objects
   - Add to `GameRecord.players`

3. **Ensure Baselines:**
   - Run `transform_player_stats_optimized.py` to compute baseline stats
   - Cache player baselines (MPG, TS%, usage) for quick lookup

4. **Regenerate Training Data:**
   - Re-run `prepare_data.py` with player data included
   - Features would now have real values instead of zeros

---

## Impact on Current Model

Since these 3 features (per team) are currently **all zeros/defaults**:
- They contribute **nothing** to predictions
- 6 features out of 23 total (26%) are wasted
- Model is essentially running on **17 effective features**
- This explains why we might be missing injury-driven mispricings in the market!

**Recommendation:** Either:
1. **Add player data** (significant effort, high impact)
2. **Remove these features** (quick fix, simplifies model)
3. **Use team-level rest/schedule features instead** (middle ground)

