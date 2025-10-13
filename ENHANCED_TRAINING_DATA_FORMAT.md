# Enhanced Training Data Format - Complete Specification

## Overview

This document describes the enhanced training data format for NBA play-by-play prediction. The format uses compact arrays with enhanced player statistics (12 values) and team statistics (8 values).

---

## Training Record Structure

### Compact Format (Gemini)

```json
{
  "A": "LAL",              // Away team abbreviation
  "H": "GSW",              // Home team abbreviation
  "as": [119.5, 116.2, 101.3, 0.356, 0.305, 0.164, 0.677, 1],  // Away team stats (8 values)
  "hs": [114.2, 112.8, 98.5, 0.412, 0.28, 0.22, 0.694, 2],     // Home team stats (8 values)
  "ap": [                  // Away players (array of 13-value arrays)
    ["LeBron James", 0.434, 0.027, 0.261, 0.279, 0.459, 0.575, 11.6, 1.5, 0.6, 36.0, 28.0, 21, 2],
    ["Anthony Davis", 0.480, 0.020, 0.088, 0.412, 0.598, 0.667, 4.2, 1.2, 2.8, 35.0, 29.0, 12, 1],
    ...
  ],
  "hp": [                  // Home players (array of 13-value arrays)
    ["Stephen Curry", 0.183, 0.065, 0.457, 0.295, 0.545, 0.923, 10.8, 1.8, 0.3, 34.0, 31.0, 15, 1],
    ...
  ],
  "l": [                   // Lineup lookup
    {"A": [0,1,2,3,4], "H": [0,1,2,3,4]},  // Lineup 0
    {"A": [0,1,2,5,6], "H": [0,1,2,3,7]},  // Lineup 1
    ...
  ],
  "p": [                   // Plays (recent context)
    [1, 697, [0,0], 0, ["A",0], "made2"],
    [1, 685, [2,0], 1, ["H",2], "miss3"],
    ...
  ],
  "y": [1, 678, [2,2], 1, ["A",1], "made2"]  // Label (next play)
}
```

---

## Team Stats Array (8 values)

### Format
```
[OEFF, DEFF, PACE, 3PAr, FTr, ORr, ASTr, REST]
```

### Field Descriptions

| Index | Field | Description | Range | Notes |
|-------|-------|-------------|-------|-------|
| 0 | **OEFF** | Offensive Efficiency (points per 100 possessions) | 90-130 | Rolling 10-game average |
| 1 | **DEFF** | Defensive Efficiency (points allowed per 100 poss) | 90-130 | Rolling 10-game average |
| 2 | **PACE** | Pace (possessions per 48 minutes) | 80-110 | Rolling 10-game average |
| 3 | **3PAr** | 3-Point Attempt Rate (3PA / FGA) | 0.25-0.50 | Rolling 10-game average |
| 4 | **FTr** | Free Throw Rate (FTA / (FGA + FTA)) | 0.15-0.35 | Rolling 10-game average |
| 5 | **ORr** | Offensive Rebound Rate (OR / (OR + Opp DR)) | 0.15-0.35 | Rolling 10-game average |
| 6 | **ASTr** | Assist Rate (AST / FGM) | 0.50-0.75 | Rolling 10-game average |
| 7 | **REST** | Rest days since last game | 0-3 | 0=back-to-back, 1=1 day rest, 2=2 days, 3=3+ days |

### Interpretation Guide

**Offensive Efficiency (OEFF)**
- **< 105**: Poor offense
- **105-110**: Below average
- **110-115**: Average
- **115-120**: Above average
- **> 120**: Elite offense

**Defensive Efficiency (DEFF)**
- **< 108**: Elite defense
- **108-112**: Above average
- **112-115**: Average
- **115-118**: Below average
- **> 118**: Poor defense

**Pace (PACE)**
- **< 95**: Slow (grind-it-out style)
- **95-100**: Average
- **> 100**: Fast (run-and-gun style)

**3-Point Attempt Rate (3PAr)**
- **< 0.30**: Low 3-point volume (traditional)
- **0.30-0.38**: Average
- **> 0.38**: High 3-point volume (modern style)

**Free Throw Rate (FTr)**
- **< 0.20**: Limited free throw attempts
- **0.20-0.28**: Average
- **> 0.28**: Aggressive attacking (gets to line often)

**Offensive Rebound Rate (ORr)**
- **< 0.22**: Poor offensive rebounding
- **0.22-0.28**: Average
- **> 0.28**: Strong offensive rebounding

**Assist Rate (ASTr)**
- **< 0.60**: Low ball movement (iso-heavy)
- **0.60-0.68**: Average
- **> 0.68**: High ball movement (motion offense)

---

## Player Stats Array (13 values + fouls placeholder)

### Format
```
[NAME, RIM%, C3%, NC3%, MID%, A2%, A3%, AST/100, STL/100, BLK/100, MPG, USG, CLUSTER, FOULS]
```

### Field Descriptions

| Index | Field | Description | Range | Notes |
|-------|-------|-------------|-------|-------|
| 0 | **NAME** | Player full name | string | e.g., "LeBron James" |
| 1 | **RIM%** | Rim Attempt Rate (rim FGA / total FGA) | 0.0-0.7 | Rolling 20-game average |
| 2 | **C3%** | Corner 3 Rate (corner 3PA / total FGA) | 0.0-0.15 | Rolling 20-game average |
| 3 | **NC3%** | Non-Corner 3 Rate (above-break 3PA / total FGA) | 0.0-0.5 | Rolling 20-game average |
| 4 | **MID%** | Mid-Range Rate (mid-range FGA / total FGA) | 0.0-0.5 | Rolling 20-game average |
| 5 | **A2%** | Assisted 2PT Rate (assisted 2PM / total 2PM) | 0.0-1.0 | Rolling 20-game average |
| 6 | **A3%** | Assisted 3PT Rate (assisted 3PM / total 3PM) | 0.0-1.0 | Rolling 20-game average |
| 7 | **AST/100** | Assists per 100 possessions | 0.0-20.0 | Rolling 20-game average |
| 8 | **STL/100** | Steals per 100 possessions | 0.0-5.0 | Rolling 20-game average |
| 9 | **BLK/100** | Blocks per 100 possessions | 0.0-5.0 | Rolling 20-game average |
| 10 | **MPG** | Minutes per game | 10.0-40.0 | Season average |
| 11 | **USG** | Usage Rate (% of team possessions used) | 10.0-35.0 | Season average |
| 12 | **CLUSTER** | Player archetype cluster ID | 0-21 or -1 | -1 = unknown |
| 13 | **FOULS** | Personal fouls in this game context | 0-6 | Updated per context |

### Shot Profile Interpretation (Indices 1-4)

The shot profile fields sum to approximately 1.0 and describe where a player takes their shots:

**Guards (example: Stephen Curry)**
```
rim=0.18, c3=0.07, nc3=0.46, mid=0.29
```
- Low rim rate (18%)
- Very high non-corner 3 rate (46%) - elite shooter
- Moderate mid-range (29%)
- Some corner 3s (7%)

**Wings (example: LeBron James)**
```
rim=0.43, c3=0.03, nc3=0.26, mid=0.28
```
- High rim rate (43%) - attacks basket
- Moderate non-corner 3 rate (26%)
- Moderate mid-range (28%)
- Few corner 3s (3%) - not a spot-up shooter

**Bigs (example: Rudy Gobert)**
```
rim=0.94, c3=0.00, nc3=0.01, mid=0.05
```
- Extremely high rim rate (94%) - rim runner
- Virtually no 3-point attempts (1%)
- Minimal mid-range (5%)

### Creation Interpretation (Indices 5-7)

**Assisted 2PT Rate (A2%)**
- **< 0.30**: Elite self-creator (e.g., Luka Doncic)
- **0.30-0.60**: Good self-creator
- **0.60-0.80**: Average
- **> 0.80**: Spot-up/finisher only

**Assisted 3PT Rate (A3%)**
- **< 0.70**: Creates own 3s (pull-ups)
- **0.70-0.90**: Mix of creation and spot-up
- **> 0.90**: Pure catch-and-shoot

**Assists per 100 (AST/100)**
- **< 3.0**: Non-playmaker
- **3.0-8.0**: Average playmaking
- **8.0-15.0**: Good playmaker
- **> 15.0**: Elite playmaker (e.g., Chris Paul)

### Defense Interpretation (Indices 8-9)

**Steals per 100 (STL/100)**
- **< 1.0**: Below average
- **1.0-2.0**: Average
- **2.0-3.0**: Above average
- **> 3.0**: Elite (e.g., Alex Caruso)

**Blocks per 100 (BLK/100)**
- **< 0.5**: Guards/non-rim protectors
- **0.5-1.5**: Average rim protection
- **1.5-3.0**: Good rim protection
- **> 3.0**: Elite rim protector (e.g., Rudy Gobert)

---

## Player Archetype Clusters (Index 12)

### Cluster IDs and Descriptions

| ID | Archetype Name | Description | Example Players |
|----|----------------|-------------|-----------------|
| 0 | Elite Ball Handlers | High usage, elite playmaking, create own shots | Luka Doncic, Trae Young |
| 1 | Perimeter Stoppers | 3&D wings, low usage, good defense | Alex Caruso, Marcus Smart |
| 3 | Floor Generals | High assists, moderate usage, distributes | Chris Paul, Tyus Jones |
| 4 | Rim Runners | High rim rate, assisted shots, rim protection | Rudy Gobert, Jarrett Allen |
| 6 | Secondary Creators | Moderate playmaking and scoring | Malcolm Brogdon, Caris LeVert |
| 7 | Spot-Up Shooters | High assisted 3PT%, low usage | Duncan Robinson, Joe Harris |
| 11 | Defensive Anchors | High blocks, rebounds, low usage | Robert Williams, Daniel Gafford |
| 12 | Mid-Range Maestros | High mid-range rate, versatile | DeMar DeRozan, Jimmy Butler |
| 13 | Offensive Bigs | Scoring bigs, moderate usage | Julius Randle, Domantas Sabonis |
| 14 | 3-Point Specialists | Very high 3PT rate, assisted | Buddy Hield, Malik Beasley |
| 15 | Volume Scorers | High usage, scoring-first guards/wings | Donovan Mitchell, Zach LaVine |
| 20 | Versatile Wings | Balanced shot profile, good creation | Kawhi Leonard, Paul George |
| 21 | Point-Forwards | Elite passing bigs/forwards, versatile | LeBron James, Nikola Jokic, Giannis |
| -1 | Unknown | Insufficient data or not in cluster database | Rookies, G-League call-ups |

### Using Cluster Information

Clusters help the model understand player archetypes and predict behavior:

1. **Shot Selection**: A player in cluster 4 (Rim Runner) is unlikely to take a 3-pointer
2. **Playmaking**: A player in cluster 21 (Point-Forward) is likely to assist when doubled
3. **Defensive Impact**: A player in cluster 11 (Defensive Anchor) is more likely to block shots at the rim
4. **Usage Patterns**: A player in cluster 7 (Spot-Up Shooter) will have lower usage in crunch time

---

## Lineup Array

### Format
```json
{"A": [0, 1, 2, 3, 4], "H": [0, 1, 2, 3, 4]}
```

- **"A"**: Away team lineup (player indices 0-based)
- **"H"**: Home team lineup (player indices 0-based)
- Players are sorted by MPG (highest first), so index 0 is typically the highest-minute player

---

## Play Array

### Format
```
[QUARTER, TIME_SECONDS, SCORE, LINEUP_ID, ACTOR, EVENT]
```

| Index | Field | Description | Example |
|-------|-------|-------------|---------|
| 0 | **QUARTER** | Quarter number (1-4, 5+ for OT) | `1` |
| 1 | **TIME_SECONDS** | Seconds remaining in quarter | `697` (11:37) |
| 2 | **SCORE** | Score delta from previous play | `[0, 2]` (home scored 2) |
| 3 | **LINEUP_ID** | Index into lineup lookup array | `0` |
| 4 | **ACTOR** | [Team, PlayerIndex] who made the play | `["A", 0]` |
| 5 | **EVENT** | Event code | `"made2"` |

---

## Event Codes

### Shooting Events
- `made2` - Made 2-pointer
- `miss2` - Missed 2-pointer
- `made3` - Made 3-pointer
- `miss3` - Missed 3-pointer
- `mft` - Made free throw
- `xft` - Missed free throw

### Rebounding Events
- `o_reb` - Offensive rebound
- `d_reb` - Defensive rebound

### Playmaking Events
- `ast` - Assist
- `tov` - Turnover

### Defensive Events
- `stl` - Steal
- `blk` - Block
- `foul` - Personal foul

### Special Events
- `sub` - Substitution
- `jump` - Jump ball
- `tech` - Technical foul
- `end_q` - End of quarter

---

## Rolling Windows

### Player Stats (20 games)
- All PBP-derived stats use a 20-game rolling window
- Captures recent form and role changes
- Falls back to all available games if < 20

### Team Stats (10 games)
- All team stats use a 10-game rolling window
- Captures recent system changes and trends
- Smaller window because team stats are more stable

### Benefits
1. **Recency**: Recent performance is more predictive than season averages
2. **Injury Recovery**: Post-injury stats reflect current ability
3. **Role Changes**: New roles are captured quickly
4. **Trade Impact**: New team context is reflected immediately

---

## Data Quality

### Cache Hit Rates
- **Player stats**: ~95% (cached per player/date/season)
- **Team stats**: ~98% (cached per team/date/season)
- **Cluster assignments**: 100% (loaded once)

### Fallback Handling
- **Missing PBP data**: Use default balanced shot profile
- **Early season** (< 20 games): Use all available games
- **No cluster**: Assign cluster ID = -1
- **Missing team data**: Use league average stats

---

## Training Dataset Statistics

### 2023-2024 Season
- **Total games**: 1,319
- **Total plays**: ~607,000
- **Training examples**: ~580,000 (remaining_plays mode)
- **Players with stats**: 412
- **Unique clusters**: 22 (0-21)

### Token Usage
- **Per training example**: ~272 tokens
- **Full season**: ~158M tokens
- **Training cost**: ~$316 @ $2/M tokens

---

## System Prompt Guidance

### For Model Training

Include this in your system prompt:

```
You are an NBA play-by-play prediction model. You receive game context with:

TEAM STATS (8 values per team):
- [OEFF, DEFF, PACE, 3PAr, FTr, ORr, ASTr, REST]
- OEFF/DEFF are per 100 possessions (90-130 range)
- Rates are 0.0-1.0 (e.g., 0.35 = 35%)
- REST is 0-3 (days since last game)

PLAYER STATS (13 values per player):
- [NAME, RIM%, C3%, NC3%, MID%, A2%, A3%, AST/100, STL/100, BLK/100, MPG, USG, CLUSTER, FOULS]
- Shot profile (RIM%, C3%, NC3%, MID%) sums to ~1.0
- Assisted rates (A2%, A3%) range 0.0-1.0
- Per-100 stats (AST/100, STL/100, BLK/100) are normalized to possessions
- CLUSTER identifies player archetype (0-21 or -1 for unknown)

Use these stats to understand:
- Team offensive/defensive systems and pace
- Player shot selection tendencies and creation ability
- Defensive capabilities (steals, blocks)
- Player roles and archetypes
- Current form (stats use 20-game rolling windows)

Predict the next play based on game context, score, time, lineup, and recent plays.
```

---

## Example Predictions

### High-Usage Star with Ball (Cluster 0)
```
Player: Luka Doncic
Stats: rim=0.32, nc3=0.38, mid=0.25, a2%=0.28, ast/100=15.2, usg=36.0, cluster=0
Context: Late in shot clock, isolation
Prediction: High likelihood of pull-up 3 (nc3) or drive (rim) with potential assist
```

### Rim Runner in Pick-and-Roll (Cluster 4)
```
Player: Rudy Gobert
Stats: rim=0.94, a2%=0.98, blk/100=3.2, cluster=4
Context: Screen set, roller to basket
Prediction: Very high likelihood of assisted rim attempt or offensive rebound
```

### Spot-Up Shooter (Cluster 7)
```
Player: Duncan Robinson
Stats: c3=0.08, nc3=0.64, a3%=0.96, cluster=7
Context: Off-ball, catch-and-shoot opportunity
Prediction: Extremely high likelihood of assisted 3-pointer (corner or above-break)
```

---

## Version History

- **v1.0** (2024-01-15): Initial compact format with 4-value team stats, 8-value player stats (PCA-based)
- **v2.0** (Current): Enhanced format with 8-value team stats, 12-value player stats (PBP-based), 22 player archetypes

---

## Contact

For questions about the training data format, contact the data engineering team or refer to:
- `ENHANCED_STATS_INTEGRATION_SUMMARY.md` - Implementation details
- `PLAYER_STATS_ROLLING_WINDOW_UPDATE.md` - Rolling window methodology
- `player_archetype_clustering.py` - Cluster generation code

