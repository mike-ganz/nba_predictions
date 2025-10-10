# Implementation Plan: Play-by-Play Player Stats

## 📋 Overview
Extract advanced player statistics from play-by-play data to supplement box score stats.

**Target Stats:** 20+ new metrics across 3 categories
**Data Source:** `[10-24-2023]-[06-17-2024]-combined-stats.csv` (607K plays)
**Architecture:** New module `player_stats_from_pbp.py` with caching system

---

## PHASE 1: POSSESSION LOGIC ✅ APPROVED

### 1.1 Possession Definition Logic

A possession ends when:

**A. Made Shot (check for offensive rebound)**
```python
if event_type == 'shot' AND result == 'made':
    # Look ahead to next play
    if next_play.event_type == 'rebound' AND next_play.type == 'rebound offensive' AND next_play.team == current_team:
        # Possession continues (offensive rebound)
        continue_possession = True
    else:
        # Possession ends, switches to opponent
        end_possession()
```

**B. Defensive Rebound**
```python
if event_type == 'rebound' AND type == 'rebound defensive':
    end_possession()
    switch_possession_to(rebounding_team)
```

**C. Turnover**
```python
if event_type == 'turnover':
    end_possession()
    switch_possession_to(opponent)
```

**D. Offensive Foul (Special Case per User)**
```python
if event_type == 'foul' AND type IN ['offensive foul', 'offensive charge', 'offensive']:
    # Check if next play is also a turnover (same action)
    if next_play.event_type == 'turnover' AND next_play.play_id == current_play.play_id + 1:
        # Skip the turnover, it's the same action
        skip_next_play = True
    end_possession()
    switch_possession_to(opponent)
```

**E. Dead Ball Violations**
```python
if event_type == 'violation' AND type IN [
    'shot clock', '8-second violation', 'backcourt', 'offensive goaltending'
]:
    end_possession()
    switch_possession_to(opponent)
```

**F. End of Period**
```python
if event_type == 'end of period':
    end_possession()
    # Next possession determined by jump ball or possession arrow
```

**G. Team Rebounds (DO NOT count)**
```python
if type == 'team rebound':
    # Do not end possession, already counted with the missed shot
    pass
```

**H. Technical Fouls (DO NOT affect possession)**
```python
if event_type == 'technical foul':
    # Possession continues for original team
    # Just note the technical FT
    pass
```

### 1.2 Possession Attribution

```python
def count_possessions(pbp_df):
    """
    Count possessions for each player based on time on court.
    
    Returns:
        dict: {(player_name, game_id): possession_count}
    """
    possessions = {}
    
    for game_id in pbp_df['game_id'].unique():
        game_plays = pbp_df[pbp_df['game_id'] == game_id].sort_values('play_id')
        
        current_possession_team = None
        possession_start_play = None
        
        for idx, play in game_plays.iterrows():
            # Determine possession team
            if play['event_type'] == 'shot':
                current_possession_team = play['team']
            elif play['possession'] is not None:
                current_possession_team = play['possession']
            
            # Check for possession-ending events
            if is_possession_ending(play, next_play):
                # Credit possession to all 5 players on court
                if play['team'] == play['home']:
                    players_on_court = [play['h1'], play['h2'], play['h3'], play['h4'], play['h5']]
                else:
                    players_on_court = [play['a1'], play['a2'], play['a3'], play['a4'], play['a5']]
                
                for player in players_on_court:
                    if pd.notna(player):
                        key = (player, game_id)
                        possessions[key] = possessions.get(key, 0) + 1
    
    return possessions
```

---

## PHASE 2: COORDINATE SYSTEM & SHOT CLASSIFICATION

### 2.1 Court Coordinate System (from diagram)

```python
# Court dimensions
COURT_WIDTH = 50  # feet
COURT_LENGTH = 94  # feet

# Key coordinates from diagram
# Bottom basket at y=25.25, top basket at ~y=68.75
BASKET_BOTTOM_Y = 25.25
BASKET_TOP_Y = 94 - 25.25  # 68.75

# Center court
CENTER_X = 25
CENTER_Y = 47

# Corner 3 boundaries
# Corner 3s are near x=0 or x=50, within certain y range
CORNER_3_X_MIN = 3  # Within 3 feet of sideline
CORNER_3_X_MAX = 47  # Within 3 feet of sideline
CORNER_3_Y_RANGE = 14  # Within 14 feet of baseline (22 ft from basket - 8 ft restricted)

# 3-point line distance: 22 feet from basket (23.75 at corners, 22 at top of key)
THREE_POINT_DISTANCE = 22
```

### 2.2 Shot Distance & Location Classification

```python
def calculate_shot_distance(x, y, shooting_end):
    """
    Calculate distance from basket.
    
    Args:
        x, y: Shot coordinates
        shooting_end: 'bottom' or 'top' (which basket)
    
    Returns:
        distance in feet
    """
    basket_x = CENTER_X  # 25
    basket_y = BASKET_BOTTOM_Y if shooting_end == 'bottom' else BASKET_TOP_Y
    
    distance = np.sqrt((x - basket_x)**2 + (y - basket_y)**2)
    return distance

def classify_shot_location(x, y, distance, shot_type):
    """
    Classify shot into: rim, short_mid, long_mid, corner_3, non_corner_3
    """
    # 3-pointers
    if '3pt' in shot_type or distance >= THREE_POINT_DISTANCE:
        # Check if corner 3
        if (x <= CORNER_3_X_MIN or x >= CORNER_3_X_MAX):
            # Near sideline, check if within corner range from baseline
            if (y <= BASKET_BOTTOM_Y + CORNER_3_Y_RANGE or 
                y >= BASKET_TOP_Y - CORNER_3_Y_RANGE):
                return 'corner_3'
        return 'non_corner_3'
    
    # Rim shots (within 4 feet)
    if distance <= 4:
        return 'rim'
    
    # Short mid-range (4-16 feet)
    if 4 < distance <= 16:
        return 'short_mid'
    
    # Long mid-range (16-22 feet)
    if 16 < distance < THREE_POINT_DISTANCE:
        return 'long_mid'
    
    return 'unknown'
```

### 2.3 Rim Attempt Classification

```python
RIM_SHOT_TYPES = {
    'dunk',
    'cutting dunk shot',
    'driving dunk',
    'driving reverse dunk shot',
    'layup',
    'driving layup',
    'cutting layup shot',
    'driving reverse layup',
    'cutting finger roll layup shot',
    'hook shot',  # Most hooks are close
    'hook bank shot'
}

def is_rim_attempt(shot_type, distance):
    """Check if shot is at rim."""
    return shot_type in RIM_SHOT_TYPES or distance <= 4
```

### 2.4 Pull-up vs Catch-and-Shoot Classification

```python
# Per user answers: both 'floating' and 'running' are pull-ups
PULLUP_3_TYPES = {
    '3pt pullup jump shot',
    '3pt running pull-up jump shot',
    '3pt step back jump shot',
    '3pt fadeaway jumper',
    '3pt turnaround jump shot',
    '3pt turnaround fadeaway',
    '3pt floating jump shot',  # User confirmed: pull-up
    '3pt running jump shot',  # User confirmed: pull-up
    '3pt step back bank jump shot',
    '3pt driving floating bank jump shot'
}

CATCH_SHOOT_3_TYPES = {
    '3pt jump shot',  # Most common (62K)
    '3pt jump bank shot'
}

def classify_3pt_shot(shot_type, assist):
    """
    Classify 3PT as pull-up or catch-and-shoot.
    
    Primary: shot type
    Secondary: assist field (if assist present, likely catch-and-shoot)
    """
    if shot_type in PULLUP_3_TYPES:
        return 'pullup'
    elif shot_type in CATCH_SHOOT_3_TYPES:
        return 'catch_shoot'
    else:
        # Use assist as tiebreaker
        return 'catch_shoot' if pd.notna(assist) else 'pullup'
```

---

## PHASE 3: SHOOTING FOUL LOGIC ✅ CLARIFIED

### 3.1 Identifying Shooting Fouls

**Per user clarification:**
```python
def is_shooting_foul_drawn(play):
    """
    Identify if player drew a shooting foul.
    
    Logic from user:
        event_type = 'foul'
        reason = 's.foul'
        opponent = player who was fouled (VICTIM)
        player = player who fouled (FOULER)
    """
    return (
        play['event_type'] == 'foul' and
        play['reason'] == 's.foul'
    )

def get_fouled_player(play):
    """Return the player who was fouled."""
    return play['opponent']  # Victim of foul

def get_fouler(play):
    """Return the player who committed the foul."""
    return play['player']  # Person who fouled
```

### 3.2 And-1 Detection

**Per user: Check for 'free throw 1/1' or 'free throw 0/1' after shooting foul**

```python
def detect_and1(plays_df, shot_idx):
    """
    Detect if a made shot was an and-1.
    
    Logic:
    1. Shot must be made
    2. Next play is shooting foul (event_type='foul', reason='s.foul')
       OR current play has foul
    3. Following play is 'free throw 1/1' or 'free throw 0/1'
    """
    current_play = plays_df.iloc[shot_idx]
    
    if current_play['result'] != 'made':
        return False
    
    # Check next 1-2 plays for shooting foul + free throw
    for offset in [1, 2]:
        if shot_idx + offset >= len(plays_df):
            break
        
        next_play = plays_df.iloc[shot_idx + offset]
        
        # Look for shooting foul
        if next_play['event_type'] == 'foul' and next_play['reason'] == 's.foul':
            # Check play after that for FT 1/1 or 0/1
            if shot_idx + offset + 1 < len(plays_df):
                ft_play = plays_df.iloc[shot_idx + offset + 1]
                if ft_play['type'] in ['free throw 1/1', 'free throw 0/1']:
                    return True
    
    return False
```

---

## PHASE 4: STAT CALCULATIONS

### 4.A: Shot Profile & Quality (Offense)

#### 1. Rim Attempt Rate
```python
rim_attempts = sum(is_rim_attempt(type, distance) for each shot by player)
total_fga = count(event_type='shot' AND player=target)
rim_attempt_rate = rim_attempts / total_fga
```

#### 2. Corner 3 & Non-Corner 3 Attempt Rate
```python
corner_3_att = count(shot_location='corner_3')
non_corner_3_att = count(shot_location='non_corner_3')

corner_3_rate = corner_3_att / total_fga
non_corner_3_rate = non_corner_3_att / total_fga
```

#### 3. Pull-up 3 vs Catch-and-Shoot 3 Rate
```python
pullup_3_att = count(classify_3pt_shot() == 'pullup')
catch_shoot_3_att = count(classify_3pt_shot() == 'catch_shoot')

pullup_3_rate = pullup_3_att / total_fga
catch_shoot_3_rate = catch_shoot_3_att / total_fga
```

#### 4. Mid-Range Rate
```python
short_mid_att = count(shot_location='short_mid')  # 4-16 ft
long_mid_att = count(shot_location='long_mid')    # 16-22 ft
total_mid_att = short_mid_att + long_mid_att

mid_range_rate = total_mid_att / total_fga
short_mid_rate = short_mid_att / total_fga
long_mid_rate = long_mid_att / total_fga
```

#### 5. Free Throw Rate (FTr)
```python
fta = count(event_type='free throw' AND player=target)
ftr = fta / total_fga
```

#### 6. Shooting Fouls Drawn per 100
```python
shooting_fouls_drawn = count(
    event_type='foul' AND 
    reason='s.foul' AND 
    opponent=target_player  # opponent is the victim
)

shooting_fouls_per_100 = (shooting_fouls_drawn / possessions) * 100
```

#### 7. And-1 Rate
```python
and1_count = count(detect_and1(play) for made shots by player)
and1_rate = (and1_count / total_fga) * 100
```

#### 8. Assisted Rate on 2s and 3s
```python
# 2-pointers
made_2pt = [shot for shot in made_shots if not is_3pt(shot)]
assisted_2pt = [shot for shot in made_2pt if pd.notna(shot['assist'])]
assisted_2pt_rate = len(assisted_2pt) / len(made_2pt) if len(made_2pt) > 0 else 0

# 3-pointers  
made_3pt = [shot for shot in made_shots if is_3pt(shot)]
assisted_3pt = [shot for shot in made_3pt if pd.notna(shot['assist'])]
assisted_3pt_rate = len(assisted_3pt) / len(made_3pt) if len(made_3pt) > 0 else 0
```

### 4.B: On-Ball Creation & Decision Making

#### 9. Assists per 100
```python
assists = count(
    event_type='shot' AND 
    result='made' AND 
    assist=target_player
)

assists_per_100 = (assists / possessions) * 100
```

#### 10. Turnovers per 100
```python
turnovers = count(
    event_type='turnover' AND 
    player=target_player
)

turnovers_per_100 = (turnovers / possessions) * 100
```

### 4.C: Defensive Impact Signals

#### 11. Steal Rate
```python
steals = count(
    event_type='turnover' AND 
    steal=target_player
)

steals_per_100 = (steals / possessions) * 100
```

#### 12. Block Rate
```python
blocks = count(
    event_type='shot' AND 
    block=target_player
)

blocks_per_100 = (blocks / possessions) * 100
```

#### 13. Defensive Rebound Share
```python
# When player is on court
player_def_rebs = count(
    event_type='rebound' AND 
    type='rebound defensive' AND 
    player=target_player
)

# Team defensive rebounds when player on court
team_def_rebs_with_player = count(
    event_type='rebound' AND
    type='rebound defensive' AND
    target_player in [h1,h2,h3,h4,h5] or [a1,a2,a3,a4,a5]
)

def_reb_share = player_def_rebs / team_def_rebs_with_player
```

#### 14. Shooting Foul Rate Committed per 100
```python
shooting_fouls_committed = count(
    event_type='foul' AND 
    reason='s.foul' AND 
    player=target_player  # player is the fouler
)

shooting_foul_rate = (shooting_fouls_committed / possessions) * 100
```

#### 15. Total Foul Rate Committed per 100
```python
total_fouls_committed = count(
    event_type='foul' AND 
    player=target_player
)

total_foul_rate = (total_fouls_committed / possessions) * 100
```

---

## PHASE 5: MODULE ARCHITECTURE

### 5.1 File Structure

```
player_stats_from_pbp.py
├── Constants & Configuration
├── Data Loading
├── Possession Tracking
├── Shot Classification
├── Stat Calculators
│   ├── Offensive Stats
│   ├── Creation Stats
│   └── Defensive Stats
├── Caching System
└── Main Interface Functions
```

### 5.2 Key Functions

```python
# Data Loading
def load_pbp_data(season_year) -> pd.DataFrame
def preprocess_pbp_data(df) -> pd.DataFrame

# Possession Tracking
def identify_possessions(pbp_df) -> dict
def is_possession_ending(play, next_play) -> bool
def attribute_possession(play, players_on_court) -> None

# Shot Classification
def classify_shot_location(x, y, distance, shot_type) -> str
def is_rim_attempt(shot_type, distance) -> bool
def classify_3pt_shot(shot_type, assist) -> str

# Foul Detection
def is_shooting_foul_drawn(play) -> bool
def detect_and1(plays_df, shot_idx) -> bool

# Stat Calculation
def calculate_shot_profile_stats(player_pbp, possessions) -> dict
def calculate_creation_stats(player_pbp, possessions) -> dict
def calculate_defensive_stats(player_pbp, possessions) -> dict

# Main Interface
def calculate_player_pbp_stats(player_name, max_date, season) -> dict
def build_pbp_cache_for_date_range(start_date, end_date, interval) -> int

# Caching
def load_pbp_cache(player_name, max_date, season) -> dict
def save_pbp_cache(player_name, max_date, season, stats) -> None
```

### 5.3 Output Schema

```python
pbp_stats = {
    # Shot Profile
    'rim_attempt_rate': 0.234,
    'corner_3_rate': 0.089,
    'non_corner_3_rate': 0.156,
    'pullup_3_rate': 0.067,
    'catch_shoot_3_rate': 0.178,
    'mid_range_rate': 0.321,
    'short_mid_rate': 0.123,
    'long_mid_rate': 0.198,
    
    # Shot Quality
    'ftr': 0.234,  # FTA/FGA
    'shooting_fouls_per_100': 4.5,
    'and1_rate': 0.023,
    'assisted_2pt_rate': 0.456,
    'assisted_3pt_rate': 0.789,
    
    # Creation
    'assists_per_100': 18.7,
    'turnovers_per_100': 12.3,
    'ast_to_ratio': 1.52,
    
    # Defense
    'steals_per_100': 2.3,
    'blocks_per_100': 1.5,
    'def_reb_share': 0.187,
    'shooting_fouls_per_100': 3.4,
    'total_fouls_per_100': 5.8,
    
    # Context
    'total_possessions': 1234,
    'total_fga': 456,
    'games_played': 15
}
```

---

## PHASE 6: PERFORMANCE OPTIMIZATION

### 6.1 Pre-processing Strategy

```python
# One-time preprocessing per season
def preprocess_season_pbp(pbp_df):
    """
    Add computed columns to avoid repeated calculations.
    """
    # Add shot classifications
    pbp_df['shot_location'] = pbp_df.apply(classify_shot_location, axis=1)
    pbp_df['is_rim'] = pbp_df.apply(is_rim_attempt, axis=1)
    pbp_df['3pt_type'] = pbp_df.apply(classify_3pt_shot, axis=1)
    
    # Add shooting foul flags
    pbp_df['is_shooting_foul'] = pbp_df.apply(is_shooting_foul_drawn, axis=1)
    
    # Add possession ending flag
    pbp_df['ends_possession'] = pbp_df.apply(check_possession_ending, axis=1)
    
    return pbp_df
```

### 6.2 Vectorization

Use pandas vectorized operations where possible:
```python
# Instead of looping
# for idx, row in df.iterrows()

# Use vectorized operations
mask = (df['event_type'] == 'shot') & (df['player'] == target)
shots = df[mask]
```

### 6.3 Caching Strategy

```
data/cache/
├── player_stats/        # Box score stats (existing)
├── pca/                 # PCA scores (existing)  
└── player_stats_pbp/    # NEW: Play-by-play stats
    ├── {hash}.json
    └── ...
```

Cache key: `MD5(player_name + max_date + season + 'pbp')`

---

## PHASE 7: VALIDATION & TESTING

### 7.1 Validation Checks

```python
def validate_stats(stats):
    """Sanity checks on calculated stats."""
    assert 0 <= stats['rim_attempt_rate'] <= 1
    assert 0 <= stats['assisted_2pt_rate'] <= 1
    assert stats['assists_per_100'] >= 0
    assert stats['turnovers_per_100'] >= 0
    
    # Shot distribution should sum to ~100%
    shot_dist_sum = (
        stats['rim_attempt_rate'] + 
        stats['corner_3_rate'] + 
        stats['non_corner_3_rate'] + 
        stats['mid_range_rate']
    )
    assert 0.95 <= shot_dist_sum <= 1.05, f"Shot dist sum: {shot_dist_sum}"
```

### 7.2 Test Cases

Pick known players and validate manually:
- LeBron James (high assists, versatile shots)
- Stephen Curry (high 3PT rate, many pull-ups)
- Rudy Gobert (high rim rate, low shot creation)

### 7.3 Cross-validation with Box Score

Compare FTA from PBP vs box score (should match exactly).

---

## PHASE 8: INTEGRATION

### 8.1 Merge with Existing Stats

```python
# In calling code
box_score_stats = calculate_player_stats(player, date, season)
pbp_stats = calculate_player_pbp_stats(player, date, season)

complete_profile = {
    **box_score_stats,
    **pbp_stats
}
```

### 8.2 Update PCA Metrics

Consider adding new PCA categories using PBP stats:
```python
pca_metrics = {
    "shot_selection": ['rim_attempt_rate', 'corner_3_rate', 'mid_range_rate'],
    "self_creation": ['pullup_3_rate', 'unassisted_2pt_rate', 'and1_rate'],
    "defensive_activity": ['steals_per_100', 'blocks_per_100', 'def_reb_share']
}
```

---

## IMPLEMENTATION CHECKLIST

### Week 1: Core Infrastructure
- [ ] Create `player_stats_from_pbp.py` module
- [ ] Implement possession tracking logic
- [ ] Implement shot classification system
- [ ] Test on small dataset (single game)

### Week 2: Stat Calculations
- [ ] Implement shot profile stats (8 metrics)
- [ ] Implement creation stats (2 metrics)
- [ ] Implement defensive stats (5 metrics)
- [ ] Add validation checks

### Week 3: Optimization & Testing
- [ ] Add caching system
- [ ] Optimize with vectorization
- [ ] Test on full season
- [ ] Validate against known players

### Week 4: Integration
- [ ] Integrate with existing pipeline
- [ ] Update PCA calculations
- [ ] Documentation
- [ ] Performance benchmarking

---

## ESTIMATED METRICS

**New Stats Added:** 20 metrics
- Shot profile: 8
- Creation: 2
- Defense: 5
- Supporting: 5 (possessions, games, etc.)

**Performance:**
- Processing time: ~10-20 seconds per player (full season)
- Cache build time: ~2-3 hours for full season (all players, all dates)
- File size: ~2KB per player-date combo

**Code Size:**
- Lines: ~1000-1200
- Functions: ~20-25
- Classes: 1-2 (optional)

---

## READY TO IMPLEMENT! 🚀

All questions answered, approach approved. Ready to start coding when you give the go-ahead.

