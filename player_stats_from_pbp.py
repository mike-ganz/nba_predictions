"""
Play-by-Play Player Statistics Module

Extracts advanced player statistics from play-by-play data including:
- Shot profile and quality metrics
- On-ball creation and decision making
- Defensive impact signals

All stats normalized per 100 possessions for fair comparison.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import hashlib
import time
from typing import Dict, Tuple, Optional, List

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

CACHE_DIR = 'data/cache/player_stats_pbp'
PBP_DATA_DIR = 'data/play_by_play/historical'

# Rolling window configuration
PLAYER_ROLLING_WINDOW = 20  # Number of games for rolling averages
MIN_PLAYER_GAMES = 10  # Minimum games required for valid stats

# Court dimensions (from official NBA court diagram)
COURT_WIDTH = 50  # feet
COURT_LENGTH = 94  # feet
CENTER_X = 25  # feet from left sideline
CENTER_Y = 47  # feet from baseline

# Basket locations
BASKET_BOTTOM_Y = 25.25  # feet from bottom baseline
BASKET_TOP_Y = 68.75  # feet from bottom baseline (94 - 25.25)

# Shot distance thresholds
RIM_DISTANCE = 4  # feet
SHORT_MID_MIN = 4  # feet
SHORT_MID_MAX = 16  # feet
LONG_MID_MIN = 16  # feet
THREE_POINT_DISTANCE = 22  # feet (23.75 in corners, but use 22 as threshold)

# Corner 3 boundaries
CORNER_3_X_THRESHOLD = 3  # Within 3 feet of sideline
CORNER_3_Y_RANGE = 14  # Within 14 feet of baseline

# ============================================================================
# IN-MEMORY CACHES (eliminates redundant calculations)
# ============================================================================

# Cache possession counts to avoid recounting for same games
# Key: frozenset of game_ids, Value: possession_counts dict
_POSSESSION_COUNT_CACHE = {}

# Cache PBP stats to avoid recalculating for same player-date-season combinations
# Key: (player_name, max_date, season, use_rolling), Value: stats dict
_PBP_STATS_MEMORY_CACHE = {}

# Shot type classifications
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
    'hook shot',
    'hook bank shot'
}

# Per user: both floating and running 3s are pull-ups
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
    '3pt jump shot',  # Most common (62K occurrences)
    '3pt jump bank shot'
}

# Possession-ending violation types
POSSESSION_ENDING_VIOLATIONS = {
    'shot clock',
    '8-second violation',
    'backcourt',
    'offensive goaltending'
}

# Offensive foul types (end possession)
OFFENSIVE_FOUL_TYPES = {
    'offensive foul',
    'offensive charge',
    'offensive'
}

# Season file mapping
SEASON_FILE_MAPPING = {
    "2022-2023": "[10-18-2022]-[06-12-2023]-combined-stats.csv",
    "2023-2024": "[10-24-2023]-[06-17-2024]-combined-stats.csv",
    "2024-2025": "[10-22-2024]-[06-22-2025]-combined-stats.csv",
}

# Global cache for loaded PBP data
_pbp_data_cache = {}

# ============================================================================
# CACHING UTILITIES
# ============================================================================

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def generate_cache_filename(player_name: str, max_date: str, season: str) -> str:
    """Generate unique cache filename using MD5 hash."""
    unique_string = f"{player_name}_{max_date}_{season}_pbp"
    hashed = hashlib.md5(unique_string.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.json")

def save_to_cache(player_name: str, max_date: str, season: str, data: dict):
    """Save player stats to cache."""
    ensure_cache_dir()
    filename = generate_cache_filename(player_name, max_date, season)
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_from_cache(player_name: str, max_date: str, season: str) -> Optional[dict]:
    """Load player stats from cache if available."""
    filename = generate_cache_filename(player_name, max_date, season)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

# ============================================================================
# DATA LOADING
# ============================================================================

def load_pbp_data(season_year: str = "2023-2024") -> pd.DataFrame:
    """
    Load play-by-play data for specified season.
    
    Args:
        season_year: Season in format "YYYY-YYYY"
    
    Returns:
        DataFrame with play-by-play data
    """
    global _pbp_data_cache
    
    # Check cache first
    if season_year in _pbp_data_cache:
        print(f" Using cached PBP data for {season_year}")
        return _pbp_data_cache[season_year]
    
    if season_year not in SEASON_FILE_MAPPING:
        raise ValueError(f"Season {season_year} not supported. Available: {list(SEASON_FILE_MAPPING.keys())}")
    
    filename = SEASON_FILE_MAPPING[season_year]
    filepath = os.path.join(PBP_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PBP data file not found: {filepath}")
    
    print(f" Loading PBP data: {filepath}")
    start_time = time.time()
    
    df = pd.read_csv(filepath)
    
    # Convert date column (format is M/D/YYYY in the CSV)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    # Sort by game and play order
    df = df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
    
    elapsed = time.time() - start_time
    print(f" Loaded {len(df):,} plays in {elapsed:.2f}s")
    
    # Cache for reuse
    _pbp_data_cache[season_year] = df
    
    return df

# ============================================================================
# COORDINATE & DISTANCE CALCULATIONS
# ============================================================================

def calculate_shot_distance(x: float, y: float, shooting_end: str = 'bottom') -> float:
    """
    Calculate distance from shot location to basket.
    
    Args:
        x, y: Shot coordinates
        shooting_end: 'bottom' or 'top' (which basket team is shooting at)
    
    Returns:
        Distance in feet
    """
    basket_x = CENTER_X
    basket_y = BASKET_BOTTOM_Y if shooting_end == 'bottom' else BASKET_TOP_Y
    
    distance = np.sqrt((x - basket_x)**2 + (y - basket_y)**2)
    return distance

def determine_shooting_end(team: str, period: int, home: str, away: str) -> str:
    """
    Determine which basket a team is shooting at.
    
    In NBA: teams switch ends at halftime
    Periods 1-2: home shoots at one end, away at other
    Periods 3-4+: teams switch
    """
    # Simplified logic - you may need to adjust based on actual coordinate system
    # Assuming away team shoots toward bottom basket in periods 1-2
    if period <= 2:
        return 'bottom' if team == away else 'top'
    else:
        return 'top' if team == away else 'bottom'

# ============================================================================
# SHOT CLASSIFICATION
# ============================================================================

def is_corner_3(x: float, y: float, shooting_end: str = 'bottom') -> bool:
    """
    Determine if 3PT shot is from corner.
    
    Args:
        x, y: Shot coordinates
        shooting_end: Which basket team is shooting at
    
    Returns:
        True if corner 3
    """
    # Check if near sideline
    near_sideline = (x <= CORNER_3_X_THRESHOLD) or (x >= COURT_WIDTH - CORNER_3_X_THRESHOLD)
    
    if not near_sideline:
        return False
    
    # Check if within corner range from baseline
    if shooting_end == 'bottom':
        near_baseline = y <= BASKET_BOTTOM_Y + CORNER_3_Y_RANGE
    else:
        near_baseline = y >= BASKET_TOP_Y - CORNER_3_Y_RANGE
    
    return near_baseline

def classify_shot_location(row: pd.Series) -> str:
    """
    Classify shot into location categories.
    
    Categories:
        - rim: within 4 feet
        - short_mid: 4-16 feet
        - long_mid: 16-22 feet
        - corner_3: 3PT from corner
        - non_corner_3: 3PT not from corner
        - unknown: can't determine
    
    Args:
        row: Play-by-play row with shot data
    
    Returns:
        Location category string
    """
    shot_type = str(row.get('type', '')).lower()
    
    # Get coordinates and distance
    x = row.get('converted_x', np.nan)
    y = row.get('converted_y', np.nan)
    distance = row.get('shot_distance', np.nan)
    
    # Determine shooting end
    shooting_end = determine_shooting_end(
        row.get('team', ''),
        row.get('period', 1),
        row.get('home', ''),
        row.get('away', '')
    )
    
    # Calculate distance if missing
    if pd.isna(distance) and not pd.isna(x) and not pd.isna(y):
        distance = calculate_shot_distance(x, y, shooting_end)
    
    # Check if 3-pointer
    is_3pt = '3pt' in shot_type or (not pd.isna(distance) and distance >= THREE_POINT_DISTANCE)
    
    if is_3pt:
        # Classify 3PT as corner or non-corner
        if not pd.isna(x) and not pd.isna(y):
            if is_corner_3(x, y, shooting_end):
                return 'corner_3'
        return 'non_corner_3'
    
    # 2-point shots
    if pd.isna(distance):
        # Use shot type as fallback
        if shot_type in RIM_SHOT_TYPES:
            return 'rim'
        return 'unknown'
    
    # Classify by distance
    if distance <= RIM_DISTANCE:
        return 'rim'
    elif distance <= SHORT_MID_MAX:
        return 'short_mid'
    elif distance < THREE_POINT_DISTANCE:
        return 'long_mid'
    
    return 'unknown'

def is_rim_attempt(shot_type: str, shot_location: str) -> bool:
    """Check if shot is at rim."""
    return (shot_type.lower() in RIM_SHOT_TYPES) or (shot_location == 'rim')

def classify_3pt_type(shot_type: str, assist: str) -> str:
    """
    Classify 3PT shot as pull-up or catch-and-shoot.
    
    Args:
        shot_type: Type of shot
        assist: Player who assisted (None if unassisted)
    
    Returns:
        'pullup', 'catch_shoot', or 'unknown'
    """
    shot_type_lower = shot_type.lower()
    
    if shot_type_lower in PULLUP_3_TYPES:
        return 'pullup'
    elif shot_type_lower in CATCH_SHOOT_3_TYPES:
        return 'catch_shoot'
    else:
        # Use assist as tiebreaker
        if pd.notna(assist) and str(assist).strip() != '':
            return 'catch_shoot'
        else:
            return 'pullup'  # Default to pull-up if no assist

# ============================================================================
# POSSESSION TRACKING
# ============================================================================

def is_possession_ending_event(play: pd.Series, next_play: Optional[pd.Series] = None) -> Tuple[bool, str]:
    """
    Determine if a play ends a possession.
    
    Args:
        play: Current play
        next_play: Next play (for context)
    
    Returns:
        (ends_possession, reason)
    """
    event_type = play.get('event_type', '')
    play_type = str(play.get('type', '')).lower()
    result = play.get('result', '')
    
    # A. Made shot (check for offensive rebound)
    if event_type == 'shot' and result == 'made':
        if next_play is not None:
            next_event = next_play.get('event_type', '')
            next_type = str(next_play.get('type', '')).lower()
            next_team = next_play.get('team', '')
            current_team = play.get('team', '')
            
            # Check if offensive rebound by same team
            if (next_event == 'rebound' and 
                'offensive' in next_type and 
                next_team == current_team):
                return False, 'offensive_rebound_continues'
        
        return True, 'made_shot'
    
    # B. Defensive rebound
    if event_type == 'rebound' and 'defensive' in play_type:
        return True, 'defensive_rebound'
    
    # C. Turnover
    if event_type == 'turnover':
        return True, 'turnover'
    
    # D. Offensive foul (with check for duplicate turnover)
    if event_type == 'foul' and play_type in OFFENSIVE_FOUL_TYPES:
        # Check if next play is turnover (same action, don't double count)
        if next_play is not None:
            next_event = next_play.get('event_type', '')
            # If next is turnover, we'll catch it there, skip here
            if next_event == 'turnover':
                return False, 'offensive_foul_with_turnover'
        return True, 'offensive_foul'
    
    # E. Dead ball violations
    if event_type == 'violation' and play_type in POSSESSION_ENDING_VIOLATIONS:
        return True, 'violation'
    
    # F. End of period
    if event_type == 'end of period':
        return True, 'end_of_period'
    
    return False, 'continues'

def count_player_possessions(pbp_df: pd.DataFrame) -> Dict[Tuple[str, int], int]:
    """
    Count possessions for each player in each game.
    
    A player is credited with a possession if they're on the court when
    their team has the ball.
    
    Args:
        pbp_df: Play-by-play DataFrame
    
    Returns:
        Dict mapping (player_name, game_id) -> possession_count
    """
    global _POSSESSION_COUNT_CACHE
    
    # Create cache key from game IDs
    game_ids = frozenset(pbp_df['game_id'].unique())
    
    # Check cache first
    if game_ids in _POSSESSION_COUNT_CACHE:
        # Silently return cached result (no print to reduce noise)
        return _POSSESSION_COUNT_CACHE[game_ids]
    
    print("Counting possessions...")
    start_time = time.time()
    
    possession_counts = {}
    
    # Vectorized per-game processing
    for game_id, game_plays in pbp_df.groupby('game_id', sort=False):
        game_plays = game_plays.reset_index(drop=True)
        
        # Pre-compute shifted columns for next-play context
        next_event = game_plays['event_type'].shift(-1)
        next_type = game_plays['type'].astype(str).str.lower().shift(-1)
        next_team = game_plays['team'].shift(-1)
        curr_event = game_plays['event_type']
        curr_type = game_plays['type'].astype(str).str.lower()
        curr_result = game_plays['result']
        curr_team = game_plays['team']
        
        # Possession-ending masks
        made_shot = (curr_event == 'shot') & (curr_result == 'made')
        made_shot_continues = made_shot & (next_event == 'rebound') & next_type.str.contains('offensive', na=False) & (next_team == curr_team)
        ends_made_shot = made_shot & (~made_shot_continues)
        
        def_reb = (curr_event == 'rebound') & curr_type.str.contains('defensive', na=False)
        turnover = (curr_event == 'turnover')
        off_foul = (curr_event == 'foul') & curr_type.isin(list(OFFENSIVE_FOUL_TYPES))
        off_foul_followed_by_tov = off_foul & (next_event == 'turnover')
        ends_off_foul = off_foul & (~off_foul_followed_by_tov)
        violation = (curr_event == 'violation') & curr_type.isin(list(POSSESSION_ENDING_VIOLATIONS))
        end_period = (curr_event == 'end of period')
        
        ends_possession_mask = ends_made_shot | def_reb | turnover | ends_off_foul | violation | end_period
        end_indices = np.flatnonzero(ends_possession_mask.values)
        
        if end_indices.size == 0:
            continue
        
        # Determine players to credit possession at each end index
        # Prefer player’s own side if active; fallback to home lineup
        for idx in end_indices:
            row = game_plays.iloc[idx]
            active_player = row.get('player')
            h_players = [row.get(f'h{i}') for i in range(1, 6)]
            a_players = [row.get(f'a{i}') for i in range(1, 6)]
            players_to_credit = h_players  # default
            if pd.notna(active_player) and active_player != '':
                if active_player in h_players:
                    players_to_credit = h_players
                elif active_player in a_players:
                    players_to_credit = a_players
            for player in players_to_credit:
                if pd.notna(player) and player != '':
                    key = (player, game_id)
                    possession_counts[key] = possession_counts.get(key, 0) + 1
    
    elapsed = time.time() - start_time
    total_possessions = sum(possession_counts.values())
    print(f" Counted {total_possessions:,} player-possessions in {elapsed:.2f}s")
    
    # Store in cache for future use
    _POSSESSION_COUNT_CACHE[game_ids] = possession_counts
    
    return possession_counts

# ============================================================================
# FOUL DETECTION
# ============================================================================

def is_shooting_foul(play: pd.Series) -> bool:
    """
    Check if play is a shooting foul.
    
    Per user: event_type='foul' AND reason='s.foul'
    """
    return (play.get('event_type') == 'foul' and 
            play.get('reason') == 's.foul')

def get_shooting_foul_victim(play: pd.Series) -> Optional[str]:
    """
    Get player who was fouled (victim).
    
    Per user: opponent field contains the victim
    """
    if is_shooting_foul(play):
        return play.get('opponent')
    return None

def get_fouler(play: pd.Series) -> Optional[str]:
    """
    Get player who committed the foul.
    
    Per user: player field contains the fouler
    """
    if play.get('event_type') == 'foul':
        return play.get('player')
    return None

def detect_and1(plays_df: pd.DataFrame, shot_idx: int) -> bool:
    """
    Detect if a made shot was an and-1.
    
    Logic per user:
    1. Shot must be made
    2. Shooting foul occurs (event_type='foul', reason='s.foul')
    3. Following play is 'free throw 1/1' or 'free throw 0/1'
    
    Args:
        plays_df: DataFrame of plays
        shot_idx: Index of the shot to check
    
    Returns:
        True if and-1
    """
    if shot_idx >= len(plays_df):
        return False
    
    current_play = plays_df.iloc[shot_idx]
    
    # Must be a made shot
    if current_play.get('result') != 'made':
        return False
    
    shooter = current_play.get('player')
    
    # Check next 1-3 plays for shooting foul + FT sequence
    for offset in range(1, 4):
        if shot_idx + offset >= len(plays_df):
            break
        
        next_play = plays_df.iloc[shot_idx + offset]
        
        # Look for shooting foul
        if is_shooting_foul(next_play):
            victim = get_shooting_foul_victim(next_play)
            
            # Victim should be the shooter
            if victim != shooter:
                continue
            
            # Check next play for FT 1/1 or 0/1
            if shot_idx + offset + 1 < len(plays_df):
                ft_play = plays_df.iloc[shot_idx + offset + 1]
                ft_type = str(ft_play.get('type', '')).lower()
                ft_shooter = ft_play.get('player')
                
                if (ft_shooter == shooter and 
                    ('free throw 1/1' in ft_type or 'free throw 0/1' in ft_type)):
                    return True
    
    return False

# ============================================================================
# STAT CALCULATIONS - SHOT PROFILE & QUALITY
# ============================================================================

def calculate_shot_profile_stats(player_pbp: pd.DataFrame, 
                                  player_name: str,
                                  possessions: int) -> dict:
    """
    Calculate shot profile and quality statistics.
    
    Returns dict with:
        - rim_attempt_rate
        - corner_3_rate
        - non_corner_3_rate
        - pullup_3_rate
        - catch_shoot_3_rate
        - mid_range_rate (short + long)
        - short_mid_rate
        - long_mid_rate
        - ftr (free throw rate)
        - shooting_fouls_per_100
        - and1_rate
        - assisted_2pt_rate
        - assisted_3pt_rate
    """
    stats = {}
    
    # Get all shot attempts by player
    shots = player_pbp[
        (player_pbp['event_type'] == 'shot') & 
        (player_pbp['player'] == player_name)
    ].copy()
    
    total_fga = len(shots)
    stats['total_fga'] = total_fga
    
    if total_fga == 0:
        # Return zeros if no shots
        return {
            'rim_attempt_rate': 0, 'corner_3_rate': 0, 'non_corner_3_rate': 0,
            'pullup_3_rate': 0, 'catch_shoot_3_rate': 0, 'mid_range_rate': 0,
            'short_mid_rate': 0, 'long_mid_rate': 0, 'ftr': 0,
            'shooting_fouls_per_100': 0, 'and1_rate': 0,
            'assisted_2pt_rate': 0, 'assisted_3pt_rate': 0,
            'total_fga': 0
        }
    
    # Add shot classifications (vectorized)
    # Prepare fields
    shot_type_str = shots['type'].astype(str).str.lower()
    x = shots['converted_x']
    y = shots['converted_y']
    distance = shots['shot_distance']
    team = shots['team']
    period = shots['period'].fillna(1)
    home = shots['home'] if 'home' in shots.columns else ''
    away = shots['away'] if 'away' in shots.columns else ''
    
    # Determine shooting end (approximate without per-row function):
    # If team equals home team, assume bottom basket in 1st half, top in 2nd; else invert
    is_home_team = team == home
    is_first_half = period.astype(int) <= 2
    shooting_end_bottom = (is_home_team & is_first_half) | ((~is_home_team) & (~is_first_half))
    
    # If distance missing and coords present, compute distance to basket end
    has_coords = x.notna() & y.notna()
    calc_dist = np.where(
        shooting_end_bottom,
        np.sqrt((x - CENTER_X) ** 2 + (y - BASKET_BOTTOM_Y) ** 2),
        np.sqrt((x - CENTER_X) ** 2 + (y - BASKET_TOP_Y) ** 2)
    )
    use_distance = distance.copy()
    use_distance[use_distance.isna() & has_coords] = calc_dist[use_distance.isna() & has_coords]
    
    # 3PT identification
    is_3pt = shot_type_str.str.contains('3pt', na=False) | (use_distance >= THREE_POINT_DISTANCE)
    
    # Corner 3 mask
    near_sideline = (x <= CORNER_3_X_THRESHOLD) | (x >= (COURT_WIDTH - CORNER_3_X_THRESHOLD))
    near_baseline = np.where(
        shooting_end_bottom,
        y <= (BASKET_BOTTOM_Y + CORNER_3_Y_RANGE),
        y >= (BASKET_TOP_Y - CORNER_3_Y_RANGE)
    )
    is_corner3 = is_3pt & has_coords & near_sideline & near_baseline
    
    # Base categories
    shot_location = pd.Series(index=shots.index, dtype='object')
    shot_location[is_3pt & is_corner3] = 'corner_3'
    shot_location[is_3pt & (~is_corner3)] = 'non_corner_3'
    shot_location[(~is_3pt) & (use_distance <= RIM_DISTANCE)] = 'rim'
    shot_location[(~is_3pt) & (use_distance > RIM_DISTANCE) & (use_distance <= SHORT_MID_MAX)] = 'short_mid'
    shot_location[(~is_3pt) & (use_distance < THREE_POINT_DISTANCE) & (use_distance > SHORT_MID_MAX)] = 'long_mid'
    
    # For rows still unknown (missing distance), use type fallbacks
    unknown_mask = shot_location.isna()
    rim_type_mask = shot_type_str.isin(list(RIM_SHOT_TYPES))
    shot_location[unknown_mask & rim_type_mask] = 'rim'
    shot_location[unknown_mask & (~rim_type_mask)] = 'unknown'
    
    shots['shot_location'] = shot_location
    
    # 1. Rim attempt rate
    rim_attempts = shots[shots['shot_location'] == 'rim']
    stats['rim_attempt_rate'] = len(rim_attempts) / total_fga
    
    # 2. Corner 3 and non-corner 3 rates
    corner_3_attempts = shots[shots['shot_location'] == 'corner_3']
    non_corner_3_attempts = shots[shots['shot_location'] == 'non_corner_3']
    stats['corner_3_rate'] = len(corner_3_attempts) / total_fga
    stats['non_corner_3_rate'] = len(non_corner_3_attempts) / total_fga
    
    # 3. Pull-up vs catch-and-shoot 3 rates
    three_pt_shots = shots[shots['shot_location'].isin(['corner_3', 'non_corner_3'])]
    pullup_3 = 0
    catch_shoot_3 = 0
    
    # Vectorize 3PT type classification (approximate: assist -> catch_shoot else pullup; override with known types)
    three_types = three_pt_shots['type'].astype(str).str.lower()
    three_assist = three_pt_shots['assist']
    is_pullup_known = three_types.isin(list(PULLUP_3_TYPES))
    is_cns_known = three_types.isin(list(CATCH_SHOOT_3_TYPES))
    is_assisted = three_assist.notna() & (three_assist.astype(str).str.strip() != '')
    # Priority: known types > assist heuristic
    pullup_mask = is_pullup_known | (~is_cns_known & ~is_assisted)
    cns_mask = is_cns_known | (~is_pullup_known & is_assisted)
    pullup_3 = int(pullup_mask.sum())
    catch_shoot_3 = int(cns_mask.sum())
    
    stats['pullup_3_rate'] = pullup_3 / total_fga if total_fga > 0 else 0
    stats['catch_shoot_3_rate'] = catch_shoot_3 / total_fga if total_fga > 0 else 0
    
    # 4. Mid-range rates
    short_mid_attempts = shots[shots['shot_location'] == 'short_mid']
    long_mid_attempts = shots[shots['shot_location'] == 'long_mid']
    stats['short_mid_rate'] = len(short_mid_attempts) / total_fga
    stats['long_mid_rate'] = len(long_mid_attempts) / total_fga
    stats['mid_range_rate'] = (len(short_mid_attempts) + len(long_mid_attempts)) / total_fga
    
    # 5. Free throw rate (FTA/FGA)
    fta = len(player_pbp[
        (player_pbp['event_type'] == 'free throw') &
        (player_pbp['player'] == player_name)
    ])
    stats['ftr'] = fta / total_fga if total_fga > 0 else 0
    
    # 6. Shooting fouls drawn per 100 possessions
    shooting_fouls_drawn = len(player_pbp[
        (player_pbp['event_type'] == 'foul') &
        (player_pbp['reason'] == 's.foul') &
        (player_pbp['opponent'] == player_name)  # opponent is victim
    ])
    stats['shooting_fouls_per_100'] = (shooting_fouls_drawn / possessions * 100) if possessions > 0 else 0
    
    # 7. And-1 rate
    and1_count = 0
    for idx in shots.index:
        if detect_and1(player_pbp, idx):
            and1_count += 1
    stats['and1_rate'] = (and1_count / total_fga * 100) if total_fga > 0 else 0
    
    # 8. Assisted rates on 2s and 3s
    made_shots = shots[shots['result'] == 'made']
    
    made_2pt = made_shots[~made_shots['shot_location'].isin(['corner_3', 'non_corner_3'])]
    made_3pt = made_shots[made_shots['shot_location'].isin(['corner_3', 'non_corner_3'])]
    
    assisted_2pt = made_2pt[made_2pt['assist'].notna() & (made_2pt['assist'] != '')]
    assisted_3pt = made_3pt[made_3pt['assist'].notna() & (made_3pt['assist'] != '')]
    
    stats['assisted_2pt_rate'] = len(assisted_2pt) / len(made_2pt) if len(made_2pt) > 0 else 0
    stats['assisted_3pt_rate'] = len(assisted_3pt) / len(made_3pt) if len(made_3pt) > 0 else 0
    
    return stats

# ============================================================================
# STAT CALCULATIONS - CREATION & DECISION MAKING
# ============================================================================

def calculate_creation_stats(player_pbp: pd.DataFrame,
                             player_name: str,
                             possessions: int) -> dict:
    """
    Calculate on-ball creation and decision making stats.
    
    Returns dict with:
        - assists_per_100
        - turnovers_per_100
        - ast_to_ratio
    """
    stats = {}
    
    # Assists
    assists = len(player_pbp[
        (player_pbp['event_type'] == 'shot') &
        (player_pbp['result'] == 'made') &
        (player_pbp['assist'] == player_name)
    ])
    
    # Turnovers
    turnovers = len(player_pbp[
        (player_pbp['event_type'] == 'turnover') &
        (player_pbp['player'] == player_name)
    ])
    
    stats['assists_per_100'] = (assists / possessions * 100) if possessions > 0 else 0
    stats['turnovers_per_100'] = (turnovers / possessions * 100) if possessions > 0 else 0
    stats['ast_to_ratio'] = assists / turnovers if turnovers > 0 else assists
    
    return stats

# ============================================================================
# STAT CALCULATIONS - DEFENSIVE IMPACT
# ============================================================================

def calculate_defensive_stats(player_pbp: pd.DataFrame,
                              player_name: str,
                              possessions: int) -> dict:
    """
    Calculate defensive impact statistics.
    
    Returns dict with:
        - steals_per_100
        - blocks_per_100
        - def_reb_share
        - shooting_fouls_per_100
        - total_fouls_per_100
    """
    stats = {}
    
    # 1. Steals
    steals = len(player_pbp[
        (player_pbp['event_type'] == 'turnover') &
        (player_pbp['steal'] == player_name)
    ])
    stats['steals_per_100'] = (steals / possessions * 100) if possessions > 0 else 0
    
    # 2. Blocks
    blocks = len(player_pbp[
        (player_pbp['event_type'] == 'shot') &
        (player_pbp['block'] == player_name)
    ])
    stats['blocks_per_100'] = (blocks / possessions * 100) if possessions > 0 else 0
    
    # 3. Defensive rebound share
    # Player's defensive rebounds
    player_def_rebs = len(player_pbp[
        (player_pbp['event_type'] == 'rebound') &
        (player_pbp['type'].str.contains('defensive', case=False, na=False)) &
        (player_pbp['player'] == player_name)
    ])
    
    # Team defensive rebounds when player on court
    # Check if player is in any of the 10 positions
    on_court_mask = (
        (player_pbp['h1'] == player_name) |
        (player_pbp['h2'] == player_name) |
        (player_pbp['h3'] == player_name) |
        (player_pbp['h4'] == player_name) |
        (player_pbp['h5'] == player_name) |
        (player_pbp['a1'] == player_name) |
        (player_pbp['a2'] == player_name) |
        (player_pbp['a3'] == player_name) |
        (player_pbp['a4'] == player_name) |
        (player_pbp['a5'] == player_name)
    )
    
    team_def_rebs = len(player_pbp[
        on_court_mask &
        (player_pbp['event_type'] == 'rebound') &
        (player_pbp['type'].str.contains('defensive', case=False, na=False))
    ])
    
    stats['def_reb_share'] = player_def_rebs / team_def_rebs if team_def_rebs > 0 else 0
    
    # 4. Shooting fouls committed
    shooting_fouls_committed = len(player_pbp[
        (player_pbp['event_type'] == 'foul') &
        (player_pbp['reason'] == 's.foul') &
        (player_pbp['player'] == player_name)  # player is the fouler
    ])
    stats['shooting_fouls_per_100'] = (shooting_fouls_committed / possessions * 100) if possessions > 0 else 0
    
    # 5. Total fouls committed
    total_fouls_committed = len(player_pbp[
        (player_pbp['event_type'] == 'foul') &
        (player_pbp['player'] == player_name)
    ])
    stats['total_fouls_per_100'] = (total_fouls_committed / possessions * 100) if possessions > 0 else 0
    
    return stats

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def calculate_player_pbp_stats(player_name: str, 
                               max_date: Optional[str] = None,
                               season: str = "2023-2024",
                               use_rolling: bool = True) -> Optional[dict]:
    """
    Calculate all play-by-play stats for a player up to a given date.
    
    Args:
        player_name: Full player name
        max_date: Calculate stats for games before this date (YYYY-MM-DD)
                 If None, use all games in season
        season: Season year in format "YYYY-YYYY"
        use_rolling: If True, use last 20 games. If False, use all games (cumulative)
    
    Returns:
        Dict with all PBP stats, or None if player not found
    """
    global _PBP_STATS_MEMORY_CACHE
    
    # Check in-memory cache first (fastest)
    memory_cache_key = (player_name, max_date, season, use_rolling)
    if memory_cache_key in _PBP_STATS_MEMORY_CACHE:
        return _PBP_STATS_MEMORY_CACHE[memory_cache_key]
    
    # Try file cache next (slower)
    cache_mode = f"roll{PLAYER_ROLLING_WINDOW}" if use_rolling else "cumulative"
    cache_key = f"{max_date or 'full_season'}_{cache_mode}"
    cached_data = load_from_cache(player_name, cache_key, season)
    if cached_data is not None:
        # Store in memory cache for future calls
        _PBP_STATS_MEMORY_CACHE[memory_cache_key] = cached_data
        return cached_data
    
    # Load PBP data
    pbp_df = load_pbp_data(season)
    
    # Filter by date if specified
    if max_date is not None:
        pbp_df = pbp_df[pbp_df['date'] < pd.to_datetime(max_date)]
    
    # Filter to games where player participated
    player_mask = (
        (pbp_df['player'] == player_name) |
        (pbp_df['assist'] == player_name) |
        (pbp_df['block'] == player_name) |
        (pbp_df['steal'] == player_name) |
        (pbp_df['h1'] == player_name) |
        (pbp_df['h2'] == player_name) |
        (pbp_df['h3'] == player_name) |
        (pbp_df['h4'] == player_name) |
        (pbp_df['h5'] == player_name) |
        (pbp_df['a1'] == player_name) |
        (pbp_df['a2'] == player_name) |
        (pbp_df['a3'] == player_name) |
        (pbp_df['a4'] == player_name) |
        (pbp_df['a5'] == player_name)
    )
    
    player_games_mask = pbp_df[player_mask]
    
    # Get unique games with dates
    game_dates = player_games_mask[['game_id', 'date']].drop_duplicates().sort_values('date')
    
    if len(game_dates) == 0:
        return None
    
    # Apply rolling window if requested
    if use_rolling and len(game_dates) > PLAYER_ROLLING_WINDOW:
        game_dates = game_dates.tail(PLAYER_ROLLING_WINDOW)
    
    player_games = game_dates['game_id'].values
    num_games = len(player_games)
    
    # Check minimum games requirement
    if num_games < MIN_PLAYER_GAMES:
        print(f" {player_name} has only {num_games} games (minimum {MIN_PLAYER_GAMES} required)")
        return None
    
    # Get plays from selected games
    player_pbp = pbp_df[pbp_df['game_id'].isin(player_games)].copy()
    
    # Count possessions
    possession_counts = count_player_possessions(player_pbp)
    
    # Get total possessions for this player
    total_possessions = sum(
        count for (pname, gid), count in possession_counts.items()
        if pname == player_name
    )
    
    if total_possessions == 0:
        print(f" No possessions found for {player_name}")
        return None
    
    # Calculate stats
    mode_str = f"rolling {num_games} games" if use_rolling else f"cumulative {num_games} games"
    print(f" Calculating PBP stats for {player_name} ({mode_str}, {total_possessions} possessions)...")
    
    shot_stats = calculate_shot_profile_stats(player_pbp, player_name, total_possessions)
    creation_stats = calculate_creation_stats(player_pbp, player_name, total_possessions)
    defensive_stats = calculate_defensive_stats(player_pbp, player_name, total_possessions)
    
    # Combine all stats
    all_stats = {
        **shot_stats,
        **creation_stats,
        **defensive_stats,
        'total_possessions': total_possessions,
        'games_played': num_games,
        'games_in_window': num_games,
        'rolling_window_used': use_rolling,
        'season': season,
        'max_date': max_date or 'full_season'
    }
    
    # Save to file cache
    save_to_cache(player_name, cache_key, season, all_stats)
    
    # Store in memory cache for future calls in this session
    _PBP_STATS_MEMORY_CACHE[memory_cache_key] = all_stats
    
    return all_stats

def get_player_pbp_stats_summary(stats: dict) -> str:
    """Generate a readable summary of player PBP stats."""
    if stats is None:
        return "No stats available"
    
    summary = f"""
 Play-by-Play Stats Summary
{'='*50}
Games: {stats.get('games_played', 0)} | Possessions: {stats.get('total_possessions', 0)}

 Shot Profile:
  Rim Rate: {stats.get('rim_attempt_rate', 0):.1%}
  Corner 3 Rate: {stats.get('corner_3_rate', 0):.1%}
  Non-Corner 3 Rate: {stats.get('non_corner_3_rate', 0):.1%}
  Pull-up 3 Rate: {stats.get('pullup_3_rate', 0):.1%}
  Catch & Shoot 3 Rate: {stats.get('catch_shoot_3_rate', 0):.1%}
  Mid-Range Rate: {stats.get('mid_range_rate', 0):.1%}

 Shot Quality:
  Free Throw Rate: {stats.get('ftr', 0):.3f}
  Shooting Fouls Drawn/100: {stats.get('shooting_fouls_per_100', 0):.1f}
  And-1 Rate: {stats.get('and1_rate', 0):.2f}%
  Assisted 2PT Rate: {stats.get('assisted_2pt_rate', 0):.1%}
  Assisted 3PT Rate: {stats.get('assisted_3pt_rate', 0):.1%}

🎨 Creation:
  Assists/100: {stats.get('assists_per_100', 0):.1f}
  Turnovers/100: {stats.get('turnovers_per_100', 0):.1f}
  Ast/TO Ratio: {stats.get('ast_to_ratio', 0):.2f}

🛡 Defense:
  Steals/100: {stats.get('steals_per_100', 0):.1f}
  Blocks/100: {stats.get('blocks_per_100', 0):.1f}
  Def Reb Share: {stats.get('def_reb_share', 0):.1%}
  Shooting Fouls/100: {stats.get('shooting_fouls_per_100', 0):.1f}
  Total Fouls/100: {stats.get('total_fouls_per_100', 0):.1f}
"""
    return summary

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def build_pbp_cache_for_date_range(start_date: str,
                                    end_date: str,
                                    date_interval_days: int = 7,
                                    season: str = "2023-2024",
                                    use_rolling: bool = True) -> int:
    """
    Build PBP stat cache for all players across a date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        date_interval_days: Days between cache points
        season: Season year
        use_rolling: If True, use rolling 20-game windows
    
    Returns:
        Number of cache entries created
    """
    mode = f"rolling {PLAYER_ROLLING_WINDOW}-game" if use_rolling else "cumulative"
    print(f" Building PBP cache from {start_date} to {end_date} ({mode})")
    print(f"   Interval: every {date_interval_days} days")
    print(f"   Season: {season}")
    
    # Generate target dates
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{date_interval_days}D')
    target_dates = [date.strftime('%Y-%m-%d') for date in date_range]
    
    print(f"📅 Target dates: {len(target_dates)}")
    
    # Load PBP data once
    pbp_df = load_pbp_data(season)
    
    # Get all unique players
    all_players = set()
    for col in ['player', 'assist', 'block', 'steal', 
                'h1', 'h2', 'h3', 'h4', 'h5',
                'a1', 'a2', 'a3', 'a4', 'a5']:
        all_players.update(pbp_df[col].dropna().unique())
    
    all_players = sorted(list(all_players))
    print(f"👥 Unique players: {len(all_players)}")
    
    # Process each date
    total_cached = 0
    for date_idx, target_date in enumerate(target_dates):
        print(f"\n📅 Processing date {date_idx + 1}/{len(target_dates)}: {target_date}")
        date_cached = 0
        
        for player_idx, player in enumerate(all_players):
            if player_idx % 50 == 0:
                print(f"   Player {player_idx + 1}/{len(all_players)}...")
            
            try:
                stats = calculate_player_pbp_stats(player, target_date, season, use_rolling)
                if stats is not None:
                    date_cached += 1
            except Exception as e:
                print(f"    Error processing {player}: {e}")
        
        total_cached += date_cached
        print(f"    Cached {date_cached} players for {target_date}")
    
    print(f"\n🎉 Cache building complete! {total_cached} total entries created.")
    return total_cached

# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def validate_pbp_stats(stats: dict) -> List[str]:
    """
    Validate calculated stats for sanity.
    
    Returns:
        List of validation warnings (empty if all good)
    """
    warnings = []
    
    if stats is None:
        return ["Stats are None"]
    
    # Check rates are between 0 and 1
    rate_fields = [
        'rim_attempt_rate', 'corner_3_rate', 'non_corner_3_rate',
        'pullup_3_rate', 'catch_shoot_3_rate', 'mid_range_rate',
        'short_mid_rate', 'long_mid_rate', 'assisted_2pt_rate', 
        'assisted_3pt_rate', 'def_reb_share'
    ]
    
    for field in rate_fields:
        value = stats.get(field, 0)
        if not (0 <= value <= 1.05):  # Allow 5% tolerance
            warnings.append(f"{field} out of range: {value:.3f}")
    
    # Check shot distribution sums to ~100%
    shot_dist_sum = (
        stats.get('rim_attempt_rate', 0) +
        stats.get('corner_3_rate', 0) +
        stats.get('non_corner_3_rate', 0) +
        stats.get('mid_range_rate', 0)
    )
    
    if not (0.8 <= shot_dist_sum <= 1.2):  # Allow 20% tolerance for unknowns
        warnings.append(f"Shot distribution sum: {shot_dist_sum:.3f} (expected ~1.0)")
    
    # Check per-100 stats are reasonable
    if stats.get('assists_per_100', 0) > 100:
        warnings.append(f"Assists/100 too high: {stats['assists_per_100']:.1f}")
    
    if stats.get('turnovers_per_100', 0) > 100:
        warnings.append(f"Turnovers/100 too high: {stats['turnovers_per_100']:.1f}")
    
    return warnings

if __name__ == "__main__":
    # Example usage
    print("🏀 NBA Play-by-Play Stats Module - Rolling Window Test")
    print("=" * 80)
    
    # Test with a known player
    player = "LeBron James"
    season = "2023-2024"
    test_date = "2024-02-15"
    
    print(f"\nTesting with: {player} ({season}, date: {test_date})")
    
    # Test 1: Rolling 20-game window (default)
    print(f"\n{'='*80}")
    print(f"TEST 1: Rolling {PLAYER_ROLLING_WINDOW}-game window")
    print(f"{'='*80}")
    stats_rolling = calculate_player_pbp_stats(player, test_date, season, use_rolling=True)
    
    if stats_rolling:
        print(f" Games: {stats_rolling['games_played']}")
        print(f"   Rim%: {stats_rolling.get('rim_attempt_rate', 0):.1%}")
        print(f"   3PT%: {stats_rolling.get('non_corner_3_rate', 0) + stats_rolling.get('corner_3_rate', 0):.1%}")
        print(f"   AST/100: {stats_rolling.get('assists_per_100', 0):.1f}")
        print(f"   Possessions: {stats_rolling['total_possessions']}")
    
    # Test 2: Cumulative (all games)
    print(f"\n{'='*80}")
    print(f"TEST 2: Cumulative (all games up to date)")
    print(f"{'='*80}")
    stats_cumulative = calculate_player_pbp_stats(player, test_date, season, use_rolling=False)
    
    if stats_cumulative:
        print(f" Games: {stats_cumulative['games_played']}")
        print(f"   Rim%: {stats_cumulative.get('rim_attempt_rate', 0):.1%}")
        print(f"   3PT%: {stats_cumulative.get('non_corner_3_rate', 0) + stats_cumulative.get('corner_3_rate', 0):.1%}")
        print(f"   AST/100: {stats_cumulative.get('assists_per_100', 0):.1f}")
        print(f"   Possessions: {stats_cumulative['total_possessions']}")
    
    # Compare
    if stats_rolling and stats_cumulative:
        print(f"\n{'='*80}")
        print(f"COMPARISON: Rolling vs Cumulative")
        print(f"{'='*80}")
        
        rim_diff = stats_rolling.get('rim_attempt_rate', 0) - stats_cumulative.get('rim_attempt_rate', 0)
        three_rolling = stats_rolling.get('non_corner_3_rate', 0) + stats_rolling.get('corner_3_rate', 0)
        three_cumulative = stats_cumulative.get('non_corner_3_rate', 0) + stats_cumulative.get('corner_3_rate', 0)
        three_diff = three_rolling - three_cumulative
        ast_diff = stats_rolling.get('assists_per_100', 0) - stats_cumulative.get('assists_per_100', 0)
        
        print(f"Games: {stats_rolling['games_played']} (rolling) vs {stats_cumulative['games_played']} (cumulative)")
        print(f"Rim% diff: {rim_diff:+.1%} ({'↑ more rim' if rim_diff > 0 else '↓ less rim'})")
        print(f"3PT% diff: {three_diff:+.1%} ({'↑ more 3PT' if three_diff > 0 else '↓ less 3PT'})")
        print(f"AST/100 diff: {ast_diff:+.1f} ({'↑ passing more' if ast_diff > 0 else '↓ passing less'})")
        
        print("\n💡 Interpretation:")
        print("   Rolling window captures recent form/hot streaks")
        print("   Cumulative shows season-long tendencies")
        print("   Differences highlight if player style is evolving")
    
    print(f"\n{'='*80}")
    print(" Rolling window test complete!")
    print(f"{'='*80}")

