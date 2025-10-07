import pandas as pd
import os
import json
import re

# Import required functions from other modules
from transform_player_stats import calculate_player_stats, get_distinct_players
from generate_team_stats import generate_team_stats, get_available_teams
from pca_optimized import get_player_pca_score
from get_team_city import get_team_city
from generate_lineup import get_lineup_by_game_id, load_all_player_boxscores

# Configuration
SEASON_YEAR = "2023-2024"  # Default season year

# ============================================================================
# COMPACT SCHEMA CONVERSION UTILITIES
# ============================================================================

def parse_time_to_seconds(time_str):
    """Convert 'MM:SS' time format to seconds remaining."""
    if pd.isna(time_str) or not time_str:
        return 0
    
    try:
        time_parts = str(time_str).split(':')
        if len(time_parts) >= 2:
            minutes = int(time_parts[-2])
            seconds = int(time_parts[-1])
            return 60 * minutes + seconds
        else:
            return 0
    except (ValueError, IndexError):
        return 0

def parse_score_string(score_str):
    """Parse 'AWAY x - HOME y' format to [away_score, home_score]."""
    if pd.isna(score_str) or not score_str:
        return [0, 0]
    
    try:
        # Extract numbers from format like "ATL 18 - DET 11"
        parts = str(score_str).split(' - ')
        if len(parts) == 2:
            away_score = int(parts[0].split()[-1])
            home_score = int(parts[1].split()[-1])
            return [away_score, home_score]
        else:
            return [0, 0]
    except (ValueError, IndexError):
        return [0, 0]

def extract_distance_from_description(description):
    """Extract distance in feet from play description."""
    if pd.isna(description):
        return None
    
    # Look for pattern like "26'" or "26 feet"
    match = re.search(r'(\d+)\'', str(description))
    if match:
        return match.group(1)
    return None


def map_structured_to_event_code(row):
    """
    Map structured play-by-play fields to compact event code.
    Uses type, result, event_type, points, and shot_distance instead of parsing descriptions.
    
    Args:
        row: pandas Series with play-by-play data containing:
            - type: detailed play type (e.g. "3pt jump shot", "driving layup")
            - result: "made" or "missed" for shots
            - event_type: high-level category ("shot", "foul", "rebound", etc.)
            - points: points scored (0, 1, 2, 3)
            - shot_distance: distance in feet for shots
            
    Returns:
        tuple: (event_code, points) where points is only included for scoring events
    """
    play_type = str(row.get('type', '')).lower().strip()
    result = str(row.get('result', '')).lower().strip()
    event_type = str(row.get('event_type', '')).lower().strip()
    points = row.get('points', 0)
    distance = row.get('shot_distance')
    
    # Format distance suffix
    dist_suffix = str(int(distance)) if pd.notna(distance) and distance > 0 else ""
    
    # Handle shots based on type and result
    if event_type == 'shot' or any(shot_word in play_type for shot_word in ['shot', 'layup', 'dunk']):
        
        # 3-point shots
        if '3pt' in play_type or 'three' in play_type:
            if result == 'made' or points == 3:
                return f"3pm{dist_suffix}", 3
            else:
                return f"3pa{dist_suffix}", None
        
        # Layups
        elif 'layup' in play_type:
            if result == 'made' or points == 2:
                return f"layup{dist_suffix}", 2
            else:
                return f"layupa{dist_suffix}", None
        
        # Dunks
        elif 'dunk' in play_type:
            if result == 'made' or points == 2:
                return f"dunk{dist_suffix}", 2
            else:
                return f"dunka{dist_suffix}", None
        
        # Hook shots
        elif 'hook' in play_type:
            if result == 'made' or points == 2:
                return f"hook{dist_suffix}", 2
            else:
                return f"hooka{dist_suffix}", None
        
        # General 2-point shots
        elif any(shot_type in play_type for shot_type in ['jump shot', 'jumper', 'fadeaway', 'floating']):
            if result == 'made' or points == 2:
                return f"2pm{dist_suffix}", 2
            else:
                return f"2pa{dist_suffix}", None
        
        # Fallback for other shots based on points
        elif points == 3:
            return f"3pm{dist_suffix}", 3
        elif points == 2:
            return f"2pm{dist_suffix}", 2
        elif points == 1:
            return "ftm", 1
        else:
            # Missed shot, try to infer type
            if distance and distance >= 23:  # 3-point range
                return f"3pa{dist_suffix}", None
            else:
                return f"2pa{dist_suffix}", None
    
    # Free throws
    elif event_type == 'free throw' or 'free throw' in play_type:
        if result == 'made' or points == 1:
            return "ftm", 1
        else:
            return "ftx", None
    
    # Rebounds  
    elif event_type == 'rebound' or 'rebound' in play_type:
        if 'offensive' in play_type:
            return "o_reb", None
        else:
            return "d_reb", None
    
    # Fouls
    elif event_type == 'foul' or 'foul' in play_type or any(foul_type in play_type for foul_type in ['personal', 'shooting', 'offensive']):
        if 'offensive' in play_type or 'charge' in play_type:
            return "o_foul", None
        elif 'shooting' in play_type or 'flagrant' in play_type:
            return "s_foul", None
        elif 'technical' in play_type:
            return "tech", None
        else:
            return "p_foul", None
    
    # Turnovers
    elif event_type == 'turnover' or any(tov_type in play_type for tov_type in ['bad pass', 'lost ball', 'traveling', 'backcourt']):
        return "tov", None
    
    # Violations
    elif event_type == 'violation' or any(viol_type in play_type for viol_type in ['shot clock', 'kicked ball', 'goaltending']):
        return "viol", None
    
    # Steals (usually tagged as turnover with steal info)
    elif pd.notna(row.get('steal')) and row.get('steal') != '':
        return "stl", None
    
    # Other events
    elif 'jump ball' in play_type:
        return "jumpball", None
    elif 'timeout' in play_type:
        return "timeout", None
    elif 'sub' in play_type:
        return "sub", None
    elif any(period_event in play_type for period_event in ['start of period', 'end of period']):
        return "period", None
    
    # Fallback based on points if we can't categorize
    elif points == 1:
        return "ftm", 1
    elif points == 2:
        return f"2pm{dist_suffix}", 2  
    elif points == 3:
        return f"3pm{dist_suffix}", 3
    else:
        return "unknown", None

def map_description_to_event_code(description, shot_details, score_delta):
    """
    Map play description and shot_details to compact event code.
    
    Args:
        description: Play description text
        shot_details: Dict with 'team' and 'points' info
        score_delta: Points change from previous play
        
    Returns:
        tuple: (event_code, points) where points is only included for scoring events
    """
    if pd.isna(description):
        return "unknown", None
    
    desc = str(description).upper()
    distance = extract_distance_from_description(description)
    dist_suffix = distance if distance else ""
    
    # Determine if this was a scoring play
    points_scored = 0
    if shot_details and shot_details.get('points') is not None:
        points_scored = shot_details['points']
    elif score_delta > 0:
        points_scored = score_delta
    
    # Shot mappings following spec patterns exactly
    # 3-point shots
    if "3PT" in desc:
        if "MISS" in desc:
            return f"3pa{dist_suffix}", None
        else:
            return f"3pm{dist_suffix}", points_scored if points_scored > 0 else 3
    
    # Layups  
    elif "LAYUP" in desc:
        if "MISS" in desc:
            return f"layupa{dist_suffix}", None
        else:
            return f"layup{dist_suffix}", points_scored if points_scored > 0 else 2
    
    # Dunks
    elif "DUNK" in desc and "TIP DUNK" not in desc:
        if "MISS" in desc:
            return f"dunka{dist_suffix}", None
        else:
            return f"dunk{dist_suffix}", points_scored if points_scored > 0 else 2
    
    # Tip dunk shots (specific pattern from spec)
    elif "TIP DUNK" in desc:
        if "MISS" in desc or points_scored == 0:
            return "tipdunk_a", None
        else:
            return "tipdunk_m", points_scored if points_scored > 0 else 2
    
    # Putbacks/tips (scoring)
    elif ("PUTBACK" in desc or "PUT BACK" in desc) and points_scored > 0:
        return "putback2", points_scored
    
    # 2-point shots (Jump, Pullup, Bank shots without 3PT)
    elif ("JUMP SHOT" in desc or "PULLUP" in desc or "BANK SHOT" in desc) and "3PT" not in desc:
        if "MISS" in desc:
            return f"2pa{dist_suffix}", None
        else:
            return f"2pm{dist_suffix}", points_scored if points_scored > 0 else 2
    
    # Free throws
    elif "FREE THROW" in desc:
        if "MISS" in desc or points_scored == 0:
            return "ftx", None
        else:
            return "ftm", 1
    
    # Rebounds
    elif "DEF.REBOUND" in desc or "DEFENSIVE REBOUND" in desc:
        return "d_reb", None
    elif "OFF.REBOUND" in desc or "OFFENSIVE REBOUND" in desc:
        return "o_reb", None
    
    # Steals and turnovers
    elif "STEAL" in desc:
        return "stl", None
    elif ("TURNOVER" in desc or "LOST BALL" in desc or "BAD PASS" in desc or 
          "OUT OF BOUNDS" in desc or "TRAVEL" in desc or "DOUBLE DRIBBLE" in desc):
        return "tov", None
    
    # Fouls
    elif "OFFENSIVE FOUL" in desc or "O.FOUL" in desc:
        return "o_foul", None  # Note: caller should also emit 'tov'
    elif "SHOOTING FOUL" in desc or "S.FOUL" in desc:
        return "s_foul", None
    elif "PERSONAL FOUL" in desc or "P.FOUL" in desc:
        return "p_foul", None
    
    # Timeouts
    elif "TIMEOUT" in desc:
        return "timeout", None
    
    # Default fallback
    else:
        return "unknown", None

def create_lineup_key(players_on_court, away_abbrev, home_abbrev, away_name_to_idx, home_name_to_idx):
    """
    Create a unique lineup key from players_on_court data.
    
    Returns:
        tuple: (away_indices, home_indices) sorted tuples of 5 player indices each
    """
    away_indices = []
    home_indices = []
    
    if not players_on_court:
        # Return default lineup if no data
        return (tuple([0, 1, 2, 3, 4]), tuple([0, 1, 2, 3, 4]))
    
    for team_data in players_on_court:
        team_abbrev = team_data.get('team', '')
        players = team_data.get('players', [])
        
        if team_abbrev == away_abbrev:
            for player in players[:5]:  # Take first 5 players
                idx = away_name_to_idx.get(player, 0)  # Default to first player if not found
                away_indices.append(idx)
        elif team_abbrev == home_abbrev:
            for player in players[:5]:  # Take first 5 players
                idx = home_name_to_idx.get(player, 0)  # Default to first player if not found
                home_indices.append(idx)
    
    # Ensure we have 5 players for each team, pad with 0s if needed
    while len(away_indices) < 5:
        away_indices.append(0)
    while len(home_indices) < 5:
        home_indices.append(0)
    
    # Return tuples preserving original order (don't sort)
    return (tuple(away_indices[:5]), tuple(home_indices[:5]))

def resolve_actor(player_name, shot_details, away_abbrev, home_abbrev, away_name_to_idx, home_name_to_idx):
    """
    Resolve player/team actor to ["A"|"H", idx] format.
    
    Returns:
        list: ["A"|"H", player_index] or ["A"|"H", -1] for team events
    """
    if pd.notna(player_name) and player_name:
        # Try to find player in away team
        if player_name in away_name_to_idx:
            return ["A", away_name_to_idx[player_name]]
        # Try to find player in home team
        elif player_name in home_name_to_idx:
            return ["H", home_name_to_idx[player_name]]
    
    # Try to infer from shot_details if player not found
    if shot_details and shot_details.get('team'):
        team = shot_details['team']
        if team == away_abbrev:
            return ["A", -1]  # Team event
        elif team == home_abbrev:
            return ["H", -1]  # Team event
    
    # Default fallback - assign to away team
    return ["A", -1]

def build_compact_training_data_direct(
    current_game_id, away_abbrev, home_abbrev, 
    away_stats, home_stats, 
    away_players, home_players,
    recent_plays_verbose, away_name_to_idx, home_name_to_idx,
    for_first_n_plays=False
):
    """
    🚀 OPTIMIZED: Build compact schema format directly without verbose intermediate step.
    
    This replaces the verbose → compact conversion with direct compact generation,
    eliminating the expensive conversion overhead.
    
    Args:
        current_game_id: Game ID for this training example
        away_abbrev, home_abbrev: Team abbreviations 
        away_stats, home_stats: Team stats dictionaries
        away_players, home_players: Player arrays already in compact format
        recent_plays_verbose: List of play dictionaries (from the verbose building process)
        away_name_to_idx, home_name_to_idx: Player name to index mappings
        
    Returns:
        dict: Compact format following the specification
    """
    
    # Extract team stats in required order: [OEFF, DEFF, PACE, REST_DAYS]
    away_stats_array = [
        round(float(away_stats.get('OEFF', 110.0)), 2),
        round(float(away_stats.get('DEFF', 110.0)), 2), 
        round(float(away_stats.get('PACE', 100.0)), 2),
        int(away_stats.get('REST_DAYS', 2))
    ]
    
    home_stats_array = [
        round(float(home_stats.get('OEFF', 110.0)), 2),
        round(float(home_stats.get('DEFF', 110.0)), 2),
        round(float(home_stats.get('PACE', 100.0)), 2), 
        int(home_stats.get('REST_DAYS', 2))
    ]
    
    # Build lineup lookup and plays array directly
    lineup_cache = {}  # Maps lineup keys to lineup IDs
    lineup_lookup = []  # List of {A: [...], H: [...]} objects
    plays_array = []
    
    prev_score = [0, 0]
    current_lineup_id = 0
    
    # Process plays to build compact arrays directly
    for play in recent_plays_verbose:
        quarter = int(play.get('quarter', 1))
        time_remaining = play.get('time_remaining', '12:00')
        time_seconds = parse_time_to_seconds(time_remaining)
        
        score_str = play.get('score', '0 - 0')
        current_score = parse_score_string(score_str)
        
        # Calculate score delta for this play
        score_delta = max(0, max(
            current_score[0] - prev_score[0],  # Away team scored
            current_score[1] - prev_score[1]   # Home team scored
        ))
        
        # Get lineup for this play  
        players_on_court = play.get('players_on_court', [])
        lineup_key = create_lineup_key(
            players_on_court, away_abbrev, home_abbrev,
            away_name_to_idx, home_name_to_idx
        )
        
        # Get or create lineup ID
        if lineup_key not in lineup_cache:
            lineup_id = len(lineup_lookup)
            lineup_cache[lineup_key] = lineup_id
            lineup_lookup.append({
                "A": list(lineup_key[0]),
                "H": list(lineup_key[1])
            })
            current_lineup_id = lineup_id
        else:
            current_lineup_id = lineup_cache[lineup_key]
        
        # Resolve actor
        player_name = play.get('player')
        shot_details = play.get('shot_details', {})
        actor = resolve_actor(
            player_name, shot_details, away_abbrev, home_abbrev,
            away_name_to_idx, home_name_to_idx
        )
        
        # Map to event code - prefer structured data when available
        if all(key in play for key in ['type', 'event_type']):
            # Use structured mapping when available (better accuracy)
            event_code, points = map_structured_to_event_code(play)
        else:
            # Fall back to description parsing
            description = play.get('description', '')
            event_code, points = map_description_to_event_code(
                description, shot_details, score_delta
            )
        
        # Handle offensive foul special case - emit both o_foul and tov
        if event_code == "o_foul":
            # Add the offensive foul
            play_tuple = [quarter, time_seconds, current_score, actor, "o_foul", 0, current_lineup_id]
            plays_array.append(play_tuple)
            
            # Add the turnover at the same timestamp
            play_tuple = [quarter, time_seconds, current_score, actor, "tov", 0, current_lineup_id]
            plays_array.append(play_tuple)
        else:
            # Build play tuple - include points only for scoring events
            play_points = int(points) if points is not None else 0
            play_tuple = [quarter, time_seconds, current_score, actor, event_code, play_points, current_lineup_id]
            
            plays_array.append(play_tuple)
        
        prev_score = current_score
    
    # Build compact format directly
    compact_record = {
        "A": away_abbrev,
        "H": home_abbrev,
        "as": away_stats_array,
        "hs": home_stats_array,
        "ap": away_players,
        "hp": home_players,
        "L": lineup_lookup
    }
    
    # For first_N_plays mode, exclude the "p" field to create clean contexts
    # For regular mode, include recent plays 
    if not for_first_n_plays:
        compact_record["p"] = plays_array
    
    return compact_record


def get_player_pca_from_cache_or_calculate(player_name, game_date, season, pca_cache):
    """
    🚀 Get PCA scores from cache if available, otherwise calculate individually.
    
    Args:
        player_name: Name of the player
        game_date: Date of the game
        season: Season year (e.g. "2023-2024")
        pca_cache: Pre-calculated PCA cache dict
        
    Returns:
        tuple: (offense, defense, shot_selection, efficiency) scores
    """
    # 🚀 OPTIMIZED: Prioritize batch cache usage
    if pca_cache and game_date in pca_cache:
        # Use pre-calculated batch PCA scores (FASTEST path - batch optimization)
        player_scores = pca_cache[game_date].get(player_name, {})
        if player_scores:  # Found in batch cache
            return (
                player_scores.get('offense', 0.0),
                player_scores.get('defense', 0.0), 
                player_scores.get('shot_selection', 0.0),
                player_scores.get('efficiency', 0.0)
            )
    
    # Fall back to individual calculation (slower path - should be rare when batch_pca=True)
    try:
        from pca_optimized import get_player_pca_score
        return get_player_pca_score(player_name, game_date, season)
    except Exception:
        # Final fallback - return zeros to avoid breaking the pipeline
        return (0.0, 0.0, 0.0, 0.0)


def test_direct_compact_builder(result_df, test_indices=[0, 1, 2]):
    """
    🧪 TEST FUNCTION: Compare direct compact builder with verbose → convert method.
    
    This validates that our optimized direct builder produces identical output
    to the existing verbose → convert pipeline.
    
    Args:
        result_df: DataFrame with training data
        test_indices: List of row indices to test with
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("🧪 Testing direct compact builder vs verbose conversion...")
    
    # Get required data (same as main loop setup)
    game_team_mapping = determine_home_away_teams(result_df)
    
    all_tests_passed = True
    
    for i in test_indices:
        if i >= len(result_df):
            continue
            
        print(f"  Testing row {i}...")
        
        try:
            current_game_id = result_df.iloc[i]['game_id']
            
            # Get team info (simplified version of main loop logic)
            team_stats = {}  # We'd need the actual team stats here
            away_abbrev = game_team_mapping.get(current_game_id, {}).get('away_team', 'AWAY')
            home_abbrev = game_team_mapping.get(current_game_id, {}).get('home_team', 'HOME')
            
            # For testing, we'll use mock data
            mock_away_stats = {'OEFF': 115.0, 'DEFF': 110.0, 'PACE': 98.0, 'REST_DAYS': 2}
            mock_home_stats = {'OEFF': 112.0, 'DEFF': 108.0, 'PACE': 101.0, 'REST_DAYS': 1}
            mock_away_players = [["Player A1", 1.0, 0.5, -0.2, 0.8, 28, 20]]
            mock_home_players = [["Player H1", 0.8, 1.2, 0.1, -0.3, 32, 18]]
            mock_away_name_to_idx = {"Player A1": 0}
            mock_home_name_to_idx = {"Player H1": 0}
            mock_recent_plays = [
                {
                    "quarter": 1,
                    "time_remaining": "10:30", 
                    "score": f"{away_abbrev} 5 - {home_abbrev} 7",
                    "description": "Test play",
                    "players_on_court": [
                        {"team": away_abbrev, "players": ["Player A1"]},
                        {"team": home_abbrev, "players": ["Player H1"]}
                    ],
                    "player": "Player A1",
                    "shot_details": {"team": None, "points": None}
                }
            ]
            
            # Test Method 1: Verbose → Convert (existing)
            verbose_json_obj = {
                "away_team": {
                    "name": away_abbrev,
                    "stats": mock_away_stats,
                    "players": [{"name": p[0], "profile": {"offense": p[1], "defense": p[2], "shot_selection": p[3], "efficiency": p[4], "MPG": p[5], "usage": p[6]}} for p in mock_away_players]
                },
                "home_team": {
                    "name": home_abbrev, 
                    "stats": mock_home_stats,
                    "players": [{"name": p[0], "profile": {"offense": p[1], "defense": p[2], "shot_selection": p[3], "efficiency": p[4], "MPG": p[5], "usage": p[6]}} for p in mock_home_players]
                },
                "recent_plays": mock_recent_plays
            }
            
            old_result = convert_verbose_to_compact(verbose_json_obj)
            
            # Test Method 2: Direct compact builder (optimized)
            new_result = build_compact_training_data_direct(
                current_game_id, away_abbrev, home_abbrev,
                mock_away_stats, mock_home_stats,
                mock_away_players, mock_home_players,
                mock_recent_plays, mock_away_name_to_idx, mock_home_name_to_idx
            )
            
            # Compare results
            if old_result == new_result:
                print(f"    ✅ Row {i}: PASS - Results identical")
            else:
                print(f"    ❌ Row {i}: FAIL - Results differ")
                print(f"       Old keys: {old_result.keys()}")
                print(f"       New keys: {new_result.keys()}")
                
                # Show detailed diff for debugging
                for key in old_result.keys():
                    if key not in new_result:
                        print(f"       Missing key in new: {key}")
                    elif old_result[key] != new_result[key]:
                        print(f"       Key '{key}' differs:")
                        print(f"         Old: {old_result[key]}")
                        print(f"         New: {new_result[key]}")
                
                all_tests_passed = False
                
        except Exception as e:
            print(f"    ❌ Row {i}: ERROR - {e}")
            all_tests_passed = False
    
    if all_tests_passed:
        print("🎉 All tests PASSED! Direct builder produces identical output.")
    else:
        print("⚠️  Some tests FAILED! Review differences above.")
    
    return all_tests_passed


def convert_verbose_to_compact(verbose_json, for_first_n_plays=False):
    """
    Convert verbose JSON format to compact schema format.
    
    Args:
        verbose_json: Dict in old format with away_team, home_team, recent_plays
        
    Returns:
        dict: Compact format following the specification
    """
    if isinstance(verbose_json, str):
        verbose_json = json.loads(verbose_json)
    
    # Extract team info
    away_team = verbose_json.get('away_team', {})
    home_team = verbose_json.get('home_team', {})
    recent_plays = verbose_json.get('recent_plays', [])
    
    away_abbrev = away_team.get('name', 'AWAY')
    home_abbrev = home_team.get('name', 'HOME')
    
    # Extract team stats in required order: [OEFF, DEFF, PACE, REST_DAYS]
    away_stats_dict = away_team.get('stats', {})
    home_stats_dict = home_team.get('stats', {})
    
    away_stats = [
        round(float(away_stats_dict.get('OEFF', 110.0)), 2),
        round(float(away_stats_dict.get('DEFF', 110.0)), 2),
        round(float(away_stats_dict.get('PACE', 100.0)), 2),
        int(away_stats_dict.get('REST_DAYS', 2))
    ]
    
    home_stats = [
        round(float(home_stats_dict.get('OEFF', 110.0)), 2),
        round(float(home_stats_dict.get('DEFF', 110.0)), 2),
        round(float(home_stats_dict.get('PACE', 100.0)), 2),
        int(home_stats_dict.get('REST_DAYS', 2))
    ]
    
    # Extract player rosters in format: [name, offense, defense, shot_selection, efficiency, MPG, usage]
    away_players = []
    home_players = []
    away_name_to_idx = {}
    home_name_to_idx = {}
    
    # Process away team players
    for idx, player in enumerate(away_team.get('players', [])):
        name = player.get('name', f'Player{idx}')
        profile = player.get('profile', {})
        
        player_array = [
            name,
            round(float(profile.get('offense', 0.0)), 2),
            round(float(profile.get('defense', 0.0)), 2),
            round(float(profile.get('shot_selection', 0.0)), 2),
            round(float(profile.get('efficiency', 0.0)), 2),
            int(profile.get('MPG', 20)),
            int(profile.get('usage', 15)),
            0  # fouls (default to 0 for start of game prediction context)
        ]
        away_players.append(player_array)
    
    # Process home team players
    for idx, player in enumerate(home_team.get('players', [])):
        name = player.get('name', f'Player{idx}')
        profile = player.get('profile', {})
        
        player_array = [
            name,
            round(float(profile.get('offense', 0.0)), 2),
            round(float(profile.get('defense', 0.0)), 2),
            round(float(profile.get('shot_selection', 0.0)), 2),
            round(float(profile.get('efficiency', 0.0)), 2),
            int(profile.get('MPG', 20)),
            int(profile.get('usage', 15)),
            0  # fouls (default to 0 for start of game prediction context)
        ]
        home_players.append(player_array)
    
    # 🎯 OPTIMIZATION: Sort players by MPG (descending) for better model learning
    # High-MPG players (starters) at low indices makes patterns easier to learn
    # Player array format: [name, offense, defense, shot_selection, efficiency, MPG, usage, fouls]
    #                       [  0,     1,       2,        3,              4,         5,    6,     7  ]
    away_players.sort(key=lambda p: p[5], reverse=True)  # p[5] is MPG
    home_players.sort(key=lambda p: p[5], reverse=True)
    
    # Build name-to-index mappings AFTER sorting
    away_name_to_idx = {player[0]: idx for idx, player in enumerate(away_players)}
    home_name_to_idx = {player[0]: idx for idx, player in enumerate(home_players)}
    
    # Process plays to build lineup lookup and play array
    lineup_cache = {}  # Maps lineup keys to lineup IDs
    lineup_lookup = []  # List of {A: [...], H: [...]} objects
    plays_array = []
    
    prev_score = [0, 0]
    current_lineup_id = 0
    
    for play in recent_plays:
        quarter = int(play.get('quarter', 1))
        time_remaining = play.get('time_remaining', '12:00')
        time_seconds = parse_time_to_seconds(time_remaining)
        
        score_str = play.get('score', '0 - 0')
        current_score = parse_score_string(score_str)
        
        # Calculate score delta for this play
        score_delta = max(0, max(
            current_score[0] - prev_score[0],  # Away team scored
            current_score[1] - prev_score[1]   # Home team scored
        ))
        
        # Get lineup for this play
        players_on_court = play.get('players_on_court', [])
        lineup_key = create_lineup_key(
            players_on_court, away_abbrev, home_abbrev, 
            away_name_to_idx, home_name_to_idx
        )
        
        # Get or create lineup ID
        if lineup_key not in lineup_cache:
            lineup_id = len(lineup_lookup)
            lineup_cache[lineup_key] = lineup_id
            lineup_lookup.append({
                "A": list(lineup_key[0]),
                "H": list(lineup_key[1])
            })
            current_lineup_id = lineup_id
        else:
            current_lineup_id = lineup_cache[lineup_key]
        
        # Resolve actor
        player_name = play.get('player')
        shot_details = play.get('shot_details', {})
        actor = resolve_actor(
            player_name, shot_details, away_abbrev, home_abbrev,
            away_name_to_idx, home_name_to_idx
        )
        
        # Map to event code - prefer structured data when available
        if all(key in play for key in ['type', 'event_type']):
            # Use structured mapping when available (better accuracy)
            event_code, points = map_structured_to_event_code(play)
        else:
            # Fall back to description parsing
            description = play.get('description', '')
            event_code, points = map_description_to_event_code(
                description, shot_details, score_delta
            )
        
        # Handle offensive foul special case - emit both o_foul and tov
        if event_code == "o_foul":
            # Add the offensive foul (non-scoring, so points=0)
            play_tuple = [quarter, time_seconds, current_score, actor, "o_foul", 0, current_lineup_id]
            plays_array.append(play_tuple)
            
            # Add the turnover at the same timestamp (non-scoring, so points=0)
            play_tuple = [quarter, time_seconds, current_score, actor, "tov", 0, current_lineup_id]
            plays_array.append(play_tuple)
        else:
            # Build play tuple - ALWAYS 7 elements [q, t, score, actor, event, points, lineup_id]
            # Use points=0 for non-scoring events
            if points is None:
                points = 0
            play_tuple = [quarter, time_seconds, current_score, actor, event_code, points, current_lineup_id]
            plays_array.append(play_tuple)
        
        prev_score = current_score
    
    # Calculate derived fields for the context
    if plays_array:
        # If we have plays, derive from the last play
        last_play = plays_array[-1]
        last_score = last_play[2]  # [away, home]
        sd = last_score[0] - last_score[1]  # score difference (away - home)
        
        # For possession, we can't easily derive it from verbose format without event code analysis
        # Default to "N" (unknown) for now - orchestrator should handle this correctly
        pos = "N"
        
        # For team fouls, we also can't easily derive from verbose without analyzing all plays
        # Default to [0, 0] - orchestrator should handle or we'd need full game analysis
        tb = [0, 0]
    else:
        # No plays - start of game defaults
        sd = 0
        pos = "N"
        tb = [0, 0]
    
    # Build compact format with new fields
    compact_record = {
        "A": away_abbrev,
        "H": home_abbrev,
        "as": away_stats,
        "hs": home_stats,
        "ap": away_players,
        "hp": home_players,
        "ap_count": len(away_players),  # Roster size metadata
        "hp_count": len(home_players),  # Roster size metadata
        "L": lineup_lookup,
        "pos": pos,   # Possession: "A", "H", or "N" (unknown)
        "tb": tb,     # Team bonus/fouls: [away_fouls_in_quarter, home_fouls_in_quarter]
        "sd": sd      # Score difference: away_score - home_score
    }
    
    # For first_N_plays mode, exclude the "p" field to create clean contexts
    # For regular mode, include recent plays
    if not for_first_n_plays:
        compact_record["p"] = plays_array
    
    return compact_record

def convert_compact_to_verbose(compact_json):
    """
    Convert compact schema format back to verbose JSON format.
    Used for backward compatibility and debugging.
    
    Args:
        compact_json: Dict in compact format
        
    Returns:
        dict: Verbose format matching original schema
    """
    if isinstance(compact_json, str):
        compact_json = json.loads(compact_json)
    
    away_abbrev = compact_json.get('A', 'AWAY')  # Use uppercase key
    home_abbrev = compact_json.get('H', 'HOME')  # Use uppercase key
    
    # Convert team stats back
    away_stats_array = compact_json.get('as', [110.0, 110.0, 100.0, 2])
    home_stats_array = compact_json.get('hs', [110.0, 110.0, 100.0, 2])
    
    away_stats = {
        'OEFF': away_stats_array[0],
        'DEFF': away_stats_array[1],
        'PACE': away_stats_array[2],
        'REST_DAYS': away_stats_array[3]
    }
    
    home_stats = {
        'OEFF': home_stats_array[0],
        'DEFF': home_stats_array[1],
        'PACE': home_stats_array[2],
        'REST_DAYS': home_stats_array[3]
    }
    
    # Convert player rosters back
    away_players = []
    for player_array in compact_json.get('ap', []):
        player = {
            'name': player_array[0],
            'profile': {
                'offense': player_array[1],
                'defense': player_array[2],
                'shot_selection': player_array[3],
                'efficiency': player_array[4],
                'MPG': player_array[5],
                'usage': player_array[6]
            }
        }
        away_players.append(player)
    
    home_players = []
    for player_array in compact_json.get('hp', []):
        player = {
            'name': player_array[0],
            'profile': {
                'offense': player_array[1],
                'defense': player_array[2],
                'shot_selection': player_array[3],
                'efficiency': player_array[4],
                'MPG': player_array[5],
                'usage': player_array[6]
            }
        }
        home_players.append(player)
    
    # Convert plays back to verbose format
    lineup_lookup = compact_json.get('L', [])
    plays_array = compact_json.get('p', [])
    
    recent_plays = []
    for play_tuple in plays_array:
        if len(play_tuple) < 6:
            continue
            
        quarter = play_tuple[0]
        time_seconds = play_tuple[1]
        score_array = play_tuple[2]
        actor = play_tuple[3]
        event_code = play_tuple[4]
        
        # Handle both scoring and non-scoring play formats
        if len(play_tuple) == 7:  # Scoring play: [q, t, score, actor, event, pts, lineup_id]
            points = play_tuple[5]
            lineup_id = play_tuple[6]
        else:  # Non-scoring play: [q, t, score, actor, event, lineup_id]
            points = None
            lineup_id = play_tuple[5]
        
        # Convert time back to MM:SS format
        minutes = time_seconds // 60
        seconds = time_seconds % 60
        time_remaining = f"{minutes:02d}:{seconds:02d}"
        
        # Build score string
        score = f"{away_abbrev} {score_array[0]} - {home_abbrev} {score_array[1]}"
        
        # Resolve player name from actor
        player_name = None
        if actor[1] != -1:  # Not a team event
            if actor[0] == "A" and actor[1] < len(away_players):
                player_name = away_players[actor[1]]['name']
            elif actor[0] == "H" and actor[1] < len(home_players):
                player_name = home_players[actor[1]]['name']
        
        # Convert event code back to description (simplified)
        description = f"Converted from {event_code}"
        
        # Build players_on_court from lineup
        players_on_court = []
        if lineup_id < len(lineup_lookup):
            lineup = lineup_lookup[lineup_id]
            away_lineup_indices = lineup.get('A', [])
            home_lineup_indices = lineup.get('H', [])
            
            away_lineup_names = []
            for idx in away_lineup_indices:
                if idx < len(away_players):
                    away_lineup_names.append(away_players[idx]['name'])
            
            home_lineup_names = []
            for idx in home_lineup_indices:
                if idx < len(home_players):
                    home_lineup_names.append(home_players[idx]['name'])
            
            players_on_court = [
                {'team': away_abbrev, 'players': away_lineup_names},
                {'team': home_abbrev, 'players': home_lineup_names}
            ]
        
        # Build shot_details
        shot_details = {'team': None, 'points': None}
        if points is not None:
            if actor[0] == "A":
                shot_details['team'] = away_abbrev
            else:
                shot_details['team'] = home_abbrev
            shot_details['points'] = points
        
        play = {
            'quarter': quarter,
            'time_remaining': time_remaining,
            'score': score,
            'players_on_court': players_on_court,
            'player': player_name,
            'description': description,
            'shot_details': shot_details
        }
        
        recent_plays.append(play)
    
    # Build verbose format
    verbose_record = {
        'away_team': {
            'name': away_abbrev,
            'stats': away_stats,
            'players': away_players
        },
        'home_team': {
            'name': home_abbrev,
            'stats': home_stats,
            'players': home_players
        },
        'recent_plays': recent_plays
    }
    
    return verbose_record

def load_play_by_play_data(season_year):
    """
    Load play-by-play data for the specified season year.
    
    Args:
        season_year (str): Season year in format "YYYY-YYYY" (e.g., "2023-2024")
    
    Returns:
        pd.DataFrame: Loaded play-by-play data
    """
    # Map season year to file path
    file_mapping = {
        "2022-2023": r"C:\Users\micha\nba_predictions\data\play_by_play\historical\[10-18-2022]-[06-12-2023]-combined-stats.csv",
        "2023-2024": r"C:\Users\micha\nba_predictions\data\play_by_play\historical\[10-24-2023]-[06-17-2024]-combined-stats.csv",
        "2024-2025": r"C:\Users\micha\nba_predictions\data\play_by_play\historical\[10-22-2024]-[06-22-2025]-combined-stats.csv"
    }
    
    if season_year not in file_mapping:
        raise ValueError(f"Season year {season_year} not supported. Available options: {list(file_mapping.keys())}")
    
    file_path = file_mapping[season_year]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Play-by-play data file not found: {file_path}")
    
    print(f"Loading play-by-play data for season {season_year}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of play-by-play data")
    
    return df

def set_season_year(season_year):
    """
    Set the season year for data loading.
    
    Args:
        season_year (str): Season year in format "YYYY-YYYY" (e.g., "2023-2024")
    """
    global SEASON_YEAR
    SEASON_YEAR = season_year
    print(f"Season year set to: {SEASON_YEAR}")

def get_current_season_year():
    """Get the currently configured season year."""
    return SEASON_YEAR

def load_training_data(season_year=None):
    """
    Load all necessary data for training data generation.
    
    Args:
        season_year (str, optional): Season year to load. If None, uses current SEASON_YEAR
    
    Returns:
        pd.DataFrame: Play-by-play data for the specified season
    """
    if season_year is None:
        season_year = SEASON_YEAR
    
    # Load the play-by-play data
    play_by_play_df = load_play_by_play_data(season_year)
    
    return play_by_play_df

# Example usage and testing functions
def test_data_loading():
    """Test function to verify all imports and data loading work correctly."""
    print("Testing data loading functionality...")
    
    # Test loading play-by-play data for both seasons
    for season in ["2022-2023", "2023-2024"]:
        try:
            print(f"\n--- Testing season {season} ---")
            df = load_play_by_play_data(season)
            print(f"Successfully loaded {len(df)} rows for {season}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading {season}: {e}")
    
    # Test imported functions
    print("\n--- Testing imported functions ---")
    try:
        players = get_distinct_players()
        print(f"Found {len(players)} distinct players")
        
        teams = get_available_teams()
        print(f"Found {len(teams)} available teams")
        
        print("All imports and basic functionality working correctly!")
        
    except Exception as e:
        print(f"Error testing imported functions: {e}")

def calculate_game_time_remaining(period, remaining_time):
    """
    Calculate total game time remaining based on period and remaining time in current period.
    
    Args:
        period (int): Current period (1-4)
        remaining_time (str): Time remaining in current period (format: "0:MM:SS")
        
    Returns:
        str: Total game time remaining (format: "MM:SS")
    """
    try:
        # Parse remaining_time string (format: "0:MM:SS")
        if pd.isna(remaining_time) or not remaining_time:
            return "00:00"
            
        time_parts = str(remaining_time).split(':')
        if len(time_parts) >= 2:
            # Get minutes and seconds from the current period
            minutes = int(time_parts[-2])  # Second to last part is minutes
            seconds = int(time_parts[-1])   # Last part is seconds
        else:
            return "00:00"
        
        # Calculate total minutes remaining based on period
        # NBA: 4 periods of 12 minutes each = 48 minutes total
        if period == 1:
            total_minutes_remaining = 36 + minutes  # 3 full periods + current period remaining
        elif period == 2:
            total_minutes_remaining = 24 + minutes  # 2 full periods + current period remaining
        elif period == 3:
            total_minutes_remaining = 12 + minutes  # 1 full period + current period remaining
        elif period == 4:
            total_minutes_remaining = minutes       # Only current period remaining
        else:
            # Handle overtime or invalid periods
            total_minutes_remaining = minutes
            
        return f"{total_minutes_remaining:02d}:{seconds:02d}"
        
    except (ValueError, IndexError, TypeError):
        return "00:00"

def convert_to_quarter_time(period, remaining_time):
    """
    Convert period and remaining_time to quarter and time_remaining for JSON format.
    
    Args:
        period (int): Current period (1-4)
        remaining_time (str): Time remaining in current period (format: "0:MM:SS")
        
    Returns:
        tuple: (quarter, time_remaining_in_quarter)
    """
    try:
        if pd.isna(remaining_time) or not remaining_time:
            return period, "00:00"
            
        time_parts = str(remaining_time).split(':')
        if len(time_parts) >= 2:
            minutes = int(time_parts[-2])
            seconds = int(time_parts[-1])
            time_in_quarter = f"{minutes:02d}:{seconds:02d}"
            return period, time_in_quarter
        else:
            return period, "00:00"
    except (ValueError, IndexError, TypeError):
        return period, "00:00"

def determine_scoring_info(prev_away_score, prev_home_score, curr_away_score, curr_home_score, away_abbrev, home_abbrev):
    """
    Determine scoring team and points scored based on score changes.
    
    Returns:
        tuple: (scoring_team, points_scored)
    """
    away_diff = curr_away_score - prev_away_score
    home_diff = curr_home_score - prev_home_score
    
    if away_diff > 0:
        return away_abbrev, away_diff
    elif home_diff > 0:
        return home_abbrev, home_diff
    else:
        return None, 0

def create_team_abbreviation_mapping():
    """
    Create mapping from 3-letter team abbreviations to full team names.
    
    Returns:
        dict: Mapping from abbreviation to full team name
    """
    return {
        'ATL': 'Atlanta Hawks',
        'BKN': 'Brooklyn Nets', 
        'BOS': 'Boston Celtics',
        'CHA': 'Charlotte Hornets',
        'CHI': 'Chicago Bulls',
        'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks',
        'DEN': 'Denver Nuggets',
        'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors',
        'HOU': 'Houston Rockets',
        'IND': 'Indiana Pacers',
        'LAC': 'Los Angeles Clippers',
        'LAL': 'Los Angeles Lakers',
        'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat',
        'MIL': 'Milwaukee Bucks',
        'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans',
        'NYK': 'New York Knicks',
        'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic',
        'PHI': 'Philadelphia 76ers',
        'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers',
        'SAC': 'Sacramento Kings',
        'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors',
        'UTA': 'Utah Jazz',
        'WAS': 'Washington Wizards'
    }

def remove_parentheses_content(text):
    """
    Remove content within parentheses (including the parentheses) from text.
    
    Args:
        text (str): Input text that may contain parentheses
        
    Returns:
        str: Text with parentheses content removed and extra spaces cleaned up
    
    Example:
        "Lebron James 3-pt make (17 pts)" -> "Lebron James 3-pt make"
    """
    if not text or pd.isna(text):
        return text
    
    # Remove content within parentheses using regex
    # \([^)]*\) matches opening paren, any chars except closing paren, closing paren
    cleaned_text = re.sub(r'\([^)]*\)', '', str(text))
    
    # Clean up extra whitespace that may result from removal
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text

def get_prior_season(current_season):
    """
    Get the prior season year from current season.
    
    Args:
        current_season (str): Current season in format "YYYY-YYYY"
        
    Returns:
        str: Prior season in format "YYYY-YYYY"
    """
    try:
        # Extract the ending year (e.g., "2023-2024" -> "2024")
        end_year = int(current_season.split('-')[1])
        prior_end_year = end_year - 1
        prior_start_year = prior_end_year - 1
        return f"{prior_start_year}-{prior_end_year}"
    except (ValueError, IndexError):
        return "2022-2023"  # Default fallback

def get_team_stats_for_game(game_df, team_mapping, target_date=None, min_games_threshold=10):
    """
    Get team stats for both teams in a game.
    Falls back to PRIOR season averages if team has played fewer than min_games_threshold
    in current season before target_date. This prevents data leakage and ensures
    sufficient sample size for reliable current season stats.
    
    Args:
        game_df (pd.DataFrame): DataFrame for a single game
        team_mapping (dict): Mapping of game_id to home/away teams
        target_date (str, optional): Date for stats calculation
        min_games_threshold (int): Minimum games before using current season (default: 10)
        
    Returns:
        dict: Team stats for home and away teams
    """
    # 🚀 OPTIMIZED: Could use pre-calculated mapping here too, but this is only called once per game
    abbrev_mapping = create_team_abbreviation_mapping()
    game_id = game_df.iloc[0]['game_id']
    
    # Get team abbreviations from mapping
    home_abbrev = team_mapping.get(game_id, {}).get('home_team', 'Unknown')
    away_abbrev = team_mapping.get(game_id, {}).get('away_team', 'Unknown')
    
    # Convert to full team names
    home_team_full = abbrev_mapping.get(home_abbrev, home_abbrev)
    away_team_full = abbrev_mapping.get(away_abbrev, away_abbrev)
    
    # Determine prior season for fallback (prevents data leakage)
    prior_season = get_prior_season(SEASON_YEAR)
    
    def get_stats_with_threshold(team_name):
        """Get team stats, falling back to prior season if < min_games_threshold"""
        current_stats = generate_team_stats(team_name, target_date)
        
        # If no current season stats OR fewer than threshold games, use prior season
        if not current_stats or current_stats.get('GAMES_PLAYED', 0) < min_games_threshold:
            fallback_stats = generate_team_stats(team_name, None, fallback_season=prior_season)
            if fallback_stats:
                # Add metadata to indicate this is a fallback
                fallback_stats['FALLBACK_REASON'] = f'Insufficient current season games ({current_stats.get("GAMES_PLAYED", 0) if current_stats else 0} < {min_games_threshold})'
                fallback_stats['USING_PRIOR_SEASON'] = True
                
                # FIX: Handle REST_DAYS properly for insufficient games
                games_played = current_stats.get('GAMES_PLAYED', 0) if current_stats else 0
                if games_played == 0:
                    # First game of season: Default to well-rested value
                    fallback_stats['REST_DAYS'] = 10
                else:
                    # Has some games: Use current season's actual rest calculation
                    fallback_stats['REST_DAYS'] = current_stats.get('REST_DAYS')
                    
            return fallback_stats or {}
        else:
            # Sufficient current season games, use current stats
            current_stats['USING_PRIOR_SEASON'] = False
            return current_stats
    
    # Get stats for both teams using the threshold logic
    home_stats = get_stats_with_threshold(home_team_full)
    away_stats = get_stats_with_threshold(away_team_full)
    
    return {
        'home_team_stats': home_stats,
        'away_team_stats': away_stats,
        'home_abbrev': home_abbrev,
        'away_abbrev': away_abbrev
    }

def determine_home_away_teams(df):
    """
    Determine which team is home and which is away for each game by tracking score increments.
    
    Args:
        df (pd.DataFrame): Play-by-play DataFrame with game_id, team, away_score, home_score columns
        
    Returns:
        dict: Dictionary mapping game_id to {'home_team': team_name, 'away_team': team_name}
    """
    game_team_mapping = {}
    
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
        
        home_team = None
        away_team = None
        prev_away_score = 0
        prev_home_score = 0
        
        for i, row in game_df.iterrows():
            current_away_score = row.get('away_score', 0) or 0
            current_home_score = row.get('home_score', 0) or 0
            team = row.get('team', '')
            
            # Check if away score incremented and we haven't identified away team yet
            if current_away_score > prev_away_score and away_team is None:
                away_team = team
                
            # Check if home score incremented and we haven't identified home team yet  
            if current_home_score > prev_home_score and home_team is None:
                home_team = team
                
            # Update previous scores
            prev_away_score = current_away_score
            prev_home_score = current_home_score
            
            # Break early if we've identified both teams
            if home_team and away_team:
                break
        
        # Store the mapping for this game
        game_team_mapping[game_id] = {
            'home_team': home_team or 'Unknown',
            'away_team': away_team or 'Unknown'
        }
    
    return game_team_mapping

def create_llm_training_data(df, n_total=5, filter_nan=True, generation_mode="remaining_plays", use_direct_compact=False, use_batch_pca=False):
    """
    Create LLM training data in compact schema format with team stats and recent plays.
    Only includes plays within the same game (respects game_id boundaries).
    
    Args:
        df (pd.DataFrame): Play-by-play DataFrame with required columns
        n_total (int): Total number of recent plays to include (default: 5)
        filter_nan (bool): Whether to filter out rows with NaN descriptions (default: True)
        generation_mode (str): Training data generation mode:
            - "remaining_plays": Skip first N plays, generate training data for plays N+1 onwards
            - "first_N_plays": Generate training data for all plays (first_N_plays mode handles the filtering)
        use_direct_compact (bool): 🚀 OPTIMIZATION #1: If True, build compact format directly 
            instead of verbose → convert (30-50% faster, identical output)
        use_batch_pca (bool): 🚀 OPTIMIZATION #2: If True, pre-calculate PCA scores for all 
            unique dates instead of individual calls (10-50x faster)
    
    Returns:
        pd.DataFrame: DataFrame with new 'json_training_data' column containing JSON strings in chosen format
            (compact if use_direct_compact=True, verbose if use_direct_compact=False)
    """
    # Work with a copy to avoid modifying original DataFrame
    result_df = df.copy()
    
    # Filter out NaN descriptions if requested
    if filter_nan:
        result_df = result_df.dropna(subset=['description']).reset_index(drop=True)
    
    # Determine home/away team mapping for all games
    print("Determining home/away team mappings...")
    game_team_mapping = determine_home_away_teams(result_df)
    
    # 🚀 OPTIMIZATION #3: Pre-calculate team abbreviation mapping (avoid ~50K+ redundant calls in main loop)
    abbrev_mapping = create_team_abbreviation_mapping()
    print("✅ Pre-calculated team abbreviation mapping")
    
    # Get team stats and lineups for each game (cache to avoid repeated calls)
    unique_games = result_df['game_id'].unique()
    print(f"Loading team stats for {len(unique_games)} unique games...")
    
    # 🚀 OPTIMIZATION #2: Batch PCA calculations
    pca_cache = {}
    if use_batch_pca:
        unique_dates = result_df['date'].dropna().unique()
        print(f"🚀 Pre-calculating PCA scores for {len(unique_dates)} unique dates...")
        
        from pca_optimized import calculate_all_pca_scores_for_date
        import time
        
        batch_start_time = time.time()
        total_dates_processed = 0
        
        for date in unique_dates:
            try:
                pca_cache[date] = calculate_all_pca_scores_for_date(date, SEASON_YEAR)
                total_dates_processed += 1
                if total_dates_processed % 10 == 0:
                    elapsed = time.time() - batch_start_time
                    print(f"  📈 Processed {total_dates_processed}/{len(unique_dates)} dates ({elapsed:.1f}s)")
            except Exception as e:
                print(f"⚠️  Warning: PCA calculation failed for {date}: {e}")
                pca_cache[date] = {}
        
        batch_elapsed = time.time() - batch_start_time
        total_players_calculated = sum(len(date_cache) for date_cache in pca_cache.values())
        print(f"✅ Batch PCA complete: {total_players_calculated:,} player-date combinations in {batch_elapsed:.1f}s")
        if total_players_calculated > 0 and batch_elapsed > 0:
            print(f"⚡ PCA Speed: {total_players_calculated / batch_elapsed:.0f} calculations/second")
        elif total_players_calculated > 0:
            print(f"⚡ PCA Speed: Instant (cached)")
        
    
    # 🚀 OPTIMIZATION #4: Smart boxscore loading - filter early to avoid loading unnecessary data
    print("Loading player boxscore data for lineups...")
    try:
        # For small datasets, pre-filter to avoid loading entire 84K+ record dataset
        if len(unique_games) <= 20:  # Testing/medium datasets - smart filtering
            print(f"🚀 SMART LOADING: Loading boxscore data for {len(unique_games)} specific games...")
            # Load full dataset but immediately filter to target games only
            boxscore_data = load_all_player_boxscores()
            original_size = len(boxscore_data)
            boxscore_data = boxscore_data[boxscore_data['GAME-ID'].isin(unique_games)]
            filtered_size = len(boxscore_data)
            savings_pct = ((original_size - filtered_size) / original_size) * 100 if original_size > 0 else 0
            print(f"✅ OPTIMIZED: {original_size:,} → {filtered_size:,} records ({savings_pct:.1f}% reduction)")
        else:
            # Full season - load all data
            boxscore_data = load_all_player_boxscores()
            print(f"Loaded boxscore data with {len(boxscore_data):,} player records")
            
    except Exception as e:
        print(f"Warning: Could not load boxscore data for lineups: {e}")
        boxscore_data = None
    
    game_team_stats = {}
    
    for game_id in unique_games:
        game_df = result_df[result_df['game_id'] == game_id]
        # Use the game date for team stats context
        game_date = game_df.iloc[0].get('date', None)
        stats = get_team_stats_for_game(game_df, game_team_mapping, target_date=game_date)
        
        # Get lineups for this game
        lineups = {}
        if boxscore_data is not None:
            try:
                lineups = get_lineup_by_game_id(game_id, boxscore_data)
            except Exception as e:
                print(f"Warning: Could not get lineups for game {game_id}: {e}")
                lineups = {}
        
        # Combine stats and lineups
        stats['lineups'] = lineups
        game_team_stats[game_id] = stats
    
    json_training_data = []
    
    # For remaining_plays mode, identify first N plays to skip for each game
    skip_indices = set()
    if generation_mode == "remaining_plays":
        print(f"🎯 remaining_plays mode: Skipping first {n_total} plays of each game")
        for game_id in unique_games:
            game_indices = result_df[result_df['game_id'] == game_id].index.tolist()
            valid_play_count = 0
            
            for idx in game_indices:
                if pd.notna(result_df.iloc[idx]['description']):
                    valid_play_count += 1
                    if valid_play_count <= n_total:
                        skip_indices.add(idx)
                    else:
                        break  # Found first N valid plays, stop skipping
        
        print(f"   • Skipping {len(skip_indices)} plays across {len(unique_games)} games")
    else:
        print(f"🎯 {generation_mode} mode: Processing all plays")
    
    # 🚀 OPTIMIZATION #5: Pre-calculate season format logic (avoid recalculating for every play)
    if SEASON_YEAR and '-' in SEASON_YEAR:
        cached_current_season = SEASON_YEAR  # Keep full format: "2023-2024"
    else:
        cached_current_season = "2023-2024"  # Default fallback
    print(f"✅ Pre-calculated season format: {cached_current_season}")
    
    for i in range(len(result_df)):
        # Skip first N plays for remaining_plays mode
        if i in skip_indices:
            json_training_data.append("{}")  # Placeholder for skipped plays
            continue
        current_game_id = result_df.iloc[i]['game_id']
        current_desc = result_df.iloc[i]['description']
        
        # Get team stats and lineups for current game
        team_stats = game_team_stats.get(current_game_id, {})
        away_stats = team_stats.get('away_team_stats', {})
        home_stats = team_stats.get('home_team_stats', {})
        away_abbrev = team_stats.get('away_abbrev', 'Unknown')
        home_abbrev = team_stats.get('home_abbrev', 'Unknown')
        lineups = team_stats.get('lineups', {})
        
        # Create players array from lineups
        away_players = []
        home_players = []
        away_name_to_idx = {}
        home_name_to_idx = {}
        
        # Get abbreviation to full name mapping for lineup matching (🚀 OPTIMIZED: use pre-calculated mapping)
        away_full_name = abbrev_mapping.get(away_abbrev, away_abbrev)
        home_full_name = abbrev_mapping.get(home_abbrev, home_abbrev)
        
        # Get game date and current season for player stats (🚀 OPTIMIZED: use pre-calculated season)
        current_game_date = result_df.iloc[i].get('date', None)
        current_season = cached_current_season
        
        # Process lineups to create player objects with stats
        for team_name, player_list in lineups.items():
            # 🚀 OPTIMIZED: Pre-calculate team name parts for faster matching
            away_parts = away_full_name.split() if away_full_name else []
            home_parts = home_full_name.split() if home_full_name else []
            
            # Determine if this lineup is for away or home team (optimized matching)
            is_away_team = (away_full_name in team_name or team_name in away_full_name or 
                           any(part in team_name for part in away_parts))
            is_home_team = (home_full_name in team_name or team_name in home_full_name or
                           any(part in team_name for part in home_parts))
            
            if is_away_team:
                for player_name in player_list:
                    try:
                        offense, defense, shot_selection, efficiency = get_player_pca_from_cache_or_calculate(
                            player_name, current_game_date, current_season, pca_cache
                        )
                        
                        # Build player array: [name, offense, defense, shot_selection, efficiency, MPG, usage]
                        player_array = [
                            player_name,
                            round(float(offense), 2) if offense is not None else 0.0,
                            round(float(defense), 2) if defense is not None else 0.0,
                            round(float(shot_selection), 2) if shot_selection is not None else 0.0,
                            round(float(efficiency), 2) if efficiency is not None else 0.0,
                            25,  # Default MPG
                            18   # Default usage
                        ]
                        away_players.append(player_array)
                        away_name_to_idx[player_name] = len(away_players) - 1
                        
                    except Exception as e:
                        print(f"Warning: Could not get PCA scores for {player_name}: {e}")
                        player_array = [player_name, 0.0, 0.0, 0.0, 0.0, 25, 18]
                        away_players.append(player_array)
                        away_name_to_idx[player_name] = len(away_players) - 1
            
            elif is_home_team:
                for player_name in player_list:
                    try:
                        offense, defense, shot_selection, efficiency = get_player_pca_from_cache_or_calculate(
                            player_name, current_game_date, current_season, pca_cache
                        )
                        
                        player_array = [
                            player_name,
                            round(float(offense), 2) if offense is not None else 0.0,
                            round(float(defense), 2) if defense is not None else 0.0,
                            round(float(shot_selection), 2) if shot_selection is not None else 0.0,
                            round(float(efficiency), 2) if efficiency is not None else 0.0,
                            25,  # Default MPG
                            18   # Default usage
                        ]
                        home_players.append(player_array)
                        home_name_to_idx[player_name] = len(home_players) - 1
                        
                    except Exception as e:
                        print(f"Warning: Could not get PCA scores for {player_name}: {e}")
                        player_array = [player_name, 0.0, 0.0, 0.0, 0.0, 25, 18]
                        home_players.append(player_array)
                        home_name_to_idx[player_name] = len(home_players) - 1
        
        # Collect recent plays in verbose format first, then convert to compact
        recent_plays_verbose = []
        collected_count = 0
        
        # Go backwards from current position to collect recent plays
        for j in range(i, -1, -1):  # Start from current row and go backwards
            row_game_id = result_df.iloc[j]['game_id']
            row_desc = result_df.iloc[j]['description']
            row_away_score = result_df.iloc[j].get('away_score', 0) or 0
            row_home_score = result_df.iloc[j].get('home_score', 0) or 0
            
            # Stop if we've moved to a different game
            if row_game_id != current_game_id:
                break
            
            # Add valid descriptions as recent plays
            if pd.notna(row_desc):
                # Get quarter and time remaining in quarter
                row_period = result_df.iloc[j].get('period', 4)
                row_remaining_time = result_df.iloc[j].get('remaining_time', '0:00:00')
                quarter, time_in_quarter = convert_to_quarter_time(row_period, row_remaining_time)
                
                # Extract actual players_on_court from lineup data (a1-a5, h1-h5)
                current_row = result_df.iloc[j]
                away_lineup = []
                home_lineup = []
                
                # Extract away team players (a1-a5)
                for k in range(1, 6):
                    player = current_row.get(f'a{k}')
                    if pd.notna(player):
                        away_lineup.append(str(player))
                
                # Extract home team players (h1-h5)
                for k in range(1, 6):
                    player = current_row.get(f'h{k}')
                    if pd.notna(player):
                        home_lineup.append(str(player))
                
                players_on_court = [
                    {'team': away_abbrev, 'players': away_lineup},
                    {'team': home_abbrev, 'players': home_lineup}
                ]
                
                # Build shot_details for compatibility with conversion
                shot_details = {'team': None, 'points': None}
                
                # Create play object in verbose format with structured fields for better event mapping
                play_obj = {
                    "quarter": int(quarter),
                    "time_remaining": str(time_in_quarter),
                    "description": remove_parentheses_content(row_desc),
                    "score": f"{away_abbrev} {int(row_away_score)} - {home_abbrev} {int(row_home_score)}",
                    "players_on_court": players_on_court,
                    "player": str(result_df.iloc[j].get('player')) if pd.notna(result_df.iloc[j].get('player')) else None,
                    "shot_details": shot_details,
                    # Include structured fields from original DataFrame with proper null handling
                    "type": str(result_df.iloc[j].get('type')) if pd.notna(result_df.iloc[j].get('type')) else None,
                    "event_type": str(result_df.iloc[j].get('event_type')) if pd.notna(result_df.iloc[j].get('event_type')) else None,
                    "result": str(result_df.iloc[j].get('result')) if pd.notna(result_df.iloc[j].get('result')) else None,
                    "points": float(result_df.iloc[j].get('points')) if pd.notna(result_df.iloc[j].get('points')) else None,
                    "shot_distance": float(result_df.iloc[j].get('shot_distance')) if pd.notna(result_df.iloc[j].get('shot_distance')) else None
                }
                
                recent_plays_verbose.insert(0, play_obj)  # Insert at beginning to maintain chronological order
                collected_count += 1
                
                # Stop if we've collected the desired total number of plays
                if collected_count >= n_total:
                    break
        
        # Handle None REST_DAYS by converting to 0
        away_rest_days = away_stats.get('REST_DAYS') if away_stats.get('REST_DAYS') is not None else 0
        home_rest_days = home_stats.get('REST_DAYS') if home_stats.get('REST_DAYS') is not None else 0
        
        # 🚀 OPTIMIZATION: Choose format based on use_direct_compact parameter
        if use_direct_compact:
            # Direct compact generation (30-50% faster)
            away_stats_dict = {
                "OEFF": float(away_stats.get('OEFF')) if away_stats.get('OEFF') is not None else 110.0,
                "DEFF": float(away_stats.get('DEFF')) if away_stats.get('DEFF') is not None else 110.0,
                "PACE": float(away_stats.get('PACE')) if away_stats.get('PACE') is not None else 100.0,
                "REST_DAYS": int(away_rest_days) if away_rest_days is not None else 2
            }
            
            home_stats_dict = {
                "OEFF": float(home_stats.get('OEFF')) if home_stats.get('OEFF') is not None else 110.0,
                "DEFF": float(home_stats.get('DEFF')) if home_stats.get('DEFF') is not None else 110.0,
                "PACE": float(home_stats.get('PACE')) if home_stats.get('PACE') is not None else 100.0,
                "REST_DAYS": int(home_rest_days) if home_rest_days is not None else 2
            }
            
            final_json_obj = build_compact_training_data_direct(
                current_game_id, away_abbrev, home_abbrev,
                away_stats_dict, home_stats_dict,
                away_players, home_players,
                recent_plays_verbose, away_name_to_idx, home_name_to_idx,
                for_first_n_plays=(generation_mode == "first_N_plays")
            )
        else:
            # Verbose format generation - keep in verbose format!
            final_json_obj = {
                "away_team": {
                    "name": str(away_abbrev) if away_abbrev else "Unknown",
                    "stats": {
                        "OEFF": float(away_stats.get('OEFF')) if away_stats.get('OEFF') is not None else 110.0,
                        "DEFF": float(away_stats.get('DEFF')) if away_stats.get('DEFF') is not None else 110.0,
                        "PACE": float(away_stats.get('PACE')) if away_stats.get('PACE') is not None else 100.0,
                        "REST_DAYS": int(away_rest_days) if away_rest_days is not None else 2
                    },
                    "players": [{"name": p[0], "profile": {"offense": p[1], "defense": p[2], "shot_selection": p[3], "efficiency": p[4], "MPG": p[5], "usage": p[6]}} for p in away_players]
                },
                "home_team": {
                    "name": str(home_abbrev) if home_abbrev else "Unknown",
                    "stats": {
                        "OEFF": float(home_stats.get('OEFF')) if home_stats.get('OEFF') is not None else 110.0,
                        "DEFF": float(home_stats.get('DEFF')) if home_stats.get('DEFF') is not None else 110.0,
                        "PACE": float(home_stats.get('PACE')) if home_stats.get('PACE') is not None else 100.0,
                        "REST_DAYS": int(home_rest_days) if home_rest_days is not None else 2
                    },
                    "players": [{"name": p[0], "profile": {"offense": p[1], "defense": p[2], "shot_selection": p[3], "efficiency": p[4], "MPG": p[5], "usage": p[6]}} for p in home_players]
                }
            }
            
            # For verbose format, include recent_plays only if not first_N_plays mode
            if generation_mode != "first_N_plays":
                final_json_obj["recent_plays"] = recent_plays_verbose
        
        # Convert to JSON string
        json_string = json.dumps(final_json_obj, separators=(',', ':'))
        json_training_data.append(json_string)
    
    # Add the JSON training data as a new column
    result_df['json_training_data'] = json_training_data
    
    # 🔇 Show PCA summary instead of individual warnings
    try:
        from pca_optimized import print_pca_summary
        print_pca_summary()
    except ImportError:
        pass  # Skip if function not available
    
    return result_df

def generate_training_data_for_game(game_id, season_year="2023-2024", n_total=5, max_plays=None):
    """
    Generate LLM training data for a specific game_id.
    
    Args:
        game_id (int): The specific game ID to generate data for
        season_year (str): Season year (e.g., "2023-2024")
        n_total (int): Total number of recent plays to include in each sequence
        max_plays (int): Maximum number of plays from the game to process (None for all)
    
    Returns:
        pd.DataFrame: DataFrame with LLM training data for the specified game
    """
    print(f"Generating training data for game_id: {game_id}")
    
    # Load the full dataset
    df = load_play_by_play_data(season_year)
    
    # Filter to the specific game
    game_df = df[df['game_id'] == game_id]
    
    if len(game_df) == 0:
        raise ValueError(f"Game ID {game_id} not found in {season_year} season data")
    
    # Get game info
    game_date = game_df.iloc[0]['date']
    total_plays = len(game_df)
    
    print(f"Found game on {game_date} with {total_plays} total plays")
    
    # Limit plays if requested
    if max_plays and max_plays < total_plays:
        game_df = game_df.head(max_plays)
        print(f"Limited to first {max_plays} plays")
    
    # Sort by play order
    game_df = game_df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
    
    # Generate training data
    print(f"Generating training sequences with n_total={n_total}...")
    result_df = create_llm_training_data(game_df, n_total=n_total, filter_nan=True)
    
    print(f"✅ Generated {len(result_df)} training records for game {game_id}")
    
    return result_df

def generate_openai_training_for_game(game_id, season_year="2023-2024", n_total=5, max_plays=None):
    """
    Generate OpenAI fine-tuning data for a specific game.
    
    Args:
        game_id (int): Game ID to generate training data for
        season_year (str): Season year
        n_total (int): Number of recent plays in context
        max_plays (int): Max plays to process (None for all)
        
    Returns:
        tuple: (training_examples_list, jsonl_filepath)
    """
    print(f"Generating OpenAI training data for game {game_id}...")
    
    # Generate our structured JSON data first
    df = generate_training_data_for_game(game_id, season_year, n_total, max_plays)
    
    # Convert to OpenAI format
    training_examples = create_openai_training_data(df)
    
    # Save as JSONL
    jsonl_path = save_openai_training_data(
        training_examples, 
        filename=f"game_{game_id}_openai_training.jsonl",
        season_year=season_year
    )
    
    return training_examples, jsonl_path

def generate_llm_dataset(season_year=None, n_total=5, sample_size=None, game_id_filter=None):
    """
    Generate a complete LLM training dataset from play-by-play data.
    
    Args:
        season_year (str, optional): Season year to use. If None, uses current SEASON_YEAR
        n_total (int): Total number of descriptions to concatenate together per row
        sample_size (int, optional): If provided, randomly sample this many rows
        game_id_filter (list, optional): If provided, only include these game IDs
    
    Returns:
        pd.DataFrame: DataFrame ready for LLM training with concatenated descriptions
    """
    if season_year is None:
        season_year = SEASON_YEAR
    
    print(f"Generating LLM dataset for season {season_year}...")
    
    # Load the play-by-play data
    df = load_play_by_play_data(season_year)
    
    # Filter by game IDs if specified
    if game_id_filter:
        df = df[df['game_id'].isin(game_id_filter)]
        print(f"Filtered to {len(df)} rows from {len(game_id_filter)} games")
    
    # Sort by game_id and play sequence to ensure proper chronological order
    df = df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
    
    # Create concatenated descriptions
    result_df = create_llm_training_data(df, n_total=n_total)
    
    # Sample if requested
    if sample_size and sample_size < len(result_df):
        result_df = result_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} rows from dataset")
    
    print(f"Generated dataset with {len(result_df)} rows")
    print(f"Each row contains up to {n_total} descriptions concatenated together")
    
    return result_df

def create_openai_training_data(df):
    """
    Convert compact JSON training data into OpenAI fine-tuning JSONL format.
    Each row becomes a user-assistant pair where:
    - User: Our compact JSON context
    - Assistant: The next play in compact tuple format
    
    Args:
        df (pd.DataFrame): DataFrame with 'json_training_data' column containing compact format
        
    Returns:
        list: List of training examples in OpenAI format
    """
    training_examples = []
    
    # Group by game to ensure we can find next plays
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
        
        for i in range(len(game_df) - 1):  # -1 because we need a next play
            current_row = game_df.iloc[i]
            next_row = game_df.iloc[i + 1]
            
            # Parse the current compact JSON context
            try:
                current_compact = json.loads(current_row['json_training_data'])
                
                # Extract team abbreviations from compact format
                away_abbrev = current_compact.get('a', 'AWAY')
                home_abbrev = current_compact.get('h', 'HOME')
                
                # Build name to index mappings
                away_name_to_idx = {}
                home_name_to_idx = {}
                
                for idx, player_array in enumerate(current_compact.get('ap', [])):
                    away_name_to_idx[player_array[0]] = idx
                
                for idx, player_array in enumerate(current_compact.get('hp', [])):
                    home_name_to_idx[player_array[0]] = idx
                
                # Create the next play in compact tuple format
                next_quarter, next_time = convert_to_quarter_time(next_row['period'], next_row['remaining_time'])
                next_time_seconds = parse_time_to_seconds(next_time)  # Convert MM:SS to seconds
                
                next_away_score = int(next_row.get('away_score', 0) or 0)
                next_home_score = int(next_row.get('home_score', 0) or 0)
                next_score_array = [next_away_score, next_home_score]
                
                # Determine scoring info for next play
                if i == 0:
                    prev_away_score = 0
                    prev_home_score = 0
                else:
                    prev_away_score = int(game_df.iloc[i-1].get('away_score', 0) or 0)
                    prev_home_score = int(game_df.iloc[i-1].get('home_score', 0) or 0)
                
                score_delta = max(0, max(
                    next_away_score - prev_away_score,  # Away team scored
                    next_home_score - prev_home_score   # Home team scored
                ))
                
                # Resolve actor for next play
                next_player = next_row.get('player')
                next_shot_details = {'team': None, 'points': None}
                actor = resolve_actor(
                    next_player, next_shot_details, away_abbrev, home_abbrev,
                    away_name_to_idx, home_name_to_idx
                )
                
                # Map to event code - prefer structured data when available
                if all(key in next_row for key in ['type', 'event_type']):
                    # Use structured mapping when available (better accuracy)
                    event_code, points = map_structured_to_event_code(next_row)
                else:
                    # Fall back to description parsing
                    description = next_row.get('description', '')
                    event_code, points = map_description_to_event_code(
                        description, next_shot_details, score_delta
                    )
                
                # Get current lineup ID (use last play's lineup or default to 0)
                current_lineups = current_compact.get('L', [])
                lineup_id = 0 if not current_lineups else len(current_lineups) - 1
                
                # Create compact next play tuple - ensure JSON serializable types
                if points is not None:
                    # Scoring play: [q, t, score, actor, event, pts, lineup_id]
                    next_play_tuple = [
                        int(next_quarter), 
                        int(next_time_seconds), 
                        [int(next_score_array[0]), int(next_score_array[1])], 
                        actor, 
                        event_code, 
                        int(points), 
                        int(lineup_id)
                    ]
                else:
                    # Non-scoring play: [q, t, score, actor, event, lineup_id]
                    next_play_tuple = [
                        int(next_quarter),
                        int(next_time_seconds),
                        [int(next_score_array[0]), int(next_score_array[1])],
                        actor,
                        event_code,
                        int(lineup_id)
                    ]
                
                # Handle offensive foul special case - need both o_foul and tov
                if event_code == "o_foul":
                    # Create assistant response with both plays
                    assistant_response = {
                        "y": [
                            [int(next_quarter), int(next_time_seconds), [int(next_score_array[0]), int(next_score_array[1])], actor, "o_foul", int(lineup_id)],
                            [int(next_quarter), int(next_time_seconds), [int(next_score_array[0]), int(next_score_array[1])], actor, "tov", int(lineup_id)]
                        ]
                    }
                else:
                    # Regular single play response
                    assistant_response = {
                        "y": next_play_tuple
                }
                
                # Create OpenAI training example
                training_example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": current_row['json_training_data']  # Our compact JSON context
                        },
                        {
                            "role": "assistant", 
                            "content": json.dumps(assistant_response, separators=(',', ':'))
                        }
                    ]
                }
                
                training_examples.append(training_example)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Skipped row due to error: {e}")
                continue
    
    return training_examples

def save_openai_training_data(training_examples, filename=None, season_year=None):
    """
    Save training examples in JSONL format for OpenAI fine-tuning.
    
    Args:
        training_examples (list): List of training examples
        filename (str, optional): Custom filename
        season_year (str, optional): Season year for filename
        
    Returns:
        str: Path to saved JSONL file
    """
    # Ensure training directory exists
    training_dir = "data/training"
    os.makedirs(training_dir, exist_ok=True)
    
    # Generate filename
    if filename is None:
        season_str = season_year if season_year else "unknown_season"
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"openai_training_{season_str}_{timestamp}.jsonl"
    
    # Ensure .jsonl extension
    if not filename.endswith('.jsonl'):
        filename = filename.replace('.csv', '') + '.jsonl'
    
    filepath = os.path.join(training_dir, filename)
    
    # Write JSONL file
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(training_examples)} training examples to: {filepath}")
    return filepath

def save_llm_dataset(df, filename=None, season_year=None):
    """
    Save the LLM training dataset to a CSV file.
    
    Args:
        df (pd.DataFrame): Dataset to save
        filename (str, optional): Custom filename. If None, auto-generates based on season
        season_year (str, optional): Season year for auto-generated filename
    """
    if filename is None:
        if season_year is None:
            season_year = SEASON_YEAR
        filename = f"llm_training_data_{season_year.replace('-', '_')}.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "data/training"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved LLM training dataset to: {filepath}")
    return filepath

def preview_llm_data(df, n_samples=3):
    """
    Preview sample JSON training data from the LLM dataset.
    
    Args:
        df (pd.DataFrame): LLM dataset with json_training_data column
        n_samples (int): Number of samples to show
    """
    print(f"\n=== Preview of LLM Training Data (showing {n_samples} samples) ===\n")
    
    # Show random samples
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
        print(f"Sample {i}:")
        print(f"Game ID: {row['game_id']}")
        print(f"Date: {row['date']}")
        print(f"Original Description: {row['description']}")
        print(f"JSON Training Data:")
        
        # Parse and pretty print JSON for readability
        try:
            json_obj = json.loads(row['json_training_data'])
            print(json.dumps(json_obj, indent=2))
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(row['json_training_data'])
            
        print("-" * 80)

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_data_loading()
    
    # Example of generating LLM training data
    print("\n" + "="*50)
    print("TESTING LLM TRAINING DATA GENERATION")
    print("="*50)
    
    try:
        # SINGLE GAME TEST MODE for speed and verification
        print("Loading single game for PCA testing...")
        df_full = load_play_by_play_data("2023-2024")
        sample_game_id = df_full['game_id'].iloc[0]
        game_date = df_full[df_full['game_id'] == sample_game_id].iloc[0]['date']
        
        print(f"Testing with game_id: {sample_game_id} on date: {game_date}")
        
        # Filter to single game, limit plays for speed
        df_test = df_full[df_full['game_id'] == sample_game_id].head(20)  # Even smaller for speed
        df_test = df_test.sort_values(['game_id', 'play_id']).reset_index(drop=True)
        
        # Verify we only have one game
        unique_test_games = df_test['game_id'].unique()
        assert len(unique_test_games) == 1, f"Expected 1 game, got {len(unique_test_games)}"
        
        print(f"Using {len(df_test)} plays from single game")
        
        # Generate training data with PCA stats
        test_df = create_llm_training_data(df_test, n_total=3, filter_nan=True)
        print(f"✅ Generated {len(test_df)} training records with PCA player stats")
        
        # Verify PCA integration worked
        if len(test_df) > 0:
            sample_json = json.loads(test_df.iloc[0]['json_training_data'])
            print(f"✅ Total players with stats: {len(sample_json['players'])}")
            
            # Show sample player to verify PCA worked
            if len(sample_json['players']) > 0:
                sample_player = sample_json['players'][0]
                print(f"✅ Sample: {sample_player['name']} ({sample_player['team']})")
                stats = sample_player['stats']
                print(f"   PCA Stats: O={stats['offense']} D={stats['defense']} S={stats['shot_selection']} E={stats['efficiency']}")
                
                if stats['offense'] is not None:
                    print("✅ PCA integration SUCCESS!")
                else:
                    print("⚠️  PCA stats are None - check date passing")
        
        # Save single game test
        saved_path = save_llm_dataset(test_df, filename="single_game_PCA_test.csv")
        print(f"✅ Single game test with PCA saved: {saved_path}")
        
        # Test OpenAI training data generation
        print("\n" + "="*50)
        print("TESTING OPENAI TRAINING DATA GENERATION")
        print("="*50)
        
        openai_examples = create_openai_training_data(test_df)
        print(f"✅ Generated {len(openai_examples)} OpenAI training examples")
        
        if len(openai_examples) > 0:
            # Show sample training example
            sample = openai_examples[0]
            print("\n📋 Sample Training Example:")
            print("USER (Context):")
            user_content = json.loads(sample['messages'][0]['content'])
            print(f"  Teams: {user_content['away_team']['name']} @ {user_content['home_team']['name']}")
            print(f"  Players: {len(user_content['players'])}")
            print(f"  Recent plays: {len(user_content['recent_plays'])}")
            
            print("\nASSISTANT (Next Play Prediction):")
            assistant_content = json.loads(sample['messages'][1]['content'])
            next_play = assistant_content['next_play']
            print(f"  Q{next_play['quarter']} {next_play['time_remaining']}: {next_play['description']}")
            print(f"  Score: {next_play['score']}")
            print(f"  Scoring: {next_play['scoring_team']} (+{next_play['points_scored']})")
            
            # Save OpenAI training data
            openai_path = save_openai_training_data(openai_examples, filename="test_openai_training.jsonl")
            print(f"✅ OpenAI training data saved: {openai_path}")
        
    except Exception as e:
        print(f"❌ Error testing LLM data generation: {e}")
        import traceback
        traceback.print_exc()
