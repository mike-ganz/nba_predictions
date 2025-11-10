"""
Validate that the LAL @ ATL game from Nov 8, 2025 now has matching
rosters and features between day-of and backlook pipelines.
"""
import json
import pandas as pd
from datetime import datetime

from scripts.player_data_loader import get_team_players_with_injuries, precompute_season_baselines
from transform_player_stats import load_player_data


def load_current_season_boxscores():
    """Load current season player boxscores."""
    import glob
    
    # Find the current season player boxscore file
    pattern = "data/player_boxscores/current/*2025*.xlsx"
    files = glob.glob(pattern)
    
    if not files:
        print("ERROR: No current season player boxscore files found")
        return None
    
    latest_file = max(files, key=lambda f: f)
    print(f"Loading player boxscores from: {latest_file}")
    
    df = pd.read_excel(latest_file, sheet_name='NBA Player Box Score')
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    return df


def load_day_of_game_data():
    """Load the day-of game data from JSONL."""
    import json
    
    file_path = "data/games_todays.jsonl"
    games = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                games.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"ERROR: Could not find {file_path}")
        return None
    
    return games


def get_player_features(players):
    """Calculate injury-related features from a player roster."""
    if not players:
        return {
            'minutes_missing_top2': 0.0,
            'star_out': 0,
            'usage_share_top2': 0.0
        }
    
    # Sort by baseline minutes
    sorted_players = sorted(players, key=lambda p: p['baseline_minutes'], reverse=True)
    
    # Calculate minutes_missing_top2
    minutes_missing_top2 = 0.0
    if len(sorted_players) >= 2:
        for player in sorted_players[:2]:
            if player['projected_minutes'] == 0.0:
                minutes_missing_top2 += player['baseline_minutes']
    
    # Calculate star_out (usage >= 27%)
    star_out = 0
    for player in sorted_players:
        if player['projected_minutes'] == 0.0 and player['baseline_usage_rate'] >= 0.27:
            star_out = 1
            break
    
    # Calculate usage_share_top2
    total_usage = sum(p['baseline_usage_rate'] for p in sorted_players)
    usage_share_top2 = 0.0
    if total_usage > 0 and len(sorted_players) >= 2:
        usage_share_top2 = sum(p['baseline_usage_rate'] for p in sorted_players[:2]) / total_usage
    
    return {
        'minutes_missing_top2': minutes_missing_top2,
        'star_out': star_out,
        'usage_share_top2': usage_share_top2
    }


def main():
    print("="*80)
    print("VALIDATING LAL @ ATL FIX (2025-11-08)")
    print("="*80)
    
    target_date = '2025-11-08'
    season = '2025-2026'
    
    # Load current season data
    print("\nStep 1: Loading current season boxscores...")
    player_boxscore_df = load_current_season_boxscores()
    if player_boxscore_df is None:
        return
    
    # Precompute baselines
    print("Step 2: Precomputing season baselines...")
    precompute_season_baselines(season, player_boxscore_df)
    
    # Process LAL roster with new method
    print("\nStep 3: Processing LAL roster with injury reconstruction...")
    lal_players = get_team_players_with_injuries(
        team_name='Los Angeles Lakers',
        game_date=target_date,
        season=season,
        player_boxscore_df=player_boxscore_df
    )
    
    print(f"\nLAL Roster ({len(lal_players)} players):")
    print("-" * 80)
    for p in lal_players:
        status = "OUT" if p['projected_minutes'] == 0.0 else "ACTIVE"
        print(f"  {p['player_name']:25s} | {p['baseline_minutes']:5.1f} mins | {status}")
    
    # Check for specific injured players
    expected_injured_lal = ['Austin Reaves']
    found_injured_lal = [p['player_name'] for p in lal_players if p['projected_minutes'] == 0.0]
    
    print(f"\nExpected injured LAL players: {expected_injured_lal}")
    print(f"Found injured LAL players: {found_injured_lal}")
    
    # Process ATL roster with new method
    print("\nStep 4: Processing ATL roster with injury reconstruction...")
    atl_players = get_team_players_with_injuries(
        team_name='Atlanta',
        game_date=target_date,
        season=season,
        player_boxscore_df=player_boxscore_df
    )
    
    print(f"\nATL Roster ({len(atl_players)} players):")
    print("-" * 80)
    for p in atl_players:
        status = "OUT" if p['projected_minutes'] == 0.0 else "ACTIVE"
        print(f"  {p['player_name']:25s} | {p['baseline_minutes']:5.1f} mins | {status}")
    
    # Check for specific injured players
    expected_injured_atl = ['Trae Young', 'Jalen Johnson']  # High-profile injuries
    found_injured_atl = [p['player_name'] for p in atl_players if p['projected_minutes'] == 0.0]
    
    print(f"\nExpected injured ATL players (at least): {expected_injured_atl}")
    print(f"Found injured ATL players: {found_injured_atl}")
    
    # Calculate features
    print("\nStep 5: Calculating injury features...")
    lal_features = get_player_features(lal_players)
    atl_features = get_player_features(atl_players)
    
    print("\nLAL (Away) Features:")
    print(f"  away_minutes_missing_top2: {lal_features['minutes_missing_top2']:.1f}")
    print(f"  away_star_out: {lal_features['star_out']}")
    print(f"  away_usage_share_top2: {lal_features['usage_share_top2']:.3f}")
    
    print("\nATL (Home) Features:")
    print(f"  home_minutes_missing_top2: {atl_features['minutes_missing_top2']:.1f}")
    print(f"  home_star_out: {atl_features['star_out']}")
    print(f"  home_usage_share_top2: {atl_features['usage_share_top2']:.3f}")
    
    # Load day-of features for comparison
    print("\nStep 6: Loading day-of features for comparison...")
    day_of_games = load_day_of_game_data()
    
    if day_of_games:
        lal_atl_game = next(
            (g for g in day_of_games if g['game_id'] == '2025-11-08-LAL-ATL'),
            None
        )
        
        if lal_atl_game:
            print("\nDay-of features (from games_todays.jsonl):")
            
            # Extract away (LAL) player features
            away_players_day_of = lal_atl_game.get('away_players', [])
            lal_features_day_of = get_player_features(away_players_day_of)
            
            # Extract home (ATL) player features
            home_players_day_of = lal_atl_game.get('home_players', [])
            atl_features_day_of = get_player_features(home_players_day_of)
            
            print("\nDay-of LAL (Away):")
            print(f"  away_minutes_missing_top2: {lal_features_day_of['minutes_missing_top2']:.1f}")
            print(f"  away_star_out: {lal_features_day_of['star_out']}")
            
            print("\nDay-of ATL (Home):")
            print(f"  home_minutes_missing_top2: {atl_features_day_of['minutes_missing_top2']:.1f}")
            print(f"  home_star_out: {atl_features_day_of['star_out']}")
            
            # Compare
            print("\n" + "="*80)
            print("COMPARISON: Backlook (new method) vs Day-of")
            print("="*80)
            
            lal_minutes_match = abs(lal_features['minutes_missing_top2'] - lal_features_day_of['minutes_missing_top2']) < 5.0
            lal_star_match = lal_features['star_out'] == lal_features_day_of['star_out']
            atl_minutes_match = abs(atl_features['minutes_missing_top2'] - atl_features_day_of['minutes_missing_top2']) < 5.0
            atl_star_match = atl_features['star_out'] == atl_features_day_of['star_out']
            
            print(f"\nLAL away_minutes_missing_top2: {'MATCH' if lal_minutes_match else 'DIFFER'}")
            print(f"  Backlook: {lal_features['minutes_missing_top2']:.1f}")
            print(f"  Day-of:   {lal_features_day_of['minutes_missing_top2']:.1f}")
            
            print(f"\nLAL away_star_out: {'MATCH' if lal_star_match else 'DIFFER'}")
            print(f"  Backlook: {lal_features['star_out']}")
            print(f"  Day-of:   {lal_features_day_of['star_out']}")
            
            print(f"\nATL home_minutes_missing_top2: {'MATCH' if atl_minutes_match else 'DIFFER'}")
            print(f"  Backlook: {atl_features['minutes_missing_top2']:.1f}")
            print(f"  Day-of:   {atl_features_day_of['minutes_missing_top2']:.1f}")
            
            print(f"\nATL home_star_out: {'MATCH' if atl_star_match else 'DIFFER'}")
            print(f"  Backlook: {atl_features['star_out']}")
            print(f"  Day-of:   {atl_features_day_of['star_out']}")
            
            # Overall validation
            all_match = lal_minutes_match and lal_star_match and atl_minutes_match and atl_star_match
            
            print("\n" + "="*80)
            if all_match:
                print("VALIDATION: SUCCESS")
                print("All injury features now match between backlook and day-of!")
            else:
                print("VALIDATION: PARTIAL")
                print("Some differences remain - this may be expected if injury data changed")
                print("or if the lookback window captures different games.")
            print("="*80)
        else:
            print("ERROR: LAL @ ATL game not found in day-of data")
    else:
        print("WARNING: Could not load day-of data for comparison")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

