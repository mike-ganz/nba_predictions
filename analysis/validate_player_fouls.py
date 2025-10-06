"""
Validate player foul tracking against official boxscore data.
"""
import sys
import os
import random
import pandas as pd
import json
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_training_data import load_play_by_play_data
from generate_training_data_OPTIMIZED import map_description_to_event_code, map_structured_to_event_code


def load_player_boxscore_data(season):
    """Load player boxscore data with PF (personal fouls) column."""
    path = f"data/player_boxscores/historical/NBA-{season}-Player-BoxScore-Dataset.xlsx"
    df = pd.read_excel(path)
    return df


def get_game_ids_from_season(season_str):
    """Get all game IDs from a season's play-by-play data."""
    df = load_play_by_play_data(season_str)
    game_ids = df['game_id'].unique().tolist()
    return game_ids


def count_player_fouls_from_play_by_play(game_id, season_str):
    """Count player fouls from play-by-play data using our logic."""
    df = load_play_by_play_data(season_str)
    game_df = df[df['game_id'] == game_id].copy()
    
    player_fouls = defaultdict(int)
    
    for idx, row in game_df.iterrows():
        play_dict = row.to_dict()
        
        # Map event code
        try:
            if play_dict.get('type') or play_dict.get('event_type'):
                ev_code, _ = map_structured_to_event_code(play_dict)
            else:
                ev_code, _ = map_description_to_event_code(
                    play_dict.get('description', ''),
                    play_dict.get('shot_details', {}),
                    0
                )
        except Exception:
            ev_code = "unknown"
        
        # Count player fouls (same logic as generate_training_data_OPTIMIZED.py)
        event_type_val = str(play_dict.get('event_type', '')).lower()
        if ev_code in ("p_foul", "s_foul", "o_foul") and event_type_val == 'foul':
            player_name = play_dict.get('player')
            if player_name and not pd.isna(player_name):
                player_name = str(player_name)
                player_fouls[player_name] += 1
    
    return dict(player_fouls)


def get_boxscore_player_fouls(game_id, boxscore_df):
    """Get player foul counts from official boxscore data."""
    game_records = boxscore_df[boxscore_df['GAME-ID'] == int(game_id)]
    
    player_fouls = {}
    for idx, row in game_records.iterrows():
        player_name = row['PLAYER \nFULL NAME']
        fouls = row['PF']
        if pd.notna(player_name) and pd.notna(fouls):
            player_fouls[str(player_name)] = int(fouls)
    
    return player_fouls


def validate_player_foul_counts(game_ids, season_str, boxscore_df):
    """Validate player foul counts for multiple games."""
    results = []
    
    for game_id in game_ids:
        try:
            # Get computed player fouls
            computed_fouls = count_player_fouls_from_play_by_play(game_id, season_str)
            
            # Get official boxscore fouls
            official_fouls = get_boxscore_player_fouls(game_id, boxscore_df)
            
            # Compare players who appear in both
            all_players = set(computed_fouls.keys()) | set(official_fouls.keys())
            
            matches = 0
            mismatches = []
            
            for player in all_players:
                comp_count = computed_fouls.get(player, 0)
                official_count = official_fouls.get(player, 0)
                
                if comp_count == official_count:
                    matches += 1
                else:
                    mismatches.append({
                        'player': player,
                        'computed': comp_count,
                        'official': official_count,
                        'diff': comp_count - official_count
                    })
            
            match_rate = matches / len(all_players) if all_players else 1.0
            
            results.append({
                'game_id': game_id,
                'total_players': len(all_players),
                'matches': matches,
                'mismatches': len(mismatches),
                'match_rate': match_rate,
                'mismatch_details': mismatches
            })
            
            print(f"Game {game_id}: {matches}/{len(all_players)} players match ({match_rate*100:.1f}%)")
            if mismatches:
                for mm in mismatches[:3]:  # Show first 3 mismatches
                    print(f"  - {mm['player']}: computed={mm['computed']}, official={mm['official']}, diff={mm['diff']}")
            
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            continue
    
    return results


def main():
    season_str = "2023-2024"
    print("Validating Player Foul Tracking")
    print("=" * 70)
    
    # Load boxscore data
    print(f"\nLoading player boxscore data for {season_str}...")
    boxscore_df = load_player_boxscore_data(season_str)
    print(f"Loaded {len(boxscore_df)} player-game records")
    
    # Get game IDs
    print(f"\nLoading play-by-play data for {season_str}...")
    game_ids = get_game_ids_from_season(season_str)
    print(f"Found {len(game_ids)} games")
    
    # Sample 25 random games
    sample_size = min(25, len(game_ids))
    sampled_game_ids = random.sample(game_ids, sample_size)
    
    print(f"\nValidating {sample_size} random games...")
    print("=" * 70)
    
    results = validate_player_foul_counts(sampled_game_ids, season_str, boxscore_df)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_players = sum(r['total_players'] for r in results)
    total_matches = sum(r['matches'] for r in results)
    overall_match_rate = total_matches / total_players if total_players > 0 else 0
    
    print(f"Games validated: {len(results)}")
    print(f"Total players: {total_players}")
    print(f"Total matches: {total_matches}")
    print(f"Overall match rate: {overall_match_rate*100:.1f}%")
    
    # Games with 100% match
    perfect_games = sum(1 for r in results if r['match_rate'] == 1.0)
    print(f"Perfect match games: {perfect_games}/{len(results)} ({perfect_games/len(results)*100:.1f}%)")
    
    # Most common mismatches
    all_mismatches = []
    for r in results:
        all_mismatches.extend(r['mismatch_details'])
    
    if all_mismatches:
        print(f"\nTotal mismatches: {len(all_mismatches)}")
        print("Most common mismatch patterns:")
        
        # Group by difference
        by_diff = defaultdict(int)
        for mm in all_mismatches:
            by_diff[mm['diff']] += 1
        
        for diff in sorted(by_diff.keys(), key=lambda x: abs(x), reverse=True):
            print(f"  Difference {diff:+d}: {by_diff[diff]} occurrences")


if __name__ == "__main__":
    main()

