#!/usr/bin/env python3
"""Debug foul counting for a specific game."""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_training_data import (
    load_play_by_play_data,
    determine_home_away_teams,
    map_structured_to_event_code,
    map_description_to_event_code
)


def debug_game_fouls(game_id: int = 22300228):
    """Debug foul counting for a specific game (MIL @ BOS)."""
    print(f"🔍 Debugging foul counting for game {game_id}")
    print("=" * 80)
    
    # Load data
    pbp_df = load_play_by_play_data("2023-2024")
    game_team_mapping = determine_home_away_teams(pbp_df)
    
    # Get team names
    team_mapping = game_team_mapping.get(game_id, {})
    away_abbrev = team_mapping.get('away_team', 'Unknown')
    home_abbrev = team_mapping.get('home_team', 'Unknown')
    
    print(f"Teams: {away_abbrev} @ {home_abbrev}")
    print()
    
    # Get game data
    game_df = pbp_df[pbp_df['game_id'] == game_id].copy()
    
    away_fouls = []
    home_fouls = []
    
    for idx, row in game_df.iterrows():
        if pd.isna(row.get('description')):
            continue
        
        # Map to event code
        try:
            if row.get('type') or row.get('event_type'):
                event_code, _ = map_structured_to_event_code(row)
            else:
                shot_details = {'team': row.get('team'), 'points': row.get('points')}
                event_code, _ = map_description_to_event_code(
                    row.get('description', ''), shot_details, 0
                )
        except Exception:
            event_code = "unknown"
        
        # Check if it's a foul event
        if event_code in ("p_foul", "s_foul", "o_foul", "tech"):
            team_name = row.get('team')
            desc = row.get('description', '')
            event_type = row.get('event_type', '')
            play_type = row.get('type', '')
            
            foul_info = {
                'quarter': row.get('period'),
                'time': row.get('remaining_time'),
                'event_code': event_code,
                'team': team_name,
                'desc': desc[:80],
                'event_type': event_type,
                'type': play_type
            }
            
            if team_name == away_abbrev:
                away_fouls.append(foul_info)
            elif team_name == home_abbrev:
                home_fouls.append(foul_info)
    
    print(f"\n{away_abbrev} fouls (counted={len(away_fouls)}, expected=19):")
    print("=" * 80)
    for i, f in enumerate(away_fouls[:30], 1):
        print(f"{i}. Q{f['quarter']} {f['time']} - {f['event_code']} - {f['desc']}")
        if f['event_type'] or f['type']:
            print(f"   event_type={f['event_type']}, type={f['type']}")
    
    print(f"\n{home_abbrev} fouls (counted={len(home_fouls)}, expected=15):")
    print("=" * 80)
    for i, f in enumerate(home_fouls[:30], 1):
        print(f"{i}. Q{f['quarter']} {f['time']} - {f['event_code']} - {f['desc']}")
        if f['event_type'] or f['type']:
            print(f"   event_type={f['event_type']}, type={f['type']}")
    
    print(f"\n" + "=" * 80)
    print(f"Summary:")
    print(f"  {away_abbrev}: counted={len(away_fouls)}, expected=19, diff={len(away_fouls)-19}")
    print(f"  {home_abbrev}: counted={len(home_fouls)}, expected=15, diff={len(home_fouls)-15}")


if __name__ == '__main__':
    debug_game_fouls()

