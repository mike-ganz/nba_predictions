#!/usr/bin/env python3
"""
Validate team foul counting by comparing computed fouls from play-by-play
against ground truth team boxscore data.
"""

import pandas as pd
import random
import sys
import os
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_training_data import (
    load_play_by_play_data,
    determine_home_away_teams,
    map_structured_to_event_code,
    map_description_to_event_code
)


def get_abbrev_to_boxscore_name_mapping() -> dict:
    """Map team abbreviations to boxscore team names."""
    return {
        'ATL': 'Atlanta', 'BOS': 'Boston', 'BKN': 'Brooklyn', 'CHA': 'Charlotte',
        'CHI': 'Chicago', 'CLE': 'Cleveland', 'DAL': 'Dallas', 'DEN': 'Denver',
        'DET': 'Detroit', 'GSW': 'Golden State', 'HOU': 'Houston', 'IND': 'Indiana',
        'LAC': 'LA Clippers', 'LAL': 'LA Lakers', 'MEM': 'Memphis', 'MIA': 'Miami',
        'MIL': 'Milwaukee', 'MIN': 'Minnesota', 'NOP': 'New Orleans', 'NYK': 'New York',
        'OKC': 'Oklahoma City', 'ORL': 'Orlando', 'PHI': 'Philadelphia', 'PHX': 'Phoenix',
        'POR': 'Portland', 'SAC': 'Sacramento', 'SAS': 'San Antonio', 'TOR': 'Toronto',
        'UTA': 'Utah', 'WAS': 'Washington'
    }


def load_team_boxscore_data(season: str = "2023-2024") -> pd.DataFrame:
    """Load team boxscore data for ground truth foul counts."""
    path = f"data/team_boxscores/historical/{season}_NBA_Box_Score_Team-Stats.xlsx"
    df = pd.read_excel(path)
    return df


def count_fouls_from_play_by_play(game_df: pd.DataFrame, away_abbrev: str, home_abbrev: str) -> Tuple[int, int]:
    """
    Count team fouls from play-by-play data using our logic.
    Returns (away_fouls, home_fouls).
    """
    away_fouls = 0
    home_fouls = 0
    
    for idx, row in game_df.iterrows():
        # Skip rows without description
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
        
        # Count team fouls (exclude tech, and exclude o_foul turnovers to avoid double-counting)
        event_type_val = str(row.get('event_type', '')).lower()
        # Only count fouls where event_type is 'foul' (not 'turnover')
        # This avoids double-counting offensive fouls which appear as both foul and turnover
        if event_code in ("p_foul", "s_foul", "o_foul") and event_type_val == 'foul':
            team_name = row.get('team')
            
            # Determine which team committed the foul
            if isinstance(team_name, str):
                if team_name == away_abbrev:
                    away_fouls += 1
                elif team_name == home_abbrev:
                    home_fouls += 1
    
    return away_fouls, home_fouls


def validate_foul_counts(sample_size: int = 25) -> None:
    """Validate foul counts for a sample of games."""
    print(f"🔍 Validating team foul counts for {sample_size} random games...")
    print("=" * 80)
    
    # Load data
    print("📂 Loading play-by-play data...")
    pbp_df = load_play_by_play_data("2023-2024")
    print(f"✅ Loaded {len(pbp_df):,} plays")
    
    print("\n📂 Loading team boxscore data...")
    boxscore_df = load_team_boxscore_data("2023-2024")
    print(f"✅ Loaded {len(boxscore_df):,} team boxscore records")
    
    # Get team mappings
    print("\n📋 Determining home/away team mappings...")
    game_team_mapping = determine_home_away_teams(pbp_df)
    
    # Sample random games
    unique_games = pbp_df['game_id'].unique()
    sampled_games = random.sample(list(unique_games), min(sample_size, len(unique_games)))
    
    print(f"\n🎲 Sampled {len(sampled_games)} random games")
    print("=" * 80)
    
    matches = 0
    mismatches = []
    
    # Get abbreviation to boxscore name mapping
    abbrev_to_boxscore = get_abbrev_to_boxscore_name_mapping()
    
    for game_id in sampled_games:
        # Get team names
        team_mapping = game_team_mapping.get(game_id, {})
        away_abbrev = team_mapping.get('away_team', 'Unknown')
        home_abbrev = team_mapping.get('home_team', 'Unknown')
        
        if away_abbrev == 'Unknown' or home_abbrev == 'Unknown':
            print(f"⚠️  Game {game_id}: Could not determine teams")
            continue
        
        # Get play-by-play for this game
        game_df = pbp_df[pbp_df['game_id'] == game_id]
        
        # Count fouls from play-by-play
        computed_away, computed_home = count_fouls_from_play_by_play(game_df, away_abbrev, home_abbrev)
        
        # Get ground truth from boxscore
        # Need to find the matching rows in boxscore
        game_boxscore = boxscore_df[boxscore_df['GAME-ID'] == game_id]
        
        if len(game_boxscore) == 0:
            print(f"⚠️  Game {game_id}: No boxscore data found")
            continue
        
        # Map abbreviations to boxscore team names
        away_boxscore_name = abbrev_to_boxscore.get(away_abbrev)
        home_boxscore_name = abbrev_to_boxscore.get(home_abbrev)
        
        if not away_boxscore_name or not home_boxscore_name:
            print(f"⚠️  Game {game_id}: Unknown team abbreviation (away={away_abbrev}, home={home_abbrev})")
            continue
        
        # Find away and home team rows using the mapped names
        away_box = game_boxscore[game_boxscore['TEAM'].str.contains(away_boxscore_name, case=False, na=False)]
        home_box = game_boxscore[game_boxscore['TEAM'].str.contains(home_boxscore_name, case=False, na=False)]
        
        if len(away_box) == 0 or len(home_box) == 0:
            print(f"⚠️  Game {game_id}: Missing team boxscore data (away={away_abbrev}/{away_boxscore_name}, home={home_abbrev}/{home_boxscore_name})")
            continue
        
        actual_away = int(away_box.iloc[0]['PF'])
        actual_home = int(home_box.iloc[0]['PF'])
        
        # Compare
        away_match = computed_away == actual_away
        home_match = computed_home == actual_home
        
        if away_match and home_match:
            matches += 1
            status = "✅"
        else:
            status = "❌"
            mismatches.append({
                'game_id': game_id,
                'away_team': away_abbrev,
                'home_team': home_abbrev,
                'computed_away': computed_away,
                'actual_away': actual_away,
                'computed_home': computed_home,
                'actual_home': actual_home
            })
        
        print(f"{status} Game {game_id} ({away_abbrev} @ {home_abbrev}):")
        print(f"     Away: computed={computed_away}, actual={actual_away} {'✓' if away_match else '✗'}")
        print(f"     Home: computed={computed_home}, actual={actual_home} {'✓' if home_match else '✗'}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📊 VALIDATION SUMMARY:")
    print(f"   Total games validated: {len(sampled_games)}")
    print(f"   Exact matches: {matches}")
    print(f"   Mismatches: {len(mismatches)}")
    print(f"   Match rate: {matches / len(sampled_games) * 100:.1f}%")
    
    if mismatches:
        print(f"\n❌ MISMATCHES DETECTED:")
        for m in mismatches[:10]:  # Show first 10
            print(f"   Game {m['game_id']} ({m['away_team']} @ {m['home_team']}):")
            print(f"      Away: {m['computed_away']} vs {m['actual_away']} (diff: {m['computed_away'] - m['actual_away']})")
            print(f"      Home: {m['computed_home']} vs {m['actual_home']} (diff: {m['computed_home'] - m['actual_home']})")


if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    validate_foul_counts(25)

