#!/usr/bin/env python3
"""
Create JSONL Directly from Play-by-Play Data
===========================================
This approach is much cleaner - we go directly from play-by-play to JSONL
without the complex CSV mapping approach.

For each play N:
- User input: team stats + player stats + recent plays (N-M to N-1)  
- Assistant output: next play (play N)

All data comes directly from the original source with no mapping issues.
"""

import json
import pandas as pd
import re
from typing import Dict, Any, List
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing systems
from generate_team_stats import generate_team_stats
from pca_optimized import get_player_pca_score

def remove_parentheses_content(text: str) -> str:
    """Remove content within parentheses from text."""
    if not isinstance(text, str):
        return str(text)
    
    # Remove content in parentheses (including the parentheses)
    cleaned_text = re.sub(r'\s*\([^)]*\)', '', text)
    
    # Clean up any double spaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text

def format_time_remaining(time_str: str) -> str:
    """Convert time format from '0:12:00' to '12:00' for consistency."""
    if not isinstance(time_str, str):
        time_str = str(time_str)
    
    # Handle "0:MM:SS" format -> "MM:SS"  
    if time_str.startswith('0:') and len(time_str.split(':')) == 3:
        parts = time_str.split(':')
        return f"{parts[1]}:{parts[2]}"
    
    return time_str

def safe_description(desc) -> str:
    """Handle NaN descriptions and clean them."""
    if pd.isna(desc) or str(desc).lower() == 'nan':
        return "Play description unavailable"
    return remove_parentheses_content(str(desc))

def extract_team_names_from_game(game_plays: pd.DataFrame) -> tuple[str, str]:
    """Extract away and home team names from game data."""
    # Collect all unique team values from the 'team' column
    team_values = set()
    for _, row in game_plays.iterrows():
        team = row.get('team')
        if pd.notna(team) and len(str(team)) == 3:  # Team abbreviations are 3 letters
            team_values.add(str(team))
    
    # Convert to list and sort for consistency
    teams = sorted(list(team_values))
    
    # Return first two teams (alphabetically) as away and home
    if len(teams) >= 2:
        return teams[0], teams[1]  # e.g., "DEN", "LAL"
    elif len(teams) == 1:
        return teams[0], "UNKNOWN"
    else:
        return "AWAY", "HOME"

def extract_players_from_game(game_plays: pd.DataFrame) -> List[str]:
    """Extract all unique players from a game."""
    players = set()
    
    for _, row in game_plays.iterrows():
        # Add lineup players (a1-a5, h1-h5)
        for i in range(1, 6):
            away_player = row.get(f'a{i}')
            home_player = row.get(f'h{i}')
            if pd.notna(away_player):
                players.add(str(away_player))
            if pd.notna(home_player):
                players.add(str(home_player))
        
        # Add player involved in the play
        play_player = row.get('player')
        if pd.notna(play_player):
            players.add(str(play_player))
    
    return list(players)

def create_play_from_row(row: pd.Series, away_team: str, home_team: str) -> Dict[str, Any]:
    """Create a play object directly from a play-by-play row."""
    
    # Get points and scoring team
    points_scored = row.get('points', 0)
    scoring_team = row.get('team', None)
    
    # Convert points to int if it's a valid number
    if pd.notna(points_scored):
        try:
            points_scored = int(float(points_scored))
        except:
            points_scored = 0
    else:
        points_scored = 0
    
    # Only assign scoring_team if points were actually scored
    if points_scored > 0 and pd.notna(scoring_team):
        final_scoring_team = scoring_team
    else:
        final_scoring_team = None
    
    return {
        "quarter": int(row.get('period', 1)),
        "time_remaining": format_time_remaining(str(row.get('remaining_time', '12:00'))),
        "description": safe_description(row.get('description')),
        "score": f"{away_team} {int(row.get('away_score', 0))} - {home_team} {int(row.get('home_score', 0))}",
        "scoring_team": final_scoring_team,
        "points_scored": points_scored
    }

def get_team_stats_for_game(game_date: str, team_abbrev: str) -> Dict[str, float]:
    """Get real team stats using the existing system."""
    try:
        if not team_abbrev or team_abbrev in ["AWAY", "HOME", "UNKNOWN"]:
            raise ValueError(f"Invalid team abbreviation: {team_abbrev}")
            
        stats = generate_team_stats(team_abbrev, target_date=game_date)
        
        if stats is None or not isinstance(stats, dict):
            raise ValueError(f"generate_team_stats returned None or invalid data for {team_abbrev}")
            
        return {
            "OEFF": float(stats.get('OEFF', 110.0)),
            "DEFF": float(stats.get('DEFF', 110.0)), 
            "PACE": float(stats.get('PACE', 100.0)),
            "REST_DAYS": int(stats.get('REST_DAYS', 1))
        }
    except Exception as e:
        print(f"⚠️ Could not get team stats for {team_abbrev} on {game_date}: {e}")
        # Return reasonable defaults
        return {
            "OEFF": 110.0,
            "DEFF": 110.0, 
            "PACE": 100.0,
            "REST_DAYS": 1
        }

def get_player_stats_for_game(game_date: str, players: List[str], team_mapping: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """Get real player PCA scores using the existing system."""
    player_stats = []
    
    for player in players:
        try:
            # Get PCA scores from the optimized system
            offense, defense, shot_selection, efficiency = get_player_pca_score(
                player, game_date, current_season="2023-2024"
            )
            
            # Determine player's team (simplified for now)
            player_team = team_mapping.get(player, "UNKNOWN") if team_mapping else "UNKNOWN"
            
            player_stats.append({
                "name": player,
                "team": player_team,
                "pca_scores": {
                    "offense": round(float(offense), 4) if offense is not None else 0.0,
                    "defense": round(float(defense), 4) if defense is not None else 0.0,
                    "shot_selection": round(float(shot_selection), 4) if shot_selection is not None else 0.0,
                    "efficiency": round(float(efficiency), 4) if efficiency is not None else 0.0
                }
            })
        except Exception as e:
            print(f"⚠️ Could not get PCA scores for {player}: {e}")
            # Add player with default scores
            player_stats.append({
                "name": player,
                "team": team_mapping.get(player, "UNKNOWN") if team_mapping else "UNKNOWN",
                "pca_scores": {
                    "offense": 0.0,
                    "defense": 0.0,
                    "shot_selection": 0.0,
                    "efficiency": 0.0
                }
            })
    
    return player_stats

def create_player_team_mapping(game_plays: pd.DataFrame) -> Dict[str, str]:
    """Create a mapping of players to their teams based on lineup data."""
    player_team_map = {}
    
    # Get team names
    away_team, home_team = extract_team_names_from_game(game_plays)
    
    for _, row in game_plays.iterrows():
        # Map away team players
        for i in range(1, 6):
            away_player = row.get(f'a{i}')
            if pd.notna(away_player):
                player_team_map[str(away_player)] = away_team
        
        # Map home team players  
        for i in range(1, 6):
            home_player = row.get(f'h{i}')
            if pd.notna(home_player):
                player_team_map[str(home_player)] = home_team
        
        # Map play player based on team column
        play_player = row.get('player')
        play_team = row.get('team')
        if pd.notna(play_player) and pd.notna(play_team):
            player_team_map[str(play_player)] = str(play_team)
    
    return player_team_map

def create_training_example_direct(game_plays: pd.DataFrame, play_index: int, recent_plays_count: int = None) -> Dict[str, Any]:
    """
    Create a training example directly from play-by-play data.
    
    Args:
        game_plays: DataFrame containing all plays for one game
        play_index: Index of the "next play" we're trying to predict
        recent_plays_count: Number of recent plays to include in context
    """
    
    # Use default if not specified
    if recent_plays_count is None:
        from config.settings import DEFAULT_N_TOTAL_PLAYS
        recent_plays_count = DEFAULT_N_TOTAL_PLAYS
    
    if play_index == 0:
        return None  # Can't create example for first play (no recent plays)
    
    # Get the next play (what we're trying to predict)
    next_play_row = game_plays.iloc[play_index]
    
    # Extract real team names from game data
    away_team, home_team = extract_team_names_from_game(game_plays)
    
    # Get recent plays (plays before the current one)
    start_index = max(0, play_index - recent_plays_count)
    recent_plays_rows = game_plays.iloc[start_index:play_index]
    
    # Create recent plays array
    recent_plays = []
    for _, row in recent_plays_rows.iterrows():
        play = create_play_from_row(row, away_team, home_team)
        recent_plays.append(play)
    
    # Create next play  
    next_play = create_play_from_row(next_play_row, away_team, home_team)
    
    # Get team stats (temporarily using defaults while we fix team extraction)
    game_date = str(next_play_row.get('date', '2023-01-01'))
    # away_stats = get_team_stats_for_game(game_date, away_team)
    # home_stats = get_team_stats_for_game(game_date, home_team)
    # Temporary defaults:
    away_stats = {"OEFF": 110.0, "DEFF": 110.0, "PACE": 100.0, "REST_DAYS": 1}
    home_stats = {"OEFF": 110.0, "DEFF": 110.0, "PACE": 100.0, "REST_DAYS": 1}
    
    # Get real player stats with team mapping
    players = extract_players_from_game(game_plays)
    team_mapping = create_player_team_mapping(game_plays)
    player_stats = get_player_stats_for_game(game_date, players, team_mapping)
    
    # Create the training context
    context = {
        "away_team": {
            "name": away_team,
            "stats": away_stats
        },
        "home_team": {
            "name": home_team, 
            "stats": home_stats
        },
        "players": player_stats,
        "recent_plays": recent_plays
    }
    
    # Create the OpenAI training example
    training_example = {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(context, separators=(',', ':'))
            },
            {
                "role": "assistant",
                "content": json.dumps({"next_play": next_play}, separators=(',', ':'))
            }
        ]
    }
    
    return training_example

def convert_playbyplay_to_jsonl_direct(
    playbyplay_path: str, 
    output_path: str, 
    sample_size: int = None,
    recent_plays_count: int = None
) -> int:
    """
    Convert play-by-play data directly to JSONL format.
    
    Args:
        playbyplay_path: Path to the original play-by-play CSV
        output_path: Path to save the JSONL output
        sample_size: Optional sample size for testing
        recent_plays_count: Number of recent plays to include in context
    """
    
    # Use default if not specified
    if recent_plays_count is None:
        from config.settings import DEFAULT_N_TOTAL_PLAYS
        recent_plays_count = DEFAULT_N_TOTAL_PLAYS
    
    print("📊 Loading play-by-play data from:", playbyplay_path)
    df = pd.read_csv(playbyplay_path)
    print(f"✅ Loaded {len(df):,} rows from play-by-play data")
    
    # Sample if requested  
    if sample_size and sample_size < len(df):
        # Sample by taking first N rows (to maintain game sequences)
        df = df.head(sample_size).reset_index(drop=True)
        print(f"🎯 Using sample of {len(df):,} rows")
    
    # Group by game_id to process each game separately
    print("📋 Grouping by game_id to create training examples...")
    games = df.groupby('game_id')
    
    examples_created = 0
    errors = 0
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for game_id, game_plays in games:
            game_plays = game_plays.sort_values('play_id').reset_index(drop=True)
            
            # Create training examples for each play in the game (except the first)
            for play_index in range(1, len(game_plays)):  # Start at 1 (need prior plays for context)
                try:
                    example = create_training_example_direct(game_plays, play_index, recent_plays_count)
                    if example:
                        f.write(json.dumps(example) + '\n')
                        examples_created += 1
                        
                        # Progress update
                        if examples_created % 10000 == 0:
                            print(f"   Progress: {examples_created:,} examples created...")
                except Exception as e:
                    print(f"⚠️ Error creating example for game {game_id}, play {play_index}: {e}")
                    errors += 1
    
    print(f"📊 Processed {len(games)} games")
    print(f"✅ Direct conversion complete!")
    print(f"   • Examples created: {examples_created:,}")
    print(f"   • Errors: {errors:,}")
    if examples_created + errors > 0:
        print(f"   • Success rate: {100 * examples_created / (examples_created + errors):.1f}%")
    
    return examples_created

def main():
    """Main function to run the direct JSONL conversion."""
    print("🔄 Play-by-Play to JSONL Direct Converter")
    print("="*50)
    
    playbyplay_path = "data/play_by_play/historical/[10-24-2023]-[06-17-2024]-combined-stats.csv"
    sample_output = "data/training/DIRECT_sample_test.jsonl"
    final_output = "data/training/DIRECT_full_season_2023_2024.jsonl"
    
    # Test with sample first
    print("🧪 Starting with sample conversion (10K rows)...")
    examples_created = convert_playbyplay_to_jsonl_direct(playbyplay_path, sample_output, sample_size=10000)
    
    if examples_created > 0:
        print(f"💾 Saved to: {sample_output}")
        print(f"✅ Sample test successful! Created {examples_created:,} examples")
        
        # Ask user if they want to continue
        print()
        continue_full = input("🤔 Continue with full conversion? (y/n): ")
        
        if continue_full.lower() == 'y':
            print("🚀 Converting full dataset...")
            examples_created = convert_playbyplay_to_jsonl_direct(playbyplay_path, final_output)
            print(f"💾 Saved to: {final_output}")
            print(f"🎉 Full conversion complete: {examples_created:,} examples created!")
        else:
            print("👋 Conversion stopped by user.")
    else:
        print("❌ Sample test failed. Please check the data.")

if __name__ == "__main__":
    main()
