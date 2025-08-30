#!/usr/bin/env python3
"""
Create Correct JSONL from CSV Training Data
==========================================
Converts the CSV training data to proper OpenAI JSONL format.
Each CSV row already contains complete training context - we don't need complex game grouping.
"""

import json
import pandas as pd
import re
from typing import Dict, Any
import os

def remove_parentheses_content(text):
    """Remove content within parentheses from text."""
    if not isinstance(text, str):
        return text
    
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

def create_openai_example_from_csv_rows(current_row: pd.Series, next_row: pd.Series, original_lookup: Dict) -> Dict[str, Any]:
    """
    Create a single OpenAI training example from a CSV row.
    
    The CSV row already contains:
    - json_training_data: Complete context (teams, players, recent plays)
    - All the info needed to extract the "next play" prediction
    
    Args:
        row: Single row from the CSV with json_training_data column
        
    Returns:
        dict: OpenAI training example
    """
    try:
        # Parse the JSON context from the current row
        context_json = json.loads(current_row['json_training_data'])
        
        # Extract team names from context to use in score
        home_team = context_json.get('home_team', {}).get('name', 'HOME')
        away_team = context_json.get('away_team', {}).get('name', 'AWAY')
        
        # Get the next play data from original lookup using game_id and play_id
        next_game_id = int(next_row.get('game_id', 0))
        next_play_id = int(next_row.get('play_id', 0))
        
        # Look up the original play data
        original_next_play = original_lookup.get((next_game_id, next_play_id))
        
        if original_next_play:
            # Use original data for accurate time/quarter info
            points_scored = original_next_play.get('points_scored', 0)
            scoring_team = original_next_play.get('scoring_team', None)
            
            # Convert points to int if it's a valid number
            if pd.notna(points_scored):
                try:
                    points_scored = int(float(points_scored))
                except:
                    points_scored = 0
            else:
                points_scored = 0
            
            next_play = {
                "quarter": int(original_next_play.get('quarter', 1)),
                "time_remaining": str(original_next_play.get('time_remaining', '12:00')),
                "description": remove_parentheses_content(str(original_next_play.get('description', 'Unknown play'))),
                "score": f"{away_team} {int(original_next_play.get('away_score', 0))} - {home_team} {int(original_next_play.get('home_score', 0))}",
                "scoring_team": scoring_team if pd.notna(scoring_team) else None,
                "points_scored": points_scored
            }
        else:
            # Fallback if lookup fails (should rarely happen)
            print(f"⚠️ Could not find original data for game {next_game_id}, play {next_play_id}")
            next_play = {
                "quarter": 1,
                "time_remaining": "12:00",
                "description": "Unknown play",
                "score": f"{away_team} 0 - {home_team} 0",
                "scoring_team": None,
                "points_scored": 0
            }
        
        # Create the OpenAI training example
        training_example = {
            "messages": [
                {
                    "role": "user",
                    "content": current_row['json_training_data']  # Complete context
                },
                {
                    "role": "assistant", 
                    "content": json.dumps({"next_play": next_play}, separators=(',', ':'))
                }
            ]
        }
        
        return training_example
        
    except Exception as e:
        print(f"⚠️ Error creating example from rows: {e}")
        return None

def load_original_playbyplay_for_reference() -> pd.DataFrame:
    """Load original play-by-play data to get proper time/quarter info for next plays."""
    try:
        original_path = "data/play_by_play/historical/[10-24-2023]-[06-17-2024]-combined-stats.csv"
        df = pd.read_csv(original_path)
        print(f"📂 Loaded original play-by-play reference: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"⚠️ Could not load original data: {e}")
        return None

def convert_csv_to_jsonl(csv_path: str, output_path: str, sample_size: int = None) -> int:
    """
    Convert CSV training data to correct OpenAI JSONL format.
    
    Args:
        csv_path: Path to the CSV training data
        output_path: Path for output JSONL file
        sample_size: If provided, only process this many rows (for testing)
        
    Returns:
        int: Number of examples created
    """
    
    print(f"📊 Loading CSV data from: {csv_path}")
    
    # Load CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} rows from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return 0
    
    # Sample if requested  
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"🎯 Using sample of {len(df):,} rows")
    
    # Check required columns
    if 'json_training_data' not in df.columns:
        print("❌ Error: CSV must contain 'json_training_data' column")
        return 0
    
    # Load original play-by-play data for proper time/quarter reference
    original_df = load_original_playbyplay_for_reference()
    if original_df is None:
        print("❌ Cannot proceed without original play-by-play data")
        return 0
    
    # Create lookup for original data by game_id and play_id  
    print("🔄 Creating lookup for original play data...")
    original_lookup = {}
    for _, row in original_df.iterrows():
        game_id = row.get('game_id')
        play_id = row.get('play_id') 
        if pd.notna(game_id) and pd.notna(play_id):
            original_lookup[(game_id, play_id)] = {
                'quarter': row.get('period', 1),  # Original uses 'period'
                'time_remaining': format_time_remaining(row.get('remaining_time', '12:00')),  # Convert time format
                'description': row.get('description', 'Unknown play'),
                'away_score': row.get('away_score', 0),
                'home_score': row.get('home_score', 0),
                'scoring_team': row.get('team', None),  # Use 'team' column for scoring team
                'points_scored': row.get('points', 0) if pd.notna(row.get('points')) else 0  # Use 'points' column
            }
    
    print(f"✅ Created lookup with {len(original_lookup):,} play entries")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Group by game_id to ensure we process games correctly
    print(f"🔄 Converting {len(df):,} rows to OpenAI format...")
    print("📋 Grouping by game_id to create proper sequence pairs...")
    
    examples_created = 0
    errors = 0
    total_processed = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Process each game separately to maintain sequence
        for game_id in df['game_id'].unique():
            game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
            
            # Create training examples from consecutive pairs
            for i in range(len(game_df) - 1):  # -1 because we need a next row
                current_row = game_df.iloc[i]
                next_row = game_df.iloc[i + 1]
                
                example = create_openai_example_from_csv_rows(current_row, next_row, original_lookup)
                if example:
                    f.write(json.dumps(example) + '\n')
                    examples_created += 1
                else:
                    errors += 1
                
                total_processed += 1
                if total_processed % 10000 == 0:
                    print(f"   Progress: {total_processed:,} pairs processed...")
    
    print(f"📊 Processed {len(df['game_id'].unique()):,} games")
    
    print(f"✅ Conversion complete!")
    print(f"   • Examples created: {examples_created:,}")
    print(f"   • Errors: {errors:,}")
    print(f"   • Success rate: {examples_created/(examples_created+errors)*100:.1f}%")
    print(f"💾 Saved to: {output_path}")
    
    return examples_created

def main():
    """Main conversion function."""
    
    print("🔄 CSV to Correct JSONL Converter")
    print("="*50)
    
    # Paths
    csv_path = "data/training/SAMPLE_optimized_season.csv"
    output_path = "data/training/CORRECTED_full_season_2023_2024.jsonl"
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # For testing, start with a sample
    print("🧪 Starting with sample conversion (10K rows)...")
    sample_output = "data/training/CORRECTED_sample_test.jsonl"
    
    examples_created = convert_csv_to_jsonl(csv_path, sample_output, sample_size=10000)
    
    if examples_created > 8000:  # Expect ~87% due to game boundaries (8.7K+ examples)
        print(f"\n✅ Sample test successful! Created {examples_created:,} examples ({examples_created/10000*100:.1f}% conversion)")
        
        user_input = input(f"\n🤔 Continue with full conversion? (y/n): ").lower().strip()
        if user_input == 'y':
            print(f"\n🚀 Converting full dataset...")
            total_examples = convert_csv_to_jsonl(csv_path, output_path)
            print(f"\n🎉 Full conversion complete: {total_examples:,} examples created!")
    else:
        print(f"\n⚠️ Sample test created only {examples_created:,}/10,000 examples")
        print("   Please check the CSV format and try again.")

if __name__ == "__main__":
    main()
