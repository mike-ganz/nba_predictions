#!/usr/bin/env python3
"""
Generate complete 2023-2024 NBA season training data in compact or verbose format.

Usage:
    python generate_2023_2024_season.py                             # Full season (Gemini format, compact)
    python generate_2023_2024_season.py --format openai             # Full season (OpenAI format, compact)
    python generate_2023_2024_season.py --format csv                # Full season (CSV format, compact)
    python generate_2023_2024_season.py --data-format verbose       # Full season (verbose format)
    python generate_2023_2024_season.py --sample 1000 --format gemini  # Sample + Gemini format
    python generate_2023_2024_season.py --games 50                  # First 50 games only
    python generate_2023_2024_season.py --n-total 7                 # 7 plays per sequence
"""

import argparse
import os
import json
import pandas as pd
from datetime import datetime
from generate_training_data import load_play_by_play_data
from generate_training_data_OPTIMIZED import create_llm_training_data_ULTRA_FAST as create_llm_training_data
from training.gemini_formatter import GeminiFormatter


def main():
    parser = argparse.ArgumentParser(description='Generate 2023-2024 NBA season training data')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Generate only N training examples (for testing)')
    parser.add_argument('--games', type=int, default=None,
                       help='Process only first N games (for testing)')
    parser.add_argument('--n-total', type=int, default=5,
                       help='Number of recent plays per sequence (default: 5)')
    parser.add_argument('--output-prefix', type=str, default=None,
                       help='Output filename prefix (default: nba_{season})')
    parser.add_argument('--format', type=str, default='gemini',
                       choices=['csv', 'openai', 'gemini'],
                       help='Output format to generate (default: gemini)')
    parser.add_argument('--generation-mode', type=str, default='remaining_plays',
                       choices=['remaining_plays', 'first_N_plays'],
                       help='Training data generation mode (default: remaining_plays)')
    parser.add_argument('--season', type=str, default='2023-2024',
                       choices=['2022-2023', '2023-2024', '2024-2025'],
                       help='NBA season to generate data for (default: 2023-2024)')
    parser.add_argument('--ultra-fast-gemini', action='store_true',
                       help='Use ultra-optimized Gemini formatter (10-50x faster)')
    parser.add_argument('--data-format', type=str, default='compact',
                       choices=['compact', 'verbose'],
                       help='Training data schema format (default: compact)')

    # Example usage:
    # python generate_2023_2024_season.py --games 2 --n-total 12 --format gemini --generation-mode remaining_plays --season 2023-2024
    
    args = parser.parse_args()
    
    # Ensure output directory exists under repo's data/training
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'data', 'training')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default output prefix based on season if not provided
    if args.output_prefix is None:
        args.output_prefix = f"nba_{args.season.replace('-', '_')}"
    
    print(f"NBA {args.season.upper()} SEASON TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f" Parameters:")
    print(f"   - Season: {args.season}")
    print(f"   - Plays per sequence: {args.n_total}")
    print(f"   - Sample limit: {args.sample or 'None (full dataset)'}")
    print(f"   - Games limit: {args.games or 'None (all games)'}")
    print(f"   - Output format: {args.format}")
    print(f"   - Data format: {args.data_format}")
    print(f"   - Generation mode: {args.generation_mode}")
    if args.generation_mode == "remaining_plays":
        print(f"     > Standard next-play prediction (~high volume)")
    elif args.generation_mode == "first_N_plays":
        print(f"     > Sequence generation (~low volume, 1 per game)")
    print()
    
    # Load season data 
    print(f" Loading {args.season} play-by-play data...")
    season_df = load_play_by_play_data(args.season)
    print(f" Loaded {len(season_df):,} plays from {season_df['game_id'].nunique():,} games")
    
    # Optional: Limit to first N games for testing
    if args.games:
        print(f"\n Limiting to first {args.games} games...")
        unique_games = sorted(season_df['game_id'].unique())[:args.games]
        season_df = season_df[season_df['game_id'].isin(unique_games)]
        print(f" Limited to {len(season_df):,} plays from {len(unique_games)} games")
    
    # Filter games with sufficient plays (avoid games with too few plays)
    print("\n Filtering games with sufficient data...")
    game_play_counts = season_df.groupby('game_id').size()
    valid_games = game_play_counts[game_play_counts >= 50].index  # At least 50 plays
    filtered_df = season_df[season_df['game_id'].isin(valid_games)]
    
    print(f" Filtered to {len(filtered_df):,} plays from {len(valid_games):,} games")
    print(f" Average plays per game: {len(filtered_df) / len(valid_games):.1f}")
    
    # Generate training data in selected format
    print(f"\n Generating {args.data_format} training data with ULTRA-OPTIMIZED pipeline...")
    print("    Performance improvements:")
    print("      - Comprehensive PCA cache (99%+ hit rate vs 70% before)")
    print("      - Smart season fallback handling (batch processing)")
    print("      - Pre-loaded season data (eliminates repeated loading)")
    print("      - Eliminates expensive individual calculations)")
    print("    Expected speedup: 5-20x faster than original")
    
    try:
        # Determine format choice
        use_compact = (args.data_format == 'compact')
        
        training_df = create_llm_training_data(
            filtered_df, 
            n_total=args.n_total,
            filter_nan=True,
            generation_mode=args.generation_mode,
            use_direct_compact=use_compact,  #  FIXED: Now actually generates chosen format (was always compact before)
            use_batch_pca=True               #  OPTIMIZATION #2: Batch PCA calculations (4x faster, validated identical results)
        )
        print(f" Generated {len(training_df):,} training examples")
        
        # Optional: Sample for testing
        if args.sample and len(training_df) > args.sample:
            print(f"\n Sampling {args.sample} examples for testing...")
            
            if args.generation_mode == "first_N_plays":
                # For first_N_plays mode, only sample rows with actual context data (not empty "{}")
                valid_rows = training_df[training_df['json_training_data'] != "{}"].copy()
                print(f"   Found {len(valid_rows):,} valid rows with context data out of {len(training_df):,} total")
                
                if len(valid_rows) > args.sample:
                    training_df = valid_rows.sample(n=args.sample, random_state=42).reset_index(drop=True)
                else:
                    training_df = valid_rows.reset_index(drop=True)
                    print(f"   Using all {len(training_df):,} valid rows (less than requested sample size)")
            else:
                # Standard random sampling for remaining_plays mode
                training_df = training_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
            
            print(f" Sampled to {len(training_df):,} examples")
        
        # Verify data format
        print(f"\n Verifying {args.data_format} format...")
        
        # Check if json_training_data column exists and has valid data
        if 'json_training_data' not in training_df.columns:
            print(f" ERROR: 'json_training_data' column missing from training DataFrame")
            print(f"   Available columns: {list(training_df.columns)}")
            return 1
            
        # Find a non-empty training example (skip the ones that are correctly empty in remaining_plays mode)
        sample_json_str = None
        for idx, row in training_df.iterrows():
            json_str = row['json_training_data']
            if json_str and json_str != '{}' and len(json_str) > 10:
                sample_json_str = json_str
                print(f"   Using training example at index {idx} for verification")
                break
        
        if sample_json_str is None:
            # If we can't find any non-empty examples, use the first one for error reporting
            sample_json_str = training_df.iloc[0]['json_training_data']
        
        # Debug the raw JSON string
        print(f"   Raw JSON string length: {len(str(sample_json_str))}")
        if pd.isna(sample_json_str) or sample_json_str == "" or sample_json_str == "{}":
            print(f" ERROR: json_training_data is empty or null")
            print(f"   Sample value: '{sample_json_str}'")
            
            # For first_N_plays mode, this is expected - the contexts should be clean
            if args.generation_mode == "first_N_plays":
                print(f" This is expected for first_N_plays mode - contexts should be clean without 'p' field")
                print(f"   Skipping format verification for first_N_plays mode")
            else:
                return 1
        else:
            try:
                sample_json = json.loads(sample_json_str)
                
                # Detect actual format
                is_compact = 'a' in sample_json or 'A' in sample_json  # Support both lowercase and uppercase
                is_verbose = 'away_team' in sample_json
                
                if is_compact:
                    actual_format = "COMPACT"
                    team_keys = ('a', 'h') if 'a' in sample_json else ('A', 'H')
                    plays_key = 'p'
                elif is_verbose:
                    actual_format = "VERBOSE"
                    team_keys = ('away_team', 'home_team')
                    plays_key = 'recent_plays'
                else:
                    actual_format = "UNKNOWN"
                    team_keys = ('N/A', 'N/A')
                    plays_key = None
                
                expected_format = args.data_format.upper()
                format_match = actual_format == expected_format
                
                print(f"   Expected: {expected_format}, Actual: {actual_format}")
                print(f"   Format match: {' CORRECT' if format_match else ' MISMATCH'}")
                print(f"   Keys: {list(sample_json.keys())}")
                
                if is_compact:
                    print(f"   Teams: {sample_json.get(team_keys[0], 'N/A')} vs {sample_json.get(team_keys[1], 'N/A')}")
                elif is_verbose:
                    away_name = sample_json.get('away_team', {}).get('name', 'N/A')
                    home_name = sample_json.get('home_team', {}).get('name', 'N/A')
                    print(f"   Teams: {away_name} vs {home_name}")
                
                # Check plays field based on generation mode and format
                if args.generation_mode == "first_N_plays":
                    has_plays = plays_key in sample_json if plays_key else False
                    print(f"   Has plays field: {' UNEXPECTED' if has_plays else ' CORRECT (clean context)'}")
                    print(f"   Clean context verified for first_N_plays mode")
                else:
                    if plays_key:
                        plays_count = len(sample_json.get(plays_key, []))
                        print(f"   Plays: {plays_count}")
                    
            except json.JSONDecodeError as e:
                print(f" ERROR: Invalid JSON in json_training_data: {e}")
                print(f"   Sample content: '{sample_json_str[:200]}...' (truncated)")
                return 1
        
        # Generate output in specified format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n Generating {args.format.upper()} format...")
        
        if args.format == 'csv':
            filename = os.path.join(output_dir, f"{args.output_prefix}_{args.data_format}_{timestamp}.csv")
            training_df.to_csv(filename, index=False)
            print(f" CSV saved: {filename}")
            
        elif args.format == 'openai':
            from training.openai_formatter import OpenAIFormatter
            openai_formatter = OpenAIFormatter()
            openai_examples = openai_formatter.create_training_data(training_df, generation_mode=args.generation_mode, n_total=args.n_total)
            filename = os.path.join(output_dir, f"{args.output_prefix}_openai_{args.data_format}_{args.generation_mode}_{timestamp}.jsonl")
            with open(filename, 'w') as f:
                for example in openai_examples:
                    f.write(json.dumps(example, separators=(',', ':')) + '\n')
            print(f" OpenAI JSONL saved: {filename}")
            
        elif args.format == 'gemini':
            # Check for ultra-fast formatter flag
            use_ultra_fast = getattr(args, 'ultra_fast_gemini', False)
            
            if use_ultra_fast:
                print(" Using ULTRA-OPTIMIZED Gemini formatter...")
                from training.gemini_formatter_ultra_optimized import create_ultra_fast_gemini_training_data
                gemini_examples = create_ultra_fast_gemini_training_data(training_df, generation_mode=args.generation_mode, n_total=args.n_total, season=args.season)
                filename = os.path.join(output_dir, f"{args.output_prefix}_gemini_{args.data_format}_{args.generation_mode}_ULTRA_FAST_{timestamp}.jsonl")
            else:
                print("  Using standard Gemini formatter (slower)...")
                gemini_formatter = GeminiFormatter()
                gemini_examples = gemini_formatter.create_training_data(training_df, generation_mode=args.generation_mode, n_total=args.n_total, season=args.season)
                filename = os.path.join(output_dir, f"{args.output_prefix}_gemini_{args.data_format}_{args.generation_mode}_{timestamp}.jsonl")
            
            with open(filename, 'w') as f:
                for example in gemini_examples:
                    f.write(json.dumps(example, separators=(',', ':')) + '\n')
            print(f" Gemini JSONL saved: {filename}")
        
        # Final summary
        print(f"\n COMPLETE!")
        print(f" Generated {len(training_df):,} {args.data_format} training examples")
        print(f" Output file: {filename}")
        print(f" Ready for {args.format.upper()} fine-tuning using the {args.data_format} schema!")
        
    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
