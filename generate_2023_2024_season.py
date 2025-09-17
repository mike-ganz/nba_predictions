#!/usr/bin/env python3
"""
Generate complete 2023-2024 NBA season training data in compact format.

Usage:
    python generate_2023_2024_season.py                             # Full season (Gemini format)
    python generate_2023_2024_season.py --format openai             # Full season (OpenAI format)
    python generate_2023_2024_season.py --format csv                # Full season (CSV format)
    python generate_2023_2024_season.py --sample 1000 --format gemini  # Sample + Gemini format
    python generate_2023_2024_season.py --games 50                  # First 50 games only
    python generate_2023_2024_season.py --n-total 7                 # 7 plays per sequence
"""

import argparse
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
    parser.add_argument('--output-prefix', type=str, default='nba_2023_2024',
                       help='Output filename prefix')
    parser.add_argument('--format', type=str, default='gemini',
                       choices=['csv', 'openai', 'gemini'],
                       help='Output format to generate (default: gemini)')
    parser.add_argument('--generation-mode', type=str, default='remaining_plays',
                       choices=['remaining_plays', 'first_N_plays'],
                       help='Training data generation mode (default: remaining_plays)')
    
    args = parser.parse_args()
    
    print("🏀 NBA 2023-2024 SEASON TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"📊 Parameters:")
    print(f"   • Plays per sequence: {args.n_total}")
    print(f"   • Sample limit: {args.sample or 'None (full dataset)'}")
    print(f"   • Games limit: {args.games or 'None (all games)'}")
    print(f"   • Output format: {args.format}")
    print(f"   • Generation mode: {args.generation_mode}")
    if args.generation_mode == "remaining_plays":
        print(f"     → Standard next-play prediction (~high volume)")
    elif args.generation_mode == "first_N_plays":
        print(f"     → Sequence generation (~low volume, 1 per game)")
    print()
    
    # Load 2023-2024 season data 
    print("📂 Loading 2023-2024 play-by-play data...")
    season_df = load_play_by_play_data("2023-2024")
    print(f"✅ Loaded {len(season_df):,} plays from {season_df['game_id'].nunique():,} games")
    
    # Optional: Limit to first N games for testing
    if args.games:
        print(f"\n🔬 Limiting to first {args.games} games...")
        unique_games = sorted(season_df['game_id'].unique())[:args.games]
        season_df = season_df[season_df['game_id'].isin(unique_games)]
        print(f"✅ Limited to {len(season_df):,} plays from {len(unique_games)} games")
    
    # Filter games with sufficient plays (avoid games with too few plays)
    print("\n🔍 Filtering games with sufficient data...")
    game_play_counts = season_df.groupby('game_id').size()
    valid_games = game_play_counts[game_play_counts >= 50].index  # At least 50 plays
    filtered_df = season_df[season_df['game_id'].isin(valid_games)]
    
    print(f"✅ Filtered to {len(filtered_df):,} plays from {len(valid_games):,} games")
    print(f"📈 Average plays per game: {len(filtered_df) / len(valid_games):.1f}")
    
    # Generate compact training data
    print(f"\n🚀 Generating compact training data with ULTRA-OPTIMIZED pipeline...")
    print("   ⚡ Performance improvements:")
    print("      • Comprehensive PCA cache (99%+ hit rate vs 70% before)")
    print("      • Smart season fallback handling (batch processing)")
    print("      • Pre-loaded season data (eliminates repeated loading)")
    print("      • Eliminates expensive individual calculations")
    print("   🎯 Expected speedup: 5-20x faster than original")
    
    try:
        training_df = create_llm_training_data(
            filtered_df, 
            n_total=args.n_total,
            filter_nan=True,
            generation_mode=args.generation_mode,
            use_direct_compact=True,  # 🚀 OPTIMIZATION #1: Direct compact builder (30-50% faster, validated)
            use_batch_pca=True        # 🚀 OPTIMIZATION #2: Batch PCA calculations (4x faster, validated identical results)
        )
        print(f"✅ Generated {len(training_df):,} training examples")
        
        # Optional: Sample for testing
        if args.sample and len(training_df) > args.sample:
            print(f"\n🔬 Sampling {args.sample} examples for testing...")
            training_df = training_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
            print(f"✅ Sampled to {len(training_df):,} examples")
        
        # Verify compact format
        print(f"\n🔍 Verifying compact format...")
        sample_json = json.loads(training_df.iloc[0]['json_training_data'])
        format_type = "✅ COMPACT" if 'a' in sample_json else "❌ VERBOSE"
        print(f"   Format: {format_type}")
        print(f"   Keys: {list(sample_json.keys())}")
        print(f"   Teams: {sample_json.get('a', 'N/A')} vs {sample_json.get('h', 'N/A')}")
        print(f"   Plays: {len(sample_json.get('p', []))}")
        
        # Generate output in specified format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n📤 Generating {args.format.upper()} format...")
        
        if args.format == 'csv':
            filename = f"{args.output_prefix}_compact_{timestamp}.csv"
            training_df.to_csv(filename, index=False)
            print(f"💾 CSV saved: {filename}")
            
        elif args.format == 'openai':
            from training.openai_formatter import OpenAIFormatter
            openai_formatter = OpenAIFormatter()
            openai_examples = openai_formatter.create_training_data(training_df, generation_mode=args.generation_mode, n_total=args.n_total)
            filename = f"{args.output_prefix}_openai_compact_{args.generation_mode}_{timestamp}.jsonl"
            with open(filename, 'w') as f:
                for example in openai_examples:
                    f.write(json.dumps(example, separators=(',', ':')) + '\n')
            print(f"💾 OpenAI JSONL saved: {filename}")
            
        elif args.format == 'gemini':
            gemini_formatter = GeminiFormatter()
            gemini_examples = gemini_formatter.create_training_data(training_df, generation_mode=args.generation_mode, n_total=args.n_total)
            filename = f"{args.output_prefix}_gemini_compact_{args.generation_mode}_{timestamp}.jsonl"
            with open(filename, 'w') as f:
                for example in gemini_examples:
                    f.write(json.dumps(example, separators=(',', ':')) + '\n')
            print(f"💾 Gemini JSONL saved: {filename}")
        
        # Final summary
        print(f"\n🎉 COMPLETE!")
        print(f"📊 Generated {len(training_df):,} compact training examples")
        print(f"📁 Output file: {filename}")
        print(f"✨ Ready for {args.format.upper()} fine-tuning using the compact schema!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
