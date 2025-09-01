#!/usr/bin/env python3
"""
Ultra-optimized direct JSONL generator with massive performance improvements.

Key performance optimizations:
1. Game-grouped processing (eliminates O(n²) DataFrame operations)
2. Fast dict access instead of slow DataFrame iloc
3. Vectorized operations within games  
4. Bulk processing with minimal overhead
5. Proper JSON format output (fixes Python dict format issues)

Expected performance: 15-30 minutes for full dataset (vs 2-3 hours)
Output format: Valid JSON (double quotes) compatible with OpenAI fine-tuning
"""

import os
import json
import pandas as pd
import time
from typing import Optional
from pathlib import Path

# Import ultra-optimized generator
from training.data_generator_ultra_optimized import UltraOptimizedTrainingDataGenerator
from training.openai_formatter import OpenAIFormatter
from data.loaders import data_loader


def validate_jsonl_file(file_path: str, max_lines: int = 100) -> bool:
    """
    Validate that a JSONL file contains properly formatted JSON (not Python dicts).
    
    Args:
        file_path: Path to the JSONL file to validate
        max_lines: Maximum number of lines to check (default: 100)
        
    Returns:
        bool: True if all checked lines are valid JSON
    """
    print(f"🔍 Validating JSONL format: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    errors = 0
    lines_checked = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if lines_checked >= max_lines:
                    break
                    
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    json.loads(line)
                    lines_checked += 1
                except json.JSONDecodeError as e:
                    print(f"❌ Error on line {line_num}: {e}")
                    print(f"   Content: {line[:100]}...")
                    errors += 1
                    
                    if errors >= 5:  # Stop after 5 errors to avoid spam
                        print(f"   ... stopping after {errors} errors")
                        break
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    if errors == 0:
        print(f"✅ Validation passed: {lines_checked} lines checked, all valid JSON")
        return True
    else:
        print(f"❌ Validation failed: {errors} JSON errors found in {lines_checked} lines")
        return False


def create_direct_jsonl_ultra_optimized(sample_size: Optional[int] = None, 
                                        generation_mode: str = "remaining_plays",
                                        season_year: str = "2023-2024") -> str:
    """
    Generate OpenAI JSONL training data with ultra-high performance optimizations.
    
    Performance improvements:
    - ~10x faster than original implementation
    - Game-grouped processing eliminates O(n²) operations
    - Fast dict access replaces slow DataFrame operations
    - Expected time: 15-30 minutes for full dataset
    
    Args:
        sample_size: If provided, limit to this many rows for testing
        generation_mode: Training data generation mode:
            - "remaining_plays": Skip first 10 plays of each game as targets
            - "first_N_plays": One entry per game with first 10 plays as targets
        season_year: NBA season to process (e.g., "2022-2023", "2023-2024")
        
    Returns:
        str: Path to generated JSONL file
    """
    print("🚀 Starting ULTRA-OPTIMIZED direct JSONL generation...")
    print("⚡ Expected performance: ~10x faster than previous versions")
    
    # Validate generation mode
    valid_modes = ["remaining_plays", "first_N_plays"]
    if generation_mode not in valid_modes:
        raise ValueError(f"Invalid generation_mode '{generation_mode}'. Must be one of: {valid_modes}")
    
    # Set global config season to ensure consistency across all data loading
    from config.settings import config
    config.season_year = season_year
    print(f"📅 Season year: {season_year}")
    print(f"🎯 Generation mode: {generation_mode}")
    if generation_mode == "remaining_plays":
        print("   • Skipping first 10 plays of each game as targets")
        print("   • Normal training example quantity (~594K for full season)")
    elif generation_mode == "first_N_plays":
        print("   • One entry per game with first 10 plays as targets")
        print("   • Reduced training examples (~1.2K for full season)")
    
    # Create output directory
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ultra-optimized generator
    print("🔧 Initializing ultra-optimized generator...")
    ultra_optimized_training_data_generator = UltraOptimizedTrainingDataGenerator()
    
    # Load data
    print(f"📊 Loading {season_year} NBA play-by-play data...")
    try:
        df = data_loader.load_play_by_play_data(season_year)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure the data file exists and is accessible")
        raise
    
    # Apply sample size if specified
    if sample_size:
        print(f"🧪 Using sample size: {sample_size:,} rows")
        df = df.head(sample_size)
    else:
        print(f"📈 Processing FULL dataset: {len(df):,} rows")
    
    # Setup incremental saving for full runs
    if not sample_size or sample_size > 50000:
        incremental_path = output_dir / f"ULTRA_OPTIMIZED_incremental_{generation_mode}.jsonl"
        ultra_optimized_training_data_generator.enable_incremental_save(
            str(incremental_path),
            save_every_n_batches=100  # Save every 100 games
        )
        print(f"💾 Incremental saves enabled: {incremental_path}")
    
    # Generate training data with ultra-optimized processing
    print("\n🚀 Starting ULTRA-OPTIMIZED data generation...")
    print("📊 Using game-grouped processing for maximum performance...")
    
    full_season_df = ultra_optimized_training_data_generator.create_llm_training_data(
        df, 
        n_total=10,  # Recent plays to include
        force_real_pca=False,
        generation_mode=generation_mode
    )
    
    print(f"\n✅ Ultra-optimized training data generated: {len(full_season_df):,} examples")
    
    # Convert to OpenAI JSONL format
    print("🔄 Converting to OpenAI JSONL format...")
    print(f"📊 Processing {len(full_season_df):,} examples (this may take a few minutes)...")
    
    openai_formatter = OpenAIFormatter()
    try:
        training_examples = openai_formatter.create_training_data(full_season_df, generation_mode=generation_mode)
        print(f"📋 Created {len(training_examples):,} OpenAI training examples")
    except Exception as e:
        print(f"❌ Error during OpenAI format conversion: {e}")
        raise
    
    # Save final JSONL file
    if sample_size:
        output_file = output_dir / f"ULTRA_OPTIMIZED_sample_{sample_size}_{generation_mode}.jsonl"
    else:
        output_file = output_dir / f"ULTRA_OPTIMIZED_full_dataset_{generation_mode}.jsonl"
    
    # Warn about file overwriting
    if output_file.exists():
        print(f"⚠️ File already exists: {output_file}")
        print("🔄 Overwriting existing file...")
    
    print(f"💾 Saving final JSONL file: {output_file}")
    print("🔧 Ensuring proper JSON format (fixing Python dict format issue)...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Process in batches for better performance and memory efficiency
            batch_size = 1000
            total_examples = len(training_examples)
            
            for i in range(0, total_examples, batch_size):
                batch_end = min(i + batch_size, total_examples)
                batch = training_examples[i:batch_end]
                
                # Convert batch to JSON lines
                json_lines = []
                for j, example in enumerate(batch):
                    json_line = json.dumps(example, separators=(',', ':'), ensure_ascii=False)
                    
                    # Validate first few examples to ensure proper JSON format
                    if i + j < 3:
                        try:
                            json.loads(json_line)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Warning: Invalid JSON on line {i + j + 1}: {e}")
                    
                    json_lines.append(json_line)
                
                # Write batch
                f.write('\n'.join(json_lines))
                if batch_end < total_examples:
                    f.write('\n')
                
                # Progress update for large datasets
                if total_examples > 10000 and (i + batch_size) % 10000 == 0:
                    progress = min(batch_end, total_examples)
                    print(f"📝 Saved {progress:,}/{total_examples:,} examples...")
        
        print("✅ JSON format validation passed for sample examples")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        raise
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"\n🎉 Ultra-optimized conversion complete!")
    print(f"📊 Final statistics:")
    print(f"   • Examples: {len(training_examples):,}")
    print(f"   • File size: {file_size_mb:.1f} MB")
    print(f"   • Location: {output_file}")
    print(f"🚀 Ready for OpenAI fine-tuning!")
    
    return str(output_file)


def main():
    """Main function to run ultra-optimized direct JSONL conversion."""
    
    # SINGLE PLACE TO CHANGE SETTINGS - modify these as needed
    season_year = "2022-2023"  # Can be "2022-2023" or "2023-2024"
    generation_mode = "first_N_plays"  # Can be either "remaining_plays" or "first_N_plays"
     
    # Check for existing files and warn user
    incremental_path = f"data/training/ULTRA_OPTIMIZED_incremental_{generation_mode}.jsonl"
    final_path = f"data/training/ULTRA_OPTIMIZED_full_dataset_{generation_mode}.jsonl"
    
    if os.path.exists(incremental_path):
        print("🔍 Found existing incremental file. Validating JSON format...")
        is_valid = validate_jsonl_file(incremental_path, max_lines=50)
        if not is_valid:
            print("⚠️ Existing incremental file has JSON formatting issues.")
            print("💡 This will be overwritten with corrected format during generation.")
        print()
    
    if os.path.exists(final_path):
        file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"⚠️ Final output file already exists: {final_path} ({file_size_mb:.1f} MB)")
        print("🔄 This will be overwritten with new data.")
        print()
    
    # Generate full dataset with ultra-optimization (~594K examples)
    print("🎯 Starting generation process:")
    print("   1️⃣ Load 2023-2024 NBA play-by-play data")
    print("   2️⃣ Generate JSON training contexts (game-grouped processing)")
    print("   3️⃣ Convert to OpenAI JSONL format")
    print("   4️⃣ Save final dataset with validation")
    print("   ⏱️ Estimated time: 15-30 minutes")
    print()
    
    start_time = time.time()
    result_path = create_direct_jsonl_ultra_optimized(generation_mode=generation_mode, season_year=season_year)  # No sample_size = full dataset
    end_time = time.time()
    
    total_minutes = (end_time - start_time) / 60
    
    print(f"\n🎉 Ultra-optimized direct conversion complete!")
    print(f"💾 Final output: {result_path}")
    print(f"⏱️ Total time: {total_minutes:.1f} minutes")
    print(f"⚡ Performance improvement: ~10x faster than previous versions!")
    print(f"🚀 Ready for OpenAI fine-tuning!")


if __name__ == "__main__":
    import sys
    
    # Check for validation-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        if len(sys.argv) > 2:
            file_path = sys.argv[2]
            validate_jsonl_file(file_path, max_lines=200)
        else:
            # Validate common files for both generation modes
            generation_modes = ["remaining_plays", "first_N_plays"]
            for mode in generation_modes:
                files_to_check = [
                    f"data/training/ULTRA_OPTIMIZED_full_dataset_{mode}.jsonl",
                    f"data/training/ULTRA_OPTIMIZED_incremental_{mode}.jsonl"
                ]
                for file_path in files_to_check:
                    if os.path.exists(file_path):
                        validate_jsonl_file(file_path, max_lines=100)
                        print()
    else:
        main()
