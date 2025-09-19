#!/usr/bin/env python3
"""
🚀 ULTRA-FAST Gemini Conversion Script

This script takes your existing compact training data and converts it to Gemini format
using the ultra-optimized formatter (10-50x faster than the standard method).

Usage:
    python convert_to_gemini_ultra_fast.py --input-file path/to/training_data.jsonl --output-file path/to/gemini_output.jsonl
    
Or use with the most recent training file:
    python convert_to_gemini_ultra_fast.py --auto-detect
"""

import argparse
import pandas as pd
import json
import os
import glob
from datetime import datetime
import sys

# Add training directory to path
sys.path.append('./training')

from training.gemini_formatter_ultra_optimized import create_ultra_fast_gemini_training_data

def find_latest_training_file():
    """Find the most recent compact training data file."""
    pattern = "data/training/nba_*_gemini_compact_remaining_plays_*.jsonl"
    files = glob.glob(pattern)
    
    if not files:
        # Try alternative pattern
        pattern = "data/training/*compact*.jsonl"
        files = glob.glob(pattern)
    
    if not files:
        print("❌ No training files found matching pattern")
        return None
    
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    latest = files[0]
    
    print(f"📁 Found latest training file: {latest}")
    return latest

def load_training_data(input_file: str) -> pd.DataFrame:
    """Load training data from JSONL file."""
    print(f"📊 Loading training data from {input_file}...")
    
    data = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {line_num}: Invalid JSON")
                continue
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df):,} training records")
    
    # Verify required columns
    required_cols = ['game_id', 'play_id', 'json_training_data']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    return df

def save_gemini_data(training_examples: list, output_file: str):
    """Save Gemini training examples to JSONL file."""
    print(f"💾 Saving {len(training_examples):,} Gemini examples to {output_file}...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example, separators=(',', ':')) + '\n')
    
    print(f"✅ Saved Gemini training data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Ultra-fast Gemini format conversion")
    parser.add_argument('--input-file', type=str, help="Input training data file")
    parser.add_argument('--output-file', type=str, help="Output Gemini format file")
    parser.add_argument('--auto-detect', action='store_true', 
                       help="Auto-detect latest training file")
    parser.add_argument('--season', type=str, default=None,
                       help="NBA season (e.g., '2023-2024')")
    
    args = parser.parse_args()
    
    print("🚀 ULTRA-FAST GEMINI CONVERSION")
    print("=" * 50)
    
    # Determine input file
    if args.auto_detect:
        input_file = find_latest_training_file()
        if not input_file:
            print("❌ Could not find training file")
            return
    else:
        input_file = args.input_file
        if not input_file:
            print("❌ Please specify --input-file or use --auto-detect")
            return
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/training/{base_name}_gemini_ultra_fast_{timestamp}.jsonl"
    
    print(f"📥 Input:  {input_file}")
    print(f"📤 Output: {output_file}")
    
    # Load training data
    df = load_training_data(input_file)
    if df is None:
        return
    
    # Convert to Gemini format using ultra-optimized formatter
    print(f"\n🚀 Starting ultra-fast Gemini conversion...")
    start_time = datetime.now()
    
    try:
        training_examples = create_ultra_fast_gemini_training_data(
            df, 
            generation_mode="remaining_plays",
            season=args.season
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        rate = len(training_examples) / duration if duration > 0 else 0
        
        print(f"\n🎉 CONVERSION COMPLETE!")
        print(f"✅ Generated {len(training_examples):,} Gemini examples")
        print(f"⏱️  Total time: {duration:.1f}s")
        print(f"⚡ Rate: {rate:.0f} examples/sec")
        
        # Save results
        save_gemini_data(training_examples, output_file)
        
        print(f"\n🎯 SUCCESS! Your Gemini training data is ready at:")
        print(f"   {output_file}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
