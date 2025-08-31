#!/usr/bin/env python3
"""
Create JSONL Directly Using EXISTING Optimized Infrastructure
=============================================================
This leverages the existing optimized training data generator but loads
play-by-play data directly instead of going through the complex CSV mapping.

Key optimizations used:
• Batch processing (5K rows at a time)
• Cached PCA data (306K+ cache files)  
• Vectorized DataFrame operations
• Team stats caching
• Incremental saving
"""

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the existing optimized infrastructure
from training.data_generator_optimized import optimized_training_data_generator
from training import openai_formatter
from config import set_season_year
import pandas as pd
import time

def create_direct_jsonl_optimized(sample_size: int = None, batch_size: int = 5000):
    """
    Create JSONL directly from play-by-play using existing optimized infrastructure.
    
    This is much faster because it uses:
    • Your existing cached PCA data
    • Optimized batch processing  
    • Team stats caching
    • All the 5-10x speed improvements you already built
    """
    
    print("🚀 Direct JSONL Creation Using EXISTING Optimized Infrastructure")
    print("="*70)
    
    # Set season
    set_season_year("2023-2024")
    
    # Configure the existing optimized generator
    optimized_training_data_generator.set_batch_size(batch_size)
    
    # Enable incremental saving
    incremental_path = "data/training/DIRECT_OPTIMIZED_incremental.jsonl"
    optimized_training_data_generator.enable_incremental_save(
        incremental_path, 
        save_every_n_batches=5  # Save progress every 5 batches
    )
    
    print(f"⚙️  Using optimized batch processing ({batch_size:,} rows/batch)")
    print(f"💾 Incremental saving enabled (every 5 batches)")
    
    # Test with sample first
    if sample_size:
        print(f"\n🧪 Testing with sample ({sample_size:,} examples)...")
        sample_start = time.time()
        
        sample_df = optimized_training_data_generator.generate_dataset(
            season_year="2023-2024",
            n_total=10,  # Recent plays count
            sample_size=sample_size,
            force_real_pca=False  # Use existing cache - this is key for speed!
        )
        
        sample_elapsed = time.time() - sample_start
        sample_rate = len(sample_df) / sample_elapsed if sample_elapsed > 0 else 0
        
        print(f"✅ Sample: {len(sample_df):,} examples in {sample_elapsed:.1f}s")
        print(f"📈 Processing rate: {sample_rate:.1f} examples/second")
        
        # Convert sample to OpenAI format
        print("🤖 Converting sample to OpenAI JSONL...")
        sample_openai = openai_formatter.create_training_data(sample_df)
        sample_path = "data/training/DIRECT_OPTIMIZED_sample.jsonl" 
        openai_formatter.save_training_data(sample_openai, filename="DIRECT_OPTIMIZED_sample.jsonl")
        
        print(f"💾 Sample saved: {sample_path}")
        print(f"📊 {len(sample_openai):,} OpenAI examples created")
        
        # Ask user if they want to continue
        user_input = input("\n🤔 Continue with full season? (y/n): ").lower().strip()
        if user_input != 'y':
            print("👋 Stopping at sample.")
            return sample_path
    
    # Generate full season dataset
    print("\n🏀 Generating full season dataset...")
    print("⚡ Using all existing optimizations (5-10x faster than baseline)")
    
    full_start = time.time()
    
    full_season_df = optimized_training_data_generator.generate_dataset(
        season_year="2023-2024",
        n_total=10,  # Recent plays count  
        sample_size=None,  # Full dataset
        force_real_pca=False  # Critical: Use existing cache for speed!
    )
    
    generation_time = time.time() - full_start
    
    print(f"✅ Generated {len(full_season_df):,} examples in {generation_time:.1f}s")
    print(f"📈 Average rate: {len(full_season_df) / generation_time:.1f} examples/second")
    
    # Convert to OpenAI format
    print("\n🤖 Converting to OpenAI JSONL format...")
    openai_start = time.time()
    
    openai_examples = openai_formatter.create_training_data(full_season_df)
    final_path = "data/training/DIRECT_OPTIMIZED_full_season_2023_2024.jsonl"
    openai_formatter.save_training_data(openai_examples, filename="DIRECT_OPTIMIZED_full_season_2023_2024.jsonl")
    
    openai_time = time.time() - openai_start
    total_time = time.time() - full_start
    
    print(f"💾 OpenAI JSONL saved: {final_path}")
    print(f"📊 {len(openai_examples):,} OpenAI examples created")
    
    # Performance report
    print("\n" + "="*60)
    print("🚀 DIRECT OPTIMIZED PERFORMANCE REPORT")
    print("="*60)
    print(f"📊 Dataset Statistics:")
    print(f"   • Training examples: {len(full_season_df):,}")
    print(f"   • OpenAI examples: {len(openai_examples):,}")
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   • Generation time: {generation_time:.1f}s ({generation_time/60:.1f} min)")
    print(f"   • OpenAI conversion: {openai_time:.1f}s") 
    print(f"   • Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   • Processing rate: {len(full_season_df) / generation_time:.1f} examples/sec")
    
    print(f"\n✅ OPTIMIZATION TECHNIQUES USED:")
    print(f"   🚀 Vectorized batch processing (5-10x faster)")
    print(f"   ✅ Cached PCA data (306K+ cache files)")
    print(f"   ✅ Team stats caching") 
    print(f"   ✅ Memory-efficient batch sizing ({batch_size:,} rows/batch)")
    print(f"   ✅ Incremental saving (prevents data loss)")
    
    return final_path

def main():
    """Main function to run the direct optimized JSONL conversion."""
    
    # Generate full dataset (~594K examples)
    result_path = create_direct_jsonl_optimized()  # Remove sample_size for full dataset
    
    print(f"\n🎉 Direct optimized conversion complete!")
    print(f"💾 Final output: {result_path}")
    print(f"🚀 Ready for OpenAI fine-tuning!")

if __name__ == "__main__":
    main()
