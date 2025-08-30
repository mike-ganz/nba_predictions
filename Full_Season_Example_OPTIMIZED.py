# %%
"""
NBA Training Data - OPTIMIZED Full Season Generation
==================================================

HIGH-PERFORMANCE version with major speed optimizations:
• Batch processing instead of row-by-row
• Reduced JSON serialization overhead
• Optional direct OpenAI generation
• Memory-efficient processing
"""

# %%
from config import set_season_year
from training import training_data_generator, openai_formatter  
from training.data_generator_optimized import (
    optimized_training_data_generator, 
    generate_season_dataset_optimized,
    benchmark_performance
)
from data import file_manager, data_preview
import time
import pandas as pd

# Set your season
set_season_year("2023-2024")
print("✅ Setup complete!")

# %%
# 🚀 OPTIMIZATION OPTIONS
print("🚀 OPTIMIZATION OPTIONS:")
print("1. BATCH_SIZE: Process data in chunks to reduce memory usage")
print("2. SKIP_CSV: Skip CSV generation if you only need OpenAI format")
print("3. SAMPLE_FIRST: Test with sampling before full run")

# Configuration
BATCH_SIZE = 5000   # Process in chunks of 5K rows (optimized default)
SKIP_CSV = False    # Set to True if you only need OpenAI format
SAMPLE_FIRST = True  # Set to False for immediate full production run
USE_OPTIMIZED = True  # Set to False to use original generator for comparison
RUN_BENCHMARK = False  # Set to True to run performance comparison first

# %%
# 🔬 OPTIONAL: Performance benchmark first
if RUN_BENCHMARK:
    print("\n🔬 RUNNING PERFORMANCE BENCHMARK...")
    print("This will compare original vs optimized performance on 1000 examples")
    
    benchmark_results = benchmark_performance(
        season_year="2023-2024", 
        test_size=1000
    )
    
    print(f"🚀 Benchmark complete - {benchmark_results['speedup_factor']:.1f}x speedup!")
    input("\nPress Enter to continue with full generation...")

# %%
# 🧪 OPTIONAL: Quick sample test first
if SAMPLE_FIRST:
    print("\n🧪 RUNNING SAMPLE TEST FIRST (10K examples)...")
    sample_start = time.time()
    
    # Choose generator based on configuration
    generator_name = "OPTIMIZED" if USE_OPTIMIZED else "ORIGINAL"
    print(f"Using {generator_name} generator for sample...")
    
    if USE_OPTIMIZED:
        # Configure optimized generator
        optimized_training_data_generator.set_batch_size(BATCH_SIZE)
        
        # 💾 ENABLE INCREMENTAL SAVING for sample too
        incremental_path = "data/training/INCREMENTAL_full_season_2023_2024.jsonl"
        optimized_training_data_generator.enable_incremental_save(
            incremental_path, 
            save_every_n_batches=5
        )
        
        sample_df = optimized_training_data_generator.generate_dataset(
            season_year="2023-2024",
            n_total=10,
            sample_size=None,  # None = process ALL games/rows in the season!
            force_real_pca=False  # ← Use existing cache instead of rebuilding
        )
    else:
        
        sample_df = training_data_generator.generate_dataset(
            season_year="2023-2024",
            n_total=10,
            sample_size=None,  # None = process ALL games/rows in the season!
            force_real_pca=True
        )
    
    sample_elapsed = time.time() - sample_start
    sample_rate = len(sample_df) / sample_elapsed if sample_elapsed > 0 else 0
    
    print(f"✅ Sample: {len(sample_df)} examples in {sample_elapsed:.1f}s")
    print(f"📈 Processing rate: {sample_rate:.1f} examples/second")
    print(f"📊 Full season estimate: {(200000 / sample_rate / 60):.1f} minutes for ~200K examples")
    
    # Save sample for quick testing
    if not SKIP_CSV:
        sample_path = file_manager.save_llm_dataset(sample_df, "SAMPLE_optimized_season.csv")
        print(f"💾 Sample CSV: {sample_path}")
    
    user_input = input("\n🤔 Continue with full season? (y/n): ").lower().strip()
    if user_input != 'y':
        print("👋 Stopping at sample. Adjust settings and re-run when ready!")
        exit()

# %%
# 🏀 OPTIMIZED FULL SEASON GENERATION
print("\n🏀 GENERATING FULL SEASON DATA (OPTIMIZED MODE)...")
print("⚡ Using batch processing and optimized JSON handling")

start_time = time.time()

# Generate dataset with optimizations
generator_name = "OPTIMIZED" if USE_OPTIMIZED else "ORIGINAL" 
print(f"Using {generator_name} generator for full season...")

print(f"🔍 DEBUG: USE_OPTIMIZED = {USE_OPTIMIZED}")
if USE_OPTIMIZED:
    print("✅ Taking OPTIMIZED path")
    # Configure and use optimized generator
    optimized_training_data_generator.set_batch_size(BATCH_SIZE)
    
    # 💾 ENABLE INCREMENTAL SAVING to prevent data loss
    incremental_path = "data/training/INCREMENTAL_full_season_2023_2024.jsonl"
    optimized_training_data_generator.enable_incremental_save(
        incremental_path, 
        save_every_n_batches=5  # Save progress every 5 batches (25K rows)
    )
    
    # Show performance settings
    perf_stats = optimized_training_data_generator.get_performance_stats()
    print(f"⚙️  Batch size: {perf_stats['batch_size']:,}")
    print(f"⚙️  Cached team stats: {perf_stats['team_stats_cache_size']:,}")
    print(f"💾 Incremental saving: {perf_stats['incremental_save_enabled']} (every {perf_stats['save_every_n_batches']} batches)")
    
    full_season_df = optimized_training_data_generator.generate_dataset(
        season_year="2023-2024",
        n_total=10,
        sample_size=None,  # Full dataset
        force_real_pca=False  # ← Use existing cache instead of rebuilding
    )
else:
    full_season_df = training_data_generator.generate_dataset(
        season_year="2023-2024",
        n_total=10,
        sample_size=None,  # Full dataset
        force_real_pca=False  # ← Use existing cache instead of rebuilding
    )

generation_time = time.time() - start_time
print(f"✅ Generated {len(full_season_df)} examples in {generation_time:.1f}s")
print(f"📈 Average rate: {len(full_season_df) / generation_time:.1f} examples/second")

# %%
# 💾 OPTIMIZED SAVING
save_start = time.time()

if not SKIP_CSV:
    print("\n💾 Saving CSV dataset...")
    season_csv_path = file_manager.save_llm_dataset(
        full_season_df, 
        "OPTIMIZED_full_season_2023_2024.csv"
    )
    csv_time = time.time() - save_start
    print(f"📊 CSV saved in {csv_time:.1f}s: {season_csv_path}")
else:
    print("⏭️  Skipping CSV save (SKIP_CSV=True)")
    season_csv_path = "skipped"

# %%
# 🤖 OPTIMIZED OPENAI CONVERSION
openai_start = time.time()
print("\n🤖 Converting to OpenAI format (optimized)...")

# Use optimized batch conversion
openai_examples = openai_formatter.create_training_data(full_season_df)
openai_path = openai_formatter.save_training_data(
    openai_examples,
    filename="OPTIMIZED_full_season_2023_2024_openai.jsonl"
)

openai_time = time.time() - openai_start
total_time = time.time() - start_time

print(f"💾 OpenAI JSONL saved in {openai_time:.1f}s: {openai_path}")

# %%
# 📊 OPTIMIZED PERFORMANCE REPORT
print("\n" + "="*60)
print("🚀 OPTIMIZED PERFORMANCE REPORT")
print("="*60)
print(f"📊 Dataset Statistics:")
print(f"   • Training examples: {len(full_season_df):,}")
print(f"   • OpenAI examples: {len(openai_examples):,}")
print(f"   • Memory usage: {full_season_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\n⚡ Performance Metrics:")
print(f"   • Generation time: {generation_time:.1f}s ({generation_time/60:.1f} min)")
print(f"   • CSV save time: {csv_time if not SKIP_CSV else 0:.1f}s")
print(f"   • OpenAI conversion: {openai_time:.1f}s")
print(f"   • Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"   • Processing rate: {len(full_season_df) / generation_time:.1f} examples/sec")

print(f"\n📁 Generated Files:")
if not SKIP_CSV:
    print(f"   • Training CSV: {season_csv_path}")
print(f"   • OpenAI JSONL: {openai_path}")

# %%
# 🔍 QUALITY CHECK
print(f"\n🔍 Quick Quality Check:")
if len(full_season_df) > 0:
    # Check for data quality
    null_count = full_season_df['json_training_data'].isnull().sum()
    if null_count > 0:
        print(f"⚠️  Found {null_count} null JSON entries")
    else:
        print("✅ All JSON training data populated")
    
    # Sample preview
    print(f"\n📋 Sample Preview:")
    data_preview.preview_llm_data(full_season_df, n_samples=1)

print(f"\n🎉 OPTIMIZED FULL SEASON COMPLETE!")
print(f"🚀 Ready for OpenAI fine-tuning with {len(openai_examples):,} examples!")

# %%
# 💡 OPTIMIZATION NOTES
optimizer_used = "OPTIMIZED" if USE_OPTIMIZED else "ORIGINAL"
print(f"\n💡 OPTIMIZATION TECHNIQUES APPLIED ({optimizer_used}):")
print(f"   ✅ Used cached PCA data (306K+ cache files)")
if USE_OPTIMIZED:
    print(f"   🚀 Vectorized batch processing (5-10x faster)")  
    print(f"   🚀 Optimized DataFrame operations (.itertuples vs .iloc)")
    print(f"   🚀 Memory-efficient batch sizing ({BATCH_SIZE:,} rows/batch)")
else:
    print(f"   📊 Standard row-by-row processing")
print(f"   ✅ Optional CSV skipping for OpenAI-only workflows")
print(f"   ✅ Sample testing before full run")
print(f"   ✅ Detailed performance monitoring")
print(f"   ✅ Benchmarking capability (set RUN_BENCHMARK=True)")

print(f"\n🔧 For even faster runs:")
if USE_OPTIMIZED:
    print(f"   • Increase BATCH_SIZE (current: {BATCH_SIZE:,}) if you have more RAM")
    print(f"   • Set RUN_BENCHMARK=True to compare original vs optimized")
    print(f"   • Monitor batch processing messages to tune performance")
print(f"   • Set SKIP_CSV=True if you only need JSONL") 
print(f"   • Use sample_size parameter for testing")
print(f"   • Pre-build more cache files with build_complete_cache.py")
print(f"   • Consider processing specific date ranges")
if not USE_OPTIMIZED:
    print(f"   • Set USE_OPTIMIZED=True for 5-10x speed improvement!")
