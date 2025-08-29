# %%
"""
NBA Training Data Generator - Complete Notebook Script
=====================================================

This notebook demonstrates how to use the modular NBA training data system.
Run each cell sequentially to generate and manage your training datasets.
"""

# %% [markdown]
"""
## 🚀 Setup and Imports
Import all the modules we'll need and set up the system.
"""

# %%
# Core imports for the modular system
from config import set_season_year, get_current_season_year, config
from training import training_data_generator, openai_formatter
from data import data_loader, file_manager, data_preview
from analysis import player_analyzer

# Standard libraries for data science
import pandas as pd
import json
from typing import Optional, List

print("✅ All imports successful!")

# %% [markdown]
"""
## ⚙️ Configuration Setup
Set your preferred season and check system status.
"""

# %%
# Configure your settings
SEASON_YEAR = "2023-2024"  # Change this to your preferred season
set_season_year(SEASON_YEAR)

# Display current configuration
print("📋 Current Configuration:")
print(f"  Season Year: {get_current_season_year()}")
print(f"  Data Directory: {config.data_dir}")
print(f"  Training Directory: {config.training_dir}")

# Check available seasons
available_seasons = data_loader.get_available_seasons()
print(f"  Available Seasons: {', '.join(available_seasons)}")

# %% [markdown]
"""
## 🧪 System Test
Verify that everything is working correctly.
"""

# %%
# Test data loading
print("🧪 Testing Data Loading...")
try:
    from data import test_data_loading
    test_data_loading()
    print("\n✅ System test completed successfully!")
except Exception as e:
    print(f"❌ System test failed: {e}")
    print("Please check your data files and configuration.")

# %% [markdown]
"""
## 🎯 Quick Single Game Test
Test with a single game to make sure everything works before scaling up.
"""

# %%
# Load play-by-play data to get a sample game ID
print("🏀 Loading sample game for testing...")
df_sample = data_loader.load_play_by_play_data(SEASON_YEAR)
sample_game_id = df_sample['game_id'].iloc[0]
sample_game_date = df_sample[df_sample['game_id'] == sample_game_id].iloc[0]['date']

print(f"Testing with:")
print(f"  Game ID: {sample_game_id}")
print(f"  Game Date: {sample_game_date}")
print(f"  Total plays in game: {len(df_sample[df_sample['game_id'] == sample_game_id])}")

# Generate training data for this single game (small sample for speed)
test_result = training_data_generator.generate_for_game(
    sample_game_id, 
    season_year=SEASON_YEAR,
    n_total=3,      # Reduced for speed
    max_plays=10    # Only first 10 plays for testing
)

print(f"\n✅ Generated {len(test_result)} training records for test game")
if len(test_result) > 0:
    print("Sample columns:", list(test_result.columns))

# %% [markdown]
"""
## 📊 Generate Training Dataset
Create your main training dataset with customizable parameters.
"""

# %%
# Configuration for dataset generation
DATASET_CONFIG = {
    'season_year': SEASON_YEAR,
    'n_total': 5,           # Number of recent plays per sequence
    'sample_size': 1000,    # Start with 1000 samples (adjust as needed)
    'game_id_filter': None  # None = all games, or specify list like [22200001, 22200002]
}

print("🏗️ Generating Training Dataset...")
print(f"Configuration: {DATASET_CONFIG}")

# Generate the dataset
training_df = training_data_generator.generate_dataset(
    season_year=DATASET_CONFIG['season_year'],
    n_total=DATASET_CONFIG['n_total'], 
    sample_size=DATASET_CONFIG['sample_size'],
    game_id_filter=DATASET_CONFIG['game_id_filter']
)

print(f"\n✅ Generated dataset with {len(training_df)} records")
print(f"Memory usage: ~{training_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# %% [markdown]
"""
## 🔍 Analyze and Preview Dataset
Examine the generated data to understand its structure and quality.
"""

# %%
# Get dataset statistics
stats = data_preview.analyze_dataset_stats(training_df)

print("📊 Dataset Analysis:")
print(f"  Total Records: {stats['total_rows']:,}")
print(f"  Unique Games: {stats['total_games']:,}")
print(f"  Memory Usage: {stats['memory_usage_mb']} MB")
print(f"  Columns: {len(stats['columns'])}")

if stats['date_range']:
    print(f"  Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")

# Preview sample training data
print("\n" + "="*60)
print("📋 SAMPLE TRAINING DATA")
print("="*60)
data_preview.preview_llm_data(training_df, n_samples=2)

# %% [markdown]
"""
## 💾 Save Dataset
Save your generated training data to files.
"""

# %%
# Save the main training dataset
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
filename = f"nba_training_{SEASON_YEAR.replace('-', '_')}_{timestamp}.csv"

saved_path = file_manager.save_llm_dataset(training_df, filename)
print(f"💾 Training dataset saved to: {saved_path}")

# Also save a small sample for quick testing
sample_df = training_df.head(100)
sample_path = file_manager.save_llm_dataset(sample_df, f"sample_{filename}")
print(f"💾 Sample dataset saved to: {sample_path}")

# %% [markdown]
"""
## 🤖 Generate OpenAI Training Format
Convert your data to OpenAI fine-tuning format.
"""

# %%
print("🤖 Converting to OpenAI Training Format...")

# Use a smaller sample for OpenAI format (it can get large quickly)
openai_sample_size = min(500, len(training_df))
openai_sample_df = training_df.head(openai_sample_size)

# Convert to OpenAI format
openai_examples = openai_formatter.create_training_data(openai_sample_df)

print(f"✅ Generated {len(openai_examples)} OpenAI training examples")

# Validate the OpenAI data
validation_results = openai_formatter.validate_training_examples(openai_examples)
print(f"Validation: {validation_results['valid_examples']}/{validation_results['total_examples']} examples valid")
print(f"Success rate: {validation_results['success_rate']:.1%}")

if validation_results['errors']:
    print(f"⚠️  Found {len(validation_results['errors'])} errors (showing first 3):")
    for error in validation_results['errors'][:3]:
        print(f"   - {error}")

# %% [markdown]
"""
## 👀 Preview OpenAI Format
Look at the OpenAI training examples to verify they're correctly formatted.
"""

# %%
print("👀 OpenAI Format Preview:")
print("="*60)
data_preview.preview_openai_training_data(openai_examples, n_samples=2)

# %% [markdown]
"""
## 💾 Save OpenAI Data
Save the OpenAI training data to JSONL format.
"""

# %%
# Save OpenAI training data
openai_filename = f"openai_training_{SEASON_YEAR.replace('-', '_')}_{timestamp}.jsonl"
openai_path = openai_formatter.save_training_data(openai_examples, openai_filename)

print(f"💾 OpenAI training data saved to: {openai_path}")
print(f"📏 File contains {len(openai_examples)} training examples")

# %% [markdown]
"""
## 📁 Dataset Management
View and manage your generated datasets.
"""

# %%
print("📁 Available Datasets:")
print("="*40)

datasets = file_manager.get_available_datasets()

for format_type, files in datasets.items():
    print(f"\n{format_type.upper()} Files:")
    if files:
        for filename in sorted(files)[-5:]:  # Show last 5 files
            try:
                info = file_manager.get_file_info(filename)
                print(f"  📄 {filename}")
                print(f"     Size: {info['size_mb']} MB")
                print(f"     Modified: {info['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print(f"  📄 {filename} (info unavailable)")
    else:
        print("  (none)")

# %% [markdown]
"""
## 🔧 Advanced Usage Examples
More advanced operations and customizations.
"""

# %%
# Example 1: Generate data for specific games only
specific_games = df_sample['game_id'].unique()[:3]  # First 3 games
print(f"🎯 Generating data for specific games: {list(specific_games)}")

specific_df = training_data_generator.generate_dataset(
    season_year=SEASON_YEAR,
    n_total=5,
    game_id_filter=list(specific_games)
)

print(f"✅ Generated {len(specific_df)} records for {len(specific_games)} games")

# Example 2: Different sequence lengths
for n_plays in [3, 5, 7]:
    print(f"\n🔄 Testing with {n_plays} recent plays per sequence...")
    test_df = training_data_generator.generate_dataset(
        season_year=SEASON_YEAR,
        n_total=n_plays,
        sample_size=100  # Small sample for testing
    )
    print(f"   Generated {len(test_df)} records")

# %% [markdown]
"""
## 🧹 Cache Management
Monitor and clear caches for memory management.
"""

# %%
print("🧹 Cache Management:")

# Check cache status
pca_cache_stats = player_analyzer.get_cache_stats()
print(f"PCA Cache: {pca_cache_stats['cached_entries']} entries (~{pca_cache_stats['memory_estimate_kb']} KB)")

# Clear caches if needed
if pca_cache_stats['cached_entries'] > 100:
    print("Clearing caches...")
    cleared_stats = training_data_generator.clear_caches()
    print(f"✅ Cleared caches: {cleared_stats}")

# %% [markdown]
"""
## 🎉 Summary and Next Steps
Review what you've accomplished and suggested next steps.
"""

# %%
print("🎉 NOTEBOOK COMPLETE!")
print("="*50)
print("✅ What you've accomplished:")
print(f"   • Tested system with season {SEASON_YEAR}")
print(f"   • Generated {len(training_df)} training records")
print(f"   • Created {len(openai_examples)} OpenAI examples")
print(f"   • Saved datasets to {config.training_dir}")

print("\n🚀 Suggested next steps:")
print("   1. Adjust sample_size to generate larger datasets")
print("   2. Experiment with different n_total values")
print("   3. Try filtering to specific games of interest")
print("   4. Use the OpenAI data for model fine-tuning")
print("   5. Integrate this into your ML pipeline")

print(f"\n📁 Your files are saved in: {config.training_dir}")

# %%
# Optional: Quick utility functions for continued exploration

def quick_generate(sample_size: int = 500, n_total: int = 5) -> pd.DataFrame:
    """Quick function to generate a dataset with specified parameters."""
    return training_data_generator.generate_dataset(
        season_year=SEASON_YEAR,
        sample_size=sample_size,
        n_total=n_total
    )

def quick_preview(df: pd.DataFrame, n_samples: int = 2):
    """Quick function to preview training data."""
    data_preview.preview_llm_data(df, n_samples=n_samples)

print("🛠️  Utility functions loaded:")
print("   • quick_generate(sample_size=500, n_total=5)")
print("   • quick_preview(df, n_samples=2)")
print("\nExample: df = quick_generate(1000); quick_preview(df)")
