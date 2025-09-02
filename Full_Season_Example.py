# %%
"""
NBA Training Data - Full Season Generation Example
=============================================

Generate training data for an entire season using the optimized system.
"""

# %%
from config import set_season_year
from config.settings import DEFAULT_N_TOTAL_PLAYS
from training import training_data_generator, openai_formatter  
from data import file_manager, data_preview
import time

# Set your season
set_season_year("2023-2024")
print("✅ Setup complete!")

# %%
# 🏀 FULL SEASON GENERATION
print("🏀 Generating training data for ENTIRE 2023-2024 season...")
print("⚠️  Warning: This will process ALL games in the season and may take significant time!")

start_time = time.time()

# Generate for entire season (no game_id_filter = all games)
full_season_df = training_data_generator.generate_dataset(
    season_year="2023-2024",
    n_total=DEFAULT_N_TOTAL_PLAYS,  # Number of plays in each sequence
    sample_size=None,  # None = process ALL games/rows in the season!
    # game_id_filter=None  # This is the default - processes ALL games
    force_real_pca=True  # 🔥 FORCE real cached PCA values (not dummy values!)
)

end_time = time.time() 
elapsed = end_time - start_time

print(f"✅ Generated {len(full_season_df)} training records from entire season")
print(f"⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")

# %%
# 💾 SAVE THE DATASET
print("\n💾 Saving full season dataset...")

# Save as CSV
season_csv_path = file_manager.save_llm_dataset(
    full_season_df, 
    "full_season_2023_2024_training.csv"
)
print(f"📊 CSV saved to: {season_csv_path}")

# Convert to OpenAI format and save
print("🤖 Converting to OpenAI format...")
openai_examples = openai_formatter.create_training_data(full_season_df)
openai_path = openai_formatter.save_training_data(
    openai_examples,
    filename="full_season_2023_2024_openai.jsonl"
)
print(f"💾 OpenAI JSONL saved to: {openai_path}")

# %%
# 📊 DATASET STATISTICS  
print("\n📊 Full Season Dataset Statistics:")
print(f"   • Total training examples: {len(full_season_df):,}")
print(f"   • OpenAI examples: {len(openai_examples):,}")
print(f"   • Processing time: {elapsed/60:.1f} minutes")

# Preview some examples
if len(full_season_df) > 0:
    print("\n🔍 Sample data preview:")
    data_preview.preview_llm_data(full_season_df, n_samples=2)

print(f"\n🎉 FULL SEASON COMPLETE!")
print(f"📁 Files generated:")
print(f"   • Training CSV: {season_csv_path}")  
print(f"   • OpenAI JSONL: {openai_path}")
print(f"🚀 Ready for fine-tuning with {len(openai_examples):,} examples!")

# %%
# 🔧 ALTERNATIVE: Generate without sampling (for production)
print("\n" + "="*60)
print("🔧 PRODUCTION VERSION (No sampling - generates ALL data)")
print("   Uncomment the code below to generate the complete dataset:")
print("="*60)

# Uncomment this section for full production dataset:
"""
production_df = training_data_generator.generate_dataset(
    season_year="2023-2024", 
    n_total=5,
    # sample_size=None,  # No sampling - get everything!
)

production_csv = file_manager.save_llm_dataset(
    production_df,
    "PRODUCTION_full_season_2023_2024.csv"  
)

production_openai = openai_formatter.create_training_data(production_df)
production_jsonl = openai_formatter.save_training_data(
    production_openai,
    filename="PRODUCTION_full_season_2023_2024.jsonl"
)

print(f"🏭 PRODUCTION dataset: {len(production_df):,} examples")
print(f"📁 Saved to: {production_csv}")
print(f"📁 OpenAI: {production_jsonl}")
"""
