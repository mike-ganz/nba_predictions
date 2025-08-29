# %%
"""
NBA Training Data - Quick Start Example  
=======================================

Fast and complete training data generation using optimized date filtering.
This is the main way to generate NBA training data efficiently.
"""

# %%
# 🚀 QUICK SETUP
from config import set_season_year
from training import training_data_generator, openai_formatter
from data import file_manager, data_preview

import time
start_time = time.time()

# Set your season
set_season_year("2023-2024")
print("✅ Setup complete!")

# %%
# 🏃‍♂️ SUPER FAST: Single Game Test (Now with date filtering!)
print("Testing with a single game using OPTIMIZED approach...")

# Generate for single game using the optimized system
single_game_df = training_data_generator.generate_for_game(
    game_id=22300221,  # Use game that exists
    n_total=5,
    max_plays=450  # Reasonable number for testing
)

end_time = time.time()
elapsed = end_time - start_time

print(f"✅ Generated {len(single_game_df)} records from 1 game")
print(f"⏱️  Total time: {elapsed:.2f} seconds")

# Preview it
if len(single_game_df) > 0:
    data_preview.preview_llm_data(single_game_df, n_samples=1)

# Save sample
sample_path = file_manager.save_llm_dataset(single_game_df, "optimized_single_game_test.csv")
print(f"💾 Saved test sample to: {sample_path}")

# Also convert single game to OpenAI format for quick testing
single_game_openai = openai_formatter.create_training_data(single_game_df)
single_game_openai_path = openai_formatter.save_training_data(
    single_game_openai,
    filename="single_game_openai_test.jsonl"
)
print(f"💾 Single game JSONL saved to: {single_game_openai_path}")

# %%
# 🎯 5 Games Test (Optimized)
print("\nTesting with 5 games using OPTIMIZED approach...")
start_time_5 = time.time()

# Use specific games for consistent testing
test_games = [22300154, 22300155, 22300156, 22300157, 22300158]

five_game_df = training_data_generator.generate_dataset(
    n_total=5,
    game_id_filter=test_games  # Only these specific games
)

end_time_5 = time.time()
elapsed_5 = end_time_5 - start_time_5

print(f"✅ Generated {len(five_game_df)} records from 5 games")
print(f"⏱️  5-game time: {elapsed_5:.2f} seconds")

# Save it
five_game_path = file_manager.save_llm_dataset(five_game_df, "optimized_five_game_test.csv")
print(f"💾 Saved to: {five_game_path}")

# %%
# 🤖 CONVERT TO OPENAI JSONL FORMAT (This is what you need for fine-tuning!)
print("\n🤖 Converting to OpenAI fine-tuning format...")

# Convert the 5-game dataset to OpenAI format
openai_examples = openai_formatter.create_training_data(five_game_df)
print(f"✅ Converted {len(openai_examples)} examples to OpenAI format")

# Save as JSONL file for OpenAI fine-tuning
openai_path = openai_formatter.save_training_data(
    openai_examples,
    filename="nba_openai_training.jsonl"
)
print(f"💾 OpenAI JSONL saved to: {openai_path}")

# Preview the OpenAI format
if len(openai_examples) > 0:
    print("\n📋 Sample OpenAI Training Example:")
    sample = openai_examples[0]
    print("USER MESSAGE (Context):")
    user_content = sample['messages'][0]['content']
    print(f"  Length: {len(user_content)} characters")
    print(f"  Preview: {user_content[:200]}...")
    
    print("\nASSISTANT MESSAGE (Next Play):")
    assistant_content = sample['messages'][1]['content']
    print(f"  Content: {assistant_content}")

# %%
print("\n🎉 OPTIMIZATION COMPLETE!")
print("=" * 50)
print("📊 Performance Results:")
print(f"   • Single game: {elapsed:.2f} seconds")
print(f"   • Five games: {elapsed_5:.2f} seconds")
print(f"   • OpenAI examples: {len(openai_examples)}")
print(f"\n📁 Generated Files:")
print(f"   • CSV: {five_game_path}")
print(f"   • JSONL: {openai_path}")
print("\n💡 The system now uses date filtering to only load relevant player data!")
print("   • No more loading 84,005 player records")
print("   • Only loads data up to the game date")
print("   • Much faster and prevents data leakage")
print(f"\n🚀 Ready for OpenAI Fine-tuning!")
print(f"   • Use the {openai_path.split('/')[-1]} file for fine-tuning")
print(f"   • Each example has user context + assistant next-play prediction")
