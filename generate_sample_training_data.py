"""
Generate sample training data with 10-value format for analysis
"""

import json
import sys
from generate_training_data import load_play_by_play_data

# Suppress emoji print statements
class NoEmojiWriter:
    def write(self, text):
        try:
            sys.__stdout__.write(text)
        except UnicodeEncodeError:
            sys.__stdout__.write(text.encode('ascii', 'ignore').decode('ascii'))
    def flush(self):
        sys.__stdout__.flush()

sys.stdout = NoEmojiWriter()

from generate_training_data_OPTIMIZED import create_llm_training_data_ULTRA_FAST as create_llm_training_data

print("Generating sample training data (100 examples)")
print("=" * 60)

# Load data
season_df = load_play_by_play_data("2023-2024")
print(f"Loaded {len(season_df):,} total plays")

# Limit to first 3 games for sample
unique_games = sorted(season_df['game_id'].unique())[:3]
test_df = season_df[season_df['game_id'].isin(unique_games)]
print(f"Using {len(test_df):,} plays from {len(unique_games)} games")

# Generate training data
print("\nGenerating training data...")
training_df = create_llm_training_data(
    test_df,
    n_total=5,
    filter_nan=True,
    generation_mode='remaining_plays',
    use_direct_compact=True,
    use_batch_pca=False
)

print(f"Generated {len(training_df):,} training examples")

# Save to file for analysis
output_file = "sample_training_data.jsonl"
with open(output_file, 'w') as f:
    for idx, row in training_df.iterrows():
        f.write(row['json_training_data'] + '\n')

print(f"\nSaved to {output_file}")
print("Ready for analysis")

