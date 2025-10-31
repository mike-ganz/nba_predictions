"""
Normalize all JSONL datasets with league-relative features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from league_normalizer import normalize_game_jsonl

print("=" * 80)
print("NORMALIZING ALL DATASETS")
print("=" * 80)

datasets = [
    ('data/games_train_with_players_90.jsonl', 'data/games_train_with_players_90_norm.jsonl'),
    ('data/games_val_with_players.jsonl', 'data/games_val_with_players_norm.jsonl'),
    ('data/games_predict_2024_2025_with_players.jsonl', 'data/games_predict_2024_2025_with_players_norm.jsonl'),
    ('data/games_2025_2026_current.jsonl', 'data/games_2025_2026_current_norm.jsonl'),  # Current season
]

for input_path, output_path in datasets:
    if not Path(input_path).exists():
        print(f"\n[SKIP] {input_path} (not found)")
        continue
    
    print(f"\n[Processing] {input_path}")
    normalize_game_jsonl(input_path, output_path)
    print(f"[Done] Wrote to {output_path}")

print("\n" + "=" * 80)
print("NORMALIZATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Update training/margin_dataset.py to use *_norm features")
print("2. Retrain model: python train_margin.py ...")
print("3. Evaluate on normalized test set")

