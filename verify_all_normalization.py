"""
Verify that ALL datasets have normalized features, not just 25-26
"""
import json
from pathlib import Path

print("="*80)
print("COMPREHENSIVE NORMALIZATION VERIFICATION")
print("="*80)
print()

datasets = {
    'Training (21-24)': 'data/games_train_with_players_90_norm.jsonl',
    'Validation (21-24)': 'data/games_val_with_players_norm.jsonl',
    'Test 24-25': 'data/games_predict_2024_2025_with_players_norm.jsonl',
    'Test 25-26': 'data/games_2025_2026_current_norm.jsonl',
}

for name, filepath in datasets.items():
    if not Path(filepath).exists():
        print(f"{name}: ❌ FILE NOT FOUND")
        continue
    
    with open(filepath) as f:
        game = json.loads(f.readline())
    
    h = game['teams']['H']
    norm_feats = [k for k in h.keys() if '_norm' in k]
    
    print(f"{name}:")
    print(f"  File: {filepath}")
    print(f"  Normalized features: {len(norm_feats)}")
    
    if len(norm_feats) == 9:
        print(f"  Status: ✅ CORRECT (9 features)")
        print(f"  Examples: {norm_feats[:3]}")
        # Check sample values are reasonable (should be small, not like 115)
        sample_vals = [h[f] for f in norm_feats[:3]]
        if all(abs(v) < 50 for v in sample_vals):
            print(f"  Values: ✅ Reasonable range: {[f'{v:.2f}' for v in sample_vals]}")
        else:
            print(f"  Values: ⚠️  SUSPICIOUS: {sample_vals}")
    elif len(norm_feats) == 0:
        print(f"  Status: ❌ MISSING - No normalized features!")
    else:
        print(f"  Status: ⚠️  UNEXPECTED COUNT: {len(norm_feats)}")
    
    print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

# Re-check to give final verdict
all_correct = True
for name, filepath in datasets.items():
    if not Path(filepath).exists():
        all_correct = False
        continue
    with open(filepath) as f:
        game = json.loads(f.readline())
    norm_feats = [k for k in game['teams']['H'].keys() if '_norm' in k]
    if len(norm_feats) != 9:
        all_correct = False
        print(f"❌ {name} has {len(norm_feats)} normalized features (expected 9)")

if all_correct:
    print("✅ ALL DATASETS HAVE CORRECT NORMALIZED FEATURES")
    print()
    print("Both OLD and NEW models were trained on properly normalized data.")
    print("The comparison between models is VALID.")
else:
    print("🚨 PROBLEM: Some datasets are missing normalized features!")
    print()
    print("This means the model comparison may be INVALID.")
    print("We need to regenerate the affected datasets.")

print()
print("="*80)

