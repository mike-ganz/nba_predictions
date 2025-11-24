#!/usr/bin/env python
"""Quick check of training data features"""
import json

print("="*80)
print("TRAINING DATA FEATURE CHECK")
print("="*80)
print()

files_to_check = [
    ('data/games_train_with_players_90_norm.jsonl', '2021-2024 (existing)'),
    ('data/games_train_2021_2025_combined_norm.jsonl', '2021-2025 (existing)'),
]

for filepath, description in files_to_check:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
            game = json.loads(line)
            
            teams_h = game.get('teams', {}).get('H', {})
            teams_a = game.get('teams', {}).get('A', {})
            
            print(f"✓ {description}")
            print(f"  File: {filepath}")
            print(f"  Season: {game.get('season', 'N/A')}")
            print()
            
            # Check for key features
            features_to_check = [
                ('off_rating', 'Offensive Rating'),
                ('off_rating_norm', 'Offensive Rating (normalized)'),
                ('def_rating_norm', 'Defensive Rating (normalized)'),
                ('pace_norm', 'Pace (normalized)'),
                ('three_pt_rate_norm', '3P Rate (normalized)'),
                ('free_throw_rate', 'Free Throw Rate (raw)'),
                ('free_throw_rate_norm', 'Free Throw Rate (normalized)'),
                ('off_reb_rate_norm', 'Offensive Rebound Rate (normalized)'),
                ('def_reb_rate_norm', 'Defensive Rebound Rate (normalized)'),
                ('turnover_rate_norm', 'Turnover Rate (normalized)'),
                ('assist_rate_norm', 'Assist Rate (normalized)'),
                ('rest_days', 'Rest Days'),
            ]
            
            print("  Features available:")
            for feat_key, feat_name in features_to_check:
                has_feat = feat_key in teams_h
                status = "✓" if has_feat else "✗"
                value = teams_h.get(feat_key, 'N/A')
                if isinstance(value, float):
                    print(f"    {status} {feat_name}: {value:.3f}")
                else:
                    print(f"    {status} {feat_name}: {value}")
            
            print()
            print("  FTR Status:")
            has_ftr_raw = 'free_throw_rate' in teams_h
            has_ftr_norm = 'free_throw_rate_norm' in teams_h
            
            if has_ftr_raw and has_ftr_norm:
                print(f"    ✓ FTR available for experiments 3 & 4")
            else:
                print(f"    ✗ FTR MISSING - experiments 3 & 4 will fail!")
            
            print()
            print("-"*80)
            print()
            
    except FileNotFoundError:
        print(f"✗ {description}")
        print(f"  File not found: {filepath}")
        print()
    except Exception as e:
        print(f"✗ {description}")
        print(f"  Error: {e}")
        print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print("Can we use existing files?")
print("  • Experiment 1 (14 feat, 21-24): Use games_train_with_players_90_norm.jsonl")
print("  • Experiment 2 (14 feat, 21-25): Use games_train_2021_2025_combined_norm.jsonl")
print("  • Experiment 3 (16 feat+FTR, 21-24): Check FTR above ☝️")
print("  • Experiment 4 (16 feat+FTR, 21-25): Check FTR above ☝️")
print()

