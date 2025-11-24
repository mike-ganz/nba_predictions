"""
Run Experiment 5: Robust Z-Score Normalization.

This script:
1. Generates Z-Score normalized training and test data.
2. Trains the XGBoost model on the Z-Score training data.
3. Predicts on the Z-Score test data.
4. Evaluates ATS performance and compares it to the Champion.
"""
import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import our internal modules
from league_normalizer import normalize_game_jsonl
from train_margin import main as train_main
# We'll invoke prediction script via subprocess or modify it to be importable
# For now, we'll just replicate the prediction logic here or use subprocess

def setup_directories():
    Path("data/experiments/zscore").mkdir(parents=True, exist_ok=True)
    Path("artifacts/experiments/exp5_zscore").mkdir(parents=True, exist_ok=True)
    Path("predictions/experiments").mkdir(parents=True, exist_ok=True)

def step_1_prepare_data():
    print("\n" + "="*80)
    print("STEP 1: Generating Z-Score Normalized Data")
    print("="*80)
    
    # 1. Training Data (2021-2024)
    raw_train = "data/games_train_with_players_90.jsonl"
    zscore_train = "data/experiments/zscore/games_train_zscore.jsonl"
    
    print(f"Normalizing Training Data (Z-Score): {raw_train} -> {zscore_train}")
    normalize_game_jsonl(raw_train, zscore_train, method='zscore')
    
    # 2. Test Data (2025-2026 Current)
    raw_test = "data/games_2025_2026_current.jsonl"
    zscore_test = "data/experiments/zscore/games_test_zscore.jsonl"
    
    print(f"Normalizing Test Data (Z-Score): {raw_test} -> {zscore_test}")
    normalize_game_jsonl(raw_test, zscore_test, method='zscore')

def step_2_train_model():
    print("\n" + "="*80)
    print("STEP 2: Training Robust Model (Exp 5)")
    print("="*80)
    
    # Create config for Z-Score experiment
    base_config_path = "configs/experiment_2124_no_ftr.yaml"
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    
    # Update description
    config['description'] = "Experiment 5: Z-Score Robust Normalization (2021-2024)"
    
    # Save temporary config
    exp_config_path = "configs/experiment_zscore.yaml"
    with open(exp_config_path, 'w') as f:
        yaml.safe_dump(config, f)
        
    # Run training via subprocess to ensure clean state
    import subprocess
    cmd = [
        sys.executable, "train_margin.py",
        "--config", exp_config_path,
        "--data", "data/experiments/zscore/games_train_zscore.jsonl",
        "--output", "artifacts/experiments/exp5_zscore",
        "--model-type", "xgboost"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def step_3_generate_predictions():
    print("\n" + "="*80)
    print("STEP 3: Predicting on 2025-26 Season")
    print("="*80)
    
    # We need to use the prediction script logic, but adapted for our Z-Score data
    # Since the predict_margin.py script might expect specific paths, we'll use the generic one
    # provided we pass the correct model and data.
    
    # However, predict_margin.py usually takes a model path and a data path.
    # Let's invoke it via subprocess.
    
    import subprocess
    cmd = [
        sys.executable, "predict_margin.py",
        "--model", "artifacts/experiments/exp5_zscore",
        "--data", "data/experiments/zscore/games_test_zscore.jsonl",
        "--output", "predictions/experiments/exp5_predictions.csv"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def step_4_evaluate():
    print("\n" + "="*80)
    print("STEP 4: Evaluating Performance")
    print("="*80)
    
    pred_file = "predictions/experiments/exp5_predictions.csv"
    df = pd.read_csv(pred_file)
    
    # Calculate metrics
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['ats_correct'] = (df['actual_home_covers'] == df['pred_home_covers']).astype(int)
    
    games_count = len(df)
    ats_correct = df['ats_correct'].sum()
    ats_pct = (ats_correct / games_count) * 100
    mae = np.abs(df['pred_margin_mu'] - df['actual_margin']).mean()
    
    print(f"Results for Experiment 5 (Z-Score Robust):")
    print(f"  Games: {games_count}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  ATS:   {ats_pct:.2f}% ({ats_correct}/{games_count})")
    
    # Load Champion for comparison
    champ_file = "predictions/current_season_champion_2025_2026_predictions.csv"
    if os.path.exists(champ_file):
        df_champ = pd.read_csv(champ_file)
        
        # Align datasets (intersection of games)
        # Assuming sorted by date/id, but let's merge on game_id to be safe if it exists, or date/teams
        # The files should be identical in length and order for 2025-26 usually
        
        champ_mae = np.abs(df_champ['pred_margin_mu'] - df_champ['actual_margin']).mean()
        
        # Recalculate champ ATS just to be sure
        df_champ['actual_home_covers'] = (df_champ['actual_margin'] > -df_champ['market_spread_home']).astype(int)
        df_champ['pred_home_covers'] = (df_champ['pred_margin_mu'] > -df_champ['market_spread_home']).astype(int)
        champ_ats_correct = (df_champ['actual_home_covers'] == df_champ['pred_home_covers']).sum()
        champ_ats_pct = (champ_ats_correct / len(df_champ)) * 100
        
        print(f"\nComparison vs Champion:")
        print(f"  Champion MAE: {champ_mae:.2f}")
        print(f"  Champion ATS: {champ_ats_pct:.2f}%")
        
        diff_ats = ats_pct - champ_ats_pct
        print(f"\n  Difference: {diff_ats:+.2f}% ATS")
        
        if diff_ats > 0:
            print("  ✅ Z-Score Normalization IMPROVED performance!")
        elif diff_ats > -1.0:
            print("  UNKNOWN: Performance is similar.")
        else:
            print("  ❌ Z-Score Normalization HURT performance.")

def main():
    setup_directories()
    step_1_prepare_data()
    step_2_train_model()
    step_3_generate_predictions()
    step_4_evaluate()

if __name__ == "__main__":
    main()

