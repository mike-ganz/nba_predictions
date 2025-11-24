"""
Run Experiment 5: Robust Z-Score Normalization - EXTENDED.

This script runs 4 variations of Z-Score normalization:
  5a. Baseline Z (14 feats, 2021-24) [ALREADY RAN]
  5b. Z + Recent (14 feats, 2021-25)
  5c. Z + FTR (16 feats, 2021-24)
  5d. Z + Full (16 feats, 2021-25)

It generates all necessary data, trains models, predicts, and produces a comparison table.
"""
import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import subprocess

# Import our internal modules
from league_normalizer import normalize_game_jsonl

def setup_directories():
    Path("data/experiments/zscore").mkdir(parents=True, exist_ok=True)
    Path("artifacts/experiments/exp5b_zscore_recent").mkdir(parents=True, exist_ok=True)
    Path("artifacts/experiments/exp5c_zscore_ftr").mkdir(parents=True, exist_ok=True)
    Path("artifacts/experiments/exp5d_zscore_full").mkdir(parents=True, exist_ok=True)
    Path("predictions/experiments").mkdir(parents=True, exist_ok=True)

def run_step(desc: str):
    print("\n" + "="*80)
    print(f"STEP: {desc}")
    print("="*80)

def normalize_if_missing(input_path: str, output_path: str, method: str = 'zscore'):
    if not os.path.exists(output_path):
        print(f"Normalizing {input_path} -> {output_path}")
        normalize_game_jsonl(input_path, output_path, method=method)
    else:
        print(f"Skipping normalization (file exists): {output_path}")

def prepare_data():
    run_step("Generating Z-Score Normalized Data")
    
    # 1. Training Data (2021-2024)
    normalize_if_missing(
        "data/games_train_with_players_90.jsonl",
        "data/experiments/zscore/games_train_2124_zscore.jsonl"
    )
    
    # 2. Training Data (2021-2025 Combined)
    normalize_if_missing(
        "data/games_train_2021_2025_combined.jsonl",
        "data/experiments/zscore/games_train_2125_zscore.jsonl"
    )
    
    # 3. Test Data (2025-2026 Current)
    normalize_if_missing(
        "data/games_2025_2026_current.jsonl",
        "data/experiments/zscore/games_test_zscore.jsonl"
    )

def train_model(exp_id: str, data_path: str, use_ftr: bool):
    run_step(f"Training {exp_id}")
    
    # Load base config
    if use_ftr:
        base_config_path = "configs/experiment_2124_with_ftr.yaml"
    else:
        base_config_path = "configs/experiment_2124_no_ftr.yaml"
        
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    
    config['description'] = f"Experiment {exp_id}: Z-Score Robust Normalization"
    
    # Save temporary config
    exp_config_path = f"configs/experiment_{exp_id}.yaml"
    with open(exp_config_path, 'w') as f:
        yaml.safe_dump(config, f)
        
    output_dir = f"artifacts/experiments/{exp_id}"
    
    cmd = [
        sys.executable, "train_margin.py",
        "--config", exp_config_path,
        "--data", data_path,
        "--output", output_dir,
        "--model-type", "xgboost"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def predict(exp_id: str):
    run_step(f"Predicting {exp_id}")
    
    model_dir = f"artifacts/experiments/{exp_id}"
    output_path = f"predictions/experiments/{exp_id}_predictions.csv"
    data_path = "data/experiments/zscore/games_test_zscore.jsonl"
    
    cmd = [
        sys.executable, "predict_margin.py",
        "--model", model_dir,
        "--data", data_path,
        "--output", output_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def evaluate_all():
    run_step("Final Evaluation & Comparison")
    
    experiments = [
        {'id': 'exp5_zscore', 'name': '5a. Z-Score Baseline (21-24)'},
        {'id': 'exp5b_zscore_recent', 'name': '5b. Z-Score + Recent (21-25)'},
        {'id': 'exp5c_zscore_ftr', 'name': '5c. Z-Score + FTR (21-24)'},
        {'id': 'exp5d_zscore_full', 'name': '5d. Z-Score + Full (21-25)'}
    ]
    
    results = []
    
    # Load Champion
    champ_file = "predictions/current_season_champion_2025_2026_predictions.csv"
    if os.path.exists(champ_file):
        df_champ = pd.read_csv(champ_file)
        df_champ['actual_home_covers'] = (df_champ['actual_margin'] > -df_champ['market_spread_home']).astype(int)
        df_champ['pred_home_covers'] = (df_champ['pred_margin_mu'] > -df_champ['market_spread_home']).astype(int)
        champ_ats = (df_champ['actual_home_covers'] == df_champ['pred_home_covers']).mean() * 100
        champ_mae = (df_champ['pred_margin_mu'] - df_champ['actual_margin']).abs().mean()
        
        results.append({
            'Experiment': 'Champion (Current)',
            'Data': '2021-24',
            'Features': '14 (No FTR)',
            'Norm': 'Center',
            'MAE': f"{champ_mae:.2f}",
            'ATS': f"{champ_ats:.2f}%",
            'Correct': f"{(df_champ['actual_home_covers'] == df_champ['pred_home_covers']).sum()}/{len(df_champ)}"
        })
    
    for exp in experiments:
        pred_file = f"predictions/experiments/{exp['id']}_predictions.csv"
        if not os.path.exists(pred_file):
            print(f"Skipping {exp['id']} (no predictions found)")
            continue
            
        df = pd.read_csv(pred_file)
        
        # Ensure actuals exist
        if 'actual_margin' not in df.columns or df['actual_margin'].isna().all():
             print(f"Skipping {exp['id']} (no actuals)")
             continue

        df = df.dropna(subset=['actual_margin'])
        
        df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
        df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
        
        ats = (df['actual_home_covers'] == df['pred_home_covers']).mean() * 100
        mae = (df['pred_margin_mu'] - df['actual_margin']).abs().mean()
        
        # Infer metadata from ID
        data_range = '2021-25' if 'recent' in exp['id'] or 'full' in exp['id'] else '2021-24'
        features = '16 (+FTR)' if 'ftr' in exp['id'] or 'full' in exp['id'] else '14 (No FTR)'
        
        results.append({
            'Experiment': exp['name'],
            'Data': data_range,
            'Features': features,
            'Norm': 'Z-Score',
            'MAE': f"{mae:.2f}",
            'ATS': f"{ats:.2f}%",
            'Correct': f"{(df['actual_home_covers'] == df['pred_home_covers']).sum()}/{len(df)}"
        })
        
    # Print Table
    df_res = pd.DataFrame(results)
    print("\nRESULTS SUMMARY:")
    print(df_res.to_string(index=False))
    
    df_res.to_csv("predictions/experiments/zscore_comparison_results.csv", index=False)
    print("\nSaved to predictions/experiments/zscore_comparison_results.csv")

def main():
    setup_directories()
    prepare_data()
    
    # We already ran 5a (Baseline Z) manually, but let's ensure the artifacts are there or re-run if needed
    # Assuming artifacts/experiments/exp5_zscore exists from previous run.
    
    # Run 5b: Z + Recent
    train_model('exp5b_zscore_recent', "data/experiments/zscore/games_train_2125_zscore.jsonl", use_ftr=False)
    predict('exp5b_zscore_recent')
    
    # Run 5c: Z + FTR
    train_model('exp5c_zscore_ftr', "data/experiments/zscore/games_train_2124_zscore.jsonl", use_ftr=True)
    predict('exp5c_zscore_ftr')
    
    # Run 5d: Z + Full
    train_model('exp5d_zscore_full', "data/experiments/zscore/games_train_2125_zscore.jsonl", use_ftr=True)
    predict('exp5d_zscore_full')
    
    evaluate_all()

if __name__ == "__main__":
    main()

