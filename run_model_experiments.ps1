#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run comprehensive model retraining experiments

.DESCRIPTION
    Trains 4 model variants and compares them against current champion on 2025-26 season:
    
    Experiment 1: 14 features (no FTR), 2021-2024 training - BASELINE (current champion replica)
    Experiment 2: 14 features (no FTR), 2021-2025 training - Add recent season
    Experiment 3: 16 features (with FTR), 2021-2024 training - Add FTR features
    Experiment 4: 16 features (with FTR), 2021-2025 training - Full experiment
    
.EXAMPLE
    .\run_model_experiments.ps1
    
.EXAMPLE
    .\run_model_experiments.ps1 -SkipDataPrep
#>

param(
    [switch]$SkipDataPrep,
    [switch]$SkipNormalization,
    [switch]$SkipTraining,
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  MODEL RETRAINING EXPERIMENTS - CHAMPION CHALLENGERS" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Experiments:" -ForegroundColor White
Write-Host "  1. Baseline    - 14 features, 2021-2024 (current champion replica)" -ForegroundColor Gray
Write-Host "  2. Add Season  - 14 features, 2021-2025 (test recent data)" -ForegroundColor Gray
Write-Host "  3. Add FTR     - 16 features, 2021-2024 (test FTR value)" -ForegroundColor Gray
Write-Host "  4. Full        - 16 features, 2021-2025 (all improvements)" -ForegroundColor Gray
Write-Host ""
Write-Host "Started at: $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

# Create experiments directory
$ExperimentsDir = "data/experiments"
New-Item -ItemType Directory -Force -Path $ExperimentsDir | Out-Null

$ArtifactsDir = "artifacts/experiments"
New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null

$PredictionsDir = "predictions/experiments"
New-Item -ItemType Directory -Force -Path $PredictionsDir | Out-Null

# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================

if (-not $SkipDataPrep) {
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host "  PHASE 1: DATA PREPARATION" -ForegroundColor Yellow
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host ""
    
    # Prepare 2021-2024 training data
    Write-Host "[1/2] Generating 2021-2024 training data..." -ForegroundColor Green
    Write-Host "      Seasons: 2021-2022, 2022-2023, 2023-2024" -ForegroundColor Gray
    Write-Host ""
    
    python -B .\prepare_data.py `
        --team-boxscores-dir data/team_boxscores/historical `
        --player-boxscores-dir data/player_boxscores/historical `
        --output data/experiments/games_train_2021_2024.jsonl `
        --seasons 2021-2022 2022-2023 2023-2024 `
        --include-players
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at 2021-2024 data generation" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "2021-2024 data generated successfully" -ForegroundColor Green
    Write-Host ""
    
    # Prepare 2021-2025 training data
    Write-Host "[2/2] Generating 2021-2025 training data..." -ForegroundColor Green
    Write-Host "      Seasons: 2021-2022, 2022-2023, 2023-2024, 2024-2025" -ForegroundColor Gray
    Write-Host ""
    
    python -B .\prepare_data.py `
        --team-boxscores-dir data/team_boxscores/historical `
        --player-boxscores-dir data/player_boxscores/historical `
        --output data/experiments/games_train_2021_2025.jsonl `
        --seasons 2021-2022 2022-2023 2023-2024 2024-2025 `
        --include-players
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at 2021-2025 data generation" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "2021-2025 data generated successfully" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host "Skipping data preparation (SkipDataPrep flag set)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# PHASE 2: NORMALIZATION
# ============================================================================

if (-not $SkipNormalization) {
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host "  PHASE 2: LEAGUE-RELATIVE NORMALIZATION" -ForegroundColor Yellow
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host ""
    
    # Normalize 2021-2024
    Write-Host "[1/2] Normalizing 2021-2024 data..." -ForegroundColor Green
    
    $NormScript1 = @'
from league_normalizer import normalize_game_jsonl
normalize_game_jsonl('data/experiments/games_train_2021_2024.jsonl', 'data/experiments/games_train_2021_2024_norm.jsonl')
'@
    
    python -B -c $NormScript1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at 2021-2024 normalization" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "2021-2024 normalized successfully" -ForegroundColor Green
    Write-Host ""
    
    # Normalize 2021-2025
    Write-Host "[2/2] Normalizing 2021-2025 data..." -ForegroundColor Green
    
    $NormScript2 = @'
from league_normalizer import normalize_game_jsonl
normalize_game_jsonl('data/experiments/games_train_2021_2025.jsonl', 'data/experiments/games_train_2021_2025_norm.jsonl')
'@
    
    python -B -c $NormScript2
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at 2021-2025 normalization" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "2021-2025 normalized successfully" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host "Skipping normalization (SkipNormalization flag set)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# PHASE 3: MODEL TRAINING
# ============================================================================

if (-not $SkipTraining) {
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host "  PHASE 3: MODEL TRAINING (4 Experiments)" -ForegroundColor Yellow
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host ""
    
    $Experiments = @(
        @{
            Name = "Experiment 1: Baseline (14 features, 2021-2024)"
            Config = "configs/experiment_2124_no_ftr.yaml"
            Data = "data/games_train_with_players_90_norm.jsonl"
            Output = "artifacts/experiments/exp1_2124_no_ftr"
            ID = "exp1"
        },
        @{
            Name = "Experiment 2: Add Season (14 features, 2021-2025)"
            Config = "configs/experiment_2125_no_ftr.yaml"
            Data = "data/games_train_2021_2025_combined_norm.jsonl"
            Output = "artifacts/experiments/exp2_2125_no_ftr"
            ID = "exp2"
        },
        @{
            Name = "Experiment 3: Add FTR (16 features, 2021-2024)"
            Config = "configs/experiment_2124_with_ftr.yaml"
            Data = "data/games_train_with_players_90_norm.jsonl"
            Output = "artifacts/experiments/exp3_2124_with_ftr"
            ID = "exp3"
        },
        @{
            Name = "Experiment 4: Full (16 features, 2021-2025)"
            Config = "configs/experiment_2125_with_ftr.yaml"
            Data = "data/games_train_2021_2025_combined_norm.jsonl"
            Output = "artifacts/experiments/exp4_2125_with_ftr"
            ID = "exp4"
        }
    )
    
    foreach ($exp in $Experiments) {
        Write-Host "[$($exp.ID)] $($exp.Name)" -ForegroundColor Green
        Write-Host "      Config: $($exp.Config)" -ForegroundColor Gray
        Write-Host "      Data:   $($exp.Data)" -ForegroundColor Gray
        Write-Host ""
        
        python -B .\train_margin.py `
            --data $exp.Data `
            --config $exp.Config `
            --output $exp.Output `
            --model-type xgboost
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED at $($exp.Name)" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "$($exp.ID) training complete" -ForegroundColor Green
        Write-Host ""
    }
    
} else {
    Write-Host "Skipping training (SkipTraining flag set)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# PHASE 4: GENERATE PREDICTIONS
# ============================================================================

if (-not $SkipEvaluation) {
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host "  PHASE 4: GENERATE PREDICTIONS (2025-26 Season)" -ForegroundColor Yellow
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host ""
    
    $CurrentSeasonData = "data/games_2025_2026_current_norm.jsonl"
    
    Write-Host "Generating predictions for all 4 models on 2025-26 season..." -ForegroundColor White
    Write-Host ""
    
    $Experiments = @(
        @{Name = "Experiment 1"; Model = "artifacts/experiments/exp1_2124_no_ftr"; Output = "predictions/experiments/exp1_predictions.csv"; ID = "exp1"},
        @{Name = "Experiment 2"; Model = "artifacts/experiments/exp2_2125_no_ftr"; Output = "predictions/experiments/exp2_predictions.csv"; ID = "exp2"},
        @{Name = "Experiment 3"; Model = "artifacts/experiments/exp3_2124_with_ftr"; Output = "predictions/experiments/exp3_predictions.csv"; ID = "exp3"},
        @{Name = "Experiment 4"; Model = "artifacts/experiments/exp4_2125_with_ftr"; Output = "predictions/experiments/exp4_predictions.csv"; ID = "exp4"}
    )
    
    foreach ($exp in $Experiments) {
        Write-Host "[$($exp.ID)] Generating predictions..." -ForegroundColor Cyan
        
        python -B .\predict_margin.py `
            --model $exp.Model `
            --data $CurrentSeasonData `
            --output $exp.Output
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED at $($exp.Name) predictions" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "$($exp.ID) predictions saved" -ForegroundColor Green
        Write-Host ""
    }
}

# ============================================================================
# PHASE 5: COMPARATIVE EVALUATION
# ============================================================================

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  PHASE 5: COMPARATIVE EVALUATION" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "Evaluating all models on 2025-26 season..." -ForegroundColor White
Write-Host ""

# Run comparison script
$ComparisonScript = @'
import pandas as pd
import numpy as np

def evaluate_model(predictions_file, model_name):
    df = pd.read_csv(predictions_file)
    df = df[df['actual_margin'].notna()].copy()
    
    # Rename columns if needed
    if 'pred_margin_mu' in df.columns:
        df = df.rename(columns={'pred_margin_mu': 'predicted_margin', 'market_spread_home': 'spread'})
    
    # Calculate metrics
    errors = df['predicted_margin'] - df['actual_margin']
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    actual_cover = (df['actual_margin'] > -df['spread']).astype(int)
    predicted_cover = (df['predicted_margin'] > -df['spread']).astype(int)
    ats_correct = (actual_cover == predicted_cover).sum()
    ats_pct = (ats_correct / len(df)) * 100
    
    return {
        'Model': model_name,
        'Games': len(df),
        'MAE': f'{mae:.2f}',
        'RMSE': f'{rmse:.2f}',
        'ATS_Correct': ats_correct,
        'ATS_Pct': f'{ats_pct:.2f}%'
    }

models = [
    ('predictions/current_season_champion_2025_2026_predictions.csv', 'Current Champion'),
    ('predictions/experiments/exp1_predictions.csv', 'Exp1: 14feat, 21-24'),
    ('predictions/experiments/exp2_predictions.csv', 'Exp2: 14feat, 21-25'),
    ('predictions/experiments/exp3_predictions.csv', 'Exp3: 16feat+FTR, 21-24'),
    ('predictions/experiments/exp4_predictions.csv', 'Exp4: 16feat+FTR, 21-25'),
]

results = []
for pred_file, model_name in models:
    try:
        result = evaluate_model(pred_file, model_name)
        results.append(result)
    except Exception as e:
        print(f'Error evaluating {model_name}: {e}')

results_df = pd.DataFrame(results)

print('\n' + '='*80)
print('MODEL COMPARISON - 2025-26 SEASON PERFORMANCE')
print('='*80)
print()
print(results_df.to_string(index=False))
print()
print('='*80)
print()
print('Key Findings:')
print('  - Best ATS: Check which model has highest ATS%')
print('  - Best MAE: Check which model has lowest prediction error')
print('  - FTR Impact: Compare Exp1 vs Exp3 (FTR with same training data)')
print('  - Recent Data Impact: Compare Exp1 vs Exp2 (2024-25 season added)')
print('  - Combined Impact: Exp4 shows if both improvements help')
print()

# Save results
results_df.to_csv('predictions/experiments/comparison_results.csv', index=False)
print('Results saved to: predictions/experiments/comparison_results.csv')
print()
'@

python -B -c $ComparisonScript

if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED at comparative evaluation" -ForegroundColor Red
    exit 1
}

# ============================================================================
# SUMMARY
# ============================================================================

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  EXPERIMENTS COMPLETE!" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Total time: $([math]::Round($Duration.TotalMinutes, 1)) minutes" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Review comparison_results.csv for best performer" -ForegroundColor Gray
Write-Host "  2. Run drift detection on best model" -ForegroundColor Gray
Write-Host "  3. If a challenger beats champion, promote to production" -ForegroundColor Gray
Write-Host ""

