# Comprehensive Experiment: Retrain All 4 Models with Optimal Hyperparameters
#
# Goal: Isolate the impact of features vs hyperparameters by:
# 1. Using the SAME optimal hyperparameters for all models
# 2. Removing away_tov_edge from all models
# 3. Testing on all datasets (training, 24-25, 25-26)
#
# This answers: "Are optimal hyperparameters enough, or do we need the new features?"

$ErrorActionPreference = "Stop"
$startTime = Get-Date

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "COMPREHENSIVE EXPERIMENT: OPTIMAL HYPERPARAMETERS + FEATURE ABLATION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Testing 4 models with IDENTICAL optimal hyperparameters:" -ForegroundColor White
Write-Host "  1. Champion:  NO home_ftr, NO role indicators" -ForegroundColor White
Write-Host "  2. Exp1:      YES home_ftr, NO role indicators" -ForegroundColor White
Write-Host "  3. Exp2:      YES home_ftr, YES role indicators" -ForegroundColor White
Write-Host "  4. Exp3:      NO home_ftr, YES role indicators" -ForegroundColor White
Write-Host ""
Write-Host "All models:" -ForegroundColor Yellow
Write-Host "  - Exclude away_tov_edge (no redundancy)" -ForegroundColor Yellow
Write-Host "  - Use Phase 1 hyperparameters (colsample=0.94, subsample=0.99, etc.)" -ForegroundColor Yellow
Write-Host "  - Tested on training, 24-25, and 25-26" -ForegroundColor Yellow
Write-Host ""

# Define the 4 models
$models = @(
    @{
        Name = "Champion_Optimal"
        Config = "configs\margin_xgboost_champion_optimal_hyperparams.yaml"
        Artifact = "artifacts\margin_champion_optimal"
        BaseConfig = "configs\margin_xgboost_optimized.yaml"
        IncludeFTR = $false
        IncludeRole = $false
        Description = "Baseline (no new features)"
    },
    @{
        Name = "Exp1_Optimal"
        Config = "configs\margin_xgboost_exp1_optimal_hyperparams.yaml"
        Artifact = "artifacts\margin_exp1_optimal"
        BaseConfig = "configs\margin_xgboost_experiment1_home_ftr.yaml"
        IncludeFTR = $true
        IncludeRole = $false
        Description = "FTR only"
    },
    @{
        Name = "Exp2_Optimal"
        Config = "configs\margin_xgboost_exp2_optimal_hyperparams.yaml"
        Artifact = "artifacts\margin_exp2_optimal"
        BaseConfig = "configs\margin_xgboost_experiment2_home_ftr_plus_role.yaml"
        IncludeFTR = $true
        IncludeRole = $true
        Description = "FTR + Role indicators"
    },
    @{
        Name = "Exp3_Optimal"
        Config = "configs\margin_xgboost_exp3_optimal_hyperparams.yaml"
        Artifact = "artifacts\margin_exp3_optimal"
        BaseConfig = "configs\margin_xgboost_experiment3_role_only.yaml"
        IncludeFTR = $false
        IncludeRole = $true
        Description = "Role indicators only"
    }
)

# Create output directory
$outputDir = "reports\optimal_hyperparams_comprehensive"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "STEP 1: Create Configs with Optimal Hyperparameters" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Optimal hyperparameters from Phase 1
$optimalParams = @"
  n_estimators: 476
  max_depth: 2
  learning_rate: 0.010159556754971632
  subsample: 0.9883195
  colsample_bytree: 0.9434709
  reg_alpha: 0.476818
  reg_lambda: 15.744418
  random_state: 42
  n_jobs: -1
  early_stopping_rounds: null
  eval_metric: rmse
"@

foreach ($model in $models) {
    Write-Host "Creating config for $($model.Name)..." -ForegroundColor Yellow
    
    # Read base config
    $baseLines = Get-Content $model.BaseConfig
    
    # Create new config with optimal hyperparameters
    $newLines = @()
    $inModelSection = $false
    $skipUntilNextSection = $false
    
    foreach ($line in $baseLines) {
        if ($line -match "^model:") {
            $newLines += $line
            $inModelSection = $true
            $skipUntilNextSection = $false
        }
        elseif ($line -match "^[a-z_]+:" -and $inModelSection) {
            # Hit next section, add optimal params
            $newLines += $optimalParams
            $newLines += "  model_type: xgboost"
            
            # Add exclude_features with away_tov_edge
            $newLines += "  exclude_features:"
            $newLines += "  - away_tov_edge  # REMOVED REDUNDANCY"
            
            # Copy existing exclusions from base
            $baseContent = $baseLines -join "`n"
            if ($baseContent -match "exclude_features:(.*?)(?=(  \w+:|^[a-z]+:|$))" -and $Matches[1]) {
                $exclusions = $Matches[1] -split "`n" | Where-Object { $_ -match "^\s*-\s*\w+" -and $_ -notmatch "away_tov_edge" }
                foreach ($excl in $exclusions) {
                    $newLines += $excl
                }
            }
            
            # Add include flags
            $newLines += "  include_diff_features: false"
            if ($model.IncludeRole) {
                $newLines += "  include_fav_underdog_features: true"
            } else {
                $newLines += "  include_fav_underdog_features: false"
            }
            
            # Add this line and continue
            $newLines += $line
            $inModelSection = $false
            $skipUntilNextSection = $false
        }
        elseif ($inModelSection) {
            # Skip lines in model section (we're replacing them)
            $skipUntilNextSection = $true
        }
        else {
            # Keep all other lines
            $newLines += $line
        }
    }
    
    # Save config
    $newLines | Set-Content $model.Config -Encoding UTF8
    Write-Host "  ✓ Saved: $($model.Config)" -ForegroundColor Green
    Write-Host "    Features: home_ftr=$($model.IncludeFTR), role=$($model.IncludeRole)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "STEP 2: Train All 4 Models" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($model in $models) {
    Write-Host "[$($models.IndexOf($model) + 1)/4] Training: $($model.Name)" -ForegroundColor Yellow
    Write-Host "  Description: $($model.Description)" -ForegroundColor White
    Write-Host ""
    
    python train_margin_extended.py `
        --data data/games_train_with_players_90_norm.jsonl `
        --config $model.Config `
        --output $model.Artifact
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Training failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "  ✓ Training complete" -ForegroundColor Green
    Write-Host ""
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "STEP 3: Evaluate on All Datasets" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$results = @()

foreach ($model in $models) {
    Write-Host "Evaluating $($model.Name)..." -ForegroundColor Yellow
    
    # Predict on 24-25
    $pred2425 = "predictions\$($model.Name)_2425.csv"
    python predict_margin.py `
        --data data/games_predict_2024_2025_with_players_norm.jsonl `
        --model $model.Artifact `
        --output $pred2425 | Out-Null
    
    python evaluate_margin.py `
        --predictions $pred2425 `
        --output "$outputDir\$($model.Name)_2425" | Out-Null
    
    # Predict on 25-26
    $pred2526 = "predictions\$($model.Name)_2526.csv"
    python predict_margin.py `
        --data data/games_2025_2026_current_norm.jsonl `
        --model $model.Artifact `
        --output $pred2526 | Out-Null
    
    python evaluate_margin.py `
        --predictions $pred2526 `
        --output "$outputDir\$($model.Name)_2526" | Out-Null
    
    # Parse results
    $summary2425 = Get-Content "$outputDir\$($model.Name)_2425\evaluation_summary.txt" -Raw
    $summary2526 = Get-Content "$outputDir\$($model.Name)_2526\evaluation_summary.txt" -Raw
    
    $ats2425 = [double][regex]::Match($summary2425, "ATS Accuracy:\s+([\d\.]+)%").Groups[1].Value
    $rmse2425 = [double][regex]::Match($summary2425, "RMSE:\s+([\d\.]+)").Groups[1].Value
    
    $ats2526 = [double][regex]::Match($summary2526, "ATS Accuracy:\s+([\d\.]+)%").Groups[1].Value
    $rmse2526 = [double][regex]::Match($summary2526, "RMSE:\s+([\d\.]+)").Groups[1].Value
    
    # Get training metrics from artifact metadata
    $metadata = Get-Content "$($model.Artifact)\metadata.yaml" -Raw
    $trainATS = [double][regex]::Match($metadata, "ats_accuracy:\s+([\d\.]+)").Groups[1].Value * 100
    $trainRMSE = [double][regex]::Match($metadata, "rmse:\s+([\d\.]+)").Groups[1].Value
    
    $results += [PSCustomObject]@{
        Model = $model.Name
        Description = $model.Description
        FTR = $model.IncludeFTR
        Role = $model.IncludeRole
        TrainATS = $trainATS
        TrainRMSE = $trainRMSE
        ATS2425 = $ats2425
        RMSE2425 = $rmse2425
        ATS2526 = $ats2526
        RMSE2526 = $rmse2526
    }
    
    Write-Host "  ✓ Complete" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "COMPREHENSIVE RESULTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "All models trained with IDENTICAL optimal hyperparameters" -ForegroundColor Yellow
Write-Host "(colsample=0.94, subsample=0.99, n_estimators=476, NO away_tov_edge)" -ForegroundColor Yellow
Write-Host ""

# Sort by 25-26 ATS
$results = $results | Sort-Object -Property ATS2526 -Descending

Write-Host "Model                    Features           │  Training   │   2024-25   │   2025-26" -ForegroundColor White
Write-Host "                                            │  ATS%  RMSE │  ATS%  RMSE │  ATS%  RMSE" -ForegroundColor White
Write-Host "────────────────────────────────────────────┼─────────────┼─────────────┼─────────────" -ForegroundColor White

foreach ($result in $results) {
    $ftrStr = if ($result.FTR) { "✓" } else { " " }
    $roleStr = if ($result.Role) { "✓" } else { " " }
    $featStr = "FTR:$ftrStr Role:$roleStr"
    
    $line = "{0,-25} {1,-18} │ {2,5:F1}% {3,5:F2} │ {4,5:F1}% {5,5:F2} │ {6,5:F1}% {7,5:F2}" -f `
        $result.Model, $featStr, `
        $result.TrainATS, $result.TrainRMSE, `
        $result.ATS2425, $result.RMSE2425, `
        $result.ATS2526, $result.RMSE2526
    
    Write-Host $line -ForegroundColor White
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "KEY INSIGHTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$best2526 = $results[0]
$champion = $results | Where-Object { $_.Model -eq "Champion_Optimal" }

Write-Host "Best Model on 2025-26:" -ForegroundColor Yellow
Write-Host "  $($best2526.Model) - $($best2526.Description)" -ForegroundColor Green
Write-Host "  ATS: $($best2526.ATS2526.ToString('F2'))%" -ForegroundColor Green
Write-Host ""

Write-Host "vs Champion with Optimal Hyperparameters:" -ForegroundColor Yellow
Write-Host "  Champion ATS: $($champion.ATS2526.ToString('F2'))%" -ForegroundColor White
Write-Host "  Improvement: $(($best2526.ATS2526 - $champion.ATS2526).ToString('+0.00;-0.00')) pp" -ForegroundColor White
Write-Host ""

Write-Host "Feature Impact Analysis:" -ForegroundColor Yellow
$exp1 = $results | Where-Object { $_.Model -eq "Exp1_Optimal" }
$exp2 = $results | Where-Object { $_.Model -eq "Exp2_Optimal" }
$exp3 = $results | Where-Object { $_.Model -eq "Exp3_Optimal" }

Write-Host "  Adding FTR only (Exp1 vs Champion):" -ForegroundColor White
Write-Host "    $(($exp1.ATS2526 - $champion.ATS2526).ToString('+0.00;-0.00')) pp on 25-26" -ForegroundColor White

Write-Host "  Adding Role only (Exp3 vs Champion):" -ForegroundColor White
Write-Host "    $(($exp3.ATS2526 - $champion.ATS2526).ToString('+0.00;-0.00')) pp on 25-26" -ForegroundColor White

Write-Host "  Adding Both (Exp2 vs Champion):" -ForegroundColor White
Write-Host "    $(($exp2.ATS2526 - $champion.ATS2526).ToString('+0.00;-0.00')) pp on 25-26" -ForegroundColor White

Write-Host ""
Write-Host "24-25 'Valley' Check:" -ForegroundColor Yellow
foreach ($result in $results) {
    $drop = $result.TrainATS - $result.ATS2425
    Write-Host "  $($result.Model): $(if ($drop -gt 5) { "⚠️ " } else { "✓ " }) $((-$drop).ToString('+0.0;-0.0')) pp" -ForegroundColor $(if ($drop -gt 5) { "Yellow" } else { "Green" })
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENT COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$elapsed = (Get-Date) - $startTime
Write-Host "  Total time: $([math]::Round($elapsed.TotalMinutes, 1)) minutes" -ForegroundColor White
Write-Host "  Results saved to: $outputDir" -ForegroundColor White
Write-Host ""

