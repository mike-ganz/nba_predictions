#!/usr/bin/env pwsh
# ATS Loss Function Experiments
#
# Compares standard MSE loss vs ATS-focused loss across all 4 model variants:
# - Champion
# - Experiment 1: home_ftr
# - Experiment 2: home_ftr + role indicators  
# - Experiment 3: role indicators only
#
# Tests on both 2024-2025 and 2025-2026 to see if ATS loss improves performance

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

# Color functions
function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

function Write-Step($number, $total, $text) {
    Write-Host ""
    Write-Host "[$number/$total] $text" -ForegroundColor Yellow
}

function Write-Success($text) {
    Write-Host "  * $text" -ForegroundColor Green
}

function Write-Info($text) {
    Write-Host "  $text" -ForegroundColor Gray
}

function Write-Warning($text) {
    Write-Host "  ! $text" -ForegroundColor Yellow
}

function Write-Error-Custom($text) {
    Write-Host "  X $text" -ForegroundColor Red
}

# Banner
Write-Header "ATS LOSS FUNCTION EXPERIMENT"
Write-Info "Comparing Standard MSE Loss vs ATS-Focused Loss"
Write-Info "Testing 4 model variants with each loss function (8 models total)"
Write-Info ("Started at: " + $StartTime.ToString('HH:mm:ss'))

# Define ATS-optimized models to train
$atsModels = @(
    @{
        Name = "Champion with ATS Loss"
        ShortName = "champion_ats"
        Config = "configs/margin_xgboost_champion_ats.yaml"
        Output = "artifacts/margin_champion_ats"
        StandardCounterpart = "champion"
        StandardOutput = "artifacts/margin_xgboost_optimized_with2425"
    },
    @{
        Name = "Exp1: home_ftr + ATS Loss"
        ShortName = "exp1_home_ftr_ats"
        Config = "configs/margin_xgboost_experiment1_home_ftr_ats.yaml"
        Output = "artifacts/margin_experiment1_home_ftr_ats"
        StandardCounterpart = "exp1_home_ftr"
        StandardOutput = "artifacts/margin_experiment1_home_ftr"
    },
    @{
        Name = "Exp2: home_ftr+role + ATS Loss"
        ShortName = "exp2_ftr_plus_role_ats"
        Config = "configs/margin_xgboost_experiment2_home_ftr_plus_role_ats.yaml"
        Output = "artifacts/margin_experiment2_ftr_plus_role_ats"
        StandardCounterpart = "exp2_ftr_plus_role"
        StandardOutput = "artifacts/margin_experiment2_ftr_plus_role"
    },
    @{
        Name = "Exp3: role only + ATS Loss"
        ShortName = "exp3_role_only_ats"
        Config = "configs/margin_xgboost_experiment3_role_only_ats.yaml"
        Output = "artifacts/margin_experiment3_role_only_ats"
        StandardCounterpart = "exp3_role_only"
        StandardOutput = "artifacts/margin_experiment3_role_only"
    }
)

# STEP 1: Train ATS models
Write-Header "STEP 1: TRAINING ATS-OPTIMIZED MODELS"

$trainedAtsModels = @()
$modelNumber = 1

foreach ($model in $atsModels) {
    Write-Step $modelNumber $atsModels.Count ("Training: " + $model.Name)
    Write-Info ("Config: " + $model.Config)
    Write-Info ("Output: " + $model.Output)
    
    $trainStart = Get-Date
    
    try {
        python train_margin_extended.py `
            --data data/games_train_with_players_90_norm.jsonl `
            --config $model.Config `
            --output $model.Output
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Training failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        $trainDuration = (Get-Date) - $trainStart
        Write-Success ("Training complete in " + [math]::Round($trainDuration.TotalMinutes, 1) + " minutes")
        
        $trainedAtsModels += $model
        $modelNumber++
        
    } catch {
        Write-Error-Custom ("Training failed: " + $_)
        exit 1
    }
}

# STEP 2: Test all models on 2024-2025
Write-Header "STEP 2: TESTING ON 2024-2025 HOLDOUT"

$testData2425 = "data/games_predict_2024_2025_with_players_norm.jsonl"

# Test ATS models
$modelNumber = 1
foreach ($model in $trainedAtsModels) {
    Write-Step $modelNumber $trainedAtsModels.Count ("Predicting 2024-2025: " + $model.Name)
    
    $predFile = "predictions/" + $model.ShortName + "_2425.csv"
    
    try {
        python predict_margin.py `
            --data $testData2425 `
            --model $model.Output `
            --output $predFile
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Prediction failed"
            $modelNumber++
            continue
        }
        
        Write-Success ("Predictions saved to " + $predFile)
        $model.PredFile2425 = $predFile
        
        # Evaluate
        $reportDir = "reports/ats_loss_experiments/" + $model.ShortName + "_2425"
        python evaluate_margin.py --predictions $predFile --output $reportDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Evaluation failed"
            $modelNumber++
            continue
        }
        
        $model.ReportDir2425 = $reportDir
        $modelNumber++
        
    } catch {
        Write-Warning ("Testing failed: " + $_)
        $modelNumber++
        continue
    }
}

# Also load standard model results (from previous run)
foreach ($model in $trainedAtsModels) {
    $standardReportDir = "reports/ftr_experiments/" + $model.StandardCounterpart
    if (Test-Path $standardReportDir) {
        $model.StandardReportDir2425 = $standardReportDir
    }
}

# STEP 3: Test all models on 2025-2026
Write-Header "STEP 3: TESTING ON 2025-2026 CURRENT SEASON"

$testData2526 = "data/games_2025_2026_current_norm.jsonl"

if (-not (Test-Path $testData2526)) {
    Write-Warning "2025-2026 data not found"
    Write-Info "Skipping current season tests"
} else {
    Write-Success "2025-2026 data found"
    
    # Test ATS models
    $modelNumber = 1
    foreach ($model in $trainedAtsModels) {
        Write-Step $modelNumber $trainedAtsModels.Count ("Predicting 2025-2026: " + $model.Name)
        
        $predFile = "predictions/" + $model.ShortName + "_2526.csv"
        
        try {
            python predict_margin.py `
                --data $testData2526 `
                --model $model.Output `
                --output $predFile
            
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Prediction failed"
                $modelNumber++
                continue
            }
            
            Write-Success ("Predictions saved to " + $predFile)
            $model.PredFile2526 = $predFile
            
            # Evaluate
            $reportDir = "reports/ats_loss_experiments/" + $model.ShortName + "_2526"
            python evaluate_margin.py --predictions $predFile --output $reportDir
            
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Evaluation failed"
                $modelNumber++
                continue
            }
            
            $model.ReportDir2526 = $reportDir
            $modelNumber++
            
        } catch {
            Write-Warning ("Testing failed: " + $_)
            $modelNumber++
            continue
        }
    }
    
    # Load standard model results
    foreach ($model in $trainedAtsModels) {
        $standardReportDir = "reports/ftr_experiments/" + $model.StandardCounterpart + "_2526"
        if (Test-Path $standardReportDir) {
            $model.StandardReportDir2526 = $standardReportDir
        }
    }
}

# STEP 4: Extract and compare metrics
Write-Header "STEP 4: COMPARISON - STANDARD LOSS vs ATS LOSS"

function Get-Metrics($reportDir) {
    $summaryFile = Join-Path $reportDir "evaluation_summary.txt"
    
    if (-not (Test-Path $summaryFile)) {
        return $null
    }
    
    $content = Get-Content $summaryFile -Raw
    
    $ats = 0.0
    $mae = 0.0
    $rmse = 0.0
    
    if ($content -match "ATS Accuracy:\s+([\d\.]+)%") {
        $ats = [double]$matches[1]
    }
    if ($content -match "Margin MAE:\s+([\d\.]+)") {
        $mae = [double]$matches[1]
    }
    if ($content -match "Margin RMSE:\s+([\d\.]+)") {
        $rmse = [double]$matches[1]
    }
    
    return @{
        ATS = $ats
        MAE = $mae
        RMSE = $rmse
    }
}

# Extract 2024-2025 metrics
Write-Info ""
Write-Info "Extracting 2024-2025 metrics..."

$comparison2425 = @()

foreach ($model in $trainedAtsModels) {
    # Get ATS loss metrics
    if ($model.ReportDir2425) {
        $atsMetrics = Get-Metrics $model.ReportDir2425
    } else {
        $atsMetrics = $null
    }
    
    # Get standard loss metrics
    if ($model.StandardReportDir2425) {
        $stdMetrics = Get-Metrics $model.StandardReportDir2425
    } else {
        $stdMetrics = $null
    }
    
    if ($atsMetrics -or $stdMetrics) {
        $comparison2425 += [PSCustomObject]@{
            Model = $model.Name -replace " with ATS Loss", "" -replace " \+ ATS Loss", ""
            StandardATS = if ($stdMetrics) { $stdMetrics.ATS } else { 0.0 }
            AtsLossATS = if ($atsMetrics) { $atsMetrics.ATS } else { 0.0 }
            Difference = if ($atsMetrics -and $stdMetrics) { $atsMetrics.ATS - $stdMetrics.ATS } else { 0.0 }
            StandardMAE = if ($stdMetrics) { $stdMetrics.MAE } else { 0.0 }
            AtsLossMAE = if ($atsMetrics) { $atsMetrics.MAE } else { 0.0 }
        }
    }
}

# Display 2024-2025 comparison
Write-Host ""
Write-Host "2024-2025 Holdout: Standard Loss vs ATS Loss" -ForegroundColor Cyan
Write-Host ""
Write-Host ("{0,-35} {1,10} {2,10} {3,10}" -f "Model", "Std ATS%", "ATS Loss%", "Diff") -ForegroundColor White
Write-Host ("-" * 80) -ForegroundColor DarkGray

foreach ($row in $comparison2425) {
    $color = if ($row.Difference -gt 1.0) { "Green" } 
             elseif ($row.Difference -lt -1.0) { "Red" }
             else { "White" }
    
    Write-Host ("{0,-35} {1,9:F2}% {2,9:F2}% {3,9:F2}" -f $row.Model, $row.StandardATS, $row.AtsLossATS, $row.Difference) -ForegroundColor $color
}

# Extract 2025-2026 metrics
if (Test-Path $testData2526) {
    Write-Info ""
    Write-Info "Extracting 2025-2026 metrics..."
    
    $comparison2526 = @()
    
    foreach ($model in $trainedAtsModels) {
        # Get ATS loss metrics
        if ($model.ReportDir2526) {
            $atsMetrics = Get-Metrics $model.ReportDir2526
        } else {
            $atsMetrics = $null
        }
        
        # Get standard loss metrics
        if ($model.StandardReportDir2526) {
            $stdMetrics = Get-Metrics $model.StandardReportDir2526
        } else {
            $stdMetrics = $null
        }
        
        if ($atsMetrics -or $stdMetrics) {
            $comparison2526 += [PSCustomObject]@{
                Model = $model.Name -replace " with ATS Loss", "" -replace " \+ ATS Loss", ""
                StandardATS = if ($stdMetrics) { $stdMetrics.ATS } else { 0.0 }
                AtsLossATS = if ($atsMetrics) { $atsMetrics.ATS } else { 0.0 }
                Difference = if ($atsMetrics -and $stdMetrics) { $atsMetrics.ATS - $stdMetrics.ATS } else { 0.0 }
                StandardMAE = if ($stdMetrics) { $stdMetrics.MAE } else { 0.0 }
                AtsLossMAE = if ($atsMetrics) { $atsMetrics.MAE } else { 0.0 }
            }
        }
    }
    
    # Display 2025-2026 comparison
    Write-Host ""
    Write-Host ""
    Write-Host "2025-2026 Current Season: Standard Loss vs ATS Loss" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("{0,-35} {1,10} {2,10} {3,10}" -f "Model", "Std ATS%", "ATS Loss%", "Diff") -ForegroundColor White
    Write-Host ("-" * 80) -ForegroundColor DarkGray
    
    foreach ($row in $comparison2526) {
        $color = if ($row.Difference -gt 1.0) { "Green" } 
                 elseif ($row.Difference -lt -1.0) { "Red" }
                 else { "White" }
        
        Write-Host ("{0,-35} {1,9:F2}% {2,9:F2}% {3,9:F2}" -f $row.Model, $row.StandardATS, $row.AtsLossATS, $row.Difference) -ForegroundColor $color
    }
}

# SUMMARY
$totalDuration = (Get-Date) - $StartTime

Write-Header "EXPERIMENT COMPLETE!"
Write-Info ("Total time: " + [math]::Round($totalDuration.TotalMinutes, 1) + " minutes")

Write-Host ""
Write-Host "Key Findings:" -ForegroundColor Yellow
Write-Host "  1. Does ATS loss improve performance on 2024-2025?"
if ($comparison2425.Count -gt 0) {
    $avgDiff2425 = ($comparison2425 | Measure-Object -Property Difference -Average).Average
    if ($avgDiff2425 -gt 1.0) {
        Write-Host ("     YES - Average improvement: +" + [math]::Round($avgDiff2425, 2) + " pp") -ForegroundColor Green
    } elseif ($avgDiff2425 -lt -1.0) {
        Write-Host ("     NO - Average degradation: " + [math]::Round($avgDiff2425, 2) + " pp") -ForegroundColor Red
    } else {
        Write-Host ("     MINIMAL - Average change: " + [math]::Round($avgDiff2425, 2) + " pp") -ForegroundColor Gray
    }
}

Write-Host "  2. Does ATS loss improve performance on 2025-2026?"
if ($comparison2526.Count -gt 0) {
    $avgDiff2526 = ($comparison2526 | Measure-Object -Property Difference -Average).Average
    if ($avgDiff2526 -gt 1.0) {
        Write-Host ("     YES - Average improvement: +" + [math]::Round($avgDiff2526, 2) + " pp") -ForegroundColor Green
    } elseif ($avgDiff2526 -lt -1.0) {
        Write-Host ("     NO - Average degradation: " + [math]::Round($avgDiff2526, 2) + " pp") -ForegroundColor Red
    } else {
        Write-Host ("     MINIMAL - Average change: " + [math]::Round($avgDiff2526, 2) + " pp") -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  - If ATS loss helps: Deploy best ATS model"
Write-Host "  - If standard loss wins: Stick with Experiment 2"
Write-Host "  - If mixed results: Consider ensemble or hybrid approach"
Write-Host ""

Write-Host ("Finished at: " + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Gray
Write-Host ""

