#!/usr/bin/env pwsh
# FTR Experiments: Compare 3 challengers vs current champion
# 
# Experiment 1: Include home_ftr
# Experiment 2: Include home_ftr + favorite/underdog indicators
# Experiment 3: Include ONLY favorite/underdog indicators
# Champion: Current optimized model (baseline)
#
# All models trained on data through 2024-2025 ONLY
# Tested on 2024-2025 held-out data (keeping 2025-2026 as final test set)

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
Write-Header "FTR EXPERIMENT COMPARISON: 3 CHALLENGERS vs CHAMPION"
Write-Info "Testing FTR and favorite/underdog feature variants"
Write-Info "Training on: data through 2024-2025"
Write-Info "Testing on: 2024-2025 holdout set"
Write-Info "Reserved for final test: 2025-2026 data"
Write-Info ("Started at: " + $StartTime.ToString('HH:mm:ss'))

# Check if champion model exists
$championModel = "artifacts/margin_xgboost_optimized_with2425"
if (-not (Test-Path $championModel)) {
    Write-Error-Custom "Champion model not found at $championModel"
    Write-Info "Please train champion model first or update the path"
    exit 1
}
Write-Success "Champion model found: $championModel"

# Define experimental models
$experiments = @(
    @{
        Name = "Experiment 1: home_ftr included"
        ShortName = "exp1_home_ftr"
        Config = "configs/margin_xgboost_experiment1_home_ftr.yaml"
        Output = "artifacts/margin_experiment1_home_ftr"
        Description = "Tests if including home_ftr improves 2024-25 performance"
        Hypothesis = "home_ftr may help given observed home FTR advantage"
    },
    @{
        Name = "Experiment 2: home_ftr + favorite/underdog indicators"
        ShortName = "exp2_ftr_plus_role"
        Config = "configs/margin_xgboost_experiment2_home_ftr_plus_role.yaml"
        Output = "artifacts/margin_experiment2_ftr_plus_role"
        Description = "Tests home_ftr + favorite/underdog x home/away indicators"
        Hypothesis = "Combining FTR with game role may capture interaction effects"
    },
    @{
        Name = "Experiment 3: favorite/underdog indicators only"
        ShortName = "exp3_role_only"
        Config = "configs/margin_xgboost_experiment3_role_only.yaml"
        Output = "artifacts/margin_experiment3_role_only"
        Description = "Tests ONLY favorite/underdog x home/away indicators"
        Hypothesis = "Role indicators alone may capture FTR patterns indirectly"
    }
)

# Add champion as reference
$allModels = $experiments + @{
    Name = "Champion: current optimized"
    ShortName = "champion"
    Config = $null
    Output = $championModel
    Description = "Baseline: current production model"
    Hypothesis = "N/A - baseline for comparison"
}

# STEP 1: Train experimental models
Write-Header "STEP 1: TRAINING EXPERIMENTAL MODELS"

$trainedModels = @()
$experimentNumber = 1

foreach ($exp in $experiments) {
    Write-Step $experimentNumber $experiments.Count ("Training: " + $exp.Name)
    Write-Info ("Config: " + $exp.Config)
    Write-Info ("Output: " + $exp.Output)
    Write-Info ("Hypothesis: " + $exp.Hypothesis)
    
    $trainStart = Get-Date
    
    try {
        # Use extended training script
        python train_margin_extended.py `
            --data data/games_train_with_players_90_norm.jsonl `
            --config $exp.Config `
            --output $exp.Output
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Training failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        $trainDuration = (Get-Date) - $trainStart
        Write-Success ("Training complete in " + [math]::Round($trainDuration.TotalMinutes, 1) + " minutes")
        
        $trainedModels += $exp
        $experimentNumber++
        
    } catch {
        Write-Error-Custom ("Training failed: " + $_)
        exit 1
    }
}

# Add champion (already trained)
Write-Info "Champion model already trained - skipping training step"
$trainedModels += $allModels[-1]

# STEP 2: Generate predictions from all models
Write-Header "STEP 2: GENERATING PREDICTIONS ON 2024-2025 HOLDOUT"

$testData = "data/games_predict_2024_2025_with_players_norm.jsonl"

$modelNumber = 1
foreach ($model in $trainedModels) {
    Write-Step $modelNumber $trainedModels.Count ("Predicting: " + $model.Name)
    
    $predFile = "predictions/" + $model.ShortName + "_ftr_experiment_2425.csv"
    
    try {
        python predict_margin.py `
            --data $testData `
            --model $model.Output `
            --output $predFile
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Prediction failed - model may have different feature set"
            Write-Warning "Skipping this model from comparison"
            $modelNumber++
            continue
        }
        
        Write-Success ("Predictions saved to " + $predFile)
        
        # Store prediction file in model object
        $model.PredictionFile = $predFile
        $modelNumber++
        
    } catch {
        Write-Error-Custom ("Prediction failed: " + $_)
        exit 1
    }
}

# STEP 3: Evaluate each model
Write-Header "STEP 3: EVALUATING MODELS"

$modelNumber = 1
foreach ($model in $trainedModels) {
    # Skip if no predictions
    if (-not $model.PredictionFile) {
        continue
    }
    
    Write-Step $modelNumber $trainedModels.Count ("Evaluating: " + $model.Name)
    
    $reportDir = "reports/ftr_experiments/" + $model.ShortName
    
    try {
        python evaluate_margin.py `
            --predictions $model.PredictionFile `
            --output $reportDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Evaluation failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        Write-Success "Evaluation complete"
        Write-Info ("Reports saved to " + $reportDir)
        
        # Store report directory in model object
        $model.ReportDir = $reportDir
        $modelNumber++
        
    } catch {
        Write-Error-Custom ("Evaluation failed: " + $_)
        exit 1
    }
}

# STEP 4: Create comparison summary
Write-Header "STEP 4: COMPARISON SUMMARY"

Write-Info "Extracting key metrics from evaluation reports..."

# Read metrics from each report
$comparisonTable = @()

foreach ($model in $trainedModels) {
    if (-not $model.ReportDir) {
        Write-Warning ("No report directory for model: " + $model.Name)
        continue
    }
    
    # Try to read metrics from evaluation summary
    $summaryFile = Join-Path $model.ReportDir "evaluation_summary.txt"
    
    Write-Info ("Checking summary file: " + $summaryFile)
    
    if (Test-Path $summaryFile) {
        Write-Success "Summary file found"
        # Read evaluation summary
        $content = Get-Content $summaryFile -Raw
        
        # Extract ATS accuracy (format: "ATS Accuracy:    XX.XX%")
        if ($content -match "ATS Accuracy:\s+([\d\.]+)%") {
            $ats = [double]$matches[1]
            Write-Info ("  ATS: " + $ats + "%")
        } else {
            $ats = 0.0
            Write-Warning "  Could not parse ATS from summary file"
        }
        
        # Extract MAE (format: "Margin MAE:      XX.XX points")
        if ($content -match "Margin MAE:\s+([\d\.]+)") {
            $mae = [double]$matches[1]
            Write-Info ("  MAE: " + $mae)
        } else {
            $mae = 0.0
            Write-Warning "  Could not parse MAE from summary file"
        }
        
        # Extract RMSE (format: "Margin RMSE:     XX.XX points")
        if ($content -match "Margin RMSE:\s+([\d\.]+)") {
            $rmse = [double]$matches[1]
            Write-Info ("  RMSE: " + $rmse)
        } else {
            $rmse = 0.0
            Write-Warning "  Could not parse RMSE from summary file"
        }
        
        $comparisonTable += [PSCustomObject]@{
            Model = $model.Name
            ShortName = $model.ShortName
            ATS_Pct = $ats
            MAE = $mae
            RMSE = $rmse
            ReportDir = $model.ReportDir
        }
    } else {
        Write-Warning ("Summary file not found: " + $summaryFile)
    }
}

# Check if we have any results
if ($comparisonTable.Count -eq 0) {
    Write-Error-Custom "No metrics found for any model!"
    Write-Info "This may indicate that evaluation failed."
    Write-Info "Check the evaluation output above for errors."
    exit 1
}

# Display comparison table
Write-Host ""
Write-Host "Performance Comparison on 2024-2025 Holdout Set:" -ForegroundColor Cyan
Write-Host ""
Write-Host ("{0,-45} {1,8} {2,8} {3,8}" -f "Model", "ATS %", "MAE", "RMSE") -ForegroundColor White
Write-Host ("-" * 80) -ForegroundColor DarkGray

# Sort by ATS (descending)
$sortedTable = $comparisonTable | Sort-Object -Property ATS_Pct -Descending

# Find champion for comparison
$championRow = $sortedTable | Where-Object { $_.ShortName -eq "champion" }
if ($championRow) {
    $championATS = $championRow.ATS_Pct
} else {
    $championATS = 0.0
    Write-Warning "Champion not found in results"
}

foreach ($row in $sortedTable) {
    $color = if ($row.ShortName -eq "champion") { "Yellow" } 
             elseif ($row.ATS_Pct -gt $championATS) { "Green" }
             else { "White" }
    
    $marker = if ($row.ATS_Pct -gt $championATS) { " ^" } else { "" }
    
    Write-Host ("{0,-45} {1,7:F2}% {2,8:F2} {3,8:F2}{4}" -f $row.Model, $row.ATS_Pct, $row.MAE, $row.RMSE, $marker) -ForegroundColor $color
}

Write-Host ""

# Find winner
if ($sortedTable.Count -gt 0) {
    $winner = $sortedTable[0]
    $improvement = $winner.ATS_Pct - $championATS

    Write-Host "RESULTS:" -ForegroundColor Cyan
    if ($winner.ShortName -eq "champion") {
        Write-Host "  Champion model remains the best performer" -ForegroundColor Yellow
    } else {
        Write-Host ("  Winner: " + $winner.Model) -ForegroundColor Green
        Write-Host ("  Improvement over champion: " + [math]::Round($improvement, 2) + " pp ATS") -ForegroundColor Green
    }
} else {
    Write-Error-Custom "No results to compare!"
}

# STEP 5: Test on 2025-2026 current season data
Write-Header "STEP 5: TESTING ON 2025-2026 CURRENT SEASON"

$currentSeasonData = "data/games_2025_2026_current_norm.jsonl"

if (-not (Test-Path $currentSeasonData)) {
    Write-Warning "2025-2026 data not found at $currentSeasonData"
    Write-Info "Skipping current season evaluation"
    Write-Info "To generate current season data, run:"
    Write-Info "  python process_current_season.py"
} else {
    Write-Success "2025-2026 data found"
    
    $modelNumber = 1
    foreach ($model in $trainedModels) {
        # Skip if model didn't have predictions on 2024-2025
        if (-not $model.PredictionFile) {
            continue
        }
        
        Write-Step $modelNumber $trainedModels.Count ("Predicting 2025-2026: " + $model.Name)
        
        $pred2526File = "predictions/" + $model.ShortName + "_2526_test.csv"
        
        try {
            python predict_margin.py `
                --data $currentSeasonData `
                --model $model.Output `
                --output $pred2526File
            
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Prediction failed for 2025-2026"
                $modelNumber++
                continue
            }
            
            Write-Success ("Predictions saved to " + $pred2526File)
            
            # Evaluate on 2025-2026
            $report2526Dir = "reports/ftr_experiments/" + $model.ShortName + "_2526"
            
            python evaluate_margin.py `
                --predictions $pred2526File `
                --output $report2526Dir
            
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Evaluation failed for 2025-2026"
                $modelNumber++
                continue
            }
            
            Write-Success ("Evaluation saved to " + $report2526Dir)
            
            # Store 2025-2026 report directory
            $model.Report2526Dir = $report2526Dir
            $modelNumber++
            
        } catch {
            Write-Warning ("2025-2026 testing failed: " + $_)
            $modelNumber++
            continue
        }
    }
    
    # Extract 2025-2026 metrics
    Write-Info ""
    Write-Info "Extracting 2025-2026 metrics..."
    
    $comparison2526Table = @()
    
    foreach ($model in $trainedModels) {
        if (-not $model.Report2526Dir) {
            continue
        }
        
        $summary2526File = Join-Path $model.Report2526Dir "evaluation_summary.txt"
        
        if (Test-Path $summary2526File) {
            $content = Get-Content $summary2526File -Raw
            
            # Extract metrics
            if ($content -match "ATS Accuracy:\s+([\d\.]+)%") {
                $ats2526 = [double]$matches[1]
            } else {
                $ats2526 = 0.0
            }
            
            if ($content -match "Margin MAE:\s+([\d\.]+)") {
                $mae2526 = [double]$matches[1]
            } else {
                $mae2526 = 0.0
            }
            
            if ($content -match "Margin RMSE:\s+([\d\.]+)") {
                $rmse2526 = [double]$matches[1]
            } else {
                $rmse2526 = 0.0
            }
            
            $comparison2526Table += [PSCustomObject]@{
                Model = $model.Name
                ShortName = $model.ShortName
                ATS_Pct = $ats2526
                MAE = $mae2526
                RMSE = $rmse2526
            }
        }
    }
    
    # Display 2025-2026 comparison
    if ($comparison2526Table.Count -gt 0) {
        Write-Host ""
        Write-Host "Performance Comparison on 2025-2026 Current Season:" -ForegroundColor Cyan
        Write-Host ""
        Write-Host ("{0,-45} {1,8} {2,8} {3,8}" -f "Model", "ATS %", "MAE", "RMSE") -ForegroundColor White
        Write-Host ("-" * 80) -ForegroundColor DarkGray
        
        $sorted2526Table = $comparison2526Table | Sort-Object -Property ATS_Pct -Descending
        
        $champion2526Row = $sorted2526Table | Where-Object { $_.ShortName -eq "champion" }
        if ($champion2526Row) {
            $champion2526ATS = $champion2526Row.ATS_Pct
        } else {
            $champion2526ATS = 0.0
        }
        
        foreach ($row in $sorted2526Table) {
            $color = if ($row.ShortName -eq "champion") { "Yellow" } 
                     elseif ($row.ATS_Pct -gt $champion2526ATS) { "Green" }
                     else { "White" }
            
            $marker = if ($row.ATS_Pct -gt $champion2526ATS) { " ^" } else { "" }
            
            Write-Host ("{0,-45} {1,7:F2}% {2,8:F2} {3,8:F2}{4}" -f $row.Model, $row.ATS_Pct, $row.MAE, $row.RMSE, $marker) -ForegroundColor $color
        }
        
        Write-Host ""
        
        if ($sorted2526Table.Count -gt 0) {
            $winner2526 = $sorted2526Table[0]
            $improvement2526 = $winner2526.ATS_Pct - $champion2526ATS
            
            Write-Host "2025-2026 RESULTS:" -ForegroundColor Cyan
            if ($winner2526.ShortName -eq "champion") {
                Write-Host "  Champion performs best on current season" -ForegroundColor Yellow
            } else {
                Write-Host ("  Winner on 2025-2026: " + $winner2526.Model) -ForegroundColor Green
                Write-Host ("  Improvement over champion: " + [math]::Round($improvement2526, 2) + " pp ATS") -ForegroundColor Green
            }
        }
    } else {
        Write-Warning "No 2025-2026 metrics available"
    }
}

# SUMMARY
$totalDuration = (Get-Date) - $StartTime

Write-Header "EXPERIMENT COMPLETE!"
Write-Info ("Total time: " + [math]::Round($totalDuration.TotalMinutes, 1) + " minutes")
Write-Info ""
Write-Info "Results Summary (2024-2025 Holdout):"
foreach ($row in $sortedTable) {
    Write-Info ("  " + $row.Model + ": " + $row.ATS_Pct.ToString("F2") + "% ATS")
}

if ($comparison2526Table.Count -gt 0) {
    Write-Info ""
    Write-Info "Results Summary (2025-2026 Current Season):"
    foreach ($row in $sorted2526Table) {
        Write-Info ("  " + $row.Model + ": " + $row.ATS_Pct.ToString("F2") + "% ATS")
    }
}

Write-Info ""
Write-Info "Detailed reports available in: reports/ftr_experiments/"

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Compare performance across both test sets"
Write-Host "     - Does the same model win on both 2024-2025 and 2025-2026?"
Write-Host "     - Does performance improve or degrade on current season?"
Write-Host "  2. If consistent winner emerges, analyze feature importance"
Write-Host "  3. If different winners, investigate why (FTR patterns, overfitting, etc.)"
Write-Host "  4. Deploy winner with monitoring for continued validation"
Write-Host ""
Write-Host "Key Question: Do FTR improvements generalize to 2025-2026?"
Write-Host ""

Write-Host ("Finished at: " + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Gray
Write-Host ""
