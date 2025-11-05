#!/usr/bin/env pwsh
# Train and compare XGBoost models with different loss functions
# Compares: Standard Loss, ATS-Focused Loss, and Current Champion

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
Write-Header "XGBOOST MODEL COMPARISON: STANDARD vs ATS-LOSS vs CHAMPION"
Write-Info "Training with new spread_line_movement feature"
Write-Info ("Started at: " + $StartTime.ToString('HH:mm:ss'))

# Check if champion model exists
$championModel = "artifacts/margin_xgboost_optimized_with2425"
if (-not (Test-Path $championModel)) {
    Write-Warning "Champion model not found at $championModel"
    Write-Info "Will compare only the two new models"
    $hasChampion = $false
} else {
    Write-Success "Champion model found: $championModel"
    $hasChampion = $true
}

# Define model configurations
$models = @(
    @{
        Name = "Standard Loss (with line movement)"
        ShortName = "standard"
        Config = "configs/margin_xgboost.yaml"
        Output = "artifacts/margin_xgboost_standard_with_line_movement"
        Description = "reg:squarederror objective + new feature"
    },
    @{
        Name = "ATS-Focused Loss (with line movement)"
        ShortName = "ats_custom"
        Config = "configs/margin_xgboost_ats_objective.yaml"
        Output = "artifacts/margin_xgboost_ats_with_line_movement"
        Description = "Custom ATS objective + new feature"
    }
)

if ($hasChampion) {
    $models += @{
        Name = "Current Champion (baseline)"
        ShortName = "champion"
        Config = $null
        Output = $championModel
        Description = "Existing production model"
    }
}

# STEP 1: Train new models
Write-Header "STEP 1: TRAINING NEW MODELS"

$trainedModels = @()

foreach ($model in $models) {
    if ($model.ShortName -eq "champion") {
        Write-Info "Skipping training for champion model (already trained)"
        $trainedModels += $model
        continue
    }
    
    Write-Step ($trainedModels.Count + 1) ($models.Count) ("Training: " + $model.Name)
    Write-Info ("Config: " + $model.Config)
    Write-Info ("Output: " + $model.Output)
    Write-Info ("Description: " + $model.Description)
    
    $trainStart = Get-Date
    
    try {
        python train_margin.py --data data/games_train_with_players_90_norm.jsonl --config $model.Config --model-type xgboost --output $model.Output
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Training failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        $trainDuration = (Get-Date) - $trainStart
        Write-Success ("Training complete in " + [math]::Round($trainDuration.TotalMinutes, 1) + " minutes")
        
        $trainedModels += $model
        
    } catch {
        Write-Error-Custom ("Training failed: " + $_)
        exit 1
    }
}

# STEP 2: Generate predictions from all models
Write-Header "STEP 2: GENERATING PREDICTIONS"

$testData = "data/games_predict_2024_2025_with_players_norm.jsonl"

foreach ($model in $trainedModels) {
    Write-Step ($trainedModels.IndexOf($model) + 1) ($trainedModels.Count) ("Predicting: " + $model.Name)
    
    $predFile = "predictions/" + $model.ShortName + "_2425_predictions.csv"
    
    try {
        python predict_margin.py --data $testData --model $model.Output --output $predFile
        
        if ($LASTEXITCODE -ne 0) {
            if ($model.ShortName -eq "champion") {
                Write-Warning "Champion model has feature mismatch (doesn't include new feature)"
                Write-Warning "Skipping champion from comparison"
                continue
            }
            Write-Error-Custom "Prediction failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        Write-Success ("Predictions saved to " + $predFile)
        
        # Store prediction file in model object
        $model.PredictionFile = $predFile
        
    } catch {
        Write-Error-Custom ("Prediction failed: " + $_)
        exit 1
    }
}

# STEP 3: Evaluate each model
Write-Header "STEP 3: EVALUATING MODELS"

foreach ($model in $trainedModels) {
    # Skip if no predictions (e.g., champion was skipped)
    if (-not $model.PredictionFile) {
        continue
    }
    
    Write-Step ($trainedModels.IndexOf($model) + 1) ($trainedModels.Count) ("Evaluating: " + $model.Name)
    
    $reportDir = "reports/" + $model.ShortName + "_2425"
    
    try {
        python evaluate_margin.py --predictions $model.PredictionFile --output $reportDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Evaluation failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        Write-Success "Evaluation complete"
        Write-Info ("Reports saved to " + $reportDir)
        
        # Store report directory in model object
        $model.ReportDir = $reportDir
        
    } catch {
        Write-Error-Custom ("Evaluation failed: " + $_)
        exit 1
    }
}

# STEP 4: Create comparison report
Write-Header "STEP 4: CREATING COMPARISON REPORT"

Write-Info "Extracting metrics from evaluation reports..."

# Run comparison script
python scripts/compare_models_metrics.py

if ($LASTEXITCODE -ne 0) {
    Write-Error-Custom "Comparison failed"
    exit $LASTEXITCODE
}

# SUMMARY
$totalDuration = (Get-Date) - $StartTime

Write-Header "COMPARISON COMPLETE!"
Write-Info ("Total time: " + [math]::Round($totalDuration.TotalMinutes, 1) + " minutes")
Write-Info ""
Write-Info "Results Summary:"
foreach ($model in $trainedModels) {
    Write-Info ("  - " + $model.Name)
    Write-Info ("    Model: " + $model.Output)
    Write-Info ("    Predictions: " + $model.PredictionFile)
    Write-Info ("    Reports: " + $model.ReportDir)
    Write-Info ""
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review the comparison table above to identify the best model"
Write-Host "  2. Check detailed reports in reports/*_2425/ directories"
Write-Host "  3. View CSV comparison: reports/model_comparison.csv"
Write-Host "  4. Analyze feature importance:"
Write-Host "     python analyze_margin_feature_importance.py --model artifacts/[winner_model]"
Write-Host ""
Write-Host "To use the winning model in production:"
Write-Host "  - Update prepare_todays_games.py to point to the best model"
Write-Host "  - Update daily_betting_recommendations.py model path"
Write-Host ""

Write-Host ("Finished at: " + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Gray
Write-Host ""

