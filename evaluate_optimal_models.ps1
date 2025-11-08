# Evaluate all 4 optimal models (already trained)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "EVALUATING OPTIMAL HYPERPARAMETER MODELS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$models = @(
    @{ Name = "Champion_Optimal"; Artifact = "artifacts\margin_champion_optimal"; Description = "Baseline (no new features)" },
    @{ Name = "Exp1_Optimal"; Artifact = "artifacts\margin_exp1_optimal"; Description = "FTR only" },
    @{ Name = "Exp2_Optimal"; Artifact = "artifacts\margin_exp2_optimal"; Description = "FTR + Role indicators" },
    @{ Name = "Exp3_Optimal"; Artifact = "artifacts\margin_exp3_optimal"; Description = "Role indicators only" }
)

$outputDir = "reports\optimal_hyperparams_evaluation"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

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
    
    # Get training metrics
    $metadata = Get-Content "$($model.Artifact)\metadata.yaml" -Raw
    $trainATS = [double][regex]::Match($metadata, "ats_accuracy:\s+([\d\.]+)").Groups[1].Value * 100
    $trainRMSE = [double][regex]::Match($metadata, "rmse:\s+([\d\.]+)").Groups[1].Value
    
    $results += [PSCustomObject]@{
        Model = $model.Name
        Description = $model.Description
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

Write-Host "Model                    Description              │  Training   │   2024-25   │   2025-26" -ForegroundColor White
Write-Host "                                                  │  ATS%  RMSE │  ATS%  RMSE │  ATS%  RMSE" -ForegroundColor White
Write-Host "──────────────────────────────────────────────────┼─────────────┼─────────────┼─────────────" -ForegroundColor White

foreach ($result in $results) {
    $line = "{0,-25} {1,-24} │ {2,5:F1}% {3,5:F2} │ {4,5:F1}% {5,5:F2} │ {6,5:F1}% {7,5:F2}" -f `
        $result.Model, $result.Description, `
        $result.TrainATS, $result.TrainRMSE, `
        $result.ATS2425, $result.RMSE2425, `
        $result.ATS2526, $result.RMSE2526
    
    $color = if ($result.ATS2526 -ge 58.0) { "Green" } elseif ($result.ATS2526 -ge 55.0) { "White" } else { "Yellow" }
    Write-Host $line -ForegroundColor $color
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "KEY INSIGHTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$best = $results[0]
$champion = $results | Where-Object { $_.Model -eq "Champion_Optimal" }

Write-Host "Best Model on 2025-26:" -ForegroundColor Yellow
Write-Host "  $($best.Model) - $($best.Description)" -ForegroundColor Green
Write-Host "  ATS: $($best.ATS2526.ToString('F2'))%, RMSE: $($best.RMSE2526.ToString('F2'))" -ForegroundColor Green
Write-Host ""

Write-Host "vs Champion with Optimal Hyperparameters:" -ForegroundColor Yellow
Write-Host "  Champion ATS: $($champion.ATS2526.ToString('F2'))%" -ForegroundColor White
Write-Host "  Improvement: $(($best.ATS2526 - $champion.ATS2526).ToString('+0.00;-0.00')) pp" -ForegroundColor White
Write-Host ""

Write-Host "24-25 'Valley' Analysis:" -ForegroundColor Yellow
foreach ($result in $results | Sort-Object Model) {
    $drop = $result.TrainATS - $result.ATS2425
    $status = if ($drop -gt 7) { "⚠️ " } else { "✓ " }
    $color = if ($drop -gt 7) { "Yellow" } else { "Green" }
    Write-Host "  $status$($result.Model): $((-$drop).ToString('+0.0;-0.0')) pp drop" -ForegroundColor $color
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "EVALUATION COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

