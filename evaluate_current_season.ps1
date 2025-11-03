# Evaluate Current Season (2025-2026) Performance
# This script processes current season data and evaluates model performance

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  CURRENT SEASON (2025-2026) EVALUATION PIPELINE" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

# Step 1: Process current season data
Write-Host "[Step 1/3] Processing current season data..." -ForegroundColor Green
Write-Host ""
python process_current_season.py `
    --team-boxscores-dir "data/team_boxscores/current" `
    --player-boxscores-dir "data/player_boxscores/current" `
    --output "data/games_2025_2026_current.jsonl" `
    --season "2025-2026" `
    --include-players

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error processing current season data" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Current season data processed successfully" -ForegroundColor Green
Write-Host ""

# Step 2: Generate predictions
Write-Host "[Step 2/3] Generating predictions with trained model..." -ForegroundColor Green
Write-Host ""
python predict_margin.py `
    --model artifacts/margin_normalized `
    --data data/games_2025_2026_current_norm.jsonl `
    --output predictions/current_season_2025_2026_predictions.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error generating predictions" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Predictions generated successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Evaluate on completed games
Write-Host "[Step 3/3] Evaluating performance on completed games..." -ForegroundColor Green
Write-Host ""
python evaluate_current_season.py `
    --predictions predictions/current_season_2025_2026_predictions.csv `
    --games data/games_2025_2026_current_norm.jsonl `
    --output predictions/current_season_evaluation.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error evaluating predictions" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  EVALUATION COMPLETE!" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: predictions/current_season_evaluation.txt" -ForegroundColor Cyan
Write-Host ""

