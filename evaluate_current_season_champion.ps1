# Evaluate Current Season (2025-2026) Performance - Champion Model
#
# This script evaluates the Champion model with unified injury handling.
#
# Model Details:
#   - Location: artifacts/champion_corrected_rest_days
#   - Features: 12 (no FTR, no role indicators, no redundant away_tov_edge)
#   - Training: 5,271 games (retrained Nov 9, 2025)
#   - Unified Injury Handling: All pipelines now consistent (Nov 9, 2025)
#   - Previous Performance: 61.83% ATS on 2025-2026 season (legacy model)
#   - Expected: Similar ATS with more consistent predictions

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  CURRENT SEASON (2025-2026) EVALUATION - CHAMPION MODEL" -ForegroundColor Yellow
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

# Step 2: Generate predictions with Champion model
Write-Host "[Step 2/3] Generating predictions with Champion model..." -ForegroundColor Green
Write-Host "  Model: artifacts/champion_corrected_rest_days" -ForegroundColor Cyan
Write-Host "  Features: 12 (corrected rest days, individually tuned, no redundancy)" -ForegroundColor Cyan
Write-Host ""
python predict_margin.py `
    --model artifacts/champion_corrected_rest_days `
    --data data/games_2025_2026_current_norm.jsonl `
    --output predictions/current_season_champion_2025_2026_predictions.csv

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
    --predictions predictions/current_season_champion_2025_2026_predictions.csv `
    --games data/games_2025_2026_current_norm.jsonl `
    --output predictions/current_season_champion_evaluation.txt

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
Write-Host "Results saved to: predictions/current_season_champion_evaluation.txt" -ForegroundColor Cyan
Write-Host ""
Write-Host "Champion Model (Unified Injury Handling):" -ForegroundColor Green
Write-Host "  - Retrained: November 9, 2025 with 5,271 games" -ForegroundColor White
Write-Host "  - Unified injury handling across all pipelines" -ForegroundColor White
Write-Host "  - Previous model: 61.83% ATS on 2025-2026 season" -ForegroundColor White
Write-Host "  - Expected: Similar performance with better consistency" -ForegroundColor White
Write-Host ""

