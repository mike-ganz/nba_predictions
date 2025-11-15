# Evaluate Current Season (2025-2026) Performance - Champion Model
#
# This script evaluates the Champion model with unified injury handling.
#
# CURRENT CHAMPION (Option B - Rest-Aware XGBoost, Nov 2025):
#   - Location: artifacts/champion_rest_schedule
#   - Features: 14 (12 core matchup/injury features + home/away_rest_days)
#   - Training: 3,560 games (2021-2024 seasons, unified injury handling)
#   - 2024-2025 backtest: ~52.2% ATS (very similar to prior champion)
#   - 2025-2026 so far: ~57.3% ATS overall, with stable late-season performance
#
# PREVIOUS CHAMPION (Corrected Rest Days, pre-Option B):
#   - Location: artifacts/champion_corrected_rest_days
#   - Features: 12 (no FTR, no rest_days, no role indicators, no redundant away_tov_edge)
#   - Training: 5,271 games (2021-2025 seasons, unified injury handling)
#   - Previous Performance: ~56.8% ATS on 2025-2026 season to date
#   - Kept for historical comparison but no longer used by this script

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
Write-Host "  Model: artifacts/champion_rest_schedule" -ForegroundColor Cyan
Write-Host "  Features: 14 (rest-aware, individually tuned, no redundancy)" -ForegroundColor Cyan
Write-Host ""
python predict_margin.py `
    --model artifacts/champion_rest_schedule `
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
Write-Host "Champion Model (Unified Injury Handling, Rest-Aware):" -ForegroundColor Green
Write-Host "  - Current Champion (Option B): artifacts/champion_rest_schedule" -ForegroundColor White
Write-Host "      • Rest_days included as features (home_rest_days, away_rest_days)" -ForegroundColor White
Write-Host "      • Trained on 3,560 games from 2021-2024 with unified injury handling" -ForegroundColor White
Write-Host "      • 2024-25 backtest: ~52.2% ATS; 2025-26 so far: ~57.3% ATS overall" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "  - Previous Champion: artifacts/champion_corrected_rest_days (12-feature variant)" -ForegroundColor White
Write-Host "      • Trained on 5,271 games (2021-2025) with unified injury handling" -ForegroundColor White
Write-Host "      • 2025-26 so far: ~56.8% ATS; slightly weaker late-season stability" -ForegroundColor White
Write-Host ""

