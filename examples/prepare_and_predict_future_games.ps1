# Example workflow for preparing future game data and generating predictions
# This demonstrates the complete pipeline from schedule to predictions

Write-Host "======================================================================"
Write-Host "Future Games Prediction Workflow Example"
Write-Host "======================================================================"
Write-Host ""

# Step 1: Prepare future game data
Write-Host "Step 1: Preparing future game data..."
Write-Host "---------------------------------------"
python prepare_future_games.py `
    --schedule data/schedules/current/2025-2026_NBA_Regular_Season_Schedule_Updated.xlsx `
    --player-boxscores-dir data/player_boxscores/historical `
    --injuries data/injuries/current_injuries.json `
    --market data/market/current_spreads.json `
    --output data/games_future_2025-2026.jsonl `
    --season 2025-2026 `
    --include-players `
    --limit 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to prepare future game data" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Generating predictions..."
Write-Host "-----------------------------------"

# Check if we have a trained model
if (Test-Path "artifacts/margin_normalized/margin_model.joblib") {
    python predict_margin.py `
        --data data/games_future_2025-2026.jsonl `
        --model artifacts/margin_normalized `
        --output predictions/future_predictions_2025-2026.csv
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "======================================================================"
        Write-Host "SUCCESS: Predictions generated successfully!"
        Write-Host "======================================================================"
        Write-Host ""
        Write-Host "Output files:"
        Write-Host "  - Game data: data/games_future_2025-2026.jsonl"
        Write-Host "  - Predictions: predictions/future_predictions_2025-2026.csv"
        Write-Host ""
    } else {
        Write-Host "ERROR: Failed to generate predictions" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "WARNING: No trained model found at artifacts/margin_normalized/" -ForegroundColor Yellow
    Write-Host "Run train_margin.py first to train a model, then run predictions."
    Write-Host ""
    Write-Host "For now, game data has been prepared at:"
    Write-Host "  data/games_future_2025-2026.jsonl"
}

