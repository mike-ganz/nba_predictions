# Train and evaluate the XGBoost margin model

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "TRAINING XGBOOST MODEL" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan

# Step 1: Train model
Write-Host "`n[1/3] Training XGBoost model on normalized training data..." -ForegroundColor Yellow
python train_margin.py --data data/games_train_with_players_90_norm.jsonl --config configs/margin_xgboost.yaml --model-type xgboost --output artifacts/margin_xgboost
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[+] Training complete!" -ForegroundColor Green

# Step 2: Generate predictions on 2024-2025 test set
Write-Host "`n[2/3] Generating predictions on 2024-2025 test set..." -ForegroundColor Yellow
python predict_margin.py --data data/games_predict_2024_2025_with_players_norm.jsonl --model artifacts/margin_xgboost --output predictions/xgboost_2425_predictions.csv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Prediction failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[+] Predictions complete!" -ForegroundColor Green

# Step 3: Evaluate predictions
Write-Host "`n[3/3] Evaluating predictions..." -ForegroundColor Yellow
python evaluate_margin.py --predictions predictions/xgboost_2425_predictions.csv --output reports/xgboost_2425
if ($LASTEXITCODE -ne 0) {
    Write-Host "Evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[+] Evaluation complete!" -ForegroundColor Green

Write-Host "`n" -NoNewline
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "COMPLETE! Check reports/xgboost_2425/ for results" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan

