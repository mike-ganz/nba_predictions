python train.py --data data/games_train_with_players_90.jsonl --output artifacts/run_with_players --config configs/default.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Training complete" -ForegroundColor Green

python evaluate.py --data data/games_val_with_players.jsonl --model artifacts/run_with_players --reports-dir reports/val_with_players --fast-eval --sharpen 0.9
if ($LASTEXITCODE -ne 0) {
    Write-Host "Holdout evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Holdout evaluation complete" -ForegroundColor Green

python evaluate.py --data data/games_predict_2024_2025_with_players.jsonl --model artifacts/run_with_players --reports-dir reports/2425_with_players --fast-eval --sharpen 0.9
if ($LASTEXITCODE -ne 0) {
    Write-Host "2024-2025 evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "2024-2025 evaluation complete" -ForegroundColor Green

python analysis/feature_importance.py --data data/games_val_with_players.jsonl --model artifacts/run_with_players --reports-dir reports/feature_importance --gh-samples 9
if ($LASTEXITCODE -ne 0) {
    Write-Host "Feature importance analysis failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Feature importance analysis complete" -ForegroundColor Green

Write-Host "`nAll tasks completed successfully!" -ForegroundColor Green