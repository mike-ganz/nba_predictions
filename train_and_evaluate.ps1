python train.py --data data/games_train_with_players_90.jsonl --output artifacts/run_with_players --config configs/default.yaml
Write-Host "Training complete"

python evaluate.py --data data/games_val_with_players.jsonl --model artifacts/run_with_players --reports-dir reports/val_with_players --fast-eval --sharpen 0.9
Write-Host "Holdout evaluation complete"

python evaluate.py --data data/games_predict_2024_2025_with_players.jsonl --model artifacts/run_with_players --reports-dir reports/2425_with_players --fast-eval --sharpen 0.9
Write-Host "2024-2025 evaluation complete"

python analysis/feature_importance.py --data data/games_val_with_players.jsonl --model artifacts/run_with_players --reports-dir reports/feature_importance --gh-samples 9
Write-Host "Feature importance analysis complete"

Write-Host "All tasks completed successfully"