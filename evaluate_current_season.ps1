# Evaluate Current Season (2025-2026) Performance
#
# Supports three modes:
#   -Mode Hybrid   (default) - Automatic regime-based model selection
#   -Mode Champion           - Force Champion model only
#   -Mode Context            - Force Context model only
#
# HYBRID APPROACH (Default):
#   - Uses variance regime detection to choose model per-game
#   - Champion model for normal variance periods
#   - Context model for high variance periods (>30% above baseline for 5+ days)
#   - Backtested: 56.63% ATS (+2.41% vs pure Champion)
#
# CHAMPION MODEL (artifacts/champion_rest_schedule):
#   - Features: 14 (12 core matchup/injury features + home/away_rest_days)
#   - Training: 3,560 games (2021-2024 seasons)
#   - Best for: Normal variance periods
#
# CONTEXT MODEL (artifacts/experiments/exp6_context):
#   - Features: 18 (14 base + 4 league volatility context)
#   - Training: 3,560 games (2021-2024 seasons)
#   - Best for: High variance periods
#
# Examples:
#   .\evaluate_current_season.ps1                    # Hybrid (default)
#   .\evaluate_current_season.ps1 -Mode Hybrid       # Explicit hybrid
#   .\evaluate_current_season.ps1 -Mode Champion     # Champion only
#   .\evaluate_current_season.ps1 -Mode Context      # Context only

param(
    [ValidateSet("Hybrid", "Champion", "Context")]
    [string]$Mode = "Hybrid"
)

# Configuration
$ChampionModel = "artifacts/champion_rest_schedule"
$ContextModel = "artifacts/experiments/exp6_context"

# Header
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  CURRENT SEASON (2025-2026) EVALUATION" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

# Display mode
switch ($Mode) {
    "Hybrid" {
        Write-Host "  MODE: HYBRID (Automatic Regime-Based Model Selection)" -ForegroundColor Magenta
        Write-Host "    - Champion model during normal variance" -ForegroundColor White
        Write-Host "    - Context model during high variance" -ForegroundColor White
    }
    "Champion" {
        Write-Host "  MODE: CHAMPION ONLY" -ForegroundColor Green
        Write-Host "    - Model: $ChampionModel" -ForegroundColor White
        Write-Host "    - Features: 14 (rest-aware)" -ForegroundColor White
    }
    "Context" {
        Write-Host "  MODE: CONTEXT ONLY" -ForegroundColor Yellow
        Write-Host "    - Model: $ContextModel" -ForegroundColor White
        Write-Host "    - Features: 18 (with league context)" -ForegroundColor White
    }
}
Write-Host ""
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

# Step 2: Generate predictions based on mode
Write-Host "[Step 2/3] Generating predictions..." -ForegroundColor Green

switch ($Mode) {
    "Hybrid" {
        Write-Host "  Using hybrid model selection (regime-based)" -ForegroundColor Cyan
        Write-Host ""
        
        # Step 2a: Generate Champion predictions (needed for hybrid)
        Write-Host "  [2a] Generating Champion predictions..." -ForegroundColor Cyan
        python predict_margin.py `
            --model $ChampionModel `
            --data data/games_2025_2026_current_norm.jsonl `
            --output predictions/current_season_champion_2025_2026_predictions.csv
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error generating Champion predictions" -ForegroundColor Red
            exit 1
        }
        
        # Step 2b: Generate Context predictions (needed for hybrid)
        Write-Host "  [2b] Preparing context-enhanced data..." -ForegroundColor Cyan
        python -c @'
from league_normalizer import normalize_game_jsonl
normalize_game_jsonl(
    "data/games_2025_2026_current.jsonl",
    "data/games_2025_2026_current_context.jsonl",
    method="center",
    include_context=True
)
print("Context data prepared successfully")
'@
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error preparing context data" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "  [2c] Generating Context predictions..." -ForegroundColor Cyan
        python predict_margin.py `
            --model $ContextModel `
            --data data/games_2025_2026_current_context.jsonl `
            --output predictions/current_season_context_2025_2026_predictions.csv
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error generating Context predictions" -ForegroundColor Red
            exit 1
        }
        
        # Step 2d: Generate hybrid predictions (blends Champion + Context based on regime)
        Write-Host "  [2d] Generating hybrid predictions (regime-based blending)..." -ForegroundColor Cyan
        python generate_hybrid_predictions.py `
            --data data/games_2025_2026_current_norm.jsonl `
            --champion-preds predictions/current_season_champion_2025_2026_predictions.csv `
            --context-preds predictions/current_season_context_2025_2026_predictions.csv `
            --output predictions/current_season_hybrid_2025_2026_predictions.csv
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error generating hybrid predictions" -ForegroundColor Red
            exit 1
        }
        
        $PredictionsFile = "predictions/current_season_hybrid_2025_2026_predictions.csv"
        $EvaluationFile = "predictions/current_season_hybrid_evaluation.txt"
    }
    "Champion" {
        Write-Host "  Model: $ChampionModel" -ForegroundColor Cyan
        Write-Host ""
        
        python predict_margin.py `
            --model $ChampionModel `
            --data data/games_2025_2026_current_norm.jsonl `
            --output predictions/current_season_champion_2025_2026_predictions.csv
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error generating Champion predictions" -ForegroundColor Red
            exit 1
        }
        
        $PredictionsFile = "predictions/current_season_champion_2025_2026_predictions.csv"
        $EvaluationFile = "predictions/current_season_champion_evaluation.txt"
    }
    "Context" {
        Write-Host "  Model: $ContextModel" -ForegroundColor Cyan
        Write-Host ""
        
        # First generate context-enhanced data
        Write-Host "  Preparing context-enhanced data..." -ForegroundColor Cyan
        python -c @"
from league_normalizer import normalize_game_jsonl
normalize_game_jsonl(
    'data/games_2025_2026_current.jsonl',
    'data/games_2025_2026_current_context.jsonl',
    method='center',
    include_context=True
)
print('Context data prepared successfully')
"@
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error preparing context data" -ForegroundColor Red
            exit 1
        }
        
        python predict_margin.py `
            --model $ContextModel `
            --data data/games_2025_2026_current_context.jsonl `
            --output predictions/current_season_context_2025_2026_predictions.csv
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error generating Context predictions" -ForegroundColor Red
            exit 1
        }
        
        $PredictionsFile = "predictions/current_season_context_2025_2026_predictions.csv"
        $EvaluationFile = "predictions/current_season_context_evaluation.txt"
    }
}

Write-Host ""
Write-Host "Predictions generated successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Evaluate on completed games
Write-Host "[Step 3/3] Evaluating performance on completed games..." -ForegroundColor Green
Write-Host ""
python evaluate_current_season.py `
    --predictions $PredictionsFile `
    --games data/games_2025_2026_current_norm.jsonl `
    --output $EvaluationFile

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

# Display results summary
Write-Host "Mode: $Mode" -ForegroundColor Magenta
Write-Host "Predictions: $PredictionsFile" -ForegroundColor Cyan
Write-Host "Evaluation: $EvaluationFile" -ForegroundColor Cyan
Write-Host ""

switch ($Mode) {
    "Hybrid" {
        Write-Host "Hybrid Model Selection:" -ForegroundColor Green
        Write-Host "  - Automatic regime detection per game date" -ForegroundColor White
        Write-Host "  - Champion model: Normal variance periods" -ForegroundColor White
        Write-Host "  - Context model: High variance periods (>30% above baseline)" -ForegroundColor White
        Write-Host "  - Backtested improvement: +2.41% ATS vs pure Champion" -ForegroundColor White
    }
    "Champion" {
        Write-Host "Champion Model (Rest-Aware XGBoost):" -ForegroundColor Green
        Write-Host "  - Location: $ChampionModel" -ForegroundColor White
        Write-Host "  - Features: 14 (12 core + rest_days)" -ForegroundColor White
        Write-Host "  - Training: 3,560 games (2021-2024)" -ForegroundColor White
    }
    "Context" {
        Write-Host "Context Model (League Volatility Aware):" -ForegroundColor Green
        Write-Host "  - Location: $ContextModel" -ForegroundColor White
        Write-Host "  - Features: 18 (14 base + 4 league context)" -ForegroundColor White
        Write-Host "  - Training: 3,560 games (2021-2024)" -ForegroundColor White
    }
}
Write-Host ""
