# Direct NBA Outcome Prediction

This repository shifts from play-by-play simulation to a market-fused bivariate Poisson framework that predicts pregame final score distributions, margins, and betting outcomes. The pipeline leans on betting market signals for offsets, enriches them with engineered matchup and availability features, and outputs calibrated probability mass functions (PMFs) for downstream decision-making.

## Contents

- `data/`: Schemas, validators, and loaders for game-level artifacts
- `features/`: Availability, market, matchup, and orchestration utilities (with historical baseline cache)
- `models/`: Bivariate Poisson elastic-net models, optional Poisson-lognormal overdispersion, distribution helpers
- `training/`: Dataset builders, trainer classes, and training CLI
- `calibration/`: Isotonic, temperature scaling, market blend, and optional sigma outputs for overdispersion
- `evaluation/`: Metrics, reporting utilities, bucketed backtests, and plot generation
- `configs/`: YAML/JSON configs for reproducible experiments
- `predict.py`: CLI to score upcoming games and emit PMFs
- `train.py`: CLI to fit models, calibrate, and serialize artifacts
- `evaluate.py`: Backtest and diagnostic report generator

Legacy simulator scripts remain for reference but are no longer part of the default workflow. Remove or archive them as needed.

## Data Schema

Each game JSON should follow the schema enforced in `data/schema.py`:

```json
{
  "game_id": "YYYY-MM-DD-AWAY-HOME",
  "season": "2024-2025",
  "date": "2024-10-15",
  "teams": {
    "A": { "team_id": "ATL", "off_rating": 115.2, ... },
    "H": { "team_id": "BOS", "off_rating": 118.9, ... }
  },
  "market": {
    "spread_home": -4.5,
    "total": 226.5,
    "moneyline_home": -180,
    "moneyline_away": 160
  },
  "players": {
    "A": [ { "player_id": "trae_young", "baseline_minutes": 35, ... }, ... ],
    "H": [ ... ]
  },
  "outcome": {
    "home_final": 112,
    "away_final": 104
  }
}
```

`data/validators.py` enforces market sanity (vigorish windows, implied score bounds) and schedule context checks.

## Feature Engineering

`features/builder.py` combines three modular blocks:

- `features/market.py`: Implied win probabilities, baseline scores, log offsets
- `features/matchup.py`: Efficiency, rebounding, turnover, pace, and context edges
- `features/availability.py`: Minutes missing, star-out flags, weighted TS%, usage aggregates

The builder returns ordered feature dictionaries (`HOME_FEATURE_KEYS`, `AWAY_FEATURE_KEYS`, `SHARED_FEATURE_KEYS`) so datasets can convert directly into NumPy arrays.

## Modeling

`models/bivariate_poisson.py` fits three elastic-net GLMs:

- Home-specific adjustment `g_H(x)`
- Away-specific adjustment `g_A(x)`
- Shared shock `h(x)` controlling correlation (`kappa`)

Baselines from the market (`lambdaH0`, `lambdaA0`) are multiplied by exponentiated adjustments to produce rates.

`models/distribution.py` converts rates into joint PMFs, win probabilities, and margin distributions.

## Training

```
python train.py --data data/games_train.jsonl --output artifacts/run_001
```

Steps performed:

1. Load and validate games (`GameDataLoader`)
2. Build training arrays (`TrainingDataset`)
3. Fit elastic-net components (`Trainer`)
4. Produce uncalibrated PMFs for the training set
5. Calibrate with isotonic CDF smoothing / temperature scaling (`Calibrator`)
6. Persist `model.joblib` and `calibrator.joblib`

Optional config overrides via `--config configs/default.yaml`.

## Prediction

```
python predict.py \
  --data data/upcoming_games.jsonl \
  --model artifacts/run_001 \
  --output predictions/upcoming.json
```

Output JSON contains:

- Posterior rates (`lambda_home`, `lambda_away`, `kappa`)
- Win probabilities (home/away)
- Integer margin PMF on configurable range

Adjust resolution with `--max-score`, `--margin-low`, `--margin-high`.

## Evaluation & Diagnostics

```
python evaluate.py \
  --data data/games_test.jsonl \
  --model artifacts/run_001
```

Reports:

- Log-loss and CRPS for home/away score distributions
- Margin mean absolute error
- Expected score diagnostics
- PIT histograms for calibration checks

## Overdispersion Toggle

Set `model.enable_overdispersion: true` (see `configs/default.yaml`) to activate Poisson-lognormal components that learn variance adjustments (`sigma_home`, `sigma_away`) from features. These are surfaced during inference and stored for calibration-aware reporting.

## Availability Baselines

Load season-level baseline minutes/TS/usage via `features.availability.load_baselines()` before feature construction to satisfy §4.2. The builder falls back to cached values when projected data is missing, ensuring `minutes_missing_top2`, `star_out`, and weighted TS remain well-defined.

## Evaluation Reporting

`evaluate.py` now supports bucketed analyses (season, spread, total) and produces:
- PIT histograms
- Win-probability reliability curves
- Margin MAE by spread/total bins
- CSV summaries saved to `reports/`

Use `--reports-dir` to customize output locations.

## Configuration

`configs/default.yaml` includes toggles for overdispersion, calibration, evaluation, and baseline data locations. Extend as needed for experiments or bespoke bucket definitions.

## Diagnostics Notebook

Seed a `notebooks/diagnostics.ipynb` (not yet committed) with feature importances, PIT plots, reliability curves, and bucket summaries generated by `evaluate.py`. Update docs once finalized.

## Experiments & Reproducibility

- Store run metadata in `configs/*.yaml`
- Version models in `artifacts/run_*`
- Keep validation splits consistent with `TrainerConfig`

## Notebooks

Create exploratory diagnostics in `notebooks/diagnostics.ipynb` (e.g., feature importances, calibration plots). Include team and market feature attribution charts.

## Cleaning Legacy Artifacts

Scripts tied to possession-level simulation (e.g., `predict_next_play.py`, `enhanced_orchestrator.py`) can be archived or removed if not needed. Ensure historical data migrations are complete before deleting.

## Requirements

`requirements.txt` now includes `pydantic`, `scikit-learn`, `scipy`, `matplotlib`, and `seaborn` to support schemas, modeling, and diagnostics.

## Next Steps

- Integrate historical REST / travel features once data is available
- Add Poisson-lognormal overdispersion toggle
- Build comprehensive unit/integration tests (pytest harness)
- Automate calibration tuning via cross-validation

## IO Schemas

Game JSONL files must conform to `data/schema.py`. Key sections:
- `teams`: labeled efficiency stats per home/away
- `market`: spread, total, moneylines
- `players`: projected/baseline minutes (top-2 used for availability aggregates)
- `outcome`: final scores (training/evaluation)

Loaders validate vigorish, implied score bounds, and schedule context.

## Usage Walkthrough

1. Load baselines (optional but recommended):
```python
   from features.availability import load_baselines
   load_baselines(path_to_csv)
   ```
2. Train and calibrate:
```bash
   python train.py --data data/games_train.jsonl --output artifacts/run1 --config configs/default.yaml
   ```
3. Predict upcoming games (includes sigma outputs when overdispersion is on):
```bash
   python predict.py --data data/upcoming.jsonl --model artifacts/run1 --output reports/upcoming.json
   ```
4. Evaluate with bucketed reports and plots:
```bash
   python evaluate.py --data data/games_test.jsonl --model artifacts/run1 --reports-dir reports/run1
   ```

