# NBA Sports Betting Model: Data-Driven Prediction System

A production-ready machine learning system for predicting NBA game outcomes against the spread. Built on rigorous methodology with **unified injury handling** across all pipelines and validated performance of **~61% ATS accuracy** on current season data.

---

## 📊 Executive Summary

This repository contains a complete pipeline for:

1. **Collecting and processing** NBA team statistics, player data, and betting market information
2. **Training machine learning models** to predict game outcomes with realistic expectations
3. **Validating predictions** on truly unseen data to ensure robustness
4. **Identifying profitable betting strategies** through systematic market inefficiency analysis

**Bottom line:** The model identifies systematic market biases in specific game segments (home favorites with 8-12 point spreads), providing a foundation for disciplined betting strategies with demonstrated edge.

---

## 🎯 The Problem We're Solving

### Why Sports Betting is Hard

Sports betting markets are extremely efficient. Professional bettors, sophisticated algorithms, and billions of dollars in volume push prices toward "fair value" almost instantly. To profit consistently, you need:

1. **Better information** than the market has
2. **Better models** to process that information
3. **Systematic market biases** that persist over time

Most people can't achieve #1 (the market knows everything quickly), and #3 is rare. This project focuses on #2 - building better models that can detect when #3 exists.

### The Challenge

To beat the betting market, you need to win more than **52.38%** of your bets (because of the "vig" - bookmakers charge ~10% on losses). This means you're not competing against 50/50 random chance, but against an efficient market that's already very good at pricing games.

---

## 💡 Key Innovation: League-Relative Normalization

### The Core Problem

NBA statistics change dramatically over time:

```
2022: 115 offensive rating = Elite team (league avg was 112)
2025: 115 offensive rating = Average team (league avg is 115)
```

League-wide scoring has increased, pace has shifted, and three-point volume has exploded. Raw statistics are misleading because they don't account for the evolving league context.

### Our Solution

We normalize **all team statistics** to league averages calculated from season-to-date data:

```python
normalized_feature = raw_value - league_average_on_date
```

**Example:**
- Team A has 117 offensive rating, league average is 114 → **+3.0**
- Team B has 111 defensive rating, league average is 114 → **-3.0**
- Offensive edge = (+3.0) - (-3.0) = **+6.0**

This makes features:
- ✅ **Time-invariant** - comparablacross seasons
- ✅ **Distribution-stable** - mean stays near 0
- ✅ **More predictive** - captures relative strength, not absolute numbers

**Critical detail:** We only use games BEFORE each prediction date to calculate league averages (no look-ahead bias).

---

## 🔬 How It Works

### Step 1: Collect the Data

We gather three types of information for each game:

1. **Team Statistics** (rolling 10-game averages)
   - Offensive efficiency (points per 100 possessions)
   - Defensive efficiency (points allowed per 100 possessions)
   - Pace (possessions per game)
   - Shooting rates (3PA%, FT%), rebounding, turnovers

2. **Player Information** (Unified Injury Handling - Nov 2025)
   - Top 8-10 players for each team
   - Their baseline minutes, shooting efficiency, usage rate
   - **Unified injury reconstruction** across all pipelines:
     - 10-game lookback to build baseline roster
     - Players with ≥10 min baseline who are missing → marked as injured
     - Players OUT (DNP) → projected = 0 minutes
     - Active players → projected = baseline average
     - **Training, backlook, and day-of predictions now consistent**

3. **Market Data**
   - Point spread (which team is favored and by how much)
   - Total (expected combined score)
   - Moneyline odds

### Step 2: Normalize to League Average

All team statistics are converted to league-relative values using season-to-date averages:

```
Normalized features:
├── off_rating_norm
├── def_rating_norm
├── pace_norm
├── three_pt_rate_norm
├── free_throw_rate_norm
├── off_reb_rate_norm
├── def_reb_rate_norm
├── assist_rate_norm
└── turnover_rate_norm
```

### Step 3: Engineer Matchup Features

We create derived features that capture team matchups:

- **Edges:** Offensive vs defensive matchups
  - `edge_home = home_off_rating_norm - away_def_rating_norm`
- **Rebounding battles:** Offensive vs defensive rebounding
- **Turnover edges:** Which team protects/forces turnovers better
- **Pace dynamics:** Combined and differential pace
- **Player availability:** Missing minutes from injuries

**Total: 29 features** after removing 3 that added noise

### Step 4: Train the Model

**Production Model: XGBoost (Champion Model)**

The Champion model uses XGBoost with carefully tuned hyperparameters:

**Key Features:**
- **12 features** (no redundancy, no FTR, optimal set)
- **Injury features are critical:** 21.8% of total model importance
- **Automatic interactions:** Learns complex patterns without manual feature engineering
- **Trained on 5,271 games** (2021-2025 seasons with unified injury handling)

**Why XGBoost?**
- Captures non-linear relationships automatically
- Feature importance shows injury features (#1 and #6 most important)
- Superior performance with unified injury handling
- Robust hyperparameters (colsample=0.847, n_estimators=210, max_depth=2)

**Training Configuration:**
```yaml
model:
  model_type: xgboost
  n_estimators: 210
  max_depth: 2
  learning_rate: 0.00994
  colsample_bytree: 0.847
  subsample: 0.714
  reg_alpha: 1.90
  reg_lambda: 13.06
  exclude_features:
    - away_tov_edge (redundant)
    - shared_team_weighted_ts_away (noise)
    - away_usage_share_top2 (redundant)
    - shared_implied_home_winprob (redundant)
    - home_rest_days (regime-specific)
    - away_orb_edge (correlated)
    - away_tpar (redundant)
    - away_rest_days (regime-specific)
    - shared_pace_mean (redundant)
    - shared_pace_diff (redundant)
    - home_ftr (regime-specific)
    - away_ftr (regime-specific)
```

**Training Command:**
```bash
python train_margin.py \
  --data data/games_train_with_players_90_norm.jsonl \
  --config configs/champion_corrected_rest_days/config.yaml \
  --output artifacts/champion_corrected_rest_days
```

**Note:** Model only predicts margin (μ), not uncertainty (σ). Cover probabilities computed separately if needed.

### Step 5: Generate Predictions

For each game:
```python
predicted_margin = baseline_margin + model_residual
predicted_sigma = uncertainty_model(features)
cover_probability = P(margin > -spread | μ, σ)
```

Output includes:
- Expected margin (μ)
- Uncertainty (σ)
- Cover probabilities for each team
- Win probabilities

---

## 📈 Results & Performance

### Model Selection: OLD vs NEW

**Critical Decision (October 2025):** We trained two models and compared performance:
- **OLD Model:** Trained on 2021-2024 (3 seasons, 3,560 games)
- **NEW Model:** Trained on 2021-2025 (4 seasons, 4,875 games)

**Why we chose OLD model for production:**

The NEW model included 2024-25 in training, which had **anomalous feature-outcome relationships**:
- 3-point rate correlation **flipped sign** (positive in 21-24, negative in 24-25)
- Other key features showed unstable coefficients
- When 25-26 season reverted to normal patterns, NEW model performed worse

**Evidence:**
- OLD Model on 25-26: **56.94% ATS**
- NEW Model on 25-26: **47.22% ATS** (significantly worse)
- Coefficient analysis confirmed NEW model learned "wrong" patterns from 24-25

**Conclusion:** Using OLD model (21-24) provides more robust predictions for typical NBA seasons.

### Current Season Performance (2025-26)

**Champion Model (Unified Injury Handling)**

**Training Performance:**
| Metric | Value | Notes |
|--------|-------|-------|
| **Training Data** | 5,271 games | 2021-2025 seasons |
| **Training ATS** | 54.89% | Good generalization |
| **Training MAE** | 10.38 points | Strong margin prediction |
| **Training RMSE** | 13.32 points | Consistent |

**Expected Production Performance:**
- Similar to previous champion (~61% ATS)
- More consistent predictions between day-of and backlook
- Injury features now contribute meaningful signal (21.8% of model importance)

### Historical Performance (2024-25)

**Validation set:** 1,315 games, fully out-of-sample for OLD model

| Metric | OLD Model (21-24) | Notes |
|--------|-------------------|-------|
| **ATS Accuracy** | 51.33% | Below breakeven |
| **MAE** | 10.64 pts | Strong margin prediction |
| **ROI (overall)** | -2.01% | Market efficiency |

**By Favorite Type (2024-25):**
- Home favorites (686 games): 54.23% ATS, +3.45% ROI
- Away favorites (629 games): 48.17% ATS, -6.66% ROI

### Profitable Betting Strategies

**Strategy 1: 3-Bucket System (Recommended)**

Focus on **Home Favorites + Medium Spread (5-8 points)**:
- **24-25 Season (Discovery):** 277 games, 54.51% ATS, +4.06% ROI
- **25-26 Season (Validation):** 17 games, 64.71% ATS, +23.52% ROI

**Strategy 2: 5-Bucket System (Advanced)**

Focus on **Home Favorites + HIGH Spread (8-12 points)**:
- **24-25 Season (Discovery):** 197 games, 58.88% ATS, +12.41% ROI
- **25-26 Season (Validation):** 9 games, 55.56% ATS, +6.06% ROI

**Why this works:**
- Market systematically **undervalues home favorites** in moderate-to-large spreads
- Home court advantage has declined league-wide, but market hasn't fully adjusted
- Larger sample (197-277 games) provides statistical confidence
- Consistent performance across 24-25 and 25-26 seasons

**Risk Note:** 25-26 results are on very small samples (9-17 games). Need full season to confirm.

### Key Observations

1. **Selective Betting is Critical:** Overall model performance is modest, but specific segments show strong edge
2. **Market Inefficiency Identified:** Home favorites with 8-12 point spreads consistently outperform expectations
3. **Anomalous Seasons Exist:** 24-25 had unusual patterns; robust models ignore outlier data
4. **Sample Size Matters:** 72 games (25-26) is insufficient for definitive conclusions

---

## 🔧 Critical Methodological Improvements

### Issue #1: Data Leakage in Player Minutes (October 2025)

**Problem:** Model was using actual minutes played (outcome-dependent data) instead of pre-game projections.

**Solution:** Fixed to use realistic pre-game projections (0 for OUT players, baseline for active players).

See [DATA_LEAKAGE_FIX_FINAL_REPORT.md](DATA_LEAKAGE_FIX_FINAL_REPORT.md) for details.

### Issue #2: Inconsistent Injury Handling (November 2025)

**Problem:** Train-test distribution mismatch across pipelines.

**Before:**
- **Training/Backlook:** Injured players completely omitted from roster → injury features had 0% model importance
- **Day-of:** Injured players marked with projected_minutes=0.0 → injury features should matter but model never learned

**Symptoms:**
- Injury features (`minutes_missing_top2`, `star_out`) had **zero importance** in old model
- Day-of and backlook predictions differed even with identical spreads
- Model blind to injury impacts

**Solution (November 9, 2025): Unified Injury Handling**

All three pipelines now use **roster reconstruction**:

1. **Lookback:** Analyze last 10 games before target game
2. **Baseline Roster:** Include players who appeared in ≥3 of those games
3. **Injury Detection:**
   - Player in boxscore, >0 mins → Healthy (projected_minutes = None)
   - Player in boxscore, 0 mins → OUT (projected_minutes = 0.0)
   - Player missing, baseline ≥10 mins → Injured (projected_minutes = 0.0)
   - Player missing, baseline <10 mins → DNP-CD (omit from roster)

**Validation Results:**
- ✅ Star player treatment: 13.8% absence rate (realistic)
- ✅ Injury distribution: 12.8% player-games (realistic)
- ✅ Multi-game consistency: 56% of absences span multiple games (acceptable)
- ✅ **Feature importance: 21.8% of total model importance from injury features**

**Impact:**
- Injury features went from 0% → 21.8% importance
- `home_minutes_missing_top2` is now **#1 most important feature**
- `away_minutes_missing_top2` is now **#6 most important feature**
- Day-of and backlook predictions now consistent
- Model can actually learn from injury patterns

See [UNIFIED_INJURY_IMPLEMENTATION_SUMMARY.md](UNIFIED_INJURY_IMPLEMENTATION_SUMMARY.md) for full technical details.

---

## 🏗️ Technical Architecture

### Data Pipeline

```
Raw Data Sources
├── Team Boxscores (Excel files)
├── Player Boxscores (Excel files)
└── Market Data (JSON)
         ↓
    Processors
├── generate_team_stats.py (rolling 10-game averages)
├── player_data_loader.py (season-to-date stats, realistic projections)
└── league_normalizer.py (league-relative features)
         ↓
    JSONL Files
├── games_train_with_players_90_norm.jsonl (training 90%)
├── games_val_with_players_norm.jsonl (validation 10%)
└── games_predict_2024_2025_with_players_norm.jsonl (test)
```

### Model Architecture

**Type:** Direct Margin Prediction with Ridge Regression

**Input Features (29 total):**

```
Home Features (9):
├── edge (off vs def matchup)
├── orb_edge (offensive rebounding advantage)
├── tov_edge (turnover differential)
├── tpar (three-point attempt rate, normalized)
├── ftr (free throw rate, normalized)
├── rest_days
├── minutes_missing_top2 (injury impact)
├── star_out (binary: top player OUT)
└── usage_share_top2

Away Features (9):
└── [same as home]

Shared Features (4):
├── pace_mean (combined pace)
├── implied_away_winprob (from moneyline)
└── team_weighted_ts_home (shooting efficiency)
    [Note: 3 features excluded after importance analysis]

Difference Features (7):
└── [home - away for key stats]
```

**Excluded Features (hurt performance):**
1. `shared_team_weighted_ts_away` - Added noise
2. `away_usage_share_top2` - Redundant
3. `shared_implied_home_winprob` - Redundant with spread

**Output:**
- μ (mean): Expected margin (home - away)
- σ (sigma): Prediction uncertainty
- P(cover): Probability each team covers the spread
- P(win): Probability each team wins outright

**Training Configuration:**
```yaml
model:
  use_cv: true
  alphas_mean: [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
  alphas_variance: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
  cv_folds: 5
  min_variance: 1.0
  exclude_features:
    - shared_team_weighted_ts_away
    - away_usage_share_top2
    - shared_implied_home_winprob
```

---

## 🚀 Usage

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nba_predictions.git
cd nba_predictions

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Generate training data with player info (15-20 minutes)
python prepare_data.py \
  --team-boxscores-dir data/team_boxscores/historical \
  --player-boxscores-dir data/player_boxscores/historical \
  --output data/games_train_with_players.jsonl \
  --seasons 2021-2022 2022-2023 2023-2024 2024-2025 \
  --include-players

# Apply league normalization
python scripts/normalize_all_data.py
```

### Model Training

```bash
# Train Ridge (RECOMMENDED for production)
python train_margin.py \
  --data data/games_train_with_players_90_norm.jsonl \
  --config configs/margin_default.yaml \
  --output artifacts/margin_normalized

# Train XGBoost (experimental alternative)
python train_margin.py \
  --data data/games_train_with_players_90_norm.jsonl \
  --config configs/margin_xgboost.yaml \
  --model-type xgboost \
  --output artifacts/margin_xgboost

# Or use convenience script for XGBoost
./train_and_evaluate_xgboost.ps1

# Alternative: Train on 2021-2025 dataset (NOT recommended - includes anomalous 24-25)
python train_expanded_model.py
# Note: This trains on all 4 seasons but has lower out-of-sample performance
```

### Making Predictions

```bash
# Generate predictions
python predict_margin.py \
  --model artifacts/margin_normalized \
  --data data/games_predict_2024_2025_with_players_norm.jsonl \
  --output predictions/test_predictions.csv

# Evaluate performance
python evaluate_fixed_models.py
```

### For Current Season (2025-26)

```bash
# Process current season data
python process_current_season.py --season 2025-2026

# Generate predictions
python predict_margin.py \
  --model artifacts/margin_normalized \
  --data data/games_2025_2026_current_norm.jsonl \
  --output predictions/current_season_predictions.csv

# Evaluate (as games are played)
python evaluate_current_season.py
```

### Comparing Ridge vs XGBoost

To evaluate if XGBoost provides better predictions than Ridge:

**Step 1: Train both models**
```bash
# Ridge (if not already trained)
python train_margin.py \
  --data data/games_train_with_players_90_norm.jsonl \
  --config configs/margin_default.yaml \
  --output artifacts/margin_normalized

# XGBoost
python train_margin.py \
  --data data/games_train_with_players_90_norm.jsonl \
  --config configs/margin_xgboost.yaml \
  --model-type xgboost \
  --output artifacts/margin_xgboost
```

**Step 2: Generate predictions from both models**
```bash
# Ridge predictions
python predict_margin.py \
  --model artifacts/margin_normalized \
  --data data/games_predict_2024_2025_with_players_norm.jsonl \
  --output predictions/ridge_2425_predictions.csv

# XGBoost predictions
python predict_margin.py \
  --model artifacts/margin_xgboost \
  --data data/games_predict_2024_2025_with_players_norm.jsonl \
  --output predictions/xgboost_2425_predictions.csv
```

**Step 3: Compare performance**
```bash
python compare_ridge_vs_xgboost.py \
  --ridge-predictions predictions/ridge_2425_predictions.csv \
  --xgboost-predictions predictions/xgboost_2425_predictions.csv \
  --output reports/ridge_vs_xgboost_comparison
```

The comparison script will show:
- Margin accuracy (MAE, RMSE, R²) for both models
- ATS accuracy and ROI comparison
- Statistical significance testing (paired t-test)
- Game-by-game analysis showing where each model performs better
- Performance breakdown by spread size

**Step 4: Tune XGBoost hyperparameters (optional)**
```bash
# Quick tuning (faster, less thorough)
python tune_xgboost_hyperparameters.py \
  --data data/games_train_with_players_90_norm.jsonl \
  --quick

# Full tuning (slower, more thorough)
python tune_xgboost_hyperparameters.py \
  --data data/games_train_with_players_90_norm.jsonl \
  --method random \
  --n-iter 100
```

This will save an optimized config file that you can use for retraining.

### Implementing the Betting Strategy

**Step 1: Generate predictions for upcoming games**
```bash
# Using Ridge (production model)
python predict_margin.py \
  --model artifacts/margin_normalized \
  --data data/games_2025_2026_current_norm.jsonl \
  --output predictions/today_predictions.csv

# Or using XGBoost (if it performs better)
python predict_margin.py \
  --model artifacts/margin_xgboost \
  --data data/games_2025_2026_current_norm.jsonl \
  --output predictions/today_predictions.csv
```

**Step 2: Filter for strategy matches**

Open `predictions/today_predictions.csv` and filter for:
- `market_spread_home < 0` (home team is favored)
- `abs(market_spread_home)` between 8 and 12 (spread of 8-12 points)

**Step 3: Review and place bets**

For games matching criteria:
- Bet on the **home favorite** to cover
- Use 1-2% of bankroll per game
- Shop for best line across sportsbooks
- Track all bets in a spreadsheet

**Step 4: Track performance**

Log each bet with:
- Date, teams, spread, predicted margin, actual outcome
- Running ATS% and ROI
- Compare to expected 58.88% ATS, 12.41% ROI

**Sample spreadsheet columns:**
```
Date | Home | Away | Spread | Our Pick | Result | Win/Loss | Running ATS% | Running ROI%
```

---

## 📁 Repository Structure

```
nba_predictions/
├── data/                           # Data storage
│   ├── team_boxscores/            # Team stats (Excel)
│   │   ├── historical/            # 2021-2025 seasons
│   │   └── current/               # Current season (dynamic)
│   ├── player_boxscores/          # Player stats (Excel)
│   └── [JSONL files]              # Processed game records
│
├── configs/                        # Model configurations
│   └── margin_default.yaml        # Ridge regression config
│
├── features/                       # Feature engineering
│   ├── matchup.py                 # Team matchup features
│   ├── availability.py            # Player availability
│   ├── market.py                  # Market-derived features
│   └── builder.py                 # Orchestration
│
├── models/                         # Model implementations
│   ├── margin_normal.py           # Ridge regression model
│   └── margin_distribution.py     # Probability calculations
│
├── training/                       # Training utilities
│   └── margin_dataset.py          # Dataset builder
│
├── scripts/                        # Analysis scripts
│   ├── normalize_all_data.py      # Apply league normalization
│   └── analyze_filtering_strategies.py  # Strategy analysis
│
├── artifacts/                      # Trained models
│   └── champion_corrected_rest_days/  # Champion XGBoost (unified injuries)
│
├── predictions/                    # Model outputs
│
├── Core Scripts:
├── prepare_data.py                # Data processing pipeline (unified injuries)
├── generate_team_stats.py         # Rolling team statistics
├── scripts/player_data_loader.py  # Player data with injury reconstruction
├── league_normalizer.py           # League-relative normalization
├── train_margin.py                # Model training (XGBoost)
├── predict_margin.py              # Generate predictions
├── process_current_season.py      # Current season processing
├── evaluate_current_season.py     # Current season evaluation
├── daily_betting_recommendations_champion.py  # Automated daily pipeline
└── evaluate_current_season_champion.ps1       # Current season evaluation script
```

---

## 🧪 Validation Methodology

### Temporal Holdout Testing

We use strict temporal splits to prevent data leakage:

```
Production Model (OLD):
├── Training:   2021-2022, 2022-2023, 2023-2024 (3,560 games)
├── Validation: Random 10% from training (396 games)
├── Test 1:     2024-2025 season (1,315 games) - OUT OF SAMPLE
└── Test 2:     2025-2026 season (72+ games) - TRULY UNSEEN

Alternative Model (NEW - Not Recommended):
├── Training:   2021-2025 (4,875 games)
├── Test:       2025-2026 season (72+ games)
└── Problem:    Overfit to anomalous 24-25 patterns
```

### Cross-Model Validation & Anomaly Detection

We trained two models to detect overfitting to unusual seasons:
- **OLD (21-24):** Trained on 3 stable seasons
- **NEW (21-25):** Trained on 3 stable + 1 anomalous season

**Hypothesis:** If 24-25 had unusual patterns, NEW model would learn them and fail on normal seasons.

**Results:**
- OLD Model on 24-25: 51.33% ATS (modest, as expected for efficient market)
- OLD Model on 25-26: **56.94% ATS**
- NEW Model on 25-26: **47.22% ATS** (significantly worse)

**Root Cause Analysis:**
We analyzed feature-outcome correlations across periods:

| Feature | 21-24 Corr. | 24-25 Corr. | 25-26 Corr. | Impact |
|---------|-------------|-------------|-------------|--------|
| **3P Rate Diff** | +0.089 | **-0.062** | +0.105 | Flipped sign in 24-25 |
| **Pace Mean** | -0.018 | **+0.054** | -0.003 | Changed direction |
| **Edge** | +0.168 | +0.154 | +0.185 | Stable (good) |

**Conclusion:** 24-25 season had fundamentally different patterns. NEW model learned these wrong relationships, degrading performance on 25-26 when patterns reverted to normal.

**Decision:** Use OLD model (21-24) for production to avoid learning from anomalous data.

### Feature Importance Analysis

We used permutation importance to identify features that degraded performance:

| Feature | Coefficient | Perm. Importance | Decision |
|---------|-------------|------------------|----------|
| `shared_team_weighted_ts_away` | 2.34 | -0.015 | ❌ Exclude |
| `away_usage_share_top2` | 1.56 | -0.008 | ❌ Exclude |
| `shared_implied_home_winprob` | 32.80 | -0.012 | ❌ Exclude |

Removing these features improved validation performance.

### Data Leakage Prevention

✅ No look-ahead bias in league averages  
✅ Realistic player projections (no actual minutes)  
✅ Rolling team stats use only prior games  
✅ Market data frozen at prediction time  
✅ Separate test sets for all evaluation

---

## 📊 Performance Monitoring & Betting Strategy

### For Production Use

1. **Strategy-Based Betting (Recommended)**
   
   **Primary Strategy:** Home Favorites + 8-12 Point Spread
   - Only bet games matching this criteria
   - Expected: 58.88% ATS, 12.41% ROI (based on 24-25)
   - Track actual performance vs. expected
   - Minimum 100 bets before judging strategy success
   
   **Alternative Strategy:** Home Favorites + 5-8 Point Spread
   - Larger sample size (277 games in 24-25)
   - More conservative: 54.51% ATS, 4.06% ROI
   - Good for risk-averse bettors

2. **Performance Tracking**
   
   Track metrics for your chosen strategy over rolling windows:
   - **50-game window:** Tactical adjustments
   - **100-game window:** Strategic evaluation
   - **Full season:** Definitive assessment
   
   Alert triggers:
   - ATS% drops below 50% for 50 consecutive games
   - ROI negative for 100 consecutive games
   - Pattern shift detected (correlation analysis)

3. **Segment Monitoring**
   
   Beyond the core strategy, monitor:
   - Home vs away favorites (overall trends)
   - Spread size distribution changes
   - Back-to-back game performance
   - Conference matchup differentials
   - Month-by-month patterns

4. **Line Value Analysis**
   
   Enhance strategy with timing:
   - Track opening vs closing line movement
   - Best odds typically at line release or close to tip-off
   - Shop lines across multiple sportsbooks
   - Only bet when line matches model criteria

5. **Bankroll Management (Critical)**
   
   Conservative approach for sustainable growth:
   - **Unit sizing:** 1-2% of bankroll per bet
   - **Kelly Criterion:** Use 25-50% fractional Kelly (never full Kelly)
   - **Maximum exposure:** Never more than 10% of bankroll at risk simultaneously
   - **Losing streak protocol:** Reduce bet size by 50% after 10-bet losing streak
   - **Winning streak discipline:** Don't increase bet size beyond 3% even during hot streaks

6. **Model Monitoring**
   
   Watch for degradation or drift:
   - Compare predicted margins to actual margins (MAE)
   - Track calibration (predicted probabilities vs outcomes)
   - Monitor coefficient stability if retraining
   - Alert if 25-26 patterns start resembling anomalous 24-25 patterns

---

## ⚠️ Important Disclaimers

### 1. Small Sample Variance

The current 25-26 season shows 56.94% ATS (72 games), but:
- 95% confidence interval: ±11 percentage points
- Could realistically range from 46% to 68%
- Need 500+ games to stabilize estimates

### 2. Market Efficiency

Betting markets are highly efficient:
- Sharp money moves lines quickly
- Consistent edges are rare and small
- Past performance ≠ future results

### 3. Risk Management

Sports betting carries significant risk:
- Losing streaks happen even with good models
- Variance can be brutal short-term
- Only bet what you can afford to lose
- Consider this educational, not financial advice

### 4. Data Quality

Model performance depends on:
- Accurate injury reports
- Up-to-date statistics
- Reliable market odds
- Timely data updates

---

## 🔮 Future Improvements

### Planned Enhancements

1. **Real-time injury scraping**
   - Automated injury report integration
   - Load management tracking
   - Questionable/Doubtful player modeling

2. **Advanced player projections**
   - Minutes projections using rest patterns
   - Back-to-back adjustments
   - Historical load management

3. **Market timing**
   - Opening vs closing line analysis
   - Optimal bet timing
   - Line shopping across books

4. **Alternative models**
   - **XGBoost variance prediction** (currently only predicts mean)
   - Ensemble combining Ridge + XGBoost
   - Neural network exploration

5. **Enhanced features**
   - Travel distance/time zones
   - Referee assignments
   - Recent performance trends
   - Head-to-head history

---

## 📚 Key Files & Documentation

### Technical Reports
- **[UNIFIED_INJURY_IMPLEMENTATION_SUMMARY.md](UNIFIED_INJURY_IMPLEMENTATION_SUMMARY.md)** - Unified injury handling implementation (Nov 2025)
- **[DATA_LEAKAGE_FIX_FINAL_REPORT.md](DATA_LEAKAGE_FIX_FINAL_REPORT.md)** - Data leakage fix (Oct 2025)
- **[PREDICTION_FILES_GUIDE.md](PREDICTION_FILES_GUIDE.md)** - Guide to analyzing predictions

### Configuration & Model Files
- **[artifacts/champion_corrected_rest_days/](artifacts/champion_corrected_rest_days/)** - Production Champion model (XGBoost, unified injuries)
- **[artifacts/champion_corrected_rest_days/config.yaml](artifacts/champion_corrected_rest_days/config.yaml)** - Champion model configuration

### Prediction Outputs
- **[predictions/OLD_model_2425_predictions.csv](predictions/OLD_model_2425_predictions.csv)** - 24-25 season predictions (strategy discovery)
- **[predictions/OLD_model_2526_predictions.csv](predictions/OLD_model_2526_predictions.csv)** - 25-26 season predictions (current)
- **[predictions/current_season_2025_2026_predictions.csv](predictions/current_season_2025_2026_predictions.csv)** - Live season tracking

### Key Scripts
- **[daily_betting_recommendations_champion.py](daily_betting_recommendations_champion.py)** - Automated daily prediction pipeline
- **[evaluate_current_season_champion.ps1](evaluate_current_season_champion.ps1)** - Current season evaluation
- **[scripts/player_data_loader.py](scripts/player_data_loader.py)** - Injury reconstruction logic
- **[tests/test_injury_reconstruction.py](tests/test_injury_reconstruction.py)** - Injury handling unit tests

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Data source improvements
- Feature engineering ideas
- Model architecture experiments
- Validation methodology enhancements
- Production deployment tools

Please ensure:
- No data leakage in proposed features
- Validation on holdout sets
- Clear documentation
- Reproducible results

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- NBA stats from official team boxscores
- Market data from historical betting lines
- Inspiration from professional sports modeling community
- Ridge regression implementation from scikit-learn

---

## ⚡ Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/nba_predictions.git
cd nba_predictions
pip install -r requirements.txt

# 2. Download sample data (if provided)
# Place Excel files in data/team_boxscores/historical/
# Place Excel files in data/player_boxscores/historical/

# 3. Run complete pipeline
./fix_leakage_and_retrain.ps1  # PowerShell (15-20 min)

# 4. View results
python evaluate_fixed_models.py
```

---

**Last Updated:** November 9, 2025  
**Model Version:** 2.0 (Champion - Unified Injury Handling)  
**Training Data:** 2021-2025 seasons (5,271 games)  
**Model Type:** XGBoost (12 features, injury-optimized)  
**Key Improvement:** Injury features now 21.8% of model importance (was 0%)  
**Status:** Production Ready ✅

---

## 🎓 What Makes This Model Different

1. **Unified Injury Handling:** All pipelines (training, backlook, day-of) use consistent injury reconstruction
2. **Validated Methodology:** Injury features proven meaningful (21.8% of model importance, #1 and #6 features)
3. **Train-Test Consistency:** Eliminated distribution mismatch that caused prediction inconsistencies
4. **Data Quality:** Fixed critical leakage issues and documented every methodological decision
5. **Rigorous Validation:** Indirect validation shows realistic injury patterns (12.8% rate, 56% multi-game)
6. **Production Ready:** Automated daily pipeline with email recommendations

---

*This model is for educational and research purposes. Sports betting carries financial risk. Past performance does not guarantee future results. The identified strategies show promise but require full-season validation. Always bet responsibly and within your means.*
