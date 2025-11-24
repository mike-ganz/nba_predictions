# Model Retraining Experiments Guide

## 🎯 Objective

Test 4 model variants against current champion to determine if:
1. Adding 2024-2025 season improves predictions
2. Including FTR features improves predictions (despite regime changes)
3. Combination of both helps

---

## 📊 The 4 Experiments

| # | Features | Training Data | Config File | Description |
|---|----------|---------------|-------------|-------------|
| **1** | 14 (no FTR) | 2021-2024 | `configs/experiment_2124_no_ftr.yaml` | **Baseline** - Current champion replica |
| **2** | 14 (no FTR) | 2021-2025 | `configs/experiment_2125_no_ftr.yaml` | Test recent data impact |
| **3** | 16 (+FTR) | 2021-2024 | `configs/experiment_2124_with_ftr.yaml` | Test FTR value |
| **4** | 16 (+FTR) | 2021-2025 | `configs/experiment_2125_with_ftr.yaml` | **Full experiment** |

**Current Champion:** 14 features (no FTR), trained on 2021-2024, ~54.22% ATS on 2025-26

---

## 🚀 How to Run

### Option 1: Full Run (Recommended first time)

```powershell
.\run_model_experiments.ps1
```

This will:
1. Generate training data for 2021-2024 and 2021-2025
2. Normalize data
3. Train all 4 models
4. Generate predictions on 2025-26 season
5. Compare all models

**Expected time:** ~30-45 minutes total

---

### Option 2: Skip Data Prep (If you've already run once)

```powershell
.\run_model_experiments.ps1 -SkipDataPrep -SkipNormalization
```

Useful for re-running experiments with different hyperparameters

**Expected time:** ~15-20 minutes

---

### Option 3: Just Evaluation (If models already trained)

```powershell
.\run_model_experiments.ps1 -SkipDataPrep -SkipNormalization -SkipTraining
```

Regenerates predictions and comparison

**Expected time:** ~2-3 minutes

---

## 📁 Output Files

### Training Data
- `data/experiments/games_train_2021_2024_norm.jsonl` - 3 seasons (21-24)
- `data/experiments/games_train_2021_2025_norm.jsonl` - 4 seasons (21-25)

### Model Artifacts
- `artifacts/experiments/exp1_2124_no_ftr/` - Experiment 1 model
- `artifacts/experiments/exp2_2125_no_ftr/` - Experiment 2 model
- `artifacts/experiments/exp3_2124_with_ftr/` - Experiment 3 model
- `artifacts/experiments/exp4_2125_with_ftr/` - Experiment 4 model

### Predictions
- `predictions/experiments/exp1_predictions.csv` - Experiment 1 predictions
- `predictions/experiments/exp2_predictions.csv` - Experiment 2 predictions
- `predictions/experiments/exp3_predictions.csv` - Experiment 3 predictions
- `predictions/experiments/exp4_predictions.csv` - Experiment 4 predictions

### Results
- `predictions/experiments/comparison_results.csv` - **Main results table**

---

## 📊 Expected Output

The script will generate a comparison table:

```
MODEL COMPARISON - 2025-26 SEASON PERFORMANCE
================================================================================

Model                      Games  MAE     RMSE   ATS_Correct  ATS_Pct
Current Champion           249    10.15   13.06  135          54.22%
Exp1: 14feat, 21-24        249    10.20   13.10  140          56.23%
Exp2: 14feat, 21-25        249    10.05   12.95  145          58.23%
Exp3: 16feat+FTR, 21-24    249    10.30   13.20  138          55.42%
Exp4: 16feat+FTR, 21-25    249    9.95    12.80  148          59.44%

(These are example numbers - actual results will vary)
```

---

## 🔍 How to Interpret Results

### Key Questions to Answer:

#### 1. Does adding 2024-2025 season help?
**Compare:** Exp1 (2021-2024) vs Exp2 (2021-2025)

- If Exp2 has **higher ATS%**: Recent data helps! The 2024-25 season captures new patterns
- If Exp2 has **lower ATS%**: Recent data hurts! The 2024-25 regime shift contaminates training

#### 2. Does including FTR help?
**Compare:** Exp1 (no FTR) vs Exp3 (with FTR)

- If Exp3 has **higher ATS%**: FTR adds value despite regime changes
- If Exp3 has **lower ATS%**: FTR is noise, exclusion was correct

#### 3. What's the best combination?
**Compare all 4** - Look for highest ATS%

- Best = New champion candidate
- Check if improvement is meaningful (>2 percentage points)

#### 4. Does current champion still win?
**Compare:** Current Champion vs all experiments

- If champion still best: Keep current model
- If challenger wins by >2 points: Promote to champion

---

## ⚡ Quick Analysis Commands

### Check how many games in each training set:

```powershell
python -c "import json; print(f'2021-2024: {sum(1 for _ in open(\"data/experiments/games_train_2021_2024_norm.jsonl\", encoding=\"utf-8\"))} games'); print(f'2021-2025: {sum(1 for _ in open(\"data/experiments/games_train_2021_2025_norm.jsonl\", encoding=\"utf-8\"))} games')"
```

### Check feature count for each model:

```powershell
python train_margin.py --data data/experiments/games_train_2021_2024_norm.jsonl --config configs/experiment_2124_no_ftr.yaml --output artifacts/temp_test
```

(Look at "Feature dimensionality" in output)

---

## 🎓 Feature Breakdown

### 14 Features (No FTR) - Experiments 1 & 2:

**Home/Away (12 = 6 per team):**
- `edge` - OEFF vs opponent DEFF
- `orb_edge` - ORr vs opponent DRr
- `tov_edge` - TOr differential
- `tpar` - 3PAr
- `rest_days` - Days of rest
- Injury features (3): `minutes_missing_top2`, `star_out`, `usage_share_top2`

**Shared (2):**
- `pace_mean`, `pace_diff`

### 16 Features (+FTR) - Experiments 3 & 4:

All 14 above, PLUS:
- `home_ftr` - Home team Free Throw Rate (normalized)
- `away_ftr` - Away team Free Throw Rate (normalized)

---

## 🚨 What to Watch For

### Red Flags:
- **Exp2/Exp4 perform WORSE than Exp1/Exp3**: 2024-25 season contaminating training with regime shift
- **All experiments worse than champion**: Hyperparameters need tuning
- **FTR helps with 2021-2024 but hurts with 2021-2025**: FTR regime change confirmed

### Green Flags:
- **Exp2 > Exp1**: Recent data helps, model adapting to new regime
- **Exp4 best overall**: Both improvements stack positively
- **Any experiment >57% ATS**: Significant improvement over champion

---

## 📈 Next Steps After Results

### If a challenger wins:

1. **Run drift detection on winner:**
   ```powershell
   python detect_model_drift.py --predictions predictions/experiments/exp[X]_predictions.csv
   ```

2. **Check rolling performance:**
   - Does it maintain advantage throughout season?
   - Or just early/late season spike?

3. **Promote to champion:**
   - Copy winning model to `artifacts/champion_v3/`
   - Update production scripts to use new model
   - Document changes in model registry

### If current champion wins:

- No immediate action needed
- Consider hyperparameter tuning on experiments
- Maybe try intermediate approaches (e.g., weighted average of historical seasons)

---

## 🔧 Troubleshooting

### "Missing required columns" error:
Check that training data has all features. Re-run data preparation.

### "Model failed to converge":
- Try different hyperparameters
- Check for NaN values in training data
- Reduce feature count

### "Predictions file not found":
Run with `-SkipEvaluation:$false` to regenerate predictions

### Very long training time (>1 hour):
- Check training data size (should be ~3500-4500 games)
- Reduce `n_estimators` in config files
- Use `-SkipDataPrep` if data already exists

---

## 💡 Advanced Usage

### Test different hyperparameters:

Edit the config files (`configs/experiment_*.yaml`) and modify:
- `n_estimators` - Number of trees (more = better fit, slower)
- `max_depth` - Tree depth (deeper = more complex, overfitting risk)
- `learning_rate` - Step size (lower = more stable, slower)

Then re-run:
```powershell
.\run_model_experiments.ps1 -SkipDataPrep -SkipNormalization
```

### Train only one experiment:

```powershell
python train_margin.py --data data/experiments/games_train_2021_2025_norm.jsonl --config configs/experiment_2125_with_ftr.yaml --output artifacts/experiments/exp4_2125_with_ftr --model-type xgboost
```

### Generate predictions only:

```powershell
python predict_margin.py --model artifacts/experiments/exp4_2125_with_ftr --data data/games_2025_2026_current_norm.jsonl --output predictions/experiments/exp4_predictions.csv
```

---

## ✅ Checklist

Before running experiments:

- [ ] Current season data processed (`data/games_2025_2026_current_norm.jsonl` exists)
- [ ] Historical boxscores in `data/team_boxscores/historical/` (5 seasons)
- [ ] Player boxscores in `data/player_boxscores/historical/` (5 seasons)
- [ ] Current champion predictions exist for comparison
- [ ] ~30-45 minutes available for full run
- [ ] ~20GB disk space available for all artifacts

After experiments complete:

- [ ] Review `comparison_results.csv`
- [ ] Check if any model beats champion by >2 percentage points
- [ ] Run drift detection on best performer
- [ ] Document findings
- [ ] Decide on promotion to champion

