# Hyperparameter Tuning Analysis

## 🚨 Current State: Suboptimal Configuration

### Evidence of Poor Tuning:

1. **Tuned on OLD Feature Set**
   - Tuning happened before `home_ftr` and role indicators were added
   - Feature set has changed significantly
   - Hyperparameters optimized for different features won't be optimal for new features

2. **Wrong Optimization Target**
   - Tuned using **MSE/RMSE** (margin error)
   - You care about **ATS Accuracy** (spread prediction)
   - These are different objectives!

3. **Small Search Space**
   - Only **50 random iterations**
   - From tuning script, full space has 5×5×5×5×5×2×5 = **15,625 combinations**
   - Searched only **0.32%** of space

4. **Very Conservative Hyperparameters**
   ```yaml
   max_depth: 2          # Extremely shallow trees
   n_estimators: 100     # Relatively few trees
   learning_rate: 0.01   # Very slow learning
   subsample: 0.6        # Only 60% of data per tree
   colsample_bytree: 0.7 # Only 70% of features per tree (causing redundancy issue!)
   ```

5. **Current Performance Suggests Undertraining**
   - Best CV RMSE: 13.24 points
   - Using redundant features as a crutch
   - colsample=1.0 models got 56.30% ATS (vs 60.50% with colsample=0.7 and redundancy)

---

## 🎯 What Needs to Be Retuned

### Critical Issues:

1. **`colsample_bytree = 0.7` is causing problems**
   - Forces model to rely on redundant features
   - Reduces information availability
   - Should be tuned higher (0.8, 0.9, or 1.0)

2. **`max_depth = 2` is very shallow**
   - Can only capture 2-level interactions
   - NBA betting has complex interactions (home underdogs, rest days × pace, etc.)
   - Should explore 3-5

3. **`n_estimators = 100` might be insufficient**
   - With learning_rate=0.01, need many trees to converge
   - Should explore 200-500

4. **Wrong objective function**
   - Currently optimizing MSE
   - Should optimize ATS accuracy

---

## 📋 Recommended Retuning Plan

### Phase 1: Quick Diagnostic (1-2 hours)
**Goal:** See if better hyperparameters exist without massive compute

**Feature Set:** Exp2 (home_ftr + role indicators, NO redundancy)
**Search Method:** Random search, 100 iterations
**Objective:** MSE (for speed, can validate ATS after)
**Search Space:**
```python
{
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],  # Focus on higher values
    'reg_alpha': [0.0, 0.1, 0.5, 1.0],
    'reg_lambda': [1.0, 5.0, 10.0, 20.0],
}
```

**Expected Outcome:** Find if we're leaving easy gains on the table

---

### Phase 2: Thorough Retuning (if Phase 1 shows promise)
**Goal:** Find truly optimal hyperparameters

**Feature Set:** Best from Exp1/Exp2/Exp3 (likely Exp2)
**Search Method:** Random search, 200-300 iterations OR Bayesian optimization
**Objective:** Custom metric combining ATS accuracy + calibration
**Search Space:** Expanded based on Phase 1 results

---

### Phase 3: Final Validation
1. Train final model with best hyperparameters
2. Test on 2024-25 holdout
3. Test on 2025-26 current season
4. Compare to current "champion"
5. Deploy if better

---

## 🔬 Key Questions to Answer

1. **Can we match 60.50% ATS (2025-26) WITHOUT redundant features?**
   - Current: 60.50% with colsample=0.7 + redundancy
   - With colsample=1.0 + no redundancy: only 56.30%
   - Gap of 4.2 percentage points to close

2. **Is `max_depth=2` limiting us?**
   - Very shallow trees = simple model
   - Complex betting markets might need deeper trees

3. **Is `n_estimators=100` enough?**
   - With learning_rate=0.01, might need 300-500 trees

4. **Should we optimize ATS directly instead of RMSE?**
   - Previous ATS loss experiment showed mixed results
   - But that was with suboptimal hyperparameters

---

## 💰 Expected Benefits

**If tuning finds better hyperparameters:**
- ✅ Remove need for redundant features
- ✅ Cleaner, more interpretable model
- ✅ Better ATS accuracy (potentially 62-65% on 2025-26)
- ✅ More confidence in predictions
- ✅ Faster inference (fewer redundant features)

**If tuning finds similar hyperparameters:**
- ✅ Confirmation current setup is near-optimal
- ✅ Peace of mind that we're not leaving easy gains
- ✅ Better understanding of model limitations

---

## 🎬 Recommendation: START WITH PHASE 1

**Action Items:**
1. Run quick hyperparameter search (100 iterations, ~1-2 hours)
2. Use Exp2 feature set WITHOUT away_tov_edge
3. Focus on higher colsample_bytree values
4. Check if we can match/beat current performance

**Why Phase 1 First:**
- Low cost (1-2 hours compute)
- High information value
- Can decide if Phase 2 is worth it

**Next Steps Based on Phase 1:**
- If finds >2pp improvement → Do Phase 2 (thorough tuning)
- If finds <2pp improvement → Keep current hyperparameters, accept redundancy
- If finds no improvement → Current hyperparameters are near-optimal

---

## 📊 Success Criteria

**Phase 1 Success:**
- ATS% on 2025-26 ≥ 58% (currently 56.30% with colsample=1.0, no redundancy)
- ATS% on 2024-25 ≥ 50% (currently 50.27%)

**Phase 2 Success (if needed):**
- ATS% on 2025-26 ≥ 60% (match current with redundancy)
- ATS% on 2024-25 ≥ 51%
- Cleaner model without redundant features

---

## 🤔 Bottom Line

**You're absolutely right to question this.** The hyperparameters were:
1. Tuned on an old feature set
2. Optimized for the wrong metric
3. Searched only 0.32% of the hyperparameter space
4. Resulted in a model that needs redundant features as a crutch

**This screams "retune me!"**

The good news: You have the tuning script already. We just need to:
1. Use the NEW feature set (Exp2 without redundancy)
2. Focus search on higher colsample_bytree values
3. Explore deeper trees (max_depth 3-5)
4. Validate on ATS accuracy

**Want me to run Phase 1 (quick diagnostic tuning)?**

