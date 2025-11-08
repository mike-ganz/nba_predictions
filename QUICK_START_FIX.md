# Rest Days Fix - Quick Start

## 🎯 TL;DR

**Bug:** Rest days were +1 too high everywhere  
**Impact:** Training data and evaluations used wrong rest days  
**Fix:** ✅ Applied to `generate_team_stats.py`  
**Action:** Regenerate data and optionally retrain model

---

## 🚀 Quick Start (Choose One)

### Option A: Fast Validation (2 minutes) ⚡
```powershell
.\fix_rest_days_and_retrain.ps1 -QuickMode
```
- Fixes current season evaluation only
- Uses existing model
- Quick validation of the fix

### Option B: Full Fix (30 minutes) 🔄
```powershell
.\fix_rest_days_and_retrain.ps1
```
- Regenerates all training data
- Retrains Champion model  
- Complete consistency

---

## 📋 Commands Cheat Sheet

### Verify the Fix Works
```powershell
python verify_rest_days_fix.py
```

### Quick Mode (Current Season Only)
```powershell
.\fix_rest_days_and_retrain.ps1 -QuickMode
```

### Full Regeneration
```powershell
.\fix_rest_days_and_retrain.ps1
```

### Regenerate Data, Keep Model
```powershell
.\fix_rest_days_and_retrain.ps1 -SkipRetrain
```

### Skip Backup (Not Recommended)
```powershell
.\fix_rest_days_and_retrain.ps1 -SkipBackup
```

---

## 🔍 Check Results

### View Corrected Evaluation
```powershell
cat predictions/current_season_champion_corrected_evaluation.txt
```

### Compare Before/After
```powershell
# Old (buggy)
cat predictions/current_season_champion_evaluation.txt

# New (corrected)
cat predictions/current_season_champion_corrected_evaluation.txt
```

### Verify Rest Days
```python
import json
with open('data/games_2025_2026_current_norm.jsonl') as f:
    for line in f:
        game = json.loads(line)
        if game['date'] == '2025-11-07':
            teams = game['teams']
            print(f"{game['game_id']}: Away={teams['A']['rest_days']}, Home={teams['H']['rest_days']}")
```

---

## 📊 What Changed

### Before Fix ❌
```
Nov 5 → Nov 7 = 2 days (WRONG!)
- Calendar days: 2
- Rest days calculated: 2
- Should be: 1
```

### After Fix ✅
```
Nov 5 → Nov 7 = 1 day (CORRECT!)
- Calendar days: 2
- Rest days calculated: 2 - 1 = 1
- Matches NBA schedule ✓
```

---

## 🎯 Expected Results

### Rest Days
- All -1 from previous values
- Match NBA schedule data
- Back-to-backs now detected

### Model Performance
- May change ±1-2%
- More accurate baseline
- Better feature consistency

### Predictions
- Daily predictions unchanged (were already correct)
- Evaluation predictions now consistent with daily

---

## 📁 Important Files

### Modified Code
- ✅ `generate_team_stats.py` - Bug fix applied

### Regenerated Data (After Running Script)
- `data/games_train_with_players_90_norm.jsonl` - Training data
- `data/games_2025_2026_current_norm.jsonl` - Current season
- `predictions/current_season_champion_corrected_predictions.csv` - New predictions

### New Model (If Full Retrain)
- `artifacts/champion_corrected_rest_days/` - Retrained model

### Backups
- `backups/pre_rest_days_fix_YYYYMMDD_HHMMSS/` - Original files

---

## ⚠️ Important Notes

1. **Daily predictions were always correct** - They use NBA schedule data
2. **Training data needs regeneration** - Model learned from wrong data
3. **Current season evaluation was wrong** - Now fixed
4. **Backup is automatic** - Original files are preserved
5. **Performance may change** - But will be more accurate

---

## 🆘 Help

### Script Fails
```powershell
# Check Python is working
python --version

# Check files exist
ls data/team_boxscores/historical/
ls data/team_boxscores/current/

# Run verification
python verify_rest_days_fix.py
```

### Need More Info
```powershell
# Read detailed guide
cat REST_DAYS_FIX_README.md

# View comparison
cat FINDINGS_rest_days_discrepancy.md

# See bug details
cat BUG_REPORT_rest_days_calculation.md
```

### Rollback to Original
```powershell
# Find your backup
ls backups/

# Copy from backup (replace TIMESTAMP)
$backup = "backups/pre_rest_days_fix_TIMESTAMP"
Copy-Item "$backup/champion_individually_tuned" artifacts/ -Recurse -Force
Copy-Item "$backup/*.jsonl" data/ -Force
```

---

## ✅ Success Checklist

After running the fix script:

- [ ] Script completed without errors
- [ ] Backup created in `backups/` directory
- [ ] Rest days decreased by 1 (verified with `verify_rest_days_fix.py`)
- [ ] Back-to-back games now detected
- [ ] Evaluation results look reasonable
- [ ] New model trained (if full regeneration)

---

**Ready?** Start with Quick Mode:

```powershell
.\fix_rest_days_and_retrain.ps1 -QuickMode
```

Then review results and run full regeneration when ready! 🚀

