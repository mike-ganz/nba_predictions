# NBA Metrics Monitoring - Enhanced Features

## Overview
The `monitor_nba_metrics.py` script now provides comprehensive analysis of both **historical deviations** and **intra-season changes** for 8 key NBA metrics.

## Tracked Metrics
1. **FTR** (Free Throw Rate) - FTA / FGA
2. **eFG%** (Effective Field Goal %) - (FGM + 0.5 * 3PM) / FGA
3. **TS%** (True Shooting %) - PTS / (2 * (FGA + 0.44 * FTA))
4. **3P Rate** - 3PA / FGA
5. **3P%** - 3PM / 3PA
6. **Pace** - Possessions per game
7. **PPG** - Points per game
8. **TOV Rate** - Turnover rate

## Two Types of Analysis

### 1. Historical Deviation (vs. Prior Seasons)
**What it tracks:**
- Current season metrics vs. multi-year historical baseline
- Z-scores (standard deviations from historical mean)
- Alert levels: Critical (>2σ), Alert (>1.5σ), Watch (>1σ), Normal

**Example Output:**
```
Free Throw Rate
  Current: 28.49%  |  Baseline: 25.03%  |  Diff: +3.46%
  Z-Score: +3.99 σ
  Status: 🚨 CRITICAL
```

### 2. Intra-Season Changes (Current Season Evolution)
**What it tracks:**
- Early season baseline (first 20-25 games)
- Current values and progression
- Season-long change and % change
- Recent trend direction (increasing/decreasing/stable)
- Moving averages (3-game and 5-game)
- Acceleration/deceleration patterns
- Inflection points (significant shifts during season)

**Example Output:**
```
📊 Free Throw Rate
   Early Season (10/30, 72 games): 29.79%
   Current (11/23, 250 games): 28.49%
   Change: -1.30% (-4.4%)
   Trajectory: Accelerating upward
   ⚡ Moderate intra-season change (>2%)
```

## Enhanced Analysis Features

### Season Progression Milestones
Shows how metrics evolved at key stages:
- Early season (≤25 games)
- Mid season (≤82 games / 1 month)
- Recent (≤164 games / 2 months)
- Current (most recent data)

### Dual-Risk Detection
Identifies metrics with **BOTH**:
- Historical deviation (high z-score)
- Intra-season instability (significant % change)

These represent the highest-risk areas for model performance.

### Trend Analysis
- **Recent Slope**: Linear regression on last 10 snapshots
- **Acceleration**: Comparing first-half vs. second-half trends
- **Volatility**: Standard deviation of recent values
- **Inflection Points**: Dates when significant shifts occurred

## Real Insights from 2025-26 Season

### Critical Findings (as of 11/23/2025):

**Historical Deviations:**
1. **Pace**: +7.04 poss/game (+5.35σ) - Highest alert
2. **Turnover Rate**: +0.80% (+4.43σ) 
3. **Free Throw Rate**: +3.46% (+3.99σ)
4. **Points Per Game**: +8.54 pts (+2.66σ)

**Intra-Season Changes:**
1. **FTR**: Started at 29.79%, now at 28.49% (-4.4% change)
   - Both historically elevated AND declining within season
   - Highest-risk metric for models
2. **Turnover Rate**: -2.8% from early season
3. **Pace**: -0.9% from early season (but still elevated vs. history)

**Key Insight:**
Free Throw Rate shows **dual-risk profile**:
- 3.99σ above historical baseline (regime change)
- -4.4% decline within current season (unstable)
- This suggests FTR spiked early, now normalizing but still elevated

## Usage Examples

### Basic Run
```bash
python monitor_nba_metrics.py
```

### Detailed Analysis
```bash
python monitor_nba_metrics.py --verbose
```
Shows:
- Historical data by season
- Early season baselines
- Moving averages
- Recent ranges
- Inflection points

### Custom Alert Threshold
```bash
python monitor_nba_metrics.py --alert-threshold 1.5
```

### Skip Charts
```bash
python monitor_nba_metrics.py --no-charts
```

## Output Files

### 1. Visual Charts
- **Location**: `analysis/metrics_progression.png`
- **Shows**: All metrics over time with historical baselines and std dev bands
- **Format**: Multi-panel visualization

### 2. CSV Export
- **Location**: `analysis/metrics_analysis_YYYYMMDD.csv`
- **Contains**: Full analysis data including:
  - Historical comparisons
  - Intra-season analysis (nested in JSON)
  - All calculated metrics

## Comparison with FTR Script

| Feature | FTR Script | Metrics Script |
|---------|-----------|----------------|
| Single metric focus | ✅ FTR only | ❌ |
| Multiple metrics | ❌ | ✅ 8 metrics |
| Model recommendation | ✅ | ❌ |
| Historical baseline | ✅ | ✅ |
| Intra-season tracking | ✅ | ✅ (Enhanced) |
| Moving averages | ✅ | ✅ |
| Acceleration detection | ❌ | ✅ |
| Inflection points | ❌ | ✅ |
| Season milestones | ❌ | ✅ |
| Dual-risk detection | ❌ | ✅ |

## Monitoring Strategy

### Weekly Workflow
1. Run `monitor_nba_metrics.py --verbose`
2. Review alert summary
3. Check dual-risk metrics (highest priority)
4. Compare to previous week's CSV export
5. Document any external factors (rule changes, etc.)

### When to Act
- **Critical (>2σ) + Intra-season instability**: Model retraining required
- **Critical (>2σ) + Stable within season**: Monitor but may not need immediate action
- **Alert (>1.5σ)**: Test model performance, prepare contingencies
- **Watch (>1σ)**: Normal monitoring

### Model Impact
Metrics with dual-risk profile (like current FTR) should trigger:
- Feature importance re-evaluation
- Model performance testing on recent games
- Potential model switching (use FTR script for recommendation)
- Ensemble approaches to hedge uncertainty

## Future Enhancements
Potential additions:
- Per-team metric tracking (identify team-level regime changes)
- Correlation analysis (which metrics move together?)
- Predictive alerts (forecast when metric will breach threshold)
- Integration with model performance tracking
- Automatic report generation and email alerts

