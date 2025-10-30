# NBA Betting Strategy Analysis - Complete Results
## 2024-2025 Season Performance

**⚠️ CORRECTED VERSION - All numbers verified against actual analysis outputs**

---

## Table of Contents
1. [Definitions & Terminology](#definitions--terminology)
2. [Against The Spread (ATS) Analysis](#against-the-spread-ats-analysis)
3. [Moneyline (ML) Analysis](#moneyline-ml-analysis)
4. [Key Insights & Recommendations](#key-insights--recommendations)

---

## Definitions & Terminology

### Favorite Type (Based on Market Spread)
- **Home Favored**: Market spread > 0 (e.g., home is -5.5 point favorite)
- **Away Favored**: Market spread < 0 (e.g., home is +5.5 point underdog, away team favored)
- **Pick'em**: Market spread = 0 (even matchup)

### Model Prediction
- **Pick Home to Cover (ATS)**: Model predicts home team will cover the spread (cover_prob_home > 50%)
- **Pick Away to Cover (ATS)**: Model predicts away team will cover the spread (cover_prob_away > 50%)
- **Pick Home to Win (ML)**: Model predicts home team will win outright (win_prob_home > 50%)
- **Pick Away to Win (ML)**: Model predicts away team will win outright (win_prob_away > 50%)

### Spread Size Buckets (ATS Analysis)
- **Tiny Spread**: |spread| < 2.5 points
- **Small Spread**: 2.5 ≤ |spread| < 5.5 points
- **Medium Spread**: 5.5 ≤ |spread| < 8.5 points
- **Large Spread**: |spread| ≥ 8.5 points

### Moneyline Odds Buckets (for the team the model picks)
- **Heavy Fav**: Odds ≤ -200 (need to bet $200 to win $100; ~67% implied probability)
- **Solid Fav**: Odds -150 to -200 (need to bet $150-200 to win $100; ~60-67% implied)
- **Slight Fav**: Odds -110 to -150 (need to bet $110-150 to win $100; ~52-60% implied)
- **Pick'em**: Odds -110 to +110 (roughly even money; ~48-52% implied)
- **Slight Dog**: Odds +110 to +150 (bet $100 to win $110-150; ~40-48% implied)
- **Solid Dog**: Odds +150 to +200 (bet $100 to win $150-200; ~33-40% implied)
- **Heavy Dog**: Odds > +200 (bet $100 to win $200+; <33% implied)

### Model Confidence (ATS Analysis)
- **Low**: 50-55% cover probability
- **Medium**: 55-65% cover probability  
- **High**: 65-75% cover probability
- **Very High**: 75%+ cover probability

### Model Confidence (ML Analysis)
- **70%+**: Model is very confident in its pick
- **60-70%**: Model is moderately confident
- **55-60%**: Model is somewhat confident
- **50-55%**: Model is barely confident (close to 50/50)

### Metrics

#### ATS Metrics
- **Games**: Number of games in this category
- **ATS%**: Against The Spread accuracy (correct picks / total games * 100)
- **ROI**: Return on Investment assuming standard -110 odds on all bets
  - Formula: (Wins * 0.909 - Losses) / Games * 100
  - Need 52.4% ATS to break even
  - Positive ROI = profitable, Negative ROI = losing money

#### ML Metrics
- **Games**: Number of games in this category
- **Wins**: Number of games where model's ML pick was correct (team won)
- **Win %**: Percentage of correct picks (Wins / Games * 100)
- **Total Profit**: Total profit/loss betting $100 per game using actual moneyline odds
- **ROI%**: Return on Investment percentage = (Total Profit / Total Wagered) * 100

### Important Notes
- **ATS Betting**: Standard -110 odds mean you need 52.4% accuracy to break even
- **ML Betting**: High win % does NOT guarantee profit! Favorites require very high accuracy due to low payouts (e.g., -400 needs 80% win rate to break even)
- **Underdogs**: Can be profitable with <50% win rate due to high payouts (e.g., +200 needs only 34% win rate to break even)

---

## Against The Spread (ATS) Analysis

### Overall ATS Performance
- **Total Games**: 1,315
- **ATS Accuracy**: 51.71%
- **ROI**: -1.23%
- **Status**: Slightly unprofitable overall, but strong edges exist in specific segments

---

### Table 1: By Favorite Type (Market Spread)

| Favorite Type | Games | ATS% | ROI |
|--------------|-------|------|-----|
| Home Favored | 529 (40.2%) | 41.40% | -20.93% |
| Away Favored | 786 (59.8%) | 58.65% | +12.02% |

**Key Finding**: Massive edge on away favorites (+12.02% ROI), home favorites are a disaster (-20.93% ROI).

---

### Table 2: By Model Prediction

| Model Picks | Games | ATS% | ROI |
|------------|-------|------|-----|
| Home to Cover | 608 (46.2%) | 50.66% | -3.24% |
| Away to Cover | 707 (53.8%) | 52.62% | +0.50% |

**Key Finding**: Slight edge picking away to cover.

---

### Table 3: By Spread Size

| Spread Size | Games | ATS% | ROI |
|------------|-------|------|-----|
| Tiny (< 2.5) | 155 (11.8%) | 45.81% | -12.51% |
| Small (2.5-5.5) | 373 (28.4%) | 46.11% | -11.92% |
| Medium (5.5-8.5) | 351 (26.7%) | 55.84% | +6.66% |
| Large (≥ 8.5) | 436 (33.2%) | 55.28% | +5.58% |

**Key Finding**: Model struggles with small spreads but excels with medium/large spreads.

---

### Table 4: By Confidence Level

| Confidence Level | Games | ATS% | ROI |
|-----------------|-------|------|-----|
| Low (50-55%) | 345 (26.2%) | 54.78% | +4.63% |
| Medium (55-65%) | 661 (50.3%) | 49.77% | -4.93% |
| High (65-75%) | 263 (20.0%) | 51.71% | -1.23% |
| Very High (75%+) | 46 (3.5%) | 56.52% | +7.96% |

**Key Finding**: **Confidence inversion** - Low confidence games are profitable! Very high confidence also works.

---

### Table 5: Favorite Type × Model Prediction

| Strategy | Games | ATS% | ROI |
|----------|-------|------|-----|
| Home Fav + Pick Home | 231 (17.6%) | 38.53% | -26.41% |
| Home Fav + Pick Away | 298 (22.7%) | 43.62% | -16.68% |
| **Away Fav + Pick Home** | **377 (28.7%)** | **58.09%** | **+10.95%** |
| **Away Fav + Pick Away** | **409 (31.1%)** | **59.17%** | **+13.01%** |

**Key Finding**: 
- Both "Away Fav" strategies are highly profitable
- Home favorites are unprofitable regardless of model pick

---

### Table 6: Favorite Type × Spread Size

| Strategy | Games | ATS% | ROI |
|----------|-------|------|-----|
| Home Fav + Small Spread | 173 (13.2%) | 43.93% | -16.09% |
| Home Fav + Med Spread | 140 (10.6%) | 40.71% | -22.24% |
| Home Fav + Large Spread | 137 (10.4%) | 39.42% | -24.72% |
| Away Fav + Small Spread | 200 (15.2%) | 48.00% | -8.32% |
| **Away Fav + Med Spread** | **211 (16.0%)** | **65.88%** | **+25.82%** |
| **Away Fav + Large Spread** | **299 (22.7%)** | **62.54%** | **+19.45%** |

**Key Finding**: 
- **Away Fav + Med Spread** = +25.82% ROI (best overall strategy!)
- **Away Fav + Large Spread** = +19.45% ROI (high volume)
- Home favorites are unprofitable at ALL spread sizes

---

### Table 7: Top 20 Strategies by ROI (Min 20 Games)

| Rank | Strategy | Games | ATS% | ROI |
|------|----------|-------|------|-----|
| 1 | **Away Fav + Pick Away + Med Spread** | 61 | 67.21% | +28.38% |
| 2 | **Away Fav + Pick Home + Large Spread** | 89 | 66.29% | +26.62% |
| 3 | **Away Fav + Med Spread** | 211 | 65.88% | +25.82% |
| 4 | **Away Fav + Large Spread** | 299 | 62.54% | +19.45% |
| 5 | Away Fav + Pick Away + Large Spread | 60 | 61.67% | +17.78% |
| 6 | Away Fav + Pick Home + Med Spread | 38 | 60.53% | +15.61% |
| 7 | **Away Fav + Pick Away** | **409** | **59.17%** | **+13.01%** |
| 8 | Pick Away + Very High Conf | 39 | 58.97% | +12.64% |
| 9 | **Away Favored (overall)** | **786** | **58.65%** | **+12.02%** |
| 10 | Away Fav + Pick Away + Small Spread | 67 | 58.21% | +11.18% |
| 11 | **Away Fav + Pick Home** | **377** | **58.09%** | **+10.95%** |
| 12 | Very High Conf (75%+) | 46 | 56.52% | +7.96% |
| 13 | Medium Spread | 351 | 55.84% | +6.66% |
| 14 | Large Spread | 436 | 55.28% | +5.58% |
| 15 | Low Conf (50-55%) | 345 | 54.78% | +4.63% |
| 16 | Pick Away | 707 | 52.62% | +0.50% |
| 17 | High Conf (65-75%) | 263 | 51.71% | -1.23% |
| 18 | Pick Home + High Conf | 102 | 50.98% | -2.63% |
| 19 | Pick Home | 608 | 50.66% | -3.24% |
| 20 | Medium Conf (55-65%) | 661 | 49.77% | -4.93% |

---

## Moneyline (ML) Analysis

### Overall ML Performance
- **Total Games**: 1,315
- **Wins**: 901 (68.52%)
- **Total Profit**: -$5,525
- **ROI**: -4.20%
- **Average ML Odds**: -369
- **Status**: Unprofitable overall (model picks too many heavy favorites)

---

### Table 1: By Favorite Type (Market)

| Favorite Type | Games | Wins | Win % | Total Profit | ROI |
|--------------|-------|------|-------|--------------|-----|
| Home Favored | 529 | 349 | 65.97% | -$2,943 | -5.56% |
| Away Favored | 786 | 552 | 70.23% | -$2,581 | -3.28% |

**Key Finding**: Even with 70% win rate on away favorites, still unprofitable due to heavy favorite odds.

---

### Table 2: By Model Prediction

| Model Picks | Games | Wins | Win % | Total Profit | ROI |
|------------|-------|------|-------|--------------|-----|
| Home to Win | 776 | 540 | 69.59% | -$3,977 | -5.13% |
| Away to Win | 539 | 361 | 66.98% | -$1,548 | -2.87% |

**Key Finding**: Higher win % picking home, but both unprofitable.

---

### Table 3: By Moneyline Odds Bucket (Model's Pick)

| ML Odds Bucket | Games | Wins | Win % | Avg Odds | Total Profit | ROI |
|---------------|-------|------|-------|----------|--------------|-----|
| Heavy Fav (< -200) | 814 | 612 | 75.18% | -519 | -$4,584 | -5.63% |
| Solid Fav (-150 to -200) | 241 | 150 | 62.24% | -176 | -$512 | -2.13% |
| Slight Fav (-110 to -150) | 187 | 106 | 56.68% | -131 | $14 | +0.07% |
| Pick'em (-110 to +110) | 44 | 22 | 50.00% | -7 | -$50 | -1.14% |
| Slight Dog (+110 to +150) | 23 | 10 | 43.48% | +118 | -$92 | -4.00% |
| Solid Dog (+150 to +200) | 5 | 0 | 0.00% | +166 | -$500 | -100.00% |

**Key Finding**: 75% win rate on heavy favorites STILL loses money (-5.63% ROI)!

---

### Table 4: By Model Confidence Level

| Confidence Level | Games | Wins | Win % | Total Profit | ROI |
|-----------------|-------|------|-------|--------------|-----|
| 70%+ Confidence | 960 | 683 | 71.15% | -$6,761 | -7.04% |
| 60-70% Confidence | 189 | 117 | 61.90% | -$58 | -0.31% |
| **55-60% Confidence** | **84** | **51** | **60.71%** | **$398** | **+4.74%** |
| **50-55% Confidence** | **82** | **50** | **60.98%** | **$896** | **+10.93%** |

**Key Finding**: **EXTREME CONFIDENCE INVERSION** - Lower confidence = higher profit!

---

### Table 5: Favorite Type × Model Prediction

| Strategy | Games | Wins | Win % | Total Profit | ROI |
|----------|-------|------|-------|--------------|-----|
| Home Fav + Pick Home | 29 | 9 | 31.03% | -$955 | -32.94% |
| Home Fav + Pick Away | 500 | 340 | 68.00% | -$1,988 | -3.98% |
| Away Fav + Pick Home | 747 | 531 | 71.08% | -$3,022 | -4.05% |
| **Away Fav + Pick Away** | **39** | **21** | **53.85%** | **$440** | **+11.29%** |

**Key Finding**: Away Fav + Pick Away is the ONLY profitable 2-way combo!

---

### Table 6: Favorite Type × ML Odds Bucket

| Strategy | Games | Wins | Win % | Total Profit | ROI |
|----------|-------|------|-------|--------------|-----|
| Home Fav + Heavy Fav | 290 | 218 | 75.17% | -$1,229 | -4.24% |
| Home Fav + Solid Fav | 115 | 68 | 59.13% | -$868 | -7.55% |
| Home Fav + Slight Fav | 91 | 51 | 56.04% | -$63 | -0.69% |
| Away Fav + Heavy Fav | 524 | 394 | 75.19% | -$3,355 | -6.40% |
| **Away Fav + Solid Fav** | **126** | **82** | **65.08%** | **$356** | **+2.82%** |
| Away Fav + Slight Fav | 96 | 55 | 57.29% | $77 | +0.80% |
| **Away Fav + Slight Dog** | **15** | **8** | **53.33%** | **$262** | **+17.47%** |

**Key Finding**: Away favorites are profitable when NOT heavy favorites.

---

### Table 7: Confidence Level × Favorite Type

| Strategy | Games | Wins | Win % | Total Profit | ROI |
|----------|-------|------|-------|--------------|-----|
| 70%+ Conf + Home Fav | 376 | 252 | 67.02% | -$3,513 | -9.34% |
| 70%+ Conf + Away Fav | 584 | 431 | 73.80% | -$3,248 | -5.56% |
| 60-70% Conf + Home Fav | 76 | 47 | 61.84% | -$344 | -4.53% |
| 60-70% Conf + Away Fav | 113 | 70 | 61.95% | $286 | +2.53% |
| **55-60% Conf + Home Fav** | **40** | **29** | **72.50%** | **$867** | **+21.67%** |
| 55-60% Conf + Away Fav | 44 | 22 | 50.00% | -$469 | -10.65% |
| 50-55% Conf + Home Fav | 37 | 21 | 56.76% | $47 | +1.27% |
| **50-55% Conf + Away Fav** | **45** | **29** | **64.44%** | **$849** | **+18.87%** |

**Key Finding**: 
- **55-60% Conf + Home Fav** = +21.67% ROI (highest ROI!)
- **50-55% Conf + Away Fav** = +18.87% ROI

---

### Table 8: Confidence Level × Model Prediction

| Strategy | Games | Wins | Win % | Total Profit | ROI |
|----------|-------|------|-------|--------------|-----|
| 70%+ Conf + Pick Home | 585 | 430 | 73.50% | -$3,539 | -6.05% |
| 70%+ Conf + Pick Away | 375 | 253 | 67.47% | -$3,222 | -8.59% |
| 60-70% Conf + Pick Home | 109 | 65 | 59.63% | -$340 | -3.12% |
| 60-70% Conf + Pick Away | 80 | 52 | 65.00% | $282 | +3.53% |
| 55-60% Conf + Pick Home | 36 | 18 | 50.00% | -$504 | -14.01% |
| **55-60% Conf + Pick Away** | **48** | **33** | **68.75%** | **$902** | **+18.80%** |
| 50-55% Conf + Pick Home | 46 | 27 | 58.70% | $406 | +8.83% |
| 50-55% Conf + Pick Away | 36 | 23 | 63.89% | $490 | +13.62% |

**Key Finding**: 
- **55-60% Conf + Pick Away** = +18.80% ROI ($902 profit - HIGHEST TOTAL)
- Low confidence + Pick Away is consistently profitable

---

### Table 9: Top 30 Strategies by ROI (Min 10 Games)

| Rank | Strategy | Games | Wins | Win % | Total Profit | ROI |
|------|----------|-------|------|-------|--------------|-----|
| 1 | **55-60% Conf + Home Fav** | 40 | 29 | 72.50% | $867 | +21.67% |
| 2 | **50-55% Conf + Away Fav** | 45 | 29 | 64.44% | $849 | +18.87% |
| 3 | **55-60% Conf + Pick Away** | 48 | 33 | 68.75% | $902 | +18.80% |
| 4 | Away Fav + Slight Dog | 15 | 8 | 53.33% | $262 | +17.47% |
| 5 | 50-55% Conf + Pick Away | 36 | 23 | 63.89% | $490 | +13.62% |
| 6 | **Away Fav + Pick Away** | **39** | **21** | **53.85%** | **$440** | **+11.29%** |
| 7 | **50-55% Confidence** | **82** | **50** | **60.98%** | **$896** | **+10.93%** |
| 8 | 50-55% Conf + Pick Home | 46 | 27 | 58.70% | $406 | +8.83% |
| 9 | Away Fav + Pick'em | 24 | 13 | 54.17% | $178 | +7.44% |
| 10 | **55-60% Confidence** | **84** | **51** | **60.71%** | **$398** | **+4.74%** |
| 11 | 60-70% Conf + Pick Away | 80 | 52 | 65.00% | $282 | +3.53% |
| 12 | Away Fav + Solid Fav | 126 | 82 | 65.08% | $356 | +2.82% |
| 13 | 60-70% Conf + Away Fav | 113 | 70 | 61.95% | $286 | +2.53% |
| 14 | 50-55% Conf + Home Fav | 37 | 21 | 56.76% | $47 | +1.27% |
| 15 | Away Fav + Slight Fav | 96 | 55 | 57.29% | $77 | +0.80% |
| 16 | 60-70% Confidence | 189 | 117 | 61.90% | -$58 | -0.31% |
| 17 | Home Fav + Slight Fav | 91 | 51 | 56.04% | -$63 | -0.69% |
| 18 | 60-70% Conf + Pick Home | 109 | 65 | 59.63% | -$340 | -3.12% |
| 19 | Home Fav + Pick Away | 500 | 340 | 68.00% | -$1,988 | -3.98% |
| 20 | Away Fav + Pick Home | 747 | 531 | 71.08% | -$3,022 | -4.05% |

---

### Table 10: Top 30 Strategies by Total Profit (Min 10 Games)

| Rank | Strategy | Games | Wins | Win % | Total Profit | ROI |
|------|----------|-------|------|-------|--------------|-----|
| 1 | **55-60% Conf + Pick Away** | 48 | 33 | 68.75% | **$902** | +18.80% |
| 2 | **50-55% Confidence** | 82 | 50 | 60.98% | **$896** | +10.93% |
| 3 | **55-60% Conf + Home Fav** | 40 | 29 | 72.50% | **$867** | +21.67% |
| 4 | **50-55% Conf + Away Fav** | 45 | 29 | 64.44% | **$849** | +18.87% |
| 5 | 50-55% Conf + Pick Away | 36 | 23 | 63.89% | $490 | +13.62% |
| 6 | **Away Fav + Pick Away** | 39 | 21 | 53.85% | **$440** | +11.29% |
| 7 | 50-55% Conf + Pick Home | 46 | 27 | 58.70% | $406 | +8.83% |
| 8 | **55-60% Confidence** | 84 | 51 | 60.71% | **$398** | +4.74% |
| 9 | Away Fav + Solid Fav | 126 | 82 | 65.08% | $356 | +2.82% |
| 10 | 60-70% Conf + Away Fav | 113 | 70 | 61.95% | $286 | +2.53% |

---

## Key Insights & Recommendations

### 🎯 Top Actionable Strategies

#### For Spread (ATS) Betting:

1. **Away Fav + Medium Spread** (211 games, 65.88% ATS, **+25.82% ROI**)
   - BEST OVERALL STRATEGY
   - Most reliable high-volume edge
   - Market systematically undervalues away favorites in medium spreads

2. **Away Fav + Large Spread** (299 games, 62.54% ATS, **+19.45% ROI**)
   - High volume + excellent ROI
   - Strong performance on big road favorites

3. **Away Fav + Pick Away** (409 games, 59.17% ATS, **+13.01% ROI**)
   - Highest volume profitable strategy (31% of all games!)
   - When market AND model agree away will cover

4. **Away Fav + Pick Home** (377 games, 58.09% ATS, **+10.95% ROI**)
   - Taking home underdogs when away team favored
   - High volume strategy

#### For Moneyline (ML) Betting:

1. **55-60% Conf + Home Fav** (40 games, 72.50% win, **+21.67% ROI**, $867)
   - Highest ROI strategy
   - When model is "somewhat confident" on home favorite

2. **55-60% Conf + Pick Away** (48 games, 68.75% win, **+18.80% ROI**, **$902**)
   - HIGHEST TOTAL PROFIT
   - Best risk-adjusted strategy

3. **50-55% Conf + Away Fav** (45 games, 64.44% win, **+18.87% ROI**, $849)
   - Model's "barely confident" away picks find value

4. **50-55% Confidence Overall** (82 games, 60.98% win, **+10.93% ROI**, $896)
   - Best volume-to-ROI ratio
   - Bet ALL low-confidence picks

5. **Away Fav + Pick Away** (39 games, 53.85% win, **+11.29% ROI**, $440)
   - When market AND model agree on away ML winner

### 🚨 Critical Findings

#### The Confidence Paradox (Applies to BOTH ATS and ML):

**ATS:**
- Low Confidence (50-55%): 54.78% ATS, +4.63% ROI ✅
- Medium Confidence (55-65%): 49.77% ATS, -4.93% ROI ❌
- High Confidence (65-75%): 51.71% ATS, -1.23% ROI ❌

**ML:**
- 50-55% Confidence: 60.98% win, **+10.93% ROI** ✅
- 55-60% Confidence: 60.71% win, **+4.74% ROI** ✅
- 60-70% Confidence: 61.90% win, -0.31% ROI ❌
- 70%+ Confidence: 71.15% win, **-7.04% ROI** ❌

**Why?** Model's confident picks are already priced into market odds. The edges exist where model is uncertain but finds subtle value the market missed.

#### Home Favorite Disaster:

**ATS:**
- Home Fav + Pick Home: 38.53% ATS, **-26.41% ROI** 💀
- Home Favored Overall: 41.40% ATS, **-20.93% ROI** 💀

**ML:**
- Home Fav + Pick Home: 31.03% win, **-32.94% ROI** 💀
- Home Favored Overall: 65.97% win, **-5.56% ROI** 💀

Market systematically overvalues home favorites (Home Court Advantage declining in NBA).

#### Away Favorite Gold Mine:

**ATS:**
- Away Favored Overall: 58.65% ATS, **+12.02% ROI** 🏆
- Away Fav + Med Spread: 65.88% ATS, **+25.82% ROI** 🏆
- Away Fav + Large Spread: 62.54% ATS, **+19.45% ROI** 🏆

**ML:**
- Away Fav + Pick Away: 53.85% win, **+11.29% ROI** 🏆

Market undervalues road favorites across the board.

### 📊 Market Efficiency Comparison

| Metric | Spread (ATS) | Moneyline (ML) |
|--------|--------------|----------------|
| Overall ROI | -1.23% | -4.20% |
| Best Strategy ROI | +28.38% | +21.67% |
| Best High-Volume Strategy | 786 games @ +12.02% | 82 games @ +10.93% |
| Market Efficiency | Moderate | Very High |
| Edge Difficulty | Easier to find | Harder to find |
| Breakeven Required | 52.4% fixed | Variable by odds |

**Conclusion**: 
- **ATS betting** offers more consistent edges and higher volume opportunities
- **ML betting** has higher peak ROI but requires precise filtering and lower volume
- Both markets show the same underlying inefficiencies (away favs, low confidence, home fav problems)

### 💡 Practical Recommendations

#### 1. Primary Strategy: Focus on ATS Betting
- More forgiving (need 52.4% vs variable breakeven on ML)
- Higher volume of profitable opportunities
- More consistent returns
- Best strategy: **Away favorites with medium/large spreads**

#### 2. ML Supplement: Use ML for Specific High-ROI Niches
- **ONLY bet 50-60% confidence games**
- Prioritize away favorites (but NOT heavy favorites)
- Avoid heavy favorites entirely (even at 75% win rate, you lose money)
- Best strategy: **Low confidence games (50-60%)**

#### 3. Never Bet (❌):
- Home favorites when model picks home (31-38% ATS, -26% to -33% ROI)
- 70%+ confidence ML picks (heavy favorites, -7% ROI despite 71% win rate)
- Home Fav + Large Spread (-24.72% ROI ATS)
- Tiny/Small spreads in general (-12% to -11% ROI)

#### 4. Always Consider (✅):
- **Away favorites with medium spreads** (+25.82% ROI, 65.88% ATS)
- **Away favorites with large spreads** (+19.45% ROI, 62.54% ATS)
- Model picks home underdog when away is favored (58-59% ATS)
- **Low confidence picks (50-60%) for ML** (+10.93% ROI overall)
- ANY "Away Favored" scenario for ATS (+12.02% ROI baseline)

#### 5. Bet Sizing Recommendations:
- **High Confidence Bets** (3-5% of bankroll):
  - Away Fav + Med Spread (ATS)
  - 55-60% Conf + Home Fav (ML)
  
- **Medium Confidence Bets** (2-3% of bankroll):
  - Away Fav + Large Spread (ATS)
  - 50-55% Confidence (ML)
  - Away Fav + Pick Away (ATS)
  
- **Volume Bets** (1-2% of bankroll):
  - Away Favored (any) (ATS)
  - Low confidence away picks (ML)

### 📈 Expected Results (Based on 2024-2025 Data)

If you bet $100 per game on these strategies for a full season:

**Conservative Portfolio (ATS Focus):**
- Away Favored (all): 786 games × $100 × 12.02% = **+$9,448**
- Total wagered: $78,600
- Total return: $88,048
- **ROI: +12.02%**

**Aggressive Portfolio (Best Strategies):**
- Away Fav + Med Spread: 211 games × $100 × 25.82% = **+$5,448**
- Away Fav + Large Spread: 299 games × $100 × 19.45% = **+$5,816**
- 50-55% Conf ML: 82 games × $100 × 10.93% = **+$896**
- Total wagered: $59,200
- Total return: $71,360
- **ROI: +20.54%**

---

*Analysis Date: October 30, 2024*  
*Data: 2024-2025 NBA Season (1,315 games)*  
*Model: Direct Margin Prediction with Normalized Features*  
*⚠️ Past performance does not guarantee future results. Always bet responsibly.*

