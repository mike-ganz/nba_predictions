# Play-by-Play Stats - Quick Start Guide

## 🚀 **Get Started in 3 Steps**

### **Step 1: Import the Module**
```python
from player_stats_from_pbp import (
    calculate_player_pbp_stats,
    get_player_pbp_stats_summary,
    build_pbp_cache_for_date_range
)
```

### **Step 2: Calculate Stats for a Player**
```python
# Get stats for any player
stats = calculate_player_pbp_stats(
    player_name="LeBron James",
    max_date="2024-01-01",  # Optional: only games before this date
    season="2023-2024"
)

# Print readable summary
if stats:
    print(get_player_pbp_stats_summary(stats))
```

### **Step 3: Access Individual Stats**
```python
# Shot profile
print(f"Rim Rate: {stats['rim_attempt_rate']:.1%}")
print(f"3PT Rate: {stats['corner_3_rate'] + stats['non_corner_3_rate']:.1%}")
print(f"Pull-up 3 Rate: {stats['pullup_3_rate']:.1%}")

# Creation
print(f"Assists/100: {stats['assists_per_100']:.1f}")
print(f"Turnovers/100: {stats['turnovers_per_100']:.1f}")
print(f"AST/TO: {stats['ast_to_ratio']:.2f}")

# Defense
print(f"Steals/100: {stats['steals_per_100']:.1f}")
print(f"Blocks/100: {stats['blocks_per_100']:.1f}")
print(f"Def Reb Share: {stats['def_reb_share']:.1%}")
```

---

## 📊 **All Available Stats**

### **Shot Location (6 metrics)**
```python
stats['rim_attempt_rate']      # % shots at rim
stats['corner_3_rate']          # % corner 3s
stats['non_corner_3_rate']      # % non-corner 3s
stats['mid_range_rate']         # % mid-range total
stats['short_mid_rate']         # % short mid (4-16 ft)
stats['long_mid_rate']          # % long mid (16-22 ft)
```

### **Shot Type (2 metrics)**
```python
stats['pullup_3_rate']          # % pull-up 3s
stats['catch_shoot_3_rate']     # % catch-and-shoot 3s
```

### **Shot Quality (5 metrics)**
```python
stats['ftr']                    # Free throw rate (FTA/FGA)
stats['shooting_fouls_per_100'] # Shooting fouls drawn per 100 poss
stats['and1_rate']              # And-1s per 100 shots
stats['assisted_2pt_rate']      # % of 2s that were assisted
stats['assisted_3pt_rate']      # % of 3s that were assisted
```

### **Creation (3 metrics)**
```python
stats['assists_per_100']        # Assists per 100 possessions
stats['turnovers_per_100']      # Turnovers per 100 possessions
stats['ast_to_ratio']           # Assist to turnover ratio
```

### **Defense (5 metrics)**
```python
stats['steals_per_100']         # Steals per 100 possessions
stats['blocks_per_100']         # Blocks per 100 possessions
stats['def_reb_share']          # % of team def rebs when on court
stats['shooting_fouls_per_100'] # Shooting fouls committed per 100
stats['total_fouls_per_100']    # Total fouls per 100 possessions
```

### **Context (4 metrics)**
```python
stats['total_possessions']      # Total possessions on court
stats['total_fga']              # Total field goal attempts
stats['games_played']           # Number of games
stats['season']                 # Season year
```

---

## 🔄 **Common Use Cases**

### **Compare Two Players**
```python
lebron = calculate_player_pbp_stats("LeBron James", "2024-01-01", "2023-2024")
curry = calculate_player_pbp_stats("Stephen Curry", "2024-01-01", "2023-2024")

print(f"LeBron Rim Rate: {lebron['rim_attempt_rate']:.1%}")
print(f"Curry Rim Rate: {curry['rim_attempt_rate']:.1%}")

print(f"\nLeBron 3PT Rate: {lebron['non_corner_3_rate']:.1%}")
print(f"Curry 3PT Rate: {curry['non_corner_3_rate']:.1%}")
```

### **Track Player Over Season**
```python
dates = ["2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01"]

for date in dates:
    stats = calculate_player_pbp_stats("LeBron James", date, "2023-2024")
    if stats:
        print(f"{date}: {stats['assists_per_100']:.1f} AST/100")
```

### **Identify Shot Profile**
```python
def get_shot_profile(player_name, date, season):
    stats = calculate_player_pbp_stats(player_name, date, season)
    if not stats:
        return "Player not found"
    
    rim = stats['rim_attempt_rate']
    three = stats['corner_3_rate'] + stats['non_corner_3_rate']
    mid = stats['mid_range_rate']
    
    if rim > 0.6:
        return "Rim Runner"
    elif three > 0.5:
        return "3PT Specialist"
    elif mid > 0.4:
        return "Mid-Range Shooter"
    else:
        return "Balanced"

print(get_shot_profile("Rudy Gobert", "2024-01-01", "2023-2024"))
# Output: "Rim Runner"

print(get_shot_profile("Stephen Curry", "2024-01-01", "2023-2024"))
# Output: "3PT Specialist"
```

### **Find Elite Defenders**
```python
players = ["LeBron James", "Stephen Curry", "Rudy Gobert"]

print("Defensive Stats:")
print(f"{'Player':<20} {'STL/100':>8} {'BLK/100':>8} {'DRB%':>8}")
print("-" * 50)

for player in players:
    stats = calculate_player_pbp_stats(player, "2024-01-01", "2023-2024")
    if stats:
        print(f"{player:<20} "
              f"{stats['steals_per_100']:>8.1f} "
              f"{stats['blocks_per_100']:>8.1f} "
              f"{stats['def_reb_share']:>7.1%}")
```

---

## 💾 **Build Cache for Performance**

### **Cache for One Date**
```python
from player_stats_from_pbp import calculate_player_pbp_stats

# First call: slow (calculates from scratch)
stats = calculate_player_pbp_stats("LeBron James", "2024-01-01", "2023-2024")

# Second call: fast (loaded from cache)
stats = calculate_player_pbp_stats("LeBron James", "2024-01-01", "2023-2024")
```

### **Build Cache for Entire Season**
```python
# Build cache for all players, every week
count = build_pbp_cache_for_date_range(
    start_date="2023-10-24",
    end_date="2024-04-14",
    date_interval_days=7,
    season="2023-2024"
)

print(f"Built {count} cache entries")
# This will take 2-3 hours but only needs to be done once
```

### **Check Cache Status**
```python
from player_stats_from_pbp import load_from_cache

# Check if player stats are cached
cached = load_from_cache("LeBron James", "2024-01-01", "2023-2024")
if cached:
    print("✓ Cached")
else:
    print("✗ Not cached")
```

---

## 🔧 **Integration with Existing System**

### **Combine with Box Score Stats**
```python
from transform_player_stats_optimized import calculate_player_stats
from player_stats_from_pbp import calculate_player_pbp_stats

player = "LeBron James"
date = "2024-01-01"
season = "2023-2024"

# Get traditional box score stats
box_score = calculate_player_stats(player, date, season)
# Returns: PPG, RPG, APG, TS%, eFG%, etc.

# Get play-by-play advanced stats
pbp_stats = calculate_player_pbp_stats(player, date, season)
# Returns: rim_attempt_rate, pullup_3_rate, assists_per_100, etc.

# Combine into complete profile
complete_profile = {
    **box_score,      # Traditional stats
    **pbp_stats       # Advanced PBP stats
}

print(f"PPG: {complete_profile['PPG']:.1f}")
print(f"Rim Rate: {complete_profile['rim_attempt_rate']:.1%}")
print(f"Assists/100: {complete_profile['assists_per_100']:.1f}")
```

### **Use with PCA**
```python
from player_stats_from_pbp import calculate_player_pbp_stats
from pca_optimized import calculate_all_pca_scores_for_date

# Get PBP stats
pbp_stats = calculate_player_pbp_stats("LeBron James", "2024-01-01", "2023-2024")

# Get PCA scores
pca_scores = calculate_all_pca_scores_for_date("2024-01-01", "2023-2024")

# Combine for ML features
ml_features = {
    **pbp_stats,
    'offense_pca': pca_scores.get('LeBron James', {}).get('offense', 0),
    'defense_pca': pca_scores.get('LeBron James', {}).get('defense', 0),
}
```

---

## 🐛 **Troubleshooting**

### **"No stats found for player"**
- Check player name spelling (use exact match from dataset)
- Check date range (player may not have played before that date)
- Check season year format ("YYYY-YYYY")

### **"PBP data file not found"**
- Ensure CSV file exists in `data/play_by_play/historical/`
- Check season mapping in `SEASON_FILE_MAPPING`

### **Stats look wrong**
- Use `validate_pbp_stats(stats)` to check for issues
- Shot distribution should sum to ~100% (±5%)
- Compare FTA with box score FTA (should match)

### **Too slow**
- Build cache first with `build_pbp_cache_for_date_range()`
- Subsequent calls will be <0.01 seconds
- Only need to build cache once per season

---

## 📝 **Pro Tips**

1. **Build cache overnight** - Cache building takes 2-3 hours for full season
2. **Use weekly intervals** - Daily cache is overkill, weekly is plenty
3. **Check validation** - Always validate stats for new players/dates
4. **Combine with box score** - PBP stats complement traditional stats
5. **Watch for unknowns** - Some shots may be classified as 'unknown' if coordinates missing

---

## 🎯 **Real-World Examples**

### **Find Shot Creators**
```python
# Players with high pull-up 3 rate = self-creators
players = ["Stephen Curry", "Damian Lillard", "Luka Doncic"]

for player in players:
    stats = calculate_player_pbp_stats(player, "2024-01-01", "2023-2024")
    if stats:
        pullup_rate = stats['pullup_3_rate']
        assisted_rate = stats['assisted_3pt_rate']
        print(f"{player}: {pullup_rate:.1%} pull-up, {assisted_rate:.1%} assisted")
```

### **Find Rim Protectors**
```python
# Players with high block rate + def reb share = rim protectors
players = ["Rudy Gobert", "Anthony Davis", "Joel Embiid"]

for player in players:
    stats = calculate_player_pbp_stats(player, "2024-01-01", "2023-2024")
    if stats:
        blk = stats['blocks_per_100']
        drb = stats['def_reb_share']
        print(f"{player}: {blk:.1f} BLK/100, {drb:.1%} DRB%")
```

### **Find Playmakers**
```python
# High assists/100 + good AST/TO = elite playmakers
players = ["Chris Paul", "LeBron James", "Nikola Jokic"]

for player in players:
    stats = calculate_player_pbp_stats(player, "2024-01-01", "2023-2024")
    if stats:
        ast = stats['assists_per_100']
        ratio = stats['ast_to_ratio']
        print(f"{player}: {ast:.1f} AST/100, {ratio:.2f} AST/TO")
```

---

## 📚 **Full Documentation**

See `PBP_STATS_IMPLEMENTATION_SUMMARY.md` for:
- Complete technical details
- Implementation architecture
- Validation results
- Performance metrics
- Advanced features

---

**Ready to use! 🎉**

