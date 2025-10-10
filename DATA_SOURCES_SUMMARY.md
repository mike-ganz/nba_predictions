# NBA Data Sources Summary

## 📊 Data Source Overview

### 1. Play-by-Play Data
**File:** `data/play_by_play/historical/[10-24-2023]-[06-17-2024]-combined-stats.csv`
- **Size:** 607,408 plays
- **Columns:** 44 fields
- **Coverage:** Full 2023-2024 season

### 2. Box Score Data  
**File:** `data/player_boxscores/historical/NBA-2023-2024-Player-BoxScore-Dataset.xlsx`
- **Size:** 28,232 player-game records
- **Columns:** 28 fields
- **Coverage:** Full 2023-2024 season

---

## 🎯 Play-by-Play Data Structure

### Key Fields (44 total)

**Game Context:**
- `game_id`, `date`, `data_set`
- `period`, `remaining_time`, `elapsed`, `play_length`
- `away_score`, `home_score`
- `team`, `away`, `home`, `opponent`

**Players on Court:**
- `a1`, `a2`, `a3`, `a4`, `a5` (Away team players)
- `h1`, `h2`, `h3`, `h4`, `h5` (Home team players)
- `player` (player involved in the play)

**Play Details:**
- `event_type` ⭐ (13 unique values)
- `type` ⭐⭐ (98 unique values!)
- `description` (text description)

**Context Fields:**
- `assist`, `block`, `steal` (player names)
- `entered`, `left` (substitutions)
- `points`, `possession`
- `reason`, `result`
- `num`, `outof` (for shots)

**Shot Location:**
- `shot_distance`
- `original_x`, `original_y`
- `converted_x`, `converted_y`

---

## 🎯 EVENT_TYPE Values (13 types)

| Event Type | Count | % | Description |
|------------|-------|---|-------------|
| **shot** | 233,788 | 38.49% | All shot attempts |
| **rebound** | 135,408 | 22.29% | Offensive & defensive rebounds |
| **free throw** | 57,071 | 9.40% | Free throw attempts |
| **substitution** | 61,847 | 10.18% | Player substitutions |
| **foul** | 50,273 | 8.28% | All foul types |
| **turnover** | 35,650 | 5.87% | Turnovers |
| **timeout** | 14,547 | 2.39% | Timeouts |
| **start of period** | 5,346 | 0.88% | Period starts |
| **end of period** | 5,346 | 0.88% | Period ends |
| **jump ball** | 2,229 | 0.37% | Jump balls |
| **violation** | 2,189 | 0.36% | Various violations |
| **technical foul** | 851 | 0.14% | Technical fouls |
| **ejection** | 73 | 0.01% | Ejections |

---

## 🎯 TYPE Values (98 types!)

### Shot Types (by distance)
**3-Point Shots (9 variations):**
- 3pt jump shot (62,349 - 10.26%)
- 3pt pullup jump shot (12,916)
- 3pt step back jump shot (7,884)
- 3pt running jump shot (5,879)
- 3pt running pull-up jump shot (2,551)
- 3pt fadeaway jumper (233)
- 3pt jump bank shot (170)
- 3pt floating jump shot (106)
- 3pt turnaround jump shot (102)

**2-Point Shots (26+ variations):**
- **Layups:** layup (34,031), driving layup (20,946), cutting layup (5,544), driving reverse layup (2,511)
- **Dunks:** dunk (8,362), cutting dunk (3,252), driving dunk (2,257)
- **Jump Shots:** jump shot (28,137), fadeaway jumper (4,476), floating jump shot (4,213), turnaround fadeaway (3,829), driving floating jump shot (11,850)
- **Hooks:** hook shot (6,794), hook bank shot (144)
- **Specialized:** driving floating bank jump shot (3,495), cutting finger roll layup (745)

### Free Throw Types (13 variations)
- free throw 1/1, 2/2, 3/3 (standard)
- free throw 1/2, 2/2 (2-shot fouls)
- free throw 1/3, 2/3, 3/3 (3-shot fouls)
- free throw technical
- free throw flagrant (various)
- free throw clear path

### Rebound Types (3)
- rebound defensive (92,556 - 15.24%)
- rebound offensive (33,892 - 5.58%)
- team rebound (8,960)

### Foul Types (9 variations)
- personal (15,205)
- shooting (25,728)
- offensive foul (4,168)
- loose ball (2,966)
- offensive (3,083)
- offensive charge (1,063)
- personal take (989)
- away from play (104)
- transition take (158)

### Technical Foul Types (7)
- technical (679)
- coach technical foul (98)
- double technical (153)
- non-unsportsmanlike technical (81)
- hanging technical (35)
- delay of game (502)
- excess timeout technical (2)

### Turnover Types (8)
- bad pass (16,581 - 2.73%)
- lost ball (7,401)
- out of bounds lost ball (2,024)
- traveling (1,925)
- discontinue dribble (56)
- double dribble (92)
- step out of bounds (868)
- palming (61)

### Violation Types (12)
- kicked ball (998)
- shot clock (1,760)
- 3-second violation (162)
- 5-second violation (63)
- 8-second violation (57)
- backcourt (202)
- defensive 3 seconds (600)
- defensive goaltending (691)
- offensive goaltending (136)
- lane violation (12)
- jump ball violation (4)
- basket from below (1)

### Other Types
- sub (substitutions - 61,847)
- jump ball (2,240)
- timeout: regular (14,218)
- ejection: other (73)
- start of period (5,346)
- end of period (5,346)
- unknown (3,740)

---

## 📦 Box Score Data Structure

### Available Fields (28 total)

**Identifiers:**
- PLAYER-ID
- PLAYER FULL NAME
- GAME-ID
- DATE

**Game Context:**
- OWN TEAM
- OPPONENT TEAM
- VENUE (R/H/N) - Road/Home/Neutral
- STARTER (Y/N)
- POSITION
- DAYS REST

**Traditional Stats:**
- MIN (minutes played)
- FG, FGA (field goals made/attempted)
- 3P, 3PA (3-pointers made/attempted)
- FT, FTA (free throws made/attempted)
- OR, DR, TOT (offensive/defensive/total rebounds)
- A (assists)
- PF (personal fouls)
- ST (steals)
- TO (turnovers)
- BL (blocks)
- PTS (points)

**Advanced:**
- USAGE RATE (%)

---

## 💡 Key Insights for Feature Engineering

### From Play-by-Play Data, We Can Calculate:

#### Shot Quality Metrics
- Shot type distribution (% of shots by type)
- Shot distance analysis (average, variance)
- Contested vs uncontested (using defender proximity)
- Catch-and-shoot vs off-dribble
- Shot location heat maps
- FG% by shot type
- FG% by distance
- FG% by quarter/time remaining

#### Playmaking Metrics
- Assisted shot %
- Assist types (to 3PT, to layup, to dunk)
- Pass quality (leading to what shot types)
- Hockey assists (player who passed to assister)

#### Defensive Metrics
- Blocks by shot type
- Steals by situation (bad pass, lost ball)
- Defensive rebounds in traffic
- Fouls drawn (opponent fouls when player has ball)
- Charges taken

#### Advanced Tempo/Rhythm
- Time to shoot after possession change
- Play types in transition vs half-court
- Late clock shots (shot clock < 5 sec)
- Clutch performance (last 5 min, close game)
- Performance by period
- Performance by rest days

#### Context-Aware Stats
- Who was on court for each play (5-man lineups)
- Performance with specific teammates
- Performance vs specific opponents
- Performance vs specific defenders
- Plus/minus at play-by-play level

#### Efficiency Patterns
- Turnover types (bad pass vs lost ball vs travel)
- Free throw situations (and-1, shooting fouls)
- Second chance points (after offensive rebounds)
- Points in paint vs mid-range vs 3PT
- Transition vs half-court scoring

#### Consistency Metrics
- Game-to-game variance in shot selection
- Shot type trends over season
- Performance streaks
- Hot hand effects (performance after makes/misses)

---

## 🚀 Opportunities for Expanded Analytics

### 1. Shot Chart Analytics
- Heat maps by location
- Shot selection quality scores
- Expected points per shot (xPPS)
- Shot quality vs shot efficiency

### 2. Situational Stats
- Performance by quarter
- Performance in clutch (last 5 min, score within 5)
- Performance on rest vs back-to-back
- Home vs road splits
- Vs winning/losing teams

### 3. Advanced Playmaking
- Assist quality (to what shot types)
- Creation % (shots created for others)
- Ball handling efficiency
- Passing lanes utilized

### 4. Defensive Impact (from play-by-play)
- Shots contested
- Deflections (steals + blocks)
- Fouls per shot contest
- Defensive rebounds contested %

### 5. Lineup & Chemistry
- On/Off court metrics
- Performance with specific teammates
- 2-man, 3-man, 5-man combinations
- Substitution pattern analysis

### 6. Tempo & Pace
- Possessions per game
- Seconds per possession
- Transition % (shots < 8 sec into possession)
- Fast break points

### 7. Consistency & Trends
- Rolling averages (5, 10, 20 game windows)
- Standard deviation of performance
- Trend lines (improving/declining)
- Streakiness vs consistency

---

## 📊 Data Quality Notes

### Box Score Data
- Clean, structured data
- 28,232 player-game records
- Comprehensive traditional stats
- Missing advanced metrics (PER, BPM, etc.)

### Play-by-Play Data
- Extremely detailed (607K plays)
- Some "unknown" types (3,740 occurrences - 0.62%)
- Shot coordinates available
- Full game context preserved
- Can reconstruct almost any stat from scratch

---

## 🎯 Next Steps

With this data, we can:
1. **Expand basic stats** (add shot selection breakdowns)
2. **Create situational stats** (clutch, home/away, vs quality opponents)
3. **Build defensive metrics** (contests, steals by type)
4. **Calculate lineup effects** (on/off court stats)
5. **Trend analysis** (rolling windows, consistency)
6. **Shot quality metrics** (expected points, location-based)
7. **Advanced playmaking** (assist quality, creation)
8. **Tempo & rhythm** (possession-based stats)

Ready for your instructions! 🚀

