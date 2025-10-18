# Compact Play-by-Play Schema with Per‑Play Lineup IDs
**Purpose:** teach an LLM (Cursor) how to convert our existing verbose basketball JSON to a compact schema that preserves all information needed for next‑play prediction.

---

## Goal
Migrate the basketball play-by-play pipeline from the verbose JSON to a **compact schema with per-play lineup IDs**, preserving all information needed for next-play prediction. Update ingest, transforms, training data writers, and evaluators to read and emit the compact format.

## What you will deliver
1) New schema types and converters.  
2) Refactors of readers/writers to use the compact schema.  
3) Backfill converter for historical data.  
4) A CLI entry point to batch-convert and validate.

---

## Old inputs to expect
- `away_team`, `home_team` with `name`, `stats`, and `players[].profile`.
- `recent_plays[]` with:
  - `quarter` (int), `time_remaining` ("MM:SS"), `score` ("AWAY x - HOME y"),
  - `players_on_court` arrays (both teams) as player **names**,
  - `player` (string or null),
  - `description` (free text),
  - `shot_details` with `team` and `points` (often null; resolve via score delta).

---

## Target schema: compact per-play lineup‑ID format
Represent a single record as:
```json
{
  "A": "CLE",                       // away team code
  "H": "IND",                       // home team code

  "as": [117.9,109.0,95.6,2],      // away stats: [OEFF, DEFF, PACE, REST_DAYS]
  "hs": [115.7,116.8,101.1,2],     // home stats

  "ap": [                          // away roster: [name, off, def, ss, eff, mpg, usg]
    ["Player Name", 0.0, 0.0, 0.0, 0.0, 0, 0]
  ],
  "hp": [                          // home roster: same shape
  ],

  "L": [                           // lineup lookup; each item is a unique on-court 10-man state
    { "A":[idx,idx,idx,idx,idx],   // away on-court indices into ap
      "H":[idx,idx,idx,idx,idx] }  // home on-court indices into hp
  ],

  "p": [                           // plays
    // ALWAYS 7 elements: [q, t_sec_remaining, [away_score, home_score], actor, event_code, pts, lineup_id]
    // actor: ["A"|"H", player_index] or ["A"|"H", -1] for team events
    // pts is 0 for non-scoring events
  ],

  "y": [q,t,[a,h],actor,ev,pts,l]   // optional label (next play), same tuple shape; pts=0 when non-scoring
}
```

**Rules**
- Use **indices** into `ap`/`hp` for actors and for `L` lineups.  
- Use **integers** for time (`t_sec_remaining`) and scores.  
- The final element of every play is the **lineup_id** (index into `L`).  
- Include `pts` for all events; use **0** for non-scoring events.

---

## Event codebook
Encode `description` + `shot_details` into compact `event_code`s:

### Shots
- Made 2s: `2pm<feet>` if distance known, else `2pm`.  
- Missed 2s: `2pa<feet>` or `2pa`.  
- Made 3s: `3pm<feet>` or `3pm`.  
- Missed 3s: `3pa<feet>` or `3pa`.  
- Layups: made `layup<feet>`; missed `layupa<feet>`.  
- Dunks: made `dunk<feet>`; missed `dunka<feet>`.  
- Putbacks/tips: made `putback2`, `tipdunk_m`; missed `tipdunk_a`.

### Clock/ball control
- Defensive rebound: `d_reb`  
- Offensive rebound: `o_reb`  
- Steal: `stl`  
- Turnover (non-offensive-foul): `tov`

### Fouls
- Personal: `p_foul`  
- Shooting: `s_foul`  
- Offensive: `o_foul` **and** also emit a `tov` at the same timestamp.

### Free throws
- Made: `ftm` with `pts=1`  
- Missed: `ftx` (omit `pts`)

### Game admin
- Timeout: `timeout` (team actor with index `-1`)

---

## Precise mapping rules
1) **Teams**  
   - `a` = `away_team.name` code. `h` = `home_team.name` code.  
   - `as`/`hs` order is `[OEFF, DEFF, PACE, REST_DAYS]`.

2) **Rosters**  
   - `ap` and `hp` are arrays of `[name, offense, defense, shot_selection, efficiency, MPG, usage]` in the **same order** as the source payload.  
   - Build `name → index` maps for each side.

3) **Lineup IDs (`L`)**  
   - For each play, read `players_on_court` and convert the 10 names to indices.  
   - Create a **combined lineup key** of those 10 indices.  
   - If unseen, append `{A:[...], H:[...]}` to `L` and assign the next integer id.  
   - The play’s last field is that `lineup_id`.  
   - If a play lacks `players_on_court`, reuse the previous `lineup_id`.

4) **Time and score**  
   - `t_sec_remaining` = `60*MM + SS` from `"MM:SS"`.  
   - Parse `score` `"AWAY x - HOME y"` to `[x,y]` in **away, home** order.

5) **Actor**  
   - If `player` matches an away name → `["A", idx]`.  
   - If it matches a home name → `["H", idx]`.  
   - If `player` is null but team can be inferred (by `shot_details.team` or description like “CLIPPERS Rebound”), set `["A",-1]` or `["H",-1]`.  
   - Resolve all actors before writing; do not leave `null`.

6) **Event parsing**  
   - Prefer `shot_details.points` if present; otherwise infer from score delta vs the previous play at the same `(q,t)` or next nearest earlier time.  
   - Extract distance by matching `(\d+)'` in `description` and append to the code (e.g., `3pm26`).  
   - Map canonical phrases:
     - `"DEF.REBOUND"` → `d_reb`, `"OFF.REBOUND"` → `o_reb`  
     - `"STEAL"` → `stl` (pair with opponent `tov` if explicitly present)  
     - `"Lost Ball Turnover" | "Bad Pass Turnover" | "Out of Bounds Lost Ball"` → `tov`  
     - `"P.FOUL"` → `p_foul`, `"S.FOUL"` → `s_foul`, `"Offensive Foul"` → `o_foul` (then also `tov`)  
     - `"Free Throw"` → choose `ftm` or `ftx` by score delta  
     - Shot patterns:
       - `MISS .* 3PT .*` → `3pa<dist?>`
       - `MISS .* (Jump|Pullup|Bank) Shot` without “3PT” → `2pa<dist?>`
       - `.* 3PT .*` (made) → `3pm<dist?>`
       - `.* (Jump|Bank) Shot` (made) → `2pm<dist?>`
       - `.* Layup` → `layup<dist?>` if made, `layupa<dist?>` if missed
       - `.* Dunk` → `dunk<dist?>` if made, `dunka<dist?>` if missed
       - `Tip Dunk Shot` → `tipdunk_m` if scored, else `tipdunk_a`

7) **Points field**  
   - Include `pts` only for scoring events (`2pm*`, `3pm*`, `layup*` made, `dunk*` made, `putback2`, `tipdunk_m`, `ftm`).  
   - Omit `pts` for all other events.

8) **Ordering at identical timestamps**  
   - Preserve the input order for events sharing the same `[q, t_sec]`.  
   - Do not merge or reorder steal/turnover or foul/turnover pairs.

9) **Label**  
   - If the record includes a `next_play`, convert it into `y` using the same tuple schema.

10) **Numeric format**  
   - Round floats in team/player profiles to **two decimals** on serialization.  
   - Keep integers elsewhere.

---

## Worked examples (golden references)

### Example A: CLE–IND (single lineup id; includes rosters/stats)
```json
{
  "A": "CLE",
  "H": "IND",
  "as": [117.9,109.0,95.6,2],
  "hs": [115.7,116.8,101.1,2],
  "ap": [
    ["Max Strus",0.51,1.2,-1.04,0.74,36,19],
    ["Evan Mobley",0.9,3.58,1.28,0.36,33,21],
    ["Jarrett Allen",2.39,1.22,1.98,-1.88,21,18],
    ["Donovan Mitchell",3.39,2.88,0.31,-1.12,36,34],
    ["Darius Garland",4.23,0.65,-0.12,-1.0,32,25],
    ["Georges Niang",-1.07,0.15,-1.38,1.87,23,16],
    ["Isaac Okoro",0.06,2.42,0.57,-0.37,29,14],
    ["Caris LeVert",1.39,0.68,-0.05,0.72,34,26],
    ["Dean Wade",-1.04,-0.28,-2.67,0.16,20,8],
    ["Tristan Thompson",-0.66,-0.79,1.99,-0.51,11,12]
  ],
  "hp": [
    ["Bennedict Mathurin",0.09,-0.44,0.6,0.51,22,23],
    ["Obi Toppin",-0.67,-0.39,0.03,-0.22,21,14],
    ["Myles Turner",0.65,3.19,0.64,-0.53,27,24],
    ["Bruce Brown",0.26,0.21,-0.85,-0.9,31,16],
    ["Tyrese Haliburton",5.68,0.73,-0.46,-1.58,32,26],
    ["Buddy Hield",0.71,-0.2,-1.94,-0.24,23,21],
    ["Andrew Nembhard",0.28,0.36,-0.34,0.85,21,22],
    ["Jalen Smith",0.36,0.15,0.41,-1.72,16,21],
    ["Aaron Nesmith",0.03,1.61,-0.43,-1.13,25,16]
  ],
  "L": [
    { "A":[2,1,3,4,0], "H":[2,1,4,3,0] }
  ],
  "p": [
    [1,672,[0,2],["H",3],"o_reb",0],
    [1,669,[0,5],["H",0],"3pm28",3,0],
    [1,656,[2,5],["A",3],"dunk2",2,0],
    [1,643,[2,5],["A",0],"p_foul",0],
    [1,629,[2,5],["H",2],"tov",0],
    [1,629,[2,5],["A",2],"stl",0],
    [1,625,[5,5],["A",3],"3pm26",3,0],
    [1,614,[5,8],["H",2],"3pm26",3,0],
    [1,603,[5,8],["A",1],"2pa16",0],
    [1,599,[5,8],["H",4],"d_reb",0],
    [1,589,[5,8],["H",0],"2pa17",0],
    [1,587,[5,8],["A",2],"d_reb",0],
    [1,581,[5,8],["H",2],"stl",0],
    [1,581,[5,8],["A",3],"tov",0],
    [1,576,[5,8],["H",0],"layupa3",0],
    [1,576,[5,8],["H",-1],"o_reb",0],
    [1,569,[5,8],["A",2],"s_foul",0],
    [1,569,[5,9],["H",2],"ftm",1,0],
    [1,569,[5,10],["H",2],"ftm",1,0],
    [1,560,[5,10],["A",0],"3pa27",0],
    [1,556,[5,10],["H",3],"d_reb",0],
    [1,551,[5,12],["H",3],"layup4",2,0]
  ],
  "y": [1,540,[7,12],["A",2],"dunk2",2,0]
}
```

### Example B: WAS–MIA (mid-segment sub, timeout, offensive foul + turnover)
Lineups `L`:
```json
[
  { "A": [3,2,1,0,4], "H": [0,2,4,3,1] },
  { "A": [3,10,1,0,4], "H": [0,2,4,3,1] }
]
```

Plays `p`:
```json
[
  [3,672,[54,62],["H",2],"2pm13",2,0],
  [3,660,[54,62],["A",1],"2pa9",0],
  [3,658,[54,62],["H",3],"d_reb",0],
  [3,652,[54,62],["H",1],"3pa24",0],
  [3,648,[54,62],["A",0],"d_reb",0],
  [3,633,[54,62],["A",1],"3pa",0],
  [3,632,[54,62],["H",4],"d_reb",0],
  [3,632,[54,62],["A",2],"p_foul",0],
  [3,618,[54,64],["H",2],"layup2",2,0],
  [3,602,[56,64],["A",1],"layup3",2,0],
  [3,590,[56,66],["H",0],"layup2",2,0],
  [3,574,[58,66],["A",0],"layup7",2,0],
  [3,562,[58,69],["H",3],"3pm26",3,0],
  [3,550,[58,69],["A",2],"o_foul",0],
  [3,550,[58,69],["A",2],"tov",0],
  [3,531,[58,71],["H",2],"2pm3",2,1],
  [3,530,[58,71],["A",-1],"timeout",1],
  [3,509,[60,71],["A",0],"dunk1",2,1],
  [3,492,[60,71],["A",1],"s_foul",1]
]
```

Label `y`:
```json
[3,492,[60,71],["H",0],"ftx",1]
```

### Example C: HOU–LAC (quarter change adds lineup id; tip-dunk sequence)
Lineups `L`:
```json
[
  { "A": [8,3,7,5,6], "H": [7,1,8,5,6] },
  { "A": [8,7,1,6,4], "H": [6,7,2,1,8] }
]
```

Plays `p`:
```json
[
  [1,74,[21,26],["H",1],"3pa27",0],
  [1,70,[21,26],["A",5],"d_reb",0],
  [1,66,[21,26],["A",6],"tov",0],
  [1,51,[21,29],["H",7],"3pm26",3,0],
  [1,40,[21,29],["A",3],"3pa26",0],
  [1,36,[21,29],["H",1],"d_reb",0],
  [1,31,[21,29],["H",6],"3pa26",0],
  [1,27,[21,29],["A",6],"d_reb",0],
  [1,7,[21,29],["A",3],"2pa13",0],
  [1,4,[21,29],["A",8],"o_reb",0],
  [1,3,[23,29],["A",3],"layup2",2,0],
  [1,0,[23,29],["H",1],"3pa44",0],
  [1,0,[23,29],["H",-1],"d_reb",0],
  [2,706,[23,29],["H",6],"tov",1],
  [2,694,[23,29],["A",8],"3pa27",1],
  [2,690,[23,29],["A",7],"o_reb",1],
  [2,688,[23,29],["A",1],"2pa5",1],
  [2,687,[23,29],["A",6],"o_reb",1],
  [2,687,[23,29],["A",6],"tipdunk_a",1],
  [2,686,[23,29],["A",6],"o_reb",1]
]
```

Label `y`:
```json
[2,684,[25,29],["A",6],"putback2",2,1]
```

---

Use this spec to drive the refactor: parse old records, emit compact records exactly as defined, and validate against the examples and rules above.
