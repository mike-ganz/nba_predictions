# NBA Play-by-Play Prediction System

A comprehensive machine learning system for predicting NBA game outcomes by generating play-by-play sequences using fine-tuned Large Language Models (LLMs). The system generates training data from historical NBA games and uses it to train models (OpenAI GPT or Google Gemini) that can simulate entire games play-by-play.

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Training Data Pipeline](#-training-data-pipeline)
- [Prediction Pipeline](#-prediction-pipeline)
- [Usage Examples](#-usage-examples)
- [Advanced Features](#-advanced-features)
- [Data Sources](#-data-sources)
- [Technical Details](#-technical-details)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Testing](#-testing)

## 🎯 Overview

This project implements two main pipelines:

1. **Training Data Generation**: Converts historical NBA play-by-play data into structured training data for LLM fine-tuning
2. **Game Prediction/Simulation**: Uses fine-tuned models to predict and simulate complete NBA games play-by-play

### Key Capabilities

- Generate training data from 3 NBA seasons (2022-2023, 2023-2024, 2024-2025)
- Support for both **compact** and **verbose** data formats
- Multiple output formats: **Gemini**, **OpenAI**, **Together.ai**, and **CSV**
- Two generation modes:
  - `remaining_plays`: Standard next-play prediction (high volume)
  - `first_N_plays`: Sequence generation (one example per game)
- Ultra-optimized pipeline: 5-20x faster than baseline with comprehensive PCA caching
- Systematic simulation orchestration with SQLite result storage
- Support for OpenAI, Gemini, and Together.ai fine-tuned models
- **Together.ai integration**: Fast, cost-effective LoRA fine-tuning with auto-hyperparameters

## 🏗️ Architecture

```
nba_predictions/
├── Training Data Pipeline
│   ├── generate_2023_2024_season.py      # Main CLI entry point
│   ├── generate_training_data.py          # Core data loader & utilities
│   ├── generate_training_data_OPTIMIZED.py # Ultra-fast optimized version
│   └── training/
│       ├── gemini_formatter.py            # Gemini format converter
│       └── openai_formatter.py            # OpenAI format converter
│
├── Prediction Pipeline
│   ├── orchestrator.py                    # Multi-game simulation orchestrator
│   ├── predict_next_play.py               # Core prediction engine
│   ├── game_context_builder.py            # Game state management
│   └── results_analyzer.py                # Results analysis tools
│
├── Data Processing
│   ├── transform_player_stats.py          # Player statistics calculation
│   ├── generate_team_stats.py             # Team statistics aggregation
│   ├── pca_optimized.py                   # Player PCA profiling
│   └── player_stats_from_pbp.py           # Play-by-play derived stats
│
├── Configuration & Utils
│   ├── config/settings.py                 # Global configuration
│   ├── game/                              # Game state utilities
│   └── data/                              # Data loading utilities
│
└── Data Storage
    └── data/
        ├── play_by_play/historical/       # Raw play-by-play CSVs
        ├── training/                       # Generated training data
        └── player_boxscores/               # Player statistics
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd nba_predictions

# Install dependencies
pip install -r requirements.txt

# Set up API keys (for prediction)
export OPENAI_API_KEY=your_key           # For OpenAI
export GEMINI_MODEL_1_ENDPOINT=endpoint  # For Gemini
export GOOGLE_CLOUD_PROJECT=project_id
```

### Generate Training Data

```bash
# Generate full 2023-2024 season in compact format (Gemini)
python generate_2023_2024_season.py

# Generate with OpenAI format
python generate_2023_2024_season.py --format openai

# Generate with Together.ai format
python generate_2023_2024_season.py --format together

# Generate verbose format
python generate_2023_2024_season.py --data-format verbose

# Sample for testing (1000 examples)
python generate_2023_2024_season.py --sample 1000 --format gemini

# Process only first 50 games
python generate_2023_2024_season.py --games 50

# Custom sequence length
python generate_2023_2024_season.py --n-total 12
```

### Run Simulations

```bash
# Run 10 simulations for 3 games
python orchestrator.py \
    --games "22200001,22200002,22200003" \
    --runs-per-game 10 \
    --season 2023-2024

# Use configuration file
python orchestrator.py --config my_config.json

# Analyze results
python orchestrator.py --analyze
```

## 📊 Training Data Pipeline

### How It Works

The training data pipeline converts raw NBA play-by-play data into structured JSON suitable for LLM fine-tuning:

```
Raw Play-by-Play CSV
    ↓
Load & Filter Games (≥50 plays)
    ↓
For Each Play:
  ├── Build Game Context
  │   ├── Team Stats (OEFF, DEFF, PACE, 3PAr, FTr, ORr, DRr, ASTr, TOr, REST)
  │   ├── Player Profiles (shot zones, creation, defense, usage, archetype)
  │   └── Recent Plays (n_total plays with full context)
  ├── Generate Training Example
  │   ├── Compact Format: Minimal JSON with encoded plays
  │   └── Verbose Format: Full descriptive JSON
  └── Convert to Platform Format
      ├── Gemini: GenerateContent format
      ├── OpenAI: Chat completion format
      └── CSV: Raw structured data
```

### Data Formats

#### Compact Format (Optimized)
```json
{
  "A": "ATL",
  "H": "DET",
  "as": [115.6, 117.8, 100.7, 0.35, 0.26, 0.24, 0.73, 0.63, 0.12, 3],
  "hs": [109.0, 115.9, 97.7, 0.33, 0.28, 0.27, 0.71, 0.60, 0.14, 2],
  "ap": [["De'Andre Hunter", 31, 0.18, 22.5, 15.2, 4.2, 1.8, 0.4, 0.35, 0.635, 0.06, 0.378, 0.22, 0.362, 0.37, 0.418, 0.52, 0.48, 0.812, 0], ...],
  "hp": [["Ausar Thompson", 32, 0.19, 18.3, 13.8, 3.5, 2.1, 0.8, 0.40, 0.622, 0.04, 0.333, 0.18, 0.288, 0.38, 0.395, 0.55, 0.50, 0.672, 1], ...],
  "L": [{"A": [0,1,2,3,4], "H": [0,1,2,3,4]}],
  "p": [[1, 375, [18,11], 7, ["A",0], 1, "made2", "mid", 0], ...],
  "tb": [2, 1],
  "sd": 7,
  "pos": "H"
}
```

**Fields:**
- `A`, `H`: Team abbreviations (away/home)
- `as`, `hs`: Team stats arrays [OEFF, DEFF, PACE, 3PAr, FTr, ORr, DRr, ASTr, TOr, REST_DAYS]
- `ap`, `hp`: Player arrays [name, MPG, usage, pts/poss, fga/poss, ast/poss, stl/poss, blk/poss, rim%, rim_fg%, c3%, c3_fg%, nc3%, nc3_fg%, mid%, mid_fg%, a2%, a3%, ft%, fouls]
  - High-level production stats first (usage, scoring, playmaking), then detailed shooting breakdown
  - Per-possession rates for individual efficiency (pts/poss ~1.0-1.2 for elite scorers)
  - Shot zones paired: each zone's frequency followed by its accuracy
  - **All percentages normalized to 0.0-1.0 scale** (usage is 0.18 instead of 18%, for consistency with all other percentage fields)
  - **fga/poss added** for shot volume context (~0.15-0.30 for most players)
- `L`: Lineup lookup (maps lineup IDs to player indices). For remaining_plays input, `L` contains only lineups observed up to the context play (no future lineups).
- `p`: Plays array (9-value tuples):
  - `[quarter, time_seconds, [away, home], margin, actor, actor_fouls, event_code, shot_zone, lineup_id]`
  - `actor` = `["A"|"H", playerIndex]` with `playerIndex >= 0` when actor is known; `-1` if unknown.
  - `actor_fouls` = live, cumulative personal fouls for the acting player at that moment (counts `p_foul`, `s_foul`, `o_foul` where `event_type == 'foul'`).
  - `event_code` = one of: made2, made3, miss2, miss3, mft, xft, d_reb, o_reb, tov, s_foul, p_foul, o_foul, sub, timeout, period, jumpball, viol, tech, unknown.
  - `shot_zone` = one of: rim, mid, nc3, c3, or `null` for non-shots/unknown.
  - `lineup_id` = index into `L` (0-based).
- `tb`: Team bonus (quarter fouls: [away, home])
- `sd`: Score difference (away - home)
- `pos`: Possession ("A", "H", or "N")

#### Verbose Format (Readable)
```json
{
  "away_team": {
    "name": "ATL",
    "stats": {"OEFF": 115.6, "DEFF": 117.8, "PACE": 100.7, "REST_DAYS": 3},
    "players": [
      {
        "name": "De'Andre Hunter",
        "profile": {
          "offense": 0.54,
          "defense": 1.77,
          "shot_selection": 0.21,
          "efficiency": -0.23,
          "MPG": 31,
          "usage": 18
        }
      }
    ]
  },
  "home_team": { ... },
  "recent_plays": [
    {
      "quarter": 1,
      "time_remaining": "06:15",
      "score": "ATL 18 - DET 11",
      "players_on_court": [...],
      "player": "De'Andre Hunter",
      "description": "Hunter 18' Jump Shot",
      "shot_details": {"team": "ATL", "points": 2}
    }
  ]
}
```

### Generation Modes

#### 1. `remaining_plays` Mode (Standard)
- **Purpose**: Next-play prediction training
- **Volume**: High (~400 examples per game)
- **Usage**: Standard supervised learning for play-by-play prediction
- **Output**: Each training example predicts the next play given context.
- **Lineup handling (no leakage)**:
  - Input `L`: only lineups observed up to the context (no future).
  - Label `y.lineup_id`:
    - If next lineup exists in `L`: emit its index (0..len(L)-1).
    - If next lineup is new: emit `lineup_id == len(L)` as an explicit “new lineup” signal (we do not mutate `L` in the input).
  - Inference: if `y.lineup_id < len(L)` use that lineup; if `y.lineup_id == len(L)`, append the new lineup to `L` downstream and proceed.

#### 2. `first_N_plays` Mode (Sequence)
- **Purpose**: Game opening sequence generation
- **Volume**: Low (1 example per game)
- **Usage**: Teaching model to generate realistic game openings
- **Output**: Clean context → first N plays of the game
- **Starting lineup seeding**: `L[0]` is seeded from prior game starters per team (replace DNPs with highest-MPG non-starters; for season opener, use last season’s final starting lineup).
- **Lineup progression**:
  - When raw on-court data (a1–a5/h1–h5) is available (Gemini path), lineup changes are tracked via `resolve_lineup_from_row` and `lineup_id` advances across the first-N sequence.
  - Input contexts remain clean (no `p`).

### Event Codes
### Actor Resolution
- We resolve actor indices using roster indices from `ap`/`hp` when the player name is present. We fall back to `-1` only when the player is truly unavailable/unmappable.

### Actor Fouls (Live)
- `actor_fouls` is a live, in-game cumulative count at the moment of the play (counts personal/shooting/offensive fouls where `event_type == 'foul'`; excludes technicals and avoids double-counting o_foul turnovers).

### Shot Zone
- For shots (made2, miss2, made3, miss3) we attempt to classify as `rim`, `mid`, `nc3` or `c3`; otherwise `null`.

### New Lineup Signal in Labels
- To avoid leaking future information into inputs, labels signal lineup changes without mutating the input `L`:
  - If the next lineup is not in `L`, we set `y.lineup_id = len(L)`.
  - Consumers should treat `lineup_id == len(L)` as “append new lineup” at inference time.

### Validation
- Use `analysis/validate_tuple_distributions.py` to validate tuple lengths and distributions for all fields.
- The validator treats `y.lineup_id == len(L)` as a valid “new lineup” signal and does not count it as invalid.

The system uses standardized event codes for play classification:

| Code | Description | Code | Description |
|------|-------------|------|-------------|
| `made2` | Made 2-pointer | `miss2` | Missed 2-pointer |
| `made3` | Made 3-pointer | `miss3` | Missed 3-pointer |
| `mft` | Made free throw | `xft` | Missed free throw |
| `o_reb` | Offensive rebound | `d_reb` | Defensive rebound |
| `p_foul` | Personal foul | `s_foul` | Shooting foul |
| `o_foul` | Offensive foul | `tech` | Technical foul |
| `tov` | Turnover | `stl` | Steal |
| `viol` | Violation | `timeout` | Timeout |
| `sub` | Substitution | `period` | Period boundary |

### Shot Zones

Shots are classified into 4 zones:
- `rim`: ≤4 feet from basket
- `mid`: 4-22 feet (mid-range)
- `c3`: 3-pointer from corner
- `nc3`: 3-pointer not from corner

## 🎮 Prediction Pipeline

The prediction system uses fine-tuned LLMs to simulate complete NBA games play-by-play. It features a sophisticated two-stage architecture with rolling predictions, validation, and comprehensive state management.

### System Architecture

```
Game Prediction Flow
├── Context Building
│   ├── Load game data (teams, players, stats)
│   ├── Build compact/verbose format context
│   └── Initialize optimized context manager
│
├── Stage 1: Initial Plays (Optional)
│   ├── Input: Clean context (no plays)
│   ├── Model: first_N_plays fine-tuned model
│   └── Output: ~20 initial plays
│
├── Stage 2: Rolling Predictions
│   ├── Input: Context + recent plays (sliding window)
│   ├── Model: next_play fine-tuned model
│   ├── Prediction: Generate next play
│   ├── Validation: Check format, time progression, game state
│   ├── Context Update: Add play, slide window
│   └── Repeat: Until game ends or max iterations
│
└── Results
    ├── Complete play sequence
    ├── Final score and game state
    ├── Performance metrics
    └── Validation statistics
```

### Core Prediction Script

`predict_next_play.py` is the main prediction engine with advanced features:

**Key Features:**
- Multi-platform support (OpenAI GPT, Google Gemini)
- Compact and verbose format support
- Optimized context management with JSON caching
- Comprehensive response validation
- Automatic error recovery and rollback
- Configurable logging levels (0-3)
- Quarter transition handling

**Basic Usage:**

```python
from predict_next_play import predict_rolling_sequence

# Run prediction with game context
results = predict_rolling_sequence(
    game_context=game_context,
    n_iterations=750,           # Number of predictions
    skip_stage1=False           # Include Stage 1 generation
)
```

**Configuration:**

```bash
# Platform selection
export PREDICTION_PLATFORM=gemini  # or "openai"

# Logging level
export PREDICTION_LOG_LEVEL=1      # 0=minimal, 1=normal, 2=verbose, 3=debug

# Validation mode
export VALIDATION_MODE=fast        # fast/normal/strict

# Rate limiting (optional)
export GENAI_ENABLE_LIMITER=1
export GENAI_MAX_CONCURRENT=2
export GENAI_TARGET_RPS=2.0
```

### Two-Stage Prediction System

#### Stage 1: Initial Play Generation (Optional)

**Purpose**: Generate realistic game opening sequences

**Input Format**:
```json
{
  "A": "ATL",
  "H": "DET",
  "as": [115.6, 117.8, 100.7, 0.35, 0.26, 0.24, 0.63, 3],
  "hs": [109.0, 115.9, 97.7, 0.33, 0.28, 0.27, 0.60, 2],
  "ap": [...],
  "hp": [...]
}
```
**Note**: No `p` field (plays array) - clean context only

**Output**: Array of ~20 initial plays to seed the game
```json
[
  [1, 720, [0,0], ["A",0], "made2", 2, 0],
  [1, 705, [2,0], ["H",1], "miss3", 0, 0],
  ...
]
```

**Test Mode (Static Stage 1)**:
- Set `STATIC_STAGE1_ENABLED=1` to enable the static Stage 1 pipeline
- Provide the file path via `STATIC_STAGE1_RESPONSE_PATH` (relative paths resolve from project root)
- Select the Stage 1 payload by key with `STATIC_STAGE1_GAME_KEY` (defaults to the game id when present)
- Optional: set `STATIC_STAGE1_SKIP_API=1` to skip the live Stage 1 endpoint entirely

This mode injects the pre-recorded Stage 1 response before the rolling stage so you can run the rest of the pipeline without making the initial endpoint request.

**Why Skip Stage 1?**
- Testing with pre-built contexts
- Resuming from saved game states
- Faster iteration during development

#### Stage 2: Rolling Predictions

**Purpose**: Iteratively predict the next play given game context

**Process**:
1. **Context**: Teams + players + recent plays (sliding window)
2. **Prediction**: Model generates next play
3. **Validation**: Format, time progression, game rules
4. **Update**: Add new play to context
5. **Slide Window**: Keep last N plays (default: 20)
6. **Repeat**: Until game ends or max iterations

**Sliding Window Example**:
```
Iteration 1: [P1, P2, P3, P4, P5] → Predict P6
Iteration 2: [P2, P3, P4, P5, P6] → Predict P7  (P1 removed)
Iteration 3: [P3, P4, P5, P6, P7] → Predict P8  (P2 removed)
...
```

### Context Management

#### Optimized Context Manager

The system uses two specialized context managers for performance:

**CompactGameContext** (for compact format):
```python
compact_context = CompactGameContext(game_context)

# Efficient operations
compact_context.add_play_tuple(new_play, max_plays=20)
json_str = compact_context.get_json()  # Cached
context_dict = compact_context.get_context_dict()
```

**OptimizedGameContext** (for verbose format):
```python
verbose_context = OptimizedGameContext(game_context)

# Efficient operations  
verbose_context.add_play_and_slide(new_play, max_plays=20)
json_str = verbose_context.get_json()  # Cached
```

**Performance Optimizations**:
- JSON caching: Avoids re-serialization (50+ cache hits)
- Hash-based cache keys: Fast lookup of play sequences
- Incremental updates: Only update changed parts
- Cache eviction: Automatic management at 50 entries

### Response Validation System

The `NBAResponseValidator` ensures all predictions follow game rules:

**Validation Modes**:
- `fast`: Essential checks only (10-20% faster)
- `normal`: Balanced validation (default)
- `strict`: Comprehensive checks (all rules)

**Validation Checks**:

| Category | Checks | Fast | Normal | Strict |
|----------|--------|------|--------|--------|
| **Format** | JSON structure, required fields | ✓ | ✓ | ✓ |
| **Time** | Time progression, quarter boundaries | ✓ | ✓ | ✓ |
| **Score** | Valid format, monotonic increase | ✓ | ✓ | ✓ |
| **Game Rules** | 4 quarters, time limits | ✓ | ✓ | ✓ |
| **Advanced** | Player validation, duplicate detection | | ✓ | ✓ |
| **Strict** | Historical consistency, statistical outliers | | | ✓ |

**Validation Results**:
```python
class ValidationResult(Enum):
    VALID = "valid"                        # Pass - continue
    RETRY = "retry"                        # Fail - retry prediction
    END_GAME = "end_game"                  # Game finished naturally
    ROLLBACK_TIME = "rollback_time"        # Time violation - rollback
    QUARTER_TRANSITION = "quarter_transition"  # New quarter needed
```

**Automatic Recovery**:

1. **Retry Logic**: Up to 6 attempts on validation failure
2. **Timestamp Rollback**: Restore context on time violations
3. **Quarter Transitions**: Inject clean quarter start contexts
4. **Error Tracking**: Detailed failure analysis and patterns

### Game State Management

#### Timestamp Rollback

When the model predicts backwards time progression:

```python
# Before rollback
Recent plays: [Q1 8:00, Q1 7:45, Q1 7:30, Q1 8:15]  # Invalid!

# Validator detects violation
validator.detect_timestamp_violation()

# Rollback to last good state
validator.get_rollback_snapshot()  # Returns [Q1 8:00, Q1 7:45, Q1 7:30]

# Resume from corrected state
context.restore_recent_plays(snapshot)
```

#### Quarter Transitions

Clean quarter start injection:

```python
# End of Q1 detected (consecutive 0:00 plays)
if consecutive_quarter_end > 3:
    # Inject Q2 start
    context.inject_quarter_start(quarter=2)
    
    # New context
    # [... other plays ..., Q2 12:00 "Start of quarter 2"]
```

### Orchestration System

For systematic multi-game simulations, the orchestrator provides:

**Configuration**:
```python
from orchestrator import NBA_Orchestrator, SimulationConfig

config = SimulationConfig(
    season_year="2023-2024",
    games=["22200001", "22200002", "22200003"],
    runs_per_game=10,
    max_iterations_per_run=2000,
    platform="gemini",
    output_db="simulation_results.db",
    skip_stage1=False,
    timeout_minutes=30
)

orchestrator = NBA_Orchestrator(config)
results = orchestrator.run_all_simulations()
```

**Features**:
- Multi-game batch processing
- Error recovery and continuation
- Progress tracking
- Results storage in SQLite
- Performance metrics
- Validation failure analysis

**Command Line**:
```bash
# Run simulations
python orchestrator.py \
    --games "22200001,22200002,22200003" \
    --runs-per-game 10 \
    --season 2023-2024 \
    --platform gemini

# Analyze results
python orchestrator.py --analyze
python orchestrator.py --analyze-game 22200001
```

### Database Schema

Results are stored in SQLite with comprehensive tracking:

```sql
CREATE TABLE simulation_runs (
    -- Identity
    run_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    season_year TEXT NOT NULL,
    platform TEXT NOT NULL,
    
    -- Timing
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_seconds REAL,
    
    -- Results
    status TEXT NOT NULL,
    total_predictions INTEGER,
    successful_predictions INTEGER,
    final_score TEXT,
    final_quarter INTEGER,
    final_time TEXT,
    termination_reason TEXT,
    
    -- Scoring Analysis
    scoring_plays INTEGER,
    non_scoring_plays INTEGER,
    scoring_rate REAL,
    
    -- Error Tracking
    error_message TEXT,
    iterations_data TEXT,
    
    -- Validation Termination Details
    validation_termination_type TEXT,
    validation_termination_reason TEXT,
    validation_trigger_condition TEXT,
    validation_termination_timestamp TEXT,
    validation_consecutive_count INTEGER,
    validation_total_attempts INTEGER,
    validation_game_state_quarter INTEGER,
    validation_game_state_time TEXT,
    validation_game_state_score TEXT,
    validation_context_json TEXT,
    
    -- Validation Failure Analysis
    validation_failure_timestamp TEXT,
    validation_total_failed_attempts INTEGER,
    validation_most_common_reason TEXT,
    validation_most_common_reason_count INTEGER,
    validation_most_common_error_type TEXT,
    validation_most_common_error_type_count INTEGER,
    validation_most_common_field TEXT,
    validation_most_common_field_count INTEGER,
    validation_unique_reasons INTEGER,
    validation_unique_error_types INTEGER,
    validation_unique_fields INTEGER,
    validation_failure_summary TEXT,
    validation_response_examples TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### Game Context Builder

Build realistic game contexts from historical data:

```python
from game_context_builder import GameContextBuilder

builder = GameContextBuilder("2023-2024")

# List available games
games = builder.get_available_games()

# Build context for specific game
context = builder.build_game_context(
    game_id="22200001",
    start_quarter=1,
    start_time="12:00",
    recent_plays_count=20,
    for_first_n_plays=False  # Include plays or not
)

# Batch build
contexts = builder.batch_build_contexts(
    game_ids=["22200001", "22200002", "22200003"],
    output_dir="game_contexts"
)
```

### Performance Monitoring

**Logging Levels**:

```bash
# Level 0: Minimal - Only essential messages
export PREDICTION_LOG_LEVEL=0

# Level 1: Normal - Standard progress (default)
export PREDICTION_LOG_LEVEL=1

# Level 2: Verbose - Detailed iteration info + cache stats
export PREDICTION_LOG_LEVEL=2

# Level 3: Debug - Full JSON dumps + validation details
export PREDICTION_LOG_LEVEL=3
```

**Example Output** (Level 1):
```
--- ITERATION 125/750 ---
Game State: Q2 5:23 | ATL 54 - DET 48
Recent plays:
   1. Q2 [05:45] Bogdanovic made 26' three-pointer
   2. Q2 [05:30] Stewart missed free throw
   3. Q2 [05:30] Stewart made free throw
   4. Q2 [05:26] Murray defensive rebound
   5. Q2 [05:23] Johnson missed jumper
```

**Cache Statistics** (Level 2+):
```
Cache stats: {
    'plays_cache_size': 23,
    'base_json_length': 4521,
    'current_plays_count': 20
}
Estimated JSON cache efficiency: 85.3%
```

## 🔗 Training ↔ Prediction Integration

### How They Work Together

The training and prediction pipelines are designed to work in harmony:

```
Historical Data → Training Pipeline → Fine-tuned Models → Prediction Pipeline → Game Simulations
```

**Training Pipeline** generates examples like:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "{\"A\":\"ORL\",\"H\":\"CLE\",...,\"p\":[...]}"
    },
    {
      "role": "assistant",
      "content": "{\"y\":[1,705,[2,3],[\"H\",3],\"made2\",2,0]}"
    }
  ]
}
```

**Prediction Pipeline** uses the fine-tuned model:
1. Sends similar context format to the model
2. Receives predictions in the same format
3. Validates and processes the response
4. Updates context and repeats

### Format Consistency

Both pipelines use the **same schemas**:

| Aspect | Training | Prediction |
|--------|----------|------------|
| **Context Format** | Compact or Verbose | Compact or Verbose |
| **Play Format** | Tuple: `[q,t,score,actor,event,pts,lineup]` | Same tuple format |
| **Team Stats** | `[OEFF,DEFF,PACE,3PAr,FGr,FTr,ORr,ASTr,REST]` | Same array format |
| **Player Profiles** | `[name,offense,defense,shot_sel,eff,MPG,usage]` | Same array format |
| **Sliding Window** | Last 20 plays | Last 20 plays |

### Stage Alignment

**Training Data Generation:**
- `first_N_plays` mode: Clean context → 20 plays
- `remaining_plays` mode: Context + plays → next play

**Prediction System:**
- Stage 1 (optional): Clean context → ~20 plays
- Stage 2 (main): Context + plays → next play

This alignment ensures the model sees during prediction exactly what it learned during training.

### Model Versioning

Keep training and prediction synchronized:

```bash
# Training (generates data compatible with format v2.0)
python generate_2023_2024_season.py --data-format compact

# Fine-tune model (external step)
# Model learns format v2.0 patterns

# Prediction (uses format v2.0 compatible context)
export PREDICTION_PLATFORM=gemini
python predict_next_play.py  # Automatically uses matching format
```

### Best Practices

1. **Use the Same Format**: If you train with compact format, predict with compact format
2. **Match Window Sizes**: Use same `DEFAULT_N_TOTAL_PLAYS` in both pipelines
3. **Consistent Team Stats**: Use same stat calculation methods
4. **Aligned PCA**: Use same player profiling approach
5. **Version Control**: Track data format versions for reproducibility

### Example: Complete Cycle

```python
# 1. Generate training data (compact format)
python generate_2023_2024_season.py \
    --season 2023-2024 \
    --format gemini \
    --data-format compact \
    --output training_data_v2.jsonl

# 2. Fine-tune model (external - Gemini/OpenAI platform)
# Upload training_data_v2.jsonl
# Get model endpoint: projects/X/locations/Y/endpoints/Z

# 3. Configure prediction
export GEMINI_MODEL_2_ENDPOINT=projects/X/locations/Y/endpoints/Z

# 4. Build game context (same format as training)
from game_context_builder import GameContextBuilder
builder = GameContextBuilder("2023-2024")
context = builder.build_game_context("22200001")

# 5. Run prediction (model recognizes format from training)
from predict_next_play import predict_rolling_sequence
results = predict_rolling_sequence(context, n_iterations=750)

# 6. Validation ensures model output matches training format
# Automatic error recovery if format deviates
```

## 🔧 Advanced Features

### Performance Optimizations

The ULTRA_FAST pipeline includes:

1. **Comprehensive PCA Cache**: 99%+ hit rate (vs 70% baseline)
   - Pre-loads all player stats for all dates
   - Smart fallback to previous seasons
   - Batch processing instead of individual calculations

2. **Game-Batched Processing**: Eliminates redundant work
   - Shared team stats and player arrays per game
   - Pre-computed lineup lookups
   - Single compact record base per game

3. **Memory Cache**: 10-30x faster player stats
   - Pre-loads all player-date combinations into memory
   - Eliminates file I/O during processing
   - Intelligent sparse caching (only relevant combinations)

4. **Event Mapping Fix**: Proper structured field handling
   - Uses `type`, `event_type`, `result` fields
   - Accurate event classification (no more 'unknown' events)
   - Shot zone detection with coordinates

**Result**: 5-20x overall speedup, 300-600+ plays/second processing rate

### Player Statistics

#### PCA Profiles
Players are profiled using PCA on:
- **Offense**: PPG, APG, TS%, TO
- **Defense**: BPG, SPG, FPG
- **Shot Selection**: 3PR, FTR
- **Efficiency**: TS%, eFG%, TO

#### Play-by-Play Stats
Additional stats derived from play-by-play:
- Shot zone percentages (rim%, corner3%, nonc3%, mid%)
- Assist rates by shot type (a2%, a3%)
- Per-possession rates for individual efficiency (pts/poss, fga/poss, ast/poss, stl/poss, blk/poss)
- Usage and minutes
- Player archetype cluster (K=25 clustering)

### Team Statistics

Teams have 10-value stat arrays:
1. **OEFF**: Offensive efficiency (points per 100 possessions)
2. **DEFF**: Defensive efficiency (points allowed per 100 possessions)
3. **PACE**: Possessions per 48 minutes
4. **3PAr**: 3-point attempt rate (3PA / FGA)
5. **FTr**: Free throw rate (FTA / FGA)
6. **ORr**: Offensive rebound rate (OR / total rebounds)
7. **DRr**: Defensive rebound rate (DR / total rebounds)
8. **ASTr**: Assist rate (AST / FGM)
9. **TOr**: Turnover rate (TO / possessions)
10. **REST**: Rest days before game

## 📈 Usage Examples

### Complete Workflow: Training to Prediction

```bash
# 1. Generate training data
python generate_2023_2024_season.py \
    --season 2023-2024 \
    --format gemini \
    --data-format compact

# 2. Fine-tune your model (external - OpenAI/Gemini platform)

# 3. Configure prediction environment (choose one platform)
# Gemini
export PREDICTION_PLATFORM=gemini
export GEMINI_MODEL_1_ENDPOINT=your_stage1_endpoint
export GEMINI_MODEL_2_ENDPOINT=your_stage2_endpoint

# Together.ai (OpenAI-compatible)
# export PREDICTION_PLATFORM=together
# export TOGETHER_API_KEY=your_together_api_key
# export TOGETHER_MODEL_1_ID=your_stage1_model
# export TOGETHER_MODEL_2_ID=your_stage2_model
# Optional override (defaults to https://api.together.xyz/v1):
# export TOGETHER_BASE_URL=https://api.together.xyz/v1

# 4. Run predictions
python predict_next_play.py

# 5. Run systematic simulations
python orchestrator.py \
    --games "22200001,22200002,22200003" \
    --runs-per-game 10 \
    --season 2023-2024

# 6. Analyze results
python orchestrator.py --analyze
```

### Prediction Examples

#### Basic Prediction

```python
from predict_next_play import predict_rolling_sequence, init_prediction_client

# Initialize client
client, model_config = init_prediction_client()

# Build or load game context (compact format)
game_context = {
    "A": "ORL",
    "H": "CLE",
    "as": [113.7, 112.7, 96.5, 0.35, 0.26, 0.24, 0.63, 2],
    "hs": [114.9, 112.5, 97.2, 0.33, 0.28, 0.27, 0.60, 2],
    "ap": [...],  # Player arrays
    "hp": [...]   # Player arrays
}

# Run prediction
results = predict_rolling_sequence(
    game_context=game_context,
    n_iterations=100,
    skip_stage1=False  # Include Stage 1 generation
)

# Access results
print(f"Iterations: {len(results['iterations'])}")
for iteration in results['iterations'][-5:]:  # Last 5 plays
    play = iteration['next_play']
    print(f"Q{play['quarter']} {play['time_remaining']}: {play['description']}")
```

#### Custom Context from Historical Game

```python
from game_context_builder import GameContextBuilder

# Build context from real game
builder = GameContextBuilder("2023-2024")
context = builder.build_game_context(
    game_id="22200001",
    start_quarter=2,
    start_time="08:30",
    recent_plays_count=20
)

# Run prediction from this state
results = predict_rolling_sequence(
    game_context=context,
    n_iterations=200
)
```

#### Skip Stage 1 (Pre-loaded Context)

```python
# Load context with plays already included
import json
with open('saved_game_state.json', 'r') as f:
    game_context = json.load(f)

# Resume prediction from saved state
results = predict_rolling_sequence(
    game_context=game_context,
    n_iterations=500,
    skip_stage1=True  # Context already has plays
)
```

#### Multi-Game Orchestration

```python
from orchestrator import NBA_Orchestrator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    season_year="2023-2024",
    games=["22200001", "22200002", "22200003"],
    runs_per_game=20,
    max_iterations_per_run=1000,
    platform="gemini",
    output_db="results.db",
    skip_stage1=False
)

# Run orchestration
orchestrator = NBA_Orchestrator(config)
summary = orchestrator.run_all_simulations()

print(f"Total simulations: {summary['total_simulations']}")
print(f"Successful: {summary['successful']}")
print(f"Failed: {summary['failed']}")

# Analyze specific game
game_analysis = orchestrator.analyze_results("22200001")
print(f"Game 22200001 success rate: {game_analysis['success_rate']:.1f}%")
```

#### Custom Validation Mode

```python
import os

# Set strict validation for high accuracy
os.environ['VALIDATION_MODE'] = 'strict'

# Or fast validation for development
os.environ['VALIDATION_MODE'] = 'fast'

# Run with chosen validation mode
results = predict_rolling_sequence(game_context, n_iterations=100)
```

#### Detailed Logging for Debugging

```python
import os

# Enable debug logging
os.environ['PREDICTION_LOG_LEVEL'] = '3'  # Full JSON dumps

# Run prediction with detailed output
results = predict_rolling_sequence(game_context, n_iterations=50)

# Output will include:
# - Full JSON context sent to model
# - Raw model responses
# - Validation details
# - Cache statistics
```

#### Rate Limiting for Production

```python
import os

# Configure rate limiting
os.environ['GENAI_ENABLE_LIMITER'] = '1'
os.environ['GENAI_MAX_CONCURRENT'] = '2'
os.environ['GENAI_TARGET_RPS'] = '2.0'
os.environ['GENAI_LIMITER_JITTER'] = '0.1'

# Run with rate limiting active
results = predict_rolling_sequence(game_context, n_iterations=1000)
```

### Fine-Tune with Together.ai (New!)

Complete workflow for fast, cost-effective fine-tuning:

```bash
# Step 1: Generate training data
python generate_2023_2024_season.py \
    --format together \
    --season 2023-2024 \
    --games 100

# Step 2: Fine-tune model
export TOGETHER_API_KEY=your_api_key
python fine_tune_together.py \
    --training-file data/training/nba_2023_2024_together_*.jsonl \
    --validation-split 0.1 \
    --n-evals 10

# Step 3: Model is ready!
# Output: your_account/ft-llama-3.1-8b-nba-abc123
```

See [TOGETHER_AI_GUIDE.md](TOGETHER_AI_GUIDE.md) for complete documentation.

### Generate Training Data for Fine-Tuning

```bash
# Full season, compact format, Gemini
python generate_2023_2024_season.py \
    --season 2023-2024 \
    --format gemini \
    --data-format compact \
    --generation-mode remaining_plays

# Together.ai format (fast, cost-effective)
python generate_2023_2024_season.py \
    --season 2023-2024 \
    --format together \
    --data-format compact \
    --games 100

# Sample dataset for testing
python generate_2023_2024_season.py \
    --sample 5000 \
    --games 10 \
    --n-total 7

# First N plays mode for game openings
python generate_2023_2024_season.py \
    --generation-mode first_N_plays \
    --n-total 12
```

### Analyze Historical Data

```python
from generate_training_data import load_play_by_play_data
from transform_player_stats import calculate_player_stats

# Load season data
df = load_play_by_play_data("2023-2024")
print(f"Loaded {len(df):,} plays from {df['game_id'].nunique()} games")

# Analyze specific player
stats = calculate_player_stats("Stephen Curry", "2024-01-15", "2023-2024")
print(f"PPG: {stats['PPG']:.1f}, TS%: {stats['TS%']:.1%}")
```

### Run Production Simulations

```bash
# Large-scale systematic analysis
cat > production_config.json << EOF
{
  "season_year": "2023-2024",
  "games": ["22200001", "22200002", "22200003"],
  "runs_per_game": 100,
  "max_iterations_per_run": 2000,
  "platform": "gemini",
  "output_db": "production_results.db"
}
EOF

python orchestrator.py --config production_config.json
python orchestrator.py --analyze
```

## 📊 Data Sources

The system requires three data sources:

1. **Play-by-Play Data**: Historical play-by-play CSVs
   - Location: `data/play_by_play/historical/`
   - Format: `[start_date]-[end_date]-combined-stats.csv`
   - Seasons: 2022-2023, 2023-2024, 2024-2025

2. **Player Box Scores**: Per-game player statistics
   - Location: `data/player_boxscores/`
   - Format: Excel files with player game stats
   - Used for: MPG, usage rates, PCA profiles

3. **Team Statistics**: Aggregated team metrics
   - Generated dynamically from play-by-play data
   - Cached for performance
   - Season-aware with fallbacks

## 🔬 Technical Details

### Smart Season Fallback

The system handles early-season data intelligently:
- Uses previous season stats when current season has <10 games
- Prevents data leakage by only using prior data
- Batch processes fallbacks for efficiency

### Validation & Quality

- Game filtering: Requires ≥50 plays per game
- Data validation: Filters NaN descriptions
- Format verification: Validates JSON structure
- Event mapping: Cross-validates with multiple sources

### Thread Safety

The orchestration system includes:
- Thread-safe database access with locks
- WAL mode for concurrent SQLite writes
- Atomic transactions for result storage
- Progress tracking across threads

## 🛠️ Configuration

Key configuration files:

- `config/settings.py`: Global settings and season configuration
- `orchestrator_config.json`: Simulation parameters
- Environment variables for API keys

## 📝 Documentation

Additional documentation files:

- `PBP_STATS_IMPLEMENTATION_SUMMARY.md`: Play-by-play stats details
- `COMPREHENSIVE_PERFORMANCE_OPTIMIZATIONS.md`: Optimization guide
- `README_ORCHESTRATION.md`: Orchestration system details
- `SEASON_HANDLING_GUIDE.md`: Multi-season support
- `compact_schema_prompt.md`: Data format specification
- `TOGETHER_AI_GUIDE.md`: **Together.ai fine-tuning guide** (new!)

## 🧪 Testing

```bash
# Test data loading
python generate_training_data.py

# Quick generation test
python generate_2023_2024_season.py --games 2 --sample 100

# Verify formats
python validate_training_data.py data/training/output.jsonl

# Run orchestration test
python run_orchestration_example.py
```

## 🤝 Contributing

This is a research project for NBA game prediction. Key areas for contribution:

1. Additional data sources and features
2. Model architecture improvements
3. Prediction accuracy enhancements
4. Performance optimizations
5. Visualization and analysis tools

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- NBA play-by-play data sources
- OpenAI and Google for LLM APIs
- Basketball analytics community

## 📧 Contact

[Your Contact Information]

---

**Status**: Active Development
**Last Updated**: 2025-10-14
**Python Version**: 3.8+

