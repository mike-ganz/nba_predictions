# NBA Prediction Orchestration System

A comprehensive orchestration layer for running multiple NBA prediction simulations systematically. Built on top of your existing `predict_next_play.py` system.

## 🎯 Overview

This orchestration system allows you to:
- **Run multiple simulations** for any number of games
- **Store results systematically** in SQLite database
- **Configure seasons, games, and parameters** easily  
- **Analyze outcomes** with built-in reporting tools
- **Handle errors gracefully** with automatic recovery
- **Track progress** across long-running experiments

## 🏗️ Architecture

```
Orchestration System
├── orchestrator.py              # Main orchestration engine
├── game_context_builder.py      # Game context creation from data
├── results_analyzer.py          # Results analysis and reporting  
├── orchestrator_config.json     # Configuration template
└── run_orchestration_example.py # Usage examples
```

## 🚀 Quick Start

### 1. Basic Usage

Run 10 simulations each for 3 different games:

```bash
python orchestrator.py --games "22200001,22200002,22200003" --runs-per-game 10 --season 2023-2024
```

### 2. Configuration File

Create a JSON configuration file:

```json
{
  "season_year": "2023-2024",
  "games": ["22200001", "22200002", "22200003"],
  "runs_per_game": 50,
  "max_iterations_per_run": 2000,
  "platform": "gemini",
  "output_db": "my_results.db"
}
```

Run with configuration:

```bash
python orchestrator.py --config my_config.json
```

### 3. Analyze Results

```bash
# Analyze all results
python orchestrator.py --analyze

# Analyze specific game
python orchestrator.py --analyze-game 22200001

# Generate detailed report with plots
python results_analyzer.py --db simulation_results.db --report analysis.txt --plots
```

## 📋 Configuration Options

### SimulationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `season_year` | str | "2023-2024" | NBA season to draw data from |
| `games` | List[str] | - | Game IDs to simulate |
| `runs_per_game` | int | 5 | Number of simulations per game |
| `max_iterations_per_run` | int | 2000 | Max predictions per simulation |
| `skip_stage1` | bool | true | Skip initial play generation |
| `platform` | str | "gemini" | AI platform (gemini/openai) |
| `output_db` | str | "simulation_results.db" | SQLite output file |
| `log_level` | str | "INFO" | Logging verbosity |
| `resume_on_error` | bool | true | Continue after errors |
| `timeout_minutes` | int | 30 | Max time per simulation |

## 📊 Database Schema

The orchestration system stores results in SQLite with this schema:

```sql
CREATE TABLE simulation_runs (
    run_id TEXT PRIMARY KEY,           -- Unique run identifier
    game_id TEXT NOT NULL,             -- NBA game ID
    season_year TEXT NOT NULL,         -- Season (YYYY-YYYY)
    platform TEXT NOT NULL,            -- AI platform used
    start_time TEXT NOT NULL,          -- Simulation start
    end_time TEXT,                     -- Simulation end  
    duration_seconds REAL,             -- Total runtime
    status TEXT NOT NULL,              -- completed/error/timeout
    total_predictions INTEGER,         -- Total predictions made
    successful_predictions INTEGER,    -- Successful predictions
    final_score TEXT,                  -- Final game score
    final_quarter INTEGER,             -- Final quarter reached
    final_time TEXT,                   -- Final time remaining
    termination_reason TEXT,           -- Why simulation ended
    scoring_plays INTEGER,             -- Number of scoring plays
    non_scoring_plays INTEGER,         -- Number of non-scoring plays
    scoring_rate REAL,                 -- Percentage of scoring plays
    error_message TEXT,                -- Error details if failed
    iterations_data TEXT,              -- JSON of first 10 iterations
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## 🎮 Game Context System

### Using Existing Game Data

The `GameContextBuilder` integrates with your existing data loading system:

```python
from game_context_builder import GameContextBuilder

# Build contexts from your historical data
builder = GameContextBuilder("2023-2024")

# Get available games from your data
games = builder.get_available_games()

# Build specific game context
context = builder.build_game_context("22200001")

# Batch build multiple contexts
contexts = builder.batch_build_contexts(["22200001", "22200002"])
```

### Custom Game Contexts

You can also create custom contexts:

```python
# Build context starting from specific game state
context = builder.build_game_context(
    game_id="22200001",
    start_quarter=2,          # Start from 2nd quarter
    start_time="08:30",       # 8:30 remaining
    recent_plays_count=15     # Include 15 recent plays
)
```

## 📈 Analysis Tools

### Built-in Analysis

```python
from orchestrator import NBA_Orchestrator

orchestrator = NBA_Orchestrator(config)
summary = orchestrator.run_all_simulations()

# Analyze specific game
analysis = orchestrator.analyze_results("22200001")
print(f"Success rate: {analysis['success_rate']:.1f}%")
print(f"Avg scoring rate: {analysis['avg_scoring_rate']:.1f}%")
```

### Advanced Analysis

```python
from results_analyzer import SimulationAnalyzer

analyzer = SimulationAnalyzer("simulation_results.db")

# Game-by-game summary
summary = analyzer.game_summary()

# Scoring patterns analysis  
patterns = analyzer.scoring_patterns_analysis()

# Platform comparison (if using multiple)
comparison = analyzer.platform_comparison()

# Generate comprehensive report
report = analyzer.generate_report("full_analysis.txt")

# Create visualizations
analyzer.create_visualizations("plots/")
```

## 🔧 Command Line Interface

### orchestrator.py

```bash
# Basic orchestration
python orchestrator.py --games "22200001,22200002" --runs-per-game 5

# Advanced configuration
python orchestrator.py \
    --games "22200001,22200002,22200003" \
    --runs-per-game 20 \
    --season 2023-2024 \
    --platform gemini \
    --max-iterations 1500 \
    --output-db my_results.db \
    --log-level DEBUG

# Analysis
python orchestrator.py --analyze
python orchestrator.py --analyze-game 22200001
```

### game_context_builder.py

```bash
# List available games
python game_context_builder.py --list-games --season 2023-2024

# Build contexts for specific games
python game_context_builder.py --games "22200001,22200002"

# Build random sample
python game_context_builder.py --sample 5 --output-dir contexts/
```

### results_analyzer.py

```bash
# Basic analysis
python results_analyzer.py --db simulation_results.db

# Full report with visualizations
python results_analyzer.py \
    --db simulation_results.db \
    --report analysis_report.txt \
    --plots \
    --plot-dir analysis_plots/
```

## 📊 Example Workflows

### 1. Small-Scale Testing

Test your setup with quick runs:

```bash
# Quick test with 3 games, 2 runs each, short simulations
python orchestrator.py \
    --games "22200001,22200002,22200003" \
    --runs-per-game 2 \
    --max-iterations 100 \
    --log-level INFO
```

### 2. Production-Scale Analysis

Large-scale systematic analysis:

```json
{
  "season_year": "2023-2024", 
  "games": ["22200001", "22200002", "22200003", "22200004", "22200005"],
  "runs_per_game": 100,
  "max_iterations_per_run": 2000,
  "platform": "gemini",
  "output_db": "production_results.db",
  "log_level": "INFO"
}
```

```bash
python orchestrator.py --config production_config.json
```

### 3. Platform Comparison

Compare OpenAI vs Gemini performance:

```bash
# Run with OpenAI
export PREDICTION_PLATFORM=openai
python orchestrator.py --games "22200001,22200002" --runs-per-game 10 --output-db openai_results.db

# Run with Gemini  
export PREDICTION_PLATFORM=gemini
python orchestrator.py --games "22200001,22200002" --runs-per-game 10 --output-db gemini_results.db

# Analyze both
python results_analyzer.py --db openai_results.db --report openai_analysis.txt
python results_analyzer.py --db gemini_results.db --report gemini_analysis.txt
```

## 🛠️ Integration with Existing System

The orchestration system is designed to work seamlessly with your existing prediction infrastructure:

### Environment Variables

```bash
# Set platform (same as predict_next_play.py)
export PREDICTION_PLATFORM=gemini  # or openai

# Set logging level for prediction script
export PREDICTION_LOG_LEVEL=1      # 0=minimal, 1=normal, 2=verbose, 3=debug

# OpenAI configuration
export OPENAI_API_KEY=your_key

# Gemini configuration  
export GEMINI_MODEL_1_ENDPOINT=your_endpoint_1
export GEMINI_MODEL_2_ENDPOINT=your_endpoint_2
export GOOGLE_CLOUD_PROJECT=your_project
```

### Data Integration

The orchestration system automatically uses:
- Your existing `config/settings.py` for configuration
- Your data loading system in `data/loaders.py`
- Your player analysis in `analysis/player_stats.py` 
- Your existing prediction client system

## 🚨 Error Handling

The system includes comprehensive error handling:

### Automatic Recovery
- Failed simulations are logged and skipped
- Database transactions are atomic 
- Progress is saved incrementally
- System continues with remaining simulations

### Error Analysis
```python
# Analyze what went wrong
analyzer = SimulationAnalyzer("results.db")
failures = analyzer.failure_analysis()

print(f"Failure rate: {failures['failure_rate']:.1f}%")
print(f"Common errors: {failures['common_errors']}")
```

### Timeout Handling
- Individual simulation timeouts prevent hanging
- Configurable timeout limits per simulation
- Graceful termination and cleanup

## 📊 Performance Monitoring

### Progress Tracking
- Real-time progress updates during orchestration
- Duration tracking per simulation
- Success/failure rates calculated on-the-fly

### Resource Usage
- Memory usage kept reasonable with data pagination
- Database writes optimized for bulk operations
- Optional cleanup of temporary data

### Logging Levels
- **INFO**: Standard progress updates
- **DEBUG**: Detailed simulation information  
- **WARNING**: Non-critical issues
- **ERROR**: Failed simulations and system errors

## 🎯 Results Analysis

### Key Metrics Tracked

1. **Success Metrics**
   - Success rate per game
   - Average predictions per simulation
   - Completion rates

2. **Game Dynamics**
   - Scoring rates and patterns
   - Game termination reasons
   - Final scores and quarters

3. **Performance Metrics**
   - Simulation durations
   - Error rates and types
   - Platform comparisons

4. **Statistical Analysis**
   - Distribution analysis
   - Variance measurements
   - Outlier detection

## 🔄 Advanced Usage

### Custom Context Loading

```python
# Create custom context loader
class CustomContextLoader:
    def load_game_context(self, game_id, season_year):
        # Your custom logic here
        return custom_context

# Use in orchestrator
orchestrator.context_loader = CustomContextLoader()
```

### Simulation Filtering

```python
# Filter results for analysis
analyzer = SimulationAnalyzer("results.db")

# Only analyze successful simulations  
df = analyzer.df[analyzer.df['status'] == 'completed']

# Analyze high-scoring simulations
high_scoring = df[df['scoring_rate'] > 25]
```

### Batch Processing

```python
# Run orchestration for multiple seasons
seasons = ["2022-2023", "2023-2024"]
for season in seasons:
    config.season_year = season
    config.output_db = f"results_{season.replace('-', '_')}.db"
    orchestrator = NBA_Orchestrator(config)
    orchestrator.run_all_simulations()
```

## 🎉 Getting Started

1. **Install dependencies** (same as your existing system)
2. **Set environment variables** for your AI platform
3. **Run the example** to verify setup:
   ```bash
   python run_orchestration_example.py
   ```
4. **Configure your first orchestration**:
   ```bash
   python orchestrator.py --games "22200001" --runs-per-game 3 --max-iterations 100
   ```
5. **Analyze results**:
   ```bash
   python orchestrator.py --analyze
   ```

The orchestration system is designed to be powerful yet easy to use, providing systematic testing capabilities while preserving all the sophistication of your existing prediction system.
