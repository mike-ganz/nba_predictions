# NBA Training Data Generator - Migration Guide

## Overview

This guide helps you transition from the original 988-line monolithic `generate_training_data.py` script to the new modular architecture. The refactoring maintains **100% backward compatibility** while providing better organization, maintainability, and extensibility.

## 🏗️ New Architecture

### Package Structure
```
nba_predictions/
├── config/                    # Configuration management
│   ├── __init__.py
│   └── settings.py           # Centralized configuration
├── data/                      # Data loading and file operations
│   ├── __init__.py
│   ├── loaders.py            # Data loading utilities
│   └── file_utils.py         # File I/O and previewing
├── game/                      # Game-specific utilities  
│   ├── __init__.py
│   ├── time_utils.py         # Time and quarter calculations
│   ├── team_utils.py         # Team management and mappings
│   └── scoring_utils.py      # Scoring analysis
├── analysis/                  # Player and statistical analysis
│   ├── __init__.py
│   └── player_stats.py       # PCA scores and lineup management
├── training/                  # Core training data generation
│   ├── __init__.py
│   ├── data_generator.py     # Main training data logic
│   └── openai_formatter.py   # OpenAI format conversion
├── main.py                    # New CLI interface
└── generate_training_data.py  # Original script (still works!)
```

## 🔄 Migration Options

### Option 1: Immediate Switch (Recommended)
Use the new CLI interface which provides all functionality:

```bash
# Replace direct script execution
python main.py test
python main.py generate-game 22200001
python main.py generate-dataset --sample-size 1000
```

### Option 2: Gradual Migration
Keep using existing code while gradually adopting new modules:

```python
# Your existing code still works unchanged
from generate_training_data import generate_llm_dataset, create_llm_training_data

# But you can also use the new modular API
from training import training_data_generator
from config import set_season_year
```

### Option 3: Keep Original (Zero Changes)
The original `generate_training_data.py` continues to work exactly as before. No changes required.

## 📋 Function Mapping

### Configuration Management
| **Old** | **New** |
|---------|---------|
| `set_season_year()` | `from config import set_season_year` |
| `get_current_season_year()` | `from config import get_current_season_year` |
| `SEASON_YEAR` global | `config.season_year` property |

### Data Loading
| **Old** | **New** |
|---------|---------|
| `load_play_by_play_data()` | `from data import load_play_by_play_data` |
| `load_training_data()` | `from data import load_training_data` |
| `test_data_loading()` | `from data import test_data_loading` |

### Core Training Data Generation
| **Old** | **New** |
|---------|---------|
| `create_llm_training_data()` | `from training import create_llm_training_data` |
| `generate_llm_dataset()` | `from training import generate_llm_dataset` |
| `generate_training_data_for_game()` | `from training import generate_training_data_for_game` |

### OpenAI Integration
| **Old** | **New** |
|---------|---------|
| `create_openai_training_data()` | `from training import create_openai_training_data` |
| `save_openai_training_data()` | `from training import save_openai_training_data` |

### Utility Functions
| **Old** | **New** |
|---------|---------|
| `calculate_game_time_remaining()` | `from game import calculate_game_time_remaining` |
| `determine_home_away_teams()` | `from game import determine_home_away_teams` |
| `get_player_pca_score()` | `from analysis import get_player_pca_score` |

## 🚀 New CLI Interface

### Basic Usage
```bash
# Test everything works
python main.py test

# Show configuration
python main.py config --show

# Set season
python main.py config --set-season 2023-2024
```

### Generate Training Data
```bash
# Single game
python main.py generate-game 22200001

# Full dataset with sampling
python main.py generate-dataset --sample-size 1000

# Specific games only
python main.py generate-dataset --games "22200001,22200002,22200003"

# OpenAI format
python main.py generate-openai --game-id 22200001
```

### Data Management
```bash
# List available datasets
python main.py list-datasets

# Preview training data
python main.py preview my_dataset.csv

# Preview OpenAI format
python main.py preview openai_training.jsonl
```

## 💡 Code Examples

### Before (Original)
```python
# Old monolithic approach
from generate_training_data import *

# Set configuration
set_season_year("2023-2024")

# Generate data
df = load_play_by_play_data("2023-2024")
result_df = create_llm_training_data(df, n_total=5)
save_llm_dataset(result_df, "training_data.csv")
```

### After (Modular)
```python
# New modular approach - same functionality, better organization
from config import set_season_year
from data import data_loader
from training import training_data_generator
from data import file_manager

# Set configuration
set_season_year("2023-2024")

# Generate data (exact same result)
df = data_loader.load_play_by_play_data("2023-2024")
result_df = training_data_generator.create_llm_training_data(df, n_total=5)
file_manager.save_llm_dataset(result_df, "training_data.csv")
```

### Advanced Usage
```python
# Object-oriented approach for better control
from training import training_data_generator
from analysis import player_analyzer

# Generate with caching control
result_df = training_data_generator.generate_dataset(
    season_year="2023-2024",
    n_total=5,
    sample_size=1000
)

# Clear caches when done
stats = training_data_generator.clear_caches()
print(f"Cleared {stats['team_stats_entries']} cached entries")
```

## 🎯 Benefits of New Architecture

### 1. **Single Responsibility Principle**
- Each module handles one concern
- Easier to test and maintain
- Clearer code organization

### 2. **Better Performance**
- Smart caching systems
- Optimizations for small vs large datasets
- Memory usage improvements

### 3. **Enhanced Developer Experience**
- Comprehensive CLI with help system
- Better error messages and logging
- Type hints and documentation

### 4. **Extensibility**
- Easy to add new data sources
- Plugin-like architecture
- Clean interfaces for customization

### 5. **Testing & Debugging**
- Isolated components for unit testing
- Better error isolation
- Configurable logging levels

## ⚠️ Important Notes

### Backward Compatibility
- **All existing imports still work**
- **All function signatures unchanged**
- **Same output formats**
- **No breaking changes**

### Performance
- New architecture includes optimizations
- Caching reduces redundant computations
- Better memory management for large datasets

### Dependencies
- No new external dependencies required
- Uses same imports as original script

## 🆘 Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/nba_predictions
python main.py test
```

### Module Not Found
If modules aren't found, check your Python path:
```python
import sys
sys.path.append('/path/to/nba_predictions')
```

### Configuration Issues
Reset configuration to defaults:
```python
from config import config
config.season_year = "2023-2024"  # Reset to default
```

## 🚀 Next Steps

1. **Start Simple**: Use `python main.py test` to verify everything works
2. **Try CLI**: Use the new command-line interface for common tasks
3. **Gradual Migration**: Slowly replace direct imports with modular ones
4. **Customize**: Extend individual modules for your specific needs

## 📞 Support

The modular architecture maintains 100% compatibility with your existing code while providing modern software engineering practices. You can adopt it at your own pace - from keeping everything exactly the same to fully embracing the new modular structure.
