# NBA Prediction Performance Optimizations

## Implementation Summary

This document outlines the performance optimizations implemented in `predict_next_play.py` to reduce local processing overhead during the 450-prediction game simulation pipeline.

## Implemented Optimizations

### 1. 🚀 JSON Serialization Optimization (Priority 1)

**Problem:** Context objects were being serialized 2-3 times per prediction:
- Once in main loop for logging
- Again in prediction client
- Additional times for debugging output

**Solution:**
- **`OptimizedGameContext` class**: Caches base context JSON (teams, players, stats) which rarely changes
- **Recent plays caching**: Maintains a cache of serialized play sequences to avoid re-serialization
- **Eliminated double serialization**: Added `predict_from_json()` method to accept pre-serialized JSON
- **ujson support**: Optional fast JSON library (2-3x faster than standard json)

**Expected Performance Gain:** 20-35% reduction in local processing time

### 2. 📝 Logging Optimization (Priority 2)

**Problem:** Extensive debug output with full JSON dumps was consuming significant time:
- Verbose output for every iteration
- Large JSON dumps printed multiple times
- String formatting overhead

**Solution:**
- **`LoggingConfig` class**: Configurable logging levels (0-3)
- **Conditional logging**: Debug output only when needed
- **Lazy evaluation**: Expensive operations only in debug mode
- **Environment variable control**: `PREDICTION_LOG_LEVEL=0-3`

**Logging Levels:**
- `0`: Minimal - Essential messages only
- `1`: Normal - Standard progress (default)
- `2`: Verbose - Detailed iteration info + cache stats  
- `3`: Debug - Full JSON dumps + validation details

**Expected Performance Gain:** 15-25% reduction in local processing time

## Usage

### Environment Variables

```bash
# Set logging level (0=minimal, 1=normal, 2=verbose, 3=debug)
export PREDICTION_LOG_LEVEL=1

# Platform selection
export PREDICTION_PLATFORM=gemini  # or openai

# Optional: Install ujson for faster JSON operations
pip install ujson
```

### Performance Features

1. **JSON Caching**: Automatically caches base context and play sequences
2. **Smart Logging**: Only shows detailed output when requested
3. **Optimized Sliding Window**: Efficient recent_plays management
4. **Eliminated Double Serialization**: Direct JSON pass-through to API clients

## Architecture Changes

### New Classes

- **`OptimizedGameContext`**: Manages cached JSON serialization
- **`LoggingConfig`**: Centralized logging level control

### Enhanced Client Interface

- **`predict_from_json()`**: Accepts pre-serialized JSON to eliminate double serialization
- **Optimized OpenAI/Gemini clients**: Use cached JSON directly

### Backward Compatibility

All changes are backward compatible. Existing code will work unchanged but benefit from optimizations.

## Performance Expectations

### Conservative Estimates (450 predictions/game)
- **JSON Optimization**: 7-13.5 seconds saved per game
- **Logging Optimization**: 6.7-9 seconds saved per game
- **Combined**: 20-25 seconds saved per game (35-45% reduction in local overhead)

### Aggressive Estimates
- **Combined**: 25-35 seconds saved per game (50-65% reduction in local overhead)

### Runtime Context
- **Total game time**: ~15-22.5 minutes (API calls dominate)
- **Local overhead reduction**: 2-4% improvement in total runtime
- **Most valuable for**: High-frequency usage, user experience during long runs

## Cache Statistics

The optimized context provides real-time caching statistics:

```python
cache_stats = optimized_context.get_cache_stats()
# Returns: {
#   "plays_cache_size": 15,
#   "base_json_length": 12450, 
#   "current_plays_count": 20
# }
```

## Testing

Run with different logging levels to observe performance differences:

```bash
# Minimal logging for maximum performance
PREDICTION_LOG_LEVEL=0 python predict_next_play.py

# Debug mode to see all optimizations in action
PREDICTION_LOG_LEVEL=3 python predict_next_play.py
```

## Implementation Notes

- **Memory management**: Cache sizes are limited to prevent memory bloat
- **Thread safety**: Not required for single-threaded prediction pipeline
- **Error handling**: Graceful fallbacks if optimizations fail
- **Monitoring**: Cache hit ratios reported in verbose/debug modes

## Future Enhancements

Potential additional optimizations:
- **Async API calls**: Parallel validation and prediction
- **Response caching**: Cache similar prediction contexts
- **Validation optimization**: Compiled regex and early exit patterns
- **Memory pooling**: Pre-allocated data structures
