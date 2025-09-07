# Error Analysis Tools Guide

Two powerful scripts to analyze SQLite write errors and validation terminations from your NBA simulations.

## 🚀 Quick Start

### Immediate Error Check (After Running Simulations)
```bash
# Check for errors in the last 24 hours
python quick_error_check.py

# Check last 6 hours only
python quick_error_check.py --hours 6

# Check last 2 hours with specific database
python quick_error_check.py --hours 2 --db my_results.db
```

### Detailed Error Analysis
```bash
# Full analysis of last 7 days
python error_analysis.py

# Last 3 days only
python error_analysis.py --days 3

# Show only database write errors
python error_analysis.py --database-errors-only --days 5

# Show only validation terminations
python error_analysis.py --validation-only --days 7

# Pattern analysis only
python error_analysis.py --patterns-only --days 10
```

### Export Comprehensive Report
```bash
# Export to Excel file
python error_analysis.py --export error_report.xlsx --days 7
```

## 📊 What You'll See

### Quick Error Check Output
```
⚡ QUICK ERROR CHECK - Last 24 hours
============================================================
🕒 Time: 2024-01-15 14:30:22

🔥 15 Database Save Errors Found!
   📊 Error Types:
      • database_locked: 12
      • disk_io_error: 3
   
   🎮 Games Affected: 3
      22400061, 22400062, 22400063
   
   🕒 Most Recent Errors:
      1. [2024-01-15 14:25:33,123] PERMANENT FAILURE: LAL_BOS_2024_0001 - Failed after 3 attempts...
      2. [2024-01-15 14:22:15,456] RETRY 2/3: game_data_2024_0002 - database is locked...
      3. [2024-01-15 14:18:42,789] database is locked - Retrying in 0.2s...

🛑 5 Validation Terminations Found!
   📊 Termination Types:
      • game_ended: 4
      • rollback_time: 1
   
   🕒 Most Recent Terminations:
      1. Game 22400063: game_ended (quarter_4_time_expired)
      2. Game 22400062: rollback_time (consecutive_same_timestamp_limit_reached)
```

### Detailed Analysis Features

#### 🔥 Database Write Errors
- **Error Classification**: `database_locked`, `disk_io_error`, `permanent_failure`, `retry_attempt`
- **Thread Tracking**: Which threads encountered errors
- **Game Impact**: Which games were affected
- **Timeline**: When errors occurred
- **Error Messages**: Full error context

#### 🛑 Validation Terminations
- **Termination Types**: `game_ended`, `rollback_time`
- **Trigger Conditions**: `quarter_4_time_expired`, `consecutive_same_timestamp_limit_reached`, etc.
- **Game State**: Quarter, time remaining, score when terminated
- **Validation Context**: Consecutive counts, total attempts
- **Pattern Analysis**: Which games terminate most frequently

#### 📈 Pattern Analysis
- **Error Hotspots**: Which games cause the most issues
- **Success Rates**: Overall simulation completion rates
- **Status Distribution**: Completed vs failed simulations
- **Trending Issues**: Are errors increasing or decreasing?

## 🔍 Understanding the Errors

### Database Write Errors

| Error Type | Cause | Solution |
|------------|-------|----------|
| `database_locked` | Multiple threads accessing SQLite simultaneously | Reduce thread count or enable WAL mode |
| `disk_io_error` | Storage issues or permissions | Check disk space and file permissions |
| `permanent_failure` | Repeated save failures after retries | Check database file integrity |
| `retry_attempt` | Temporary failures being retried | Normal during high concurrency |

### Validation Terminations

| Termination Type | Trigger | Meaning |
|------------------|---------|---------|
| `game_ended` | `quarter_4_time_expired` | Game naturally ended (Q4 00:00) |
| `game_ended` | `consecutive_endgame_scenarios_limit_reached` | Too many end-game situations |
| `rollback_time` | `consecutive_same_timestamp_limit_reached` | Too many plays with same timestamp |

## 🛠️ Troubleshooting Common Issues

### High Database Lock Errors
```bash
# Check if you're running too many threads
python error_analysis.py --patterns-only --days 3

# Reduce thread count in your orchestrator runs
python enhanced_orchestrator.py --games "..." --threads 3  # Instead of 10
```

### Frequent Validation Terminations
```bash
# Analyze which games terminate most
python error_analysis.py --validation-only --days 7

# Look for patterns in specific games
python error_analysis.py --export validation_analysis.xlsx --days 14
```

### Missing Simulation Results
```bash
# Check if simulations ran but results weren't saved
python error_analysis.py --database-errors-only --days 5

# Compare with orchestrator logs to identify missing results
```

## 📁 File Locations

- **Database Error Logs**: `database_errors_*.log` (timestamped filenames)
- **Simulation Database**: `enhanced_simulation_results.db` (default)
- **Export Reports**: `error_report.xlsx` or custom filename

## 🎯 Best Practices

1. **Run Quick Check** after each simulation session
2. **Weekly Analysis** to identify trends
3. **Export Reports** for detailed investigation
4. **Monitor Thread Count** vs error rate correlation
5. **Check Disk Space** if seeing I/O errors

## 🚨 When to Be Concerned

- **>10% database save failure rate** - Consider reducing thread count
- **Frequent rollback_time terminations** - May indicate model issues
- **Disk I/O errors** - Check storage health
- **All games failing** - Configuration or environment issue

## 📞 Integration with Other Tools

```bash
# Use with existing results analysis
python results_analysis.py
python quick_error_check.py  # Check for issues with recent results

# Chain commands for comprehensive analysis
python enhanced_orchestrator.py --games "..." --threads 5 && python quick_error_check.py --hours 1
```
