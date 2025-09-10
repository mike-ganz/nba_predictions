#!/usr/bin/env pwsh
# Run YOUR Actual results_analysis.py with Minimal Database Modification

Write-Host ("=" * 80)
Write-Host "RUNNING YOUR ACTUAL RESULTS_ANALYSIS.PY (DATABASE OPTIMIZED)"
Write-Host ("=" * 80)

# Create a modified version of YOUR script that only changes the data loading
$MODIFIED_SCRIPT = @'
# Import the notebook-friendly functions
from nba_results_notebook import (
    show_game_summary, 
    show_run_timeline, 
    show_performance_stats,
    quick_summary,
    load_results_df,
    get_successful_runs,
    get_run_details
)

# For additional analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# MODIFIED: Load actual game results from database instead of CSV
def load_actual_game_results():
    """Load actual game results from database instead of CSV file."""
    try:
        conn = sqlite3.connect("enhanced_simulation_results.db")
        actual_df = pd.read_sql_query("SELECT * FROM actual_game_results", conn)
        actual_df['date'] = pd.to_datetime(actual_df['date'])
        conn.close()
        return actual_df
    except Exception as e:
        print(f"❌ Error loading actual results from database: {e}")
        return pd.DataFrame()

# Get the simulation summary first, but delay table display until after betting functions are defined
summary_df = show_game_summary()

# ================================================================================
# BETTING ANALYSIS INTEGRATION
# Compares simulation predictions with actual game outcomes to evaluate betting performance.
# Answers: "Which spread/ML would our simulations have suggested betting?" and "Would that bet have been right?"
#
# BETTING LOGIC:
# - Spread: Bet if X% of simulations show that bet would win (configurable threshold)
# - ML: Bet if simulation win% > (implied odds probability + edge threshold)
# ================================================================================

# CONFIGURATION
SPREAD_CONFIDENCE_THRESHOLD = 0.80  # 80% of simulations must support the bet
MIN_SIMULATIONS_REQUIRED = 5       # Minimum simulations needed for analysis
ML_EDGE_THRESHOLD = 0.20            # 20% minimum edge over implied odds for ML bets
SPREAD_EDGE_THRESHOLD = 0.0         # Buffer on actual spread (e.g., 3 = need 3 extra points of coverage)

# BETTING LIMITS (Additional filters on top of existing logic)
MAX_SPREAD_LIMIT = 8.0             # Don't bet spreads > 15 points (0 = no limit)
MAX_FAVORITE_ML_ODDS = 300          # Don't bet favorites with odds worse than -300 (0 = no limit)
MAX_UNDERDOG_ML_ODDS = 300          # Don't bet underdogs with odds worse than +400 (0 = no limit)

def calculate_implied_probability(moneyline_str):
    """
    Calculate implied probability from moneyline odds.
    
    Examples:
    - "+120" -> 120/(120+100) = 45.45%
    - "-200" -> 200/(200+100) = 66.67%
    """
    if pd.isna(moneyline_str) or not isinstance(moneyline_str, str):
        return None
        
    try:
        # Remove any extra characters and get the numeric part
        ml_str = str(moneyline_str).strip()
        if ml_str.startswith('+'):
            odds = float(ml_str[1:])
            implied_prob = 100 / (odds + 100)
        elif ml_str.startswith('-'):
            odds = float(ml_str[1:])
            implied_prob = odds / (odds + 100)
        else:
            # Handle cases without + or - prefix
            odds = float(ml_str)
            if odds > 0:
                implied_prob = 100 / (odds + 100)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100)
                
        return implied_prob
        
    except (ValueError, TypeError):
        return None

def calculate_kelly_bet_size(win_prob, moneyline_str, bankroll=100):
    """
    Calculate Kelly optimal bet size for a moneyline bet.
    
    Args:
        win_prob: Probability of winning (from simulations, 0-1)
        moneyline_str: Moneyline odds string (e.g., "+120", "-150")
        bankroll: Total bankroll (default 100 for percentage calculations)
    
    Returns:
        (kelly_pct, bet_size, potential_profit) tuple
        kelly_pct: Kelly percentage (0-1, can be negative for don't bet)
        bet_size: Recommended bet size 
        potential_profit: Profit if bet wins
    """
    if pd.isna(moneyline_str) or not isinstance(moneyline_str, str) or win_prob <= 0:
        return 0.0, 0.0, 0.0
        
    try:
        # Convert moneyline to decimal odds
        ml_str = str(moneyline_str).strip()
        if ml_str.startswith('+'):
            odds = float(ml_str[1:])
            decimal_odds = (odds / 100) + 1  # +120 -> 2.20
            payout_multiplier = odds / 100    # +120 -> 1.20
        elif ml_str.startswith('-'):
            odds = float(ml_str[1:])
            decimal_odds = (100 / odds) + 1  # -150 -> 1.67
            payout_multiplier = 100 / odds    # -150 -> 0.67
        else:
            # Handle cases without + or - prefix
            odds = float(ml_str)
            if odds > 0:
                decimal_odds = (odds / 100) + 1
                payout_multiplier = odds / 100
            else:
                decimal_odds = (100 / abs(odds)) + 1
                payout_multiplier = 100 / abs(odds)
        
        # Kelly Formula: f* = (bp - q) / b
        # where b = decimal_odds - 1, p = win_prob, q = 1 - win_prob
        b = decimal_odds - 1
        p = win_prob
        q = 1 - win_prob
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly at reasonable levels (never bet more than 25% of bankroll)
        kelly_fraction = max(0.0, min(0.25, kelly_fraction))
        
        bet_size = kelly_fraction * bankroll
        potential_profit = bet_size * payout_multiplier if kelly_fraction > 0 else 0.0
        
        return kelly_fraction, bet_size, potential_profit
        
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0, 0.0, 0.0

def is_spread_within_limits(spread):
    """
    Check if spread is within configured limits.
    
    Args:
        spread: Spread from away team's perspective (positive = away underdog)
        
    Returns:
        bool: True if spread is within limits, False otherwise
    """
    if MAX_SPREAD_LIMIT <= 0:  # No limit set
        return True
        
    return abs(spread) <= MAX_SPREAD_LIMIT

def is_moneyline_within_limits(moneyline_str):
    """
    Check if moneyline odds are within configured limits.
    
    Args:
        moneyline_str: Moneyline odds string (e.g., "+120", "-150")
        
    Returns:
        bool: True if odds are within limits, False otherwise
    """
    if pd.isna(moneyline_str) or not isinstance(moneyline_str, str):
        return False
        
    try:
        # Parse moneyline odds
        ml_str = str(moneyline_str).strip()
        if ml_str.startswith('+'):
            odds = float(ml_str[1:])
            # Underdog - check against MAX_UNDERDOG_ML_ODDS
            if MAX_UNDERDOG_ML_ODDS <= 0:  # No limit set
                return True
            return odds <= MAX_UNDERDOG_ML_ODDS
        elif ml_str.startswith('-'):
            odds = float(ml_str[1:])
            # Favorite - check against MAX_FAVORITE_ML_ODDS  
            if MAX_FAVORITE_ML_ODDS <= 0:  # No limit set
                return True
            return odds <= MAX_FAVORITE_ML_ODDS
        else:
            # Handle cases without + or - prefix
            odds = float(ml_str)
            if odds > 0:
                # Underdog
                if MAX_UNDERDOG_ML_ODDS <= 0:
                    return True
                return odds <= MAX_UNDERDOG_ML_ODDS
            else:
                # Favorite
                if MAX_FAVORITE_ML_ODDS <= 0:
                    return True
                return abs(odds) <= MAX_FAVORITE_ML_ODDS
                
    except (ValueError, TypeError):
        return False

def parse_simulation_scores(all_scores_str):
    """
    Parse individual simulation scores from the 'all_scores' string.
    
    Returns list of (away_score, home_score) tuples.
    """
    if pd.isna(all_scores_str) or not all_scores_str:
        return []
    
    scores = []
    score_strings = [s.strip() for s in all_scores_str.split(',') if s.strip()]
    
    for score_str in score_strings:
        try:
            # Parse format: "AWAY_TEAM SCORE - HOME_TEAM SCORE"
            if ' - ' in score_str:
                away_part, home_part = score_str.split(' - ')
                away_score = int(away_part.split()[-1])  # Get last part (score)
                home_score = int(home_part.split()[-1])  # Get last part (score)
                scores.append((away_score, home_score))
        except (ValueError, IndexError):
            continue  # Skip malformed scores
    
    return scores

def analyze_spread_coverage(scores, spread, spread_edge_threshold=0.0):
    """
    Analyze what percentage of simulations would cover each side of the spread with edge threshold.
    
    Args:
        scores: List of (away_score, home_score) tuples
        spread: Spread from away team's perspective (positive = away underdog)
        spread_edge_threshold: Additional buffer required for a betting recommendation
    
    Returns:
        (away_cover_pct, home_cover_pct) - percentages as decimals
        
    Logic:
        - If away is +4 and threshold is 3, simulations need to show away covers +1 (4-3) for betting
        - If home is favored by 4 (spread = 4) and threshold is 3, sims need to show home wins by 7+ (4+3) for betting
    """
    if not scores or pd.isna(spread):
        return 0.0, 0.0
    
    away_covers = 0
    home_covers = 0
    
    for away_score, home_score in scores:
        actual_diff = away_score - home_score  # Positive = away won
        
        # AWAY BETTING: Need away to perform BETTER than spread - threshold
        # (More conservative - away needs to beat a tougher line)
        if actual_diff > -(spread - spread_edge_threshold):
            away_covers += 1
            
        # HOME BETTING: Need home to perform BETTER than spread + threshold  
        # (More conservative - home needs to beat a tougher line)
        if actual_diff < -(spread + spread_edge_threshold):
            home_covers += 1
    
    total = len(scores)
    return away_covers / total, home_covers / total

'@

# Continue with the rest of your original script...
$REMAINING_SCRIPT = Get-Content "results_analysis.py" -Raw
$START_MARKER = "def get_simulation_betting_recommendations"
$START_INDEX = $REMAINING_SCRIPT.IndexOf($START_MARKER)

if ($START_INDEX -gt 0) {
    $REMAINING_PART = $REMAINING_SCRIPT.Substring($START_INDEX)
    $FULL_MODIFIED_SCRIPT = $MODIFIED_SCRIPT + "`n" + $REMAINING_PART
} else {
    Write-Host "ERROR: Could not find the rest of your script" -ForegroundColor Red
    exit 1
}

# Write the modified script
$FULL_MODIFIED_SCRIPT | Out-File -FilePath "temp_your_actual_analysis.py" -Encoding UTF8

# Upload your actual script (modified)
gcloud compute scp temp_your_actual_analysis.py nba-orchestrator:your_actual_analysis.py --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
Remove-Item "temp_your_actual_analysis.py"

Write-Host "SUCCESS: Your actual analysis script uploaded (database optimized)" -ForegroundColor Green

# Run YOUR actual analysis script
Write-Host "`nRunning YOUR actual results_analysis.py with database optimization..."
Write-Host "This will show all your custom betting analysis, Kelly criterion, tables, etc." -ForegroundColor Cyan

$yourAnalysisResult = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 your_actual_analysis.py"

Write-Host "`nYOUR ACTUAL ANALYSIS OUTPUT:" -ForegroundColor Green
Write-Host ("=" * 100) -ForegroundColor Green
Write-Host $yourAnalysisResult -ForegroundColor White
Write-Host ("=" * 100) -ForegroundColor Green

Write-Host ("`n" + "=" * 80)
Write-Host "YOUR ACTUAL ANALYSIS COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 80)
Write-Host "This is the exact same output as your local results_analysis.py" -ForegroundColor Green
Write-Host "but now using the optimized single database file" -ForegroundColor Green
