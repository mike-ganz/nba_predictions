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

# ================================================================================
# CONFIGURATION - Change these values to adjust filtering and analysis
# ================================================================================

# ITERATION FILTERING CONFIGURATION
MAX_ITERATIONS_THRESHOLD = 550      # Maximum iterations to consider realistic (filters out runaway simulations)
MIN_ITERATIONS_THRESHOLD = 325      # Minimum iterations to consider complete (filters out stuck/incomplete games)

# SCORE FILTERING CONFIGURATION
MAX_TEAM_SCORE_THRESHOLD = 175      # Maximum team score to consider realistic (filters out unrealistic high-scoring games)

# BETTING ANALYSIS CONFIGURATION
SPREAD_CONFIDENCE_THRESHOLD = 0.60  # 60% of simulations must support the bet
MIN_SIMULATIONS_REQUIRED = 1       # Minimum simulations needed for analysis
ML_EDGE_THRESHOLD = 0.10            # 10% minimum edge over implied odds for ML bets
SPREAD_EDGE_THRESHOLD = 0.0         # Buffer on actual spread (e.g., 3 = need 3 extra points of coverage)


# BETTING LIMITS (Additional filters on top of existing logic)
MAX_SPREAD_LIMIT = 0.0             # Don't bet spreads > N points (0 = no limit)
MAX_FAVORITE_ML_ODDS = 600          # Don't bet favorites with odds worse than -300 (0 = no limit)
MAX_UNDERDOG_ML_ODDS = 600          # Don't bet underdogs with odds worse than +400 (0 = no limit)

# ================================================================================
# DATABASE CONFIGURATION - Change database path here only
# ================================================================================
# To use a different database file, simply change the path below:
# Examples:
#   DATABASE_PATH = "enhanced_simulation_results_20250915_113427.db"  # Specific timestamp
#   DATABASE_PATH = "enhanced_simulation_results_latest.db"           # Latest results
#   DATABASE_PATH = "enhanced_simulation_results_current.db"          # Current results
DATABASE_PATH = "enhanced_simulation_results.db"

# Historical database path (contains previous runs to include in analysis)
HISTORICAL_DATABASE_PATH = ""

# Define helper functions needed for filtering and combining databases
def has_realistic_scores(final_score_str):
    """
    Check if a simulation has realistic team scores (both teams < MAX_TEAM_SCORE_THRESHOLD).
    
    Args:
        final_score_str: Single final score string (e.g., "AWAY_TEAM 110 - HOME_TEAM 105")
        
    Returns:
        bool: True if both scores are realistic, False if either team scored >= MAX_TEAM_SCORE_THRESHOLD
    """
    if pd.isna(final_score_str) or not final_score_str.strip():
        return False
    
    try:
        # Parse format: "AWAY_TEAM SCORE - HOME_TEAM SCORE"
        if ' - ' in final_score_str:
            away_part, home_part = final_score_str.split(' - ')
            away_score = int(away_part.split()[-1])  # Get last part (score)
            home_score = int(home_part.split()[-1])  # Get last part (score)
            
            # Both scores must be below threshold
            return away_score < MAX_TEAM_SCORE_THRESHOLD and home_score < MAX_TEAM_SCORE_THRESHOLD
    except (ValueError, IndexError):
        return False  # Skip malformed scores
    
    return False

def get_combined_raw_simulation_data(db_paths=[DATABASE_PATH, HISTORICAL_DATABASE_PATH]):
    """
    Get raw simulation data from multiple databases and combine before grouping.
    This prevents duplicate games when the same game exists in multiple databases.
    
    Args:
        db_paths: List of database paths to query
    
    Returns:
        Combined pandas DataFrame with all raw simulation records
    """
    combined_data = []
    
    for db_path in db_paths:
        if not Path(db_path).exists():
            continue
            
        try:
            conn = sqlite3.connect(db_path)
            # Get ALL simulation records (raw data) instead of pre-grouped data
            raw_query = """
            SELECT 
                game_id,
                season_year,
                status,
                successful_predictions,
                duration_seconds,
                final_score
            FROM simulation_runs
            """
            
            df = pd.read_sql_query(raw_query, conn)
            conn.close()
            
            if not df.empty:
                df['source_database'] = db_path  # Track which database each record came from
                combined_data.append(df)
                
        except Exception as e:
            continue
    
    if combined_data:
        result = pd.concat(combined_data, ignore_index=True)
        return result
    else:
        return pd.DataFrame()

def get_combined_simulation_data(query_type="summary", db_paths=[DATABASE_PATH, HISTORICAL_DATABASE_PATH]):
    """
    Get properly aggregated simulation data from multiple databases.
    Combines raw data first, then groups to avoid duplicates.
    
    Args:
        query_type: Type of aggregation needed ("summary" or "betting")
        db_paths: List of database paths to query
    
    Returns:
        Properly grouped DataFrame without duplicates
    """
    # Get all raw data first
    raw_data = get_combined_raw_simulation_data(db_paths)
    
    if raw_data.empty:
        return pd.DataFrame()
    
    if query_type == "summary":
        # Group the combined raw data for summary display
        # Filter by iteration count, natural termination status, AND realistic scores
        completed_games = raw_data[
            (raw_data['status'].isin(['game_ended', 'completed'])) & 
            (raw_data['successful_predictions'] <= MAX_ITERATIONS_THRESHOLD) &
            (raw_data['successful_predictions'] >= MIN_ITERATIONS_THRESHOLD) &
            (raw_data['final_score'].apply(has_realistic_scores))  # Filter out unrealistic high scores
        ]
        
        if completed_games.empty:
            return pd.DataFrame()
        
        # Group by game_id and season_year across ALL databases
        summary_grouped = completed_games.groupby(['game_id', 'season_year']).agg({
            'status': 'count',  # total_sims (count any column)
            'successful_predictions': ['mean', 'min', 'max'],
            'duration_seconds': 'mean',
            'final_score': lambda x: ','.join(x.dropna().astype(str)),  # all_scores
            'source_database': lambda x: '|'.join(set(x))  # track all source databases
        }).reset_index()
        
        # Flatten multi-level column names properly
        if isinstance(summary_grouped.columns, pd.MultiIndex):
            summary_grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in summary_grouped.columns.values]
        
        # Create a mapping based on actual column names
        current_columns = list(summary_grouped.columns)
        
        # Find and rename columns based on patterns
        column_renames = {}
        
        for col in current_columns:
            if 'status' in str(col) and 'count' in str(col):
                column_renames[col] = 'completed_sims'
            elif 'successful_predictions' in str(col) and 'mean' in str(col):
                column_renames[col] = 'avg_predictions'
            elif 'successful_predictions' in str(col) and 'min' in str(col):
                column_renames[col] = 'min_predictions'
            elif 'successful_predictions' in str(col) and 'max' in str(col):
                column_renames[col] = 'max_predictions'
            elif 'duration_seconds' in str(col) and 'mean' in str(col):
                column_renames[col] = 'avg_duration_min'
            elif 'final_score' in str(col):
                column_renames[col] = 'all_scores'
            elif 'source_database' in str(col):
                column_renames[col] = 'source_databases'
        
        # Apply all renames at once
        summary_grouped = summary_grouped.rename(columns=column_renames)
        
        # Convert duration from seconds to minutes (with safety check)
        if 'avg_duration_min' in summary_grouped.columns:
            summary_grouped['avg_duration_min'] = (summary_grouped['avg_duration_min'] / 60).round(1)
        if 'avg_predictions' in summary_grouped.columns:
            summary_grouped['avg_predictions'] = summary_grouped['avg_predictions'].round(1)
        
        # Add other required fields for compatibility
        if 'completed_sims' in summary_grouped.columns:
            summary_grouped['total_sims'] = summary_grouped['completed_sims']  # For now, assume all are completed
        else:
            summary_grouped['total_sims'] = 0
            
        summary_grouped['error_sims'] = 0  # We filtered to only completed games
        
        if 'all_scores' in summary_grouped.columns:
            summary_grouped['final_scores'] = summary_grouped['all_scores']  # For compatibility
        else:
            summary_grouped['final_scores'] = ''
        
        return summary_grouped
        
    elif query_type == "betting":
        # Group for betting analysis
        # Filter by iteration count, natural termination status, AND realistic scores
        completed_games = raw_data[
            (raw_data['status'].isin(['game_ended', 'completed'])) & 
            (raw_data['successful_predictions'] <= MAX_ITERATIONS_THRESHOLD) &
            (raw_data['successful_predictions'] >= MIN_ITERATIONS_THRESHOLD) &
            (raw_data['final_score'].apply(has_realistic_scores))  # Filter out unrealistic high scores
        ]
        
        if completed_games.empty:
            return pd.DataFrame()
            
        betting_grouped = completed_games.groupby(['game_id', 'season_year']).agg({
            'status': 'count',  # total_sims (count any column)
            'final_score': lambda x: ','.join(x.dropna().astype(str)),  # all_scores
            'source_database': lambda x: '|'.join(set(x))
        }).reset_index()
        
        betting_grouped.columns = [
            'game_id', 'season_year', 'completed_sims', 'all_scores', 'source_databases'
        ]
        betting_grouped['total_sims'] = betting_grouped['completed_sims']
        
        return betting_grouped
    
    return pd.DataFrame()

# Define helper functions needed for the table display
def load_actual_game_results():
    """Load actual game results from CSV file."""
    try:
        actual_df = pd.read_csv(r"data\game_results_2024-2025.csv")
        actual_df['date'] = pd.to_datetime(actual_df['date'])
        return actual_df
    except FileNotFoundError:
        print("❌ Actual game results file not found: data/game_results_2024-2025.csv")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading actual results: {e}")
        return pd.DataFrame()

# Get the simulation summary first, but delay table display until after betting functions are defined
# Use combined data from both current and historical databases
combined_summary = get_combined_simulation_data("summary")

# Process the combined data to create the summary DataFrame (similar to show_game_summary)
if not combined_summary.empty:
    # Calculate success rate
    combined_summary['success_rate'] = (combined_summary['completed_sims'] / combined_summary['total_sims'] * 100).round(1)
    
    # Parse scores and calculate statistics (similar to original show_game_summary function)
    def parse_scores_for_stats(all_scores_str):
        """Parse scores from the all_scores string and calculate median statistics.
        Only includes simulations with complete iteration counts (350-550 predictions)."""
        if pd.isna(all_scores_str) or not all_scores_str:
            return None, None, None
            
        scores = []
        score_strings = [s.strip() for s in all_scores_str.split(',') if s.strip()]
        
        for score_str in score_strings:
            try:
                # Parse format: "AWAY_TEAM SCORE - HOME_TEAM SCORE"
                if ' - ' in score_str:
                    away_part, home_part = score_str.split(' - ')
                    away_score = int(away_part.split()[-1])
                    home_score = int(home_part.split()[-1])
                    
                    # Accept all scores - filtering will be done by iteration count in the query
                    scores.append((away_score, home_score))
            except (ValueError, IndexError):
                continue
        
        if scores:
            away_scores = [s[0] for s in scores]
            home_scores = [s[1] for s in scores]
            score_diffs = [away - home for away, home in scores]
            
            return np.median(score_diffs), np.median(away_scores), np.median(home_scores)
        return None, None, None
    
    # Calculate median statistics for each game
    combined_summary[['med_score_diff', 'med_away_score', 'med_home_score']] = combined_summary['all_scores'].apply(
        lambda x: pd.Series(parse_scores_for_stats(x))
    )
    
    # Format the 'Done' column to match original display (checkmarks + count)
    def format_done_column(completed_sims):
        if completed_sims > 0:
            return f"✅ {int(completed_sims)}"
        else:
            return "❌ 0"
    
    combined_summary['done_formatted'] = combined_summary['completed_sims'].apply(format_done_column)
    
    # Reorder columns to match the expected structure from show_game_summary
    # The display_betting_table function expects specific column positions
    summary_df = combined_summary[[
        'game_id',           # 0
        'season_year',       # 1  
        'total_sims',        # 2
        'done_formatted',    # 3 - Done (formatted with checkmarks)
        'error_sims',        # 4
        'avg_predictions',   # 5
        'min_predictions',   # 6
        'max_predictions',   # 7
        'avg_duration_min',  # 8
        'med_score_diff',    # 9 - MedDiff
        'med_away_score',    # 10 - MedAway  
        'med_home_score',    # 11 - MedHome
        'success_rate',      # 12
        'all_scores',        # 13
        'final_scores',      # 14
        'source_databases'   # 15
    ]]
else:
    summary_df = pd.DataFrame()

# ================================================================================
# DATA FILTERING AND BETTING ANALYSIS INTEGRATION
# 
# FILTERING LOGIC:
# - Only includes simulations with complete iteration counts (350-550 predictions)
# - Filters out runaway simulations (>550) that didn't terminate properly
# - Filters out stuck/incomplete simulations (<350) that terminated prematurely
# - Filters out unrealistic high-scoring games (any team scoring >= 150 points)
# - Only includes simulations with 'game_ended' or 'completed' status (natural termination)
# - This combination captures realistic NBA games that completed naturally without getting stuck or producing unrealistic scores
#
# BETTING ANALYSIS:
# Compares simulation predictions with actual game outcomes to evaluate betting performance.
# Answers: "Which spread/ML would our simulations have suggested betting?" and "Would that bet have been right?"
#
# BETTING LOGIC:
# - Spread: Bet if X% of simulations show that bet would win (configurable threshold)
# - ML: Bet if simulation win% > (implied odds probability + edge threshold)
# ================================================================================


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
    Note: Filtering by iteration count and score thresholds is now done in the query, so we accept all valid scores.
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
                
                # Accept all valid scores - filtering is done by iteration count and score thresholds in the query
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

def load_actual_game_results():
    """Load actual game results from CSV file."""
    try:
        actual_df = pd.read_csv(r"data\game_results_2024-2025.csv")
        actual_df['date'] = pd.to_datetime(actual_df['date'])
        return actual_df
    except FileNotFoundError:
        print("❌ Actual game results file not found: data/game_results_2024-2025.csv")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading actual results: {e}")
        return pd.DataFrame()

def get_simulation_betting_recommendations(actual_results_df, use_combined_data=True):
    """
    Determine betting recommendations based on simulation results using new logic:
    
    - Spread: Bet if X% of simulations show that bet would win
    - ML: Bet if simulation win% > implied probability from actual ML odds
    
    Returns DataFrame with detailed betting analysis for each game.
    """
    
    if use_combined_data:
        # Get properly grouped simulation data from combined databases (no duplicates)
        sim_df = get_combined_simulation_data("betting")
    else:
        # Legacy single database approach (kept for backward compatibility)
        if not Path(DATABASE_PATH).exists():
            print(f"❌ Database file not found: {DATABASE_PATH}")
            return pd.DataFrame()
        
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Get raw simulation data with individual scores - filter by iteration count and realistic scores
        query = """
        SELECT 
            game_id,
            season_year,
            final_score,
            status
        FROM simulation_runs 
        WHERE status IN ('game_ended', 'completed') 
            AND successful_predictions <= """ + str(MAX_ITERATIONS_THRESHOLD) + """
            AND successful_predictions >= """ + str(MIN_ITERATIONS_THRESHOLD) + """  -- Only include realistic simulation lengths
        ORDER BY game_id, season_year
        """
        
        raw_sim_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Apply score filtering and then group
        filtered_sim_df = raw_sim_df[raw_sim_df['final_score'].apply(has_realistic_scores)]
        
        if filtered_sim_df.empty:
            sim_df = pd.DataFrame()
        else:
            sim_df = filtered_sim_df.groupby(['game_id', 'season_year']).agg({
                'status': 'count',  # total_sims
                'final_score': lambda x: ','.join(x)  # all_scores
            }).reset_index()
            sim_df.columns = ['game_id', 'season_year', 'completed_sims', 'all_scores']
    
    if sim_df.empty:
        print("⚠️ No simulation data found!")
        return pd.DataFrame()
    
    print(f"📊 Processing {len(sim_df)} games with simulation data")
    
    betting_recs = []
    
    for _, sim_row in sim_df.iterrows():
        game_id = str(sim_row['game_id'])
        total_sims = sim_row['completed_sims']
        all_scores_str = sim_row['all_scores']
        
        # Skip if not enough simulations
        if total_sims < MIN_SIMULATIONS_REQUIRED:
            # print(f"⚠️  Skipping game {game_id}: only {total_sims} simulations (need {MIN_SIMULATIONS_REQUIRED}+)")
            continue
            
        # Parse individual simulation scores
        scores = parse_simulation_scores(all_scores_str)
        
        if not scores:
            print(f"⚠️  Skipping game {game_id}: could not parse simulation scores")
            continue
            
        # Find matching actual game data for spreads and ML odds
        # Convert both to strings for matching
        actual_game = actual_results_df[actual_results_df['game_id'].astype(str) == str(game_id)]
        
        if actual_game.empty:
            print(f"⚠️  Skipping game {game_id}: no actual game data found")
            continue
            
        actual_row = actual_game.iloc[0]
        actual_spread = actual_row['final_spread']  # From away team perspective
        away_ml = actual_row['away_moneyline'] 
        home_ml = actual_row['home_moneyline']
        
        # Calculate simulation win percentages
        home_wins = sum(1 for away_score, home_score in scores if home_score > away_score)
        away_wins = sum(1 for away_score, home_score in scores if away_score > home_score)
        total_games = len(scores)
        
        home_win_pct = home_wins / total_games
        away_win_pct = away_wins / total_games
        
        # Analyze spread coverage with edge threshold
        away_cover_pct, home_cover_pct = analyze_spread_coverage(scores, actual_spread, SPREAD_EDGE_THRESHOLD)
        
        # SPREAD BETTING LOGIC
        spread_recommendation = 'pass'
        spread_confidence = 0.0
        
        # First check if spread meets confidence threshold
        potential_spread_rec = 'pass'
        potential_spread_conf = 0.0
        
        if away_cover_pct >= SPREAD_CONFIDENCE_THRESHOLD:
            potential_spread_rec = 'away'
            potential_spread_conf = away_cover_pct
        elif home_cover_pct >= SPREAD_CONFIDENCE_THRESHOLD:
            potential_spread_rec = 'home'
            potential_spread_conf = home_cover_pct
        
        # Then check if spread is within configured limits
        if potential_spread_rec != 'pass' and is_spread_within_limits(actual_spread):
            spread_recommendation = potential_spread_rec
            spread_confidence = potential_spread_conf
            
        # MONEYLINE BETTING LOGIC WITH KELLY CRITERION
        ml_recommendation = 'pass'
        ml_confidence = 0.0
        ml_kelly_pct = 0.0
        ml_bet_size = 0.0
        ml_potential_profit = 0.0
        
        # Calculate implied probabilities from actual odds
        away_implied_prob = calculate_implied_probability(away_ml)
        home_implied_prob = calculate_implied_probability(home_ml)
        
        # Calculate Kelly optimal bet sizes for both sides
        away_kelly_pct, away_bet_size, away_profit = calculate_kelly_bet_size(away_win_pct, away_ml)
        home_kelly_pct, home_bet_size, home_profit = calculate_kelly_bet_size(home_win_pct, home_ml)
        
        # Bet if our simulation probability exceeds market implied probability + threshold
        # AND Kelly suggests a reasonable bet size (> 1% of bankroll)
        # AND moneyline odds are within configured limits
        if (away_implied_prob is not None and 
            away_win_pct > (away_implied_prob + ML_EDGE_THRESHOLD) and 
            away_kelly_pct >= 0.01 and
            is_moneyline_within_limits(away_ml)):
            ml_recommendation = 'away'
            ml_confidence = away_win_pct - away_implied_prob
            ml_kelly_pct = away_kelly_pct
            ml_bet_size = away_bet_size
            ml_potential_profit = away_profit
        elif (home_implied_prob is not None and 
              home_win_pct > (home_implied_prob + ML_EDGE_THRESHOLD) and
              home_kelly_pct >= 0.01 and
              is_moneyline_within_limits(home_ml)):
            ml_recommendation = 'home' 
            ml_confidence = home_win_pct - home_implied_prob
            ml_kelly_pct = home_kelly_pct
            ml_bet_size = home_bet_size
            ml_potential_profit = home_profit
            
        # Calculate median scores for reference
        away_scores = [score[0] for score in scores]
        home_scores = [score[1] for score in scores]
        med_away = np.median(away_scores)
        med_home = np.median(home_scores)
        
        betting_recs.append({
            'game_id': game_id,
            'total_simulations': total_games,
            'predicted_away_score': med_away,
            'predicted_home_score': med_home,
            'predicted_score_diff': med_away - med_home,
            'predicted_home_win_prob': home_win_pct,
            'predicted_away_win_prob': away_win_pct,
            
            # Spread analysis
            'away_cover_pct': away_cover_pct,
            'home_cover_pct': home_cover_pct,
            'spread_recommendation': spread_recommendation,
            'spread_confidence': spread_confidence,
            
            # Moneyline analysis
            'away_implied_prob': away_implied_prob,
            'home_implied_prob': home_implied_prob,
            'ml_recommendation': ml_recommendation,
            'ml_confidence': ml_confidence,
            'ml_kelly_pct': ml_kelly_pct,
            'ml_bet_size': ml_bet_size,
            'ml_potential_profit': ml_potential_profit,
        })
    
    return pd.DataFrame(betting_recs)

def analyze_betting_performance():
    """
    Main function to analyze betting performance by comparing simulation predictions
    with actual game outcomes using the new statistical betting logic.
    """
    
    # print(f"\n" + "="*100)
    # print("🎯 BETTING ANALYSIS: Simulations vs Reality (Enhanced with Kelly)")
    # print("="*100)
    # print(f"📊 Configuration:")
    # print(f"   Spread confidence threshold: {SPREAD_CONFIDENCE_THRESHOLD:.0%}")
    # print(f"   Spread edge threshold: {SPREAD_EDGE_THRESHOLD} points")
    # print(f"   ML edge threshold: {ML_EDGE_THRESHOLD:.0%} above implied odds")
    # print(f"   Minimum simulations required: {MIN_SIMULATIONS_REQUIRED}")
    # print(f"   📏 BETTING LIMITS:")
    # print(f"     Max spread: {MAX_SPREAD_LIMIT} points" + (" (no limit)" if MAX_SPREAD_LIMIT <= 0 else ""))
    # print(f"     Max favorite ML: -{MAX_FAVORITE_ML_ODDS}" + (" (no limit)" if MAX_FAVORITE_ML_ODDS <= 0 else ""))
    # print(f"     Max underdog ML: +{MAX_UNDERDOG_ML_ODDS}" + (" (no limit)" if MAX_UNDERDOG_ML_ODDS <= 0 else ""))
    
    # Load actual game results
    actual_results = load_actual_game_results()
    if actual_results.empty:
        print("❌ Cannot perform betting analysis - actual results not available")
        return
    
    # Get betting recommendations using new logic
    betting_recs = get_simulation_betting_recommendations(actual_results)
    if betting_recs.empty:
        print("❌ Cannot extract betting recommendations from simulation data")
        return
    
    print(f"📊 Analysis Summary:")
    print(f"   Actual games available: {len(actual_results)}")
    print(f"   Games with sufficient simulation data: {len(betting_recs)}")
    
    # Ensure consistent data types for merging
    betting_recs['game_id'] = betting_recs['game_id'].astype(str)
    actual_results_copy = actual_results.copy()
    actual_results_copy['game_id'] = actual_results_copy['game_id'].astype(str)
    
    merged_df = betting_recs.merge(actual_results_copy, on='game_id', how='inner')
    
    if merged_df.empty:
        print("❌ No matching games found between simulations and actual results")
        return
    
    print(f"   Games available for betting analysis: {len(merged_df)}")
    
    # Calculate betting outcomes for each game
    betting_results = []
    
    for _, row in merged_df.iterrows():
        game_id = row['game_id']
        
        # Actual game results
        actual_away = row['away_score']
        actual_home = row['home_score']
        actual_diff = actual_away - actual_home  # Positive = away won
        actual_spread = row['final_spread']
        
        # Betting recommendations from simulations
        spread_rec = row['spread_recommendation']
        ml_rec = row['ml_recommendation']
        
        # Calculate if spread bets would have been correct
        spread_bet_correct = None
        if spread_rec == 'away':
            # Away covers if actual_diff > -spread
            spread_bet_correct = actual_diff > -actual_spread
        elif spread_rec == 'home':
            # Home covers if actual_diff < -spread  
            spread_bet_correct = actual_diff < -actual_spread
            
        # Calculate if ML bets would have been correct
        ml_bet_correct = None
        if ml_rec == 'away':
            ml_bet_correct = actual_away > actual_home
        elif ml_rec == 'home':
            ml_bet_correct = actual_home > actual_away
            
        betting_results.append({
            'game_id': game_id,
            'away_team': row['away_team'],
            'home_team': row['home_team'],
            'date': row['date'],
            'total_simulations': row['total_simulations'],
            
            # Predictions vs Reality
            'pred_away': row['predicted_away_score'],
            'pred_home': row['predicted_home_score'],
            'pred_diff': row['predicted_score_diff'],
            'pred_home_prob': row['predicted_home_win_prob'],
            'pred_away_prob': row['predicted_away_win_prob'],
            'actual_away': actual_away,
            'actual_home': actual_home,
            'actual_diff': actual_diff,
            'actual_spread': actual_spread,
            
            # Spread betting analysis
            'away_cover_pct': row['away_cover_pct'],
            'home_cover_pct': row['home_cover_pct'],
            'spread_recommendation': spread_rec,
            'spread_bet_correct': spread_bet_correct,
            'spread_confidence': row['spread_confidence'],
            
            # Moneyline betting analysis
            'away_implied_prob': row['away_implied_prob'],
            'home_implied_prob': row['home_implied_prob'],
            'ml_recommendation': ml_rec,
            'ml_bet_correct': ml_bet_correct,
            'ml_confidence': row['ml_confidence'],
            'ml_kelly_pct': row.get('ml_kelly_pct', 0.0),
            'ml_bet_size': row.get('ml_bet_size', 0.0),
            'ml_potential_profit': row.get('ml_potential_profit', 0.0),
            
            # Prediction accuracy
            'score_diff_error': abs(row['predicted_score_diff'] - actual_diff),
            'away_score_error': abs(row['predicted_away_score'] - actual_away),
            'home_score_error': abs(row['predicted_home_score'] - actual_home),
        })
    
    results_df = pd.DataFrame(betting_results)
    
    if results_df.empty:
        print("❌ No complete betting analysis data available")
        return
    
    # Calculate overall performance using new logic
    print(f"\n🎯 BETTING PERFORMANCE SUMMARY (New Statistical Logic)")
    print("-" * 80)
    
    # Spread betting performance
    spread_bets = results_df[results_df['spread_recommendation'] != 'pass']
    if len(spread_bets) > 0:
        spread_correct = int(spread_bets['spread_bet_correct'].sum())
        spread_total = len(spread_bets)
        spread_win_rate = spread_correct / spread_total
        
        print(f"📊 SPREAD BETTING (≥{SPREAD_CONFIDENCE_THRESHOLD:.0%} confidence + {SPREAD_EDGE_THRESHOLD} point edge):")
        print(f"   Total spread bets: {spread_total}")
        print(f"   Correct: {spread_correct}")
        print(f"   Win rate: {spread_win_rate:.1%}")
        print(f"   Average confidence: {spread_bets['spread_confidence'].mean():.1%} of simulations")
        
        # Show spread betting breakdown
        spread_away = spread_bets[spread_bets['spread_recommendation'] == 'away']
        spread_home = spread_bets[spread_bets['spread_recommendation'] == 'home']
        
        if len(spread_away) > 0:
            away_correct = int(spread_away['spread_bet_correct'].sum())
            avg_away_conf = spread_away['spread_confidence'].mean()
            print(f"   Away bets: {away_correct}/{len(spread_away)} ({away_correct/len(spread_away):.1%}) - Avg {avg_away_conf:.1%} confidence")
        
        if len(spread_home) > 0:
            home_correct = int(spread_home['spread_bet_correct'].sum())
            avg_home_conf = spread_home['spread_confidence'].mean()
            print(f"   Home bets: {home_correct}/{len(spread_home)} ({home_correct/len(spread_home):.1%}) - Avg {avg_home_conf:.1%} confidence")
            
        # Show games where we passed on spread bets
        no_spread_bets = len(results_df[results_df['spread_recommendation'] == 'pass'])
        print(f"   Passed on spread bets: {no_spread_bets} games (insufficient confidence or outside limits)")
        
    else:
        print(f"📊 SPREAD BETTING: No bets met {SPREAD_CONFIDENCE_THRESHOLD:.0%} confidence + {SPREAD_EDGE_THRESHOLD} point edge threshold and betting limits")
    
    # Moneyline betting performance with Kelly ROI analysis
    ml_bets = results_df[results_df['ml_recommendation'] != 'pass']
    if len(ml_bets) > 0:
        ml_correct = int(ml_bets['ml_bet_correct'].sum())
        ml_total = len(ml_bets)
        ml_win_rate = ml_correct / ml_total
        
        # Calculate Kelly-optimized ROI
        total_bet_amount = ml_bets['ml_bet_size'].sum()
        total_winnings = ml_bets.apply(lambda row: 
            row['ml_potential_profit'] if row['ml_bet_correct'] else -row['ml_bet_size'], axis=1).sum()
        total_roi = (total_winnings / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
        
        print(f"\n💰 MONEYLINE BETTING (Kelly-Optimized, Simulation % > Market Implied % + {ML_EDGE_THRESHOLD:.0%}):")
        print(f"   Total ML bets: {ml_total}")
        print(f"   Correct: {ml_correct}")
        print(f"   Win rate: {ml_win_rate:.1%}")
        print(f"   Average edge: {ml_bets['ml_confidence'].mean():.1%} over market odds")
        print(f"   Average Kelly bet size: {ml_bets['ml_kelly_pct'].mean():.1%} of bankroll")
        
        print(f"\n   📊 KELLY ROI ANALYSIS (per $100 bankroll):")
        print(f"   Total amount bet: ${total_bet_amount:.2f}")
        print(f"   Total profit/loss: ${total_winnings:.2f}")
        print(f"   ROI: {total_roi:.1f}%")
        
        # Show ML betting breakdown with Kelly info
        ml_away = ml_bets[ml_bets['ml_recommendation'] == 'away']
        ml_home = ml_bets[ml_bets['ml_recommendation'] == 'home']
        
        if len(ml_away) > 0:
            away_correct = int(ml_away['ml_bet_correct'].sum())
            avg_away_edge = ml_away['ml_confidence'].mean()
            avg_away_kelly = ml_away['ml_kelly_pct'].mean()
            away_bet_total = ml_away['ml_bet_size'].sum()
            away_profit = ml_away.apply(lambda row: 
                row['ml_potential_profit'] if row['ml_bet_correct'] else -row['ml_bet_size'], axis=1).sum()
            away_roi = (away_profit / away_bet_total * 100) if away_bet_total > 0 else 0.0
            
            print(f"   Away bets: {away_correct}/{len(ml_away)} ({away_correct/len(ml_away):.1%}) - "
                  f"Avg {avg_away_edge:.1%} edge, {avg_away_kelly:.1%} Kelly, {away_roi:.1f}% ROI")
        
        if len(ml_home) > 0:
            home_correct = int(ml_home['ml_bet_correct'].sum())
            avg_home_edge = ml_home['ml_confidence'].mean()
            avg_home_kelly = ml_home['ml_kelly_pct'].mean()
            home_bet_total = ml_home['ml_bet_size'].sum()
            home_profit = ml_home.apply(lambda row: 
                row['ml_potential_profit'] if row['ml_bet_correct'] else -row['ml_bet_size'], axis=1).sum()
            home_roi = (home_profit / home_bet_total * 100) if home_bet_total > 0 else 0.0
            
            print(f"   Home bets: {home_correct}/{len(ml_home)} ({home_correct/len(ml_home):.1%}) - "
                  f"Avg {avg_home_edge:.1%} edge, {avg_home_kelly:.1%} Kelly, {home_roi:.1f}% ROI")
            
        # Show games where we passed on ML bets
        no_ml_bets = len(results_df[results_df['ml_recommendation'] == 'pass'])
        print(f"   Passed on ML bets: {no_ml_bets} games (insufficient edge over market or outside limits)")
        
    else:
        print(f"\n💰 MONEYLINE BETTING: No bets met {ML_EDGE_THRESHOLD:.0%} edge threshold over market odds and betting limits")
    
    # Prediction accuracy
    print(f"\n🎯 PREDICTION ACCURACY:")
    print(f"   Average score difference error: {results_df['score_diff_error'].mean():.1f} points")
    print(f"   Average away score error: {results_df['away_score_error'].mean():.1f} points") 
    print(f"   Average home score error: {results_df['home_score_error'].mean():.1f} points")
    
    # Summary statistics
    total_games = len(results_df)
    games_with_spread_bets = len(results_df[results_df['spread_recommendation'] != 'pass'])
    games_with_ml_bets = len(results_df[results_df['ml_recommendation'] != 'pass'])
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total games analyzed: {total_games}")
    print(f"   Games with spread bets: {games_with_spread_bets} ({games_with_spread_bets/total_games:.1%})")
    print(f"   Games with ML bets: {games_with_ml_bets} ({games_with_ml_bets/total_games:.1%})")
    print(f"   Average simulations per game: {results_df['total_simulations'].mean():.1f}")
    
    return results_df

def display_betting_table(summary_df):
    """Display the enhanced game results table with betting information."""
    if summary_df is None or summary_df.empty:
        print("\n❌ No data to display or summary_df is None")
        return
    
    print("\n" + "="*100)
    print("🏀 DETAILED GAME RESULTS TABLE")
    print("="*100)
    
    # Import tabulate for nice table formatting
    from tabulate import tabulate
    
    # Get betting recommendations to enhance the display
    actual_results = load_actual_game_results()
    betting_recs = get_simulation_betting_recommendations(actual_results) if not actual_results.empty else pd.DataFrame()
    
    # Create enhanced display DataFrame with betting info
    display_rows = []
    
    for _, row in summary_df.iterrows():
        game_id = str(row.iloc[0])  # Game ID is first column
        
        # Get actual game results and betting recommendations
        actual_game = actual_results[actual_results['game_id'].astype(str) == game_id] if not actual_results.empty else pd.DataFrame()
        betting_info = betting_recs[betting_recs['game_id'].astype(str) == game_id] if not betting_recs.empty else pd.DataFrame()
        
        if not actual_game.empty:
            actual_row = actual_game.iloc[0]
            away_team = actual_row['away_team']
            home_team = actual_row['home_team'] 
            actual_spread = actual_row['final_spread']
            away_ml = actual_row['away_moneyline']
            home_ml = actual_row['home_moneyline']
            actual_away_score = actual_row['away_score']
            actual_home_score = actual_row['home_score']
        else:
            away_team = "Unknown"
            home_team = "Unknown"
            actual_spread = "N/A"
            away_ml = "N/A" 
            home_ml = "N/A"
            actual_away_score = "N/A"
            actual_home_score = "N/A"
        
        # Format new columns
        best_bet_cover = "N/A"
        best_bet_ml = "N/A"
        recommended_cover = "PASS"
        recommended_ml = "PASS"
        
        if not betting_info.empty and not actual_game.empty:
            bet_row = betting_info.iloc[0]
            away_cover_pct = bet_row.get('away_cover_pct', 0)
            home_cover_pct = bet_row.get('home_cover_pct', 0)
            away_win_pct = bet_row.get('predicted_away_win_prob', 0) 
            home_win_pct = bet_row.get('predicted_home_win_prob', 0)
            
            # BEST BET COVER (always show higher percentage, regardless of thresholds)
            if away_cover_pct > home_cover_pct:
                # Betting on away team
                if actual_spread >= 0:
                    best_bet_cover = f"AWAY +{actual_spread} ({away_cover_pct:.0%})"  # Away gets points
                else:
                    best_bet_cover = f"AWAY {actual_spread} ({away_cover_pct:.0%})"    # Away gives points
            else:
                # Betting on home team - need to flip the spread sign
                if actual_spread >= 0:
                    best_bet_cover = f"HOME -{actual_spread} ({home_cover_pct:.0%})"   # Home gives points
                else:
                    best_bet_cover = f"HOME +{abs(actual_spread)} ({home_cover_pct:.0%})"  # Home gets points
            
            # BEST BET ML (show actual ML recommendation if any, otherwise show higher win percentage)
            ml_rec = bet_row.get('ml_recommendation', 'pass')
            if ml_rec != 'pass':
                if ml_rec == 'away':
                    best_bet_ml = f"AWAY {away_ml} ({away_win_pct:.0%})"
                else:  # home
                    best_bet_ml = f"HOME {home_ml} ({home_win_pct:.0%})"
            else:
                # No ML recommendation, show higher win percentage for reference
                if away_win_pct > home_win_pct:
                    best_bet_ml = f"AWAY {away_ml} ({away_win_pct:.0%}) - No Bet"
                else:
                    best_bet_ml = f"HOME {home_ml} ({home_win_pct:.0%}) - No Bet"
            
            # RECOMMENDED COVER (based on actual recommendation + outcome)
            spread_rec = bet_row.get('spread_recommendation', 'pass')
            if spread_rec != 'pass' and actual_away_score != "N/A" and actual_home_score != "N/A":
                actual_diff = actual_away_score - actual_home_score
                
                if spread_rec == 'away':
                    # Betting on away team
                    if actual_spread >= 0:
                        rec_text = f"AWAY +{actual_spread}"  # Away gets points
                    else:
                        rec_text = f"AWAY {actual_spread}"   # Away gives points
                    outcome = "✅" if actual_diff > -actual_spread else "❌"
                else:  # home
                    # Betting on home team - need to flip the spread sign
                    if actual_spread >= 0:
                        rec_text = f"HOME -{actual_spread}"  # Home gives points
                    else:
                        rec_text = f"HOME +{abs(actual_spread)}"  # Home gets points
                    outcome = "✅" if actual_diff < -actual_spread else "❌"
                
                recommended_cover = f"{rec_text} {outcome}"
            elif spread_rec != 'pass':
                # We have a recommendation but no actual scores
                if spread_rec == 'away':
                    if actual_spread >= 0:
                        recommended_cover = f"AWAY +{actual_spread} ?"  # Away gets points
                    else:
                        recommended_cover = f"AWAY {actual_spread} ?"   # Away gives points
                else:  # home
                    if actual_spread >= 0:
                        recommended_cover = f"HOME -{actual_spread} ?"  # Home gives points
                    else:
                        recommended_cover = f"HOME +{abs(actual_spread)} ?"  # Home gets points
            
            # RECOMMENDED ML (based on actual recommendation + outcome)
            ml_rec = bet_row.get('ml_recommendation', 'pass')
            if ml_rec != 'pass' and actual_away_score != "N/A" and actual_home_score != "N/A":
                if ml_rec == 'away':
                    rec_text = f"AWAY {away_ml}"
                    outcome = "✅" if actual_away_score > actual_home_score else "❌"
                else:  # home
                    rec_text = f"HOME {home_ml}"
                    outcome = "✅" if actual_home_score > actual_away_score else "❌"
                
                recommended_ml = f"{rec_text} {outcome}"
            elif ml_rec != 'pass':
                # We have a recommendation but no actual scores
                if ml_rec == 'away':
                    recommended_ml = f"AWAY {away_ml} ?"
                else:  # home
                    recommended_ml = f"HOME {home_ml} ?"

        display_rows.append({
            'Game': game_id,
            'Season': row.iloc[1],  # Season is second column
            'Done': row.iloc[3],    # Completed sims
            'MedDiff': row.iloc[9], # Med Score Diff
            'MedAway': row.iloc[10], # Med Away
            'MedHome': row.iloc[11], # Med Home
            'Actual Away': actual_away_score if actual_away_score != "N/A" else "N/A",
            'Actual Home': actual_home_score if actual_home_score != "N/A" else "N/A", 
            'Actual Diff': actual_away_score - actual_home_score if actual_away_score != "N/A" and actual_home_score != "N/A" else "N/A",
            'Spread': f"HOME -{actual_spread}" if actual_spread > 0 else f"AWAY -{abs(actual_spread)}" if actual_spread < 0 else "PICK" if actual_spread != "N/A" else "N/A",
            'ML': f"{away_ml}/{home_ml}" if away_ml != "N/A" and home_ml != "N/A" else "N/A",
            'Best Bet Cover': best_bet_cover,
            'Best Bet ML': best_bet_ml,
            'Recommended Cover': recommended_cover,
            'Recommended ML': recommended_ml
        })
    
    display_df = pd.DataFrame(display_rows)
    
    # Print main summary table
    print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
    
# Display the betting table first
display_betting_table(summary_df)

# Run the betting analysis
if summary_df is not None and not summary_df.empty:
    betting_analysis_results = analyze_betting_performance()