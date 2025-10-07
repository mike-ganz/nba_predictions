import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from nba_results_notebook import (
    show_game_summary, 
    show_run_timeline, 
    show_performance_stats,
    quick_summary,
    load_results_df,
    get_successful_runs,
    get_run_details
)

# Page configuration
st.set_page_config(
    page_title="NBA Betting Analysis Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stDataFrame {
        font-size: 12px;
    }
    h1 {
        color: #FF6B35;
    }
    h2 {
        color: #004E89;
    }
    h3 {
        color: #1A659E;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================================================================
# SIDEBAR - CONFIGURATION
# ================================================================================

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Database Configuration
st.sidebar.header("📊 Database Settings")
DATABASE_PATH = st.sidebar.text_input(
    "Database Path",
    value="enhanced_simulation_results.db",
    help="Path to the main simulation results database"
)

HISTORICAL_DATABASE_PATH = st.sidebar.text_input(
    "Historical Database Path",
    value="",
    help="Path to historical database (optional)"
)

st.sidebar.markdown("---")

# Iteration Filtering
st.sidebar.header("🔢 Iteration Filtering")
MAX_ITERATIONS_THRESHOLD = st.sidebar.number_input(
    "Max Iterations Threshold",
    min_value=100,
    max_value=1000,
    value=550,
    step=10,
    help="Maximum iterations to consider realistic (filters out runaway simulations)"
)

MIN_ITERATIONS_THRESHOLD = st.sidebar.number_input(
    "Min Iterations Threshold",
    min_value=100,
    max_value=1000,
    value=325,
    step=10,
    help="Minimum iterations to consider complete (filters out stuck/incomplete games)"
)

st.sidebar.markdown("---")

# Score Filtering
st.sidebar.header("🎯 Score Filtering")
MAX_TEAM_SCORE_THRESHOLD = st.sidebar.number_input(
    "Max Team Score Threshold",
    min_value=100,
    max_value=250,
    value=175,
    step=5,
    help="Maximum team score to consider realistic"
)

st.sidebar.markdown("---")

# Betting Analysis
st.sidebar.header("💰 Betting Analysis")
SPREAD_CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Spread Confidence Threshold",
    min_value=0.5,
    max_value=0.95,
    value=0.60,
    step=0.05,
    format="%.0%",
    help="% of simulations must support the spread bet"
)

MIN_SIMULATIONS_REQUIRED = st.sidebar.number_input(
    "Min Simulations Required",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
    help="Minimum simulations needed for analysis"
)

ML_EDGE_THRESHOLD = st.sidebar.slider(
    "ML Edge Threshold",
    min_value=0.0,
    max_value=0.30,
    value=0.10,
    step=0.01,
    format="%.0%",
    help="Minimum edge over implied odds for ML bets"
)

SPREAD_EDGE_THRESHOLD = st.sidebar.number_input(
    "Spread Edge Threshold",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Buffer on actual spread (e.g., 3 = need 3 extra points of coverage)"
)

st.sidebar.markdown("---")

# Betting Limits
st.sidebar.header("🚫 Betting Limits")
MAX_SPREAD_LIMIT = st.sidebar.number_input(
    "Max Spread Limit",
    min_value=0.0,
    max_value=30.0,
    value=0.0,
    step=0.5,
    help="Don't bet spreads > N points (0 = no limit)"
)

MAX_FAVORITE_ML_ODDS = st.sidebar.number_input(
    "Max Favorite ML Odds",
    min_value=0,
    max_value=1000,
    value=600,
    step=10,
    help="Don't bet favorites with odds worse than -N (0 = no limit)"
)

MAX_UNDERDOG_ML_ODDS = st.sidebar.number_input(
    "Max Underdog ML Odds",
    min_value=0,
    max_value=1000,
    value=600,
    step=10,
    help="Don't bet underdogs with odds worse than +N (0 = no limit)"
)

# ================================================================================
# HELPER FUNCTIONS (from original script)
# ================================================================================

def has_realistic_scores(final_score_str):
    """Check if a simulation has realistic team scores."""
    if pd.isna(final_score_str) or not final_score_str.strip():
        return False
    
    try:
        if ' - ' in final_score_str:
            away_part, home_part = final_score_str.split(' - ')
            away_score = int(away_part.split()[-1])
            home_score = int(home_part.split()[-1])
            return away_score < MAX_TEAM_SCORE_THRESHOLD and home_score < MAX_TEAM_SCORE_THRESHOLD
    except (ValueError, IndexError):
        return False
    
    return False

def get_combined_raw_simulation_data(db_paths):
    """Get raw simulation data from multiple databases."""
    combined_data = []
    script_dir = Path(__file__).parent
    
    for db_path in db_paths:
        if not db_path:
            continue
        
        # Convert to Path and make absolute if needed
        path_obj = Path(db_path)
        if not path_obj.is_absolute():
            path_obj = script_dir / path_obj
        
        if not path_obj.exists():
            continue
            
        try:
            conn = sqlite3.connect(str(path_obj))
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
                df['source_database'] = str(path_obj)
                combined_data.append(df)
                
        except Exception as e:
            st.error(f"Error reading {path_obj}: {e}")
            continue
    
    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        return pd.DataFrame()

def get_combined_simulation_data(query_type="summary", db_paths=None):
    """Get properly aggregated simulation data from multiple databases."""
    if db_paths is None:
        db_paths = [DATABASE_PATH, HISTORICAL_DATABASE_PATH]
    
    raw_data = get_combined_raw_simulation_data(db_paths)
    
    if raw_data.empty:
        return pd.DataFrame()
    
    if query_type == "summary":
        completed_games = raw_data[
            (raw_data['status'].isin(['game_ended', 'completed'])) & 
            (raw_data['successful_predictions'] <= MAX_ITERATIONS_THRESHOLD) &
            (raw_data['successful_predictions'] >= MIN_ITERATIONS_THRESHOLD) &
            (raw_data['final_score'].apply(has_realistic_scores))
        ]
        
        if completed_games.empty:
            return pd.DataFrame()
        
        summary_grouped = completed_games.groupby(['game_id', 'season_year']).agg({
            'status': 'count',
            'successful_predictions': ['mean', 'min', 'max'],
            'duration_seconds': 'mean',
            'final_score': lambda x: ','.join(x.dropna().astype(str)),
            'source_database': lambda x: '|'.join(set(x))
        }).reset_index()
        
        if isinstance(summary_grouped.columns, pd.MultiIndex):
            summary_grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in summary_grouped.columns.values]
        
        current_columns = list(summary_grouped.columns)
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
        
        summary_grouped = summary_grouped.rename(columns=column_renames)
        
        if 'avg_duration_min' in summary_grouped.columns:
            summary_grouped['avg_duration_min'] = (summary_grouped['avg_duration_min'] / 60).round(1)
        if 'avg_predictions' in summary_grouped.columns:
            summary_grouped['avg_predictions'] = summary_grouped['avg_predictions'].round(1)
        
        if 'completed_sims' in summary_grouped.columns:
            summary_grouped['total_sims'] = summary_grouped['completed_sims']
        else:
            summary_grouped['total_sims'] = 0
            
        summary_grouped['error_sims'] = 0
        
        if 'all_scores' in summary_grouped.columns:
            summary_grouped['final_scores'] = summary_grouped['all_scores']
        else:
            summary_grouped['final_scores'] = ''
        
        return summary_grouped
        
    elif query_type == "betting":
        completed_games = raw_data[
            (raw_data['status'].isin(['game_ended', 'completed'])) & 
            (raw_data['successful_predictions'] <= MAX_ITERATIONS_THRESHOLD) &
            (raw_data['successful_predictions'] >= MIN_ITERATIONS_THRESHOLD) &
            (raw_data['final_score'].apply(has_realistic_scores))
        ]
        
        if completed_games.empty:
            return pd.DataFrame()
            
        betting_grouped = completed_games.groupby(['game_id', 'season_year']).agg({
            'status': 'count',
            'final_score': lambda x: ','.join(x.dropna().astype(str)),
            'source_database': lambda x: '|'.join(set(x))
        }).reset_index()
        
        betting_grouped.columns = [
            'game_id', 'season_year', 'completed_sims', 'all_scores', 'source_databases'
        ]
        betting_grouped['total_sims'] = betting_grouped['completed_sims']
        
        return betting_grouped
    
    return pd.DataFrame()

def load_actual_game_results():
    """Load actual game results from CSV file."""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        csv_path = script_dir / "data" / "game_results_2024-2025.csv"
        
        actual_df = pd.read_csv(csv_path)
        actual_df['date'] = pd.to_datetime(actual_df['date'])
        return actual_df
    except FileNotFoundError:
        st.error(f"❌ Actual game results file not found: {csv_path}")
        st.info(f"Looking for file at: {csv_path.absolute()}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading actual results: {e}")
        return pd.DataFrame()

def parse_scores_for_stats(all_scores_str):
    """Parse scores from the all_scores string and calculate median statistics."""
    if pd.isna(all_scores_str) or not all_scores_str:
        return None, None, None
        
    scores = []
    score_strings = [s.strip() for s in all_scores_str.split(',') if s.strip()]
    
    for score_str in score_strings:
        try:
            if ' - ' in score_str:
                away_part, home_part = score_str.split(' - ')
                away_score = int(away_part.split()[-1])
                home_score = int(home_part.split()[-1])
                scores.append((away_score, home_score))
        except (ValueError, IndexError):
            continue
    
    if scores:
        away_scores = [s[0] for s in scores]
        home_scores = [s[1] for s in scores]
        score_diffs = [away - home for away, home in scores]
        
        return np.median(score_diffs), np.median(away_scores), np.median(home_scores)
    return None, None, None

def calculate_implied_probability(moneyline_str):
    """Calculate implied probability from moneyline odds."""
    if pd.isna(moneyline_str) or not isinstance(moneyline_str, str):
        return None
        
    try:
        ml_str = str(moneyline_str).strip()
        if ml_str.startswith('+'):
            odds = float(ml_str[1:])
            implied_prob = 100 / (odds + 100)
        elif ml_str.startswith('-'):
            odds = float(ml_str[1:])
            implied_prob = odds / (odds + 100)
        else:
            odds = float(ml_str)
            if odds > 0:
                implied_prob = 100 / (odds + 100)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100)
                
        return implied_prob
        
    except (ValueError, TypeError):
        return None

def calculate_kelly_bet_size(win_prob, moneyline_str, bankroll=100):
    """Calculate Kelly optimal bet size for a moneyline bet."""
    if pd.isna(moneyline_str) or not isinstance(moneyline_str, str) or win_prob <= 0:
        return 0.0, 0.0, 0.0
        
    try:
        ml_str = str(moneyline_str).strip()
        if ml_str.startswith('+'):
            odds = float(ml_str[1:])
            decimal_odds = (odds / 100) + 1
            payout_multiplier = odds / 100
        elif ml_str.startswith('-'):
            odds = float(ml_str[1:])
            decimal_odds = (100 / odds) + 1
            payout_multiplier = 100 / odds
        else:
            odds = float(ml_str)
            if odds > 0:
                decimal_odds = (odds / 100) + 1
                payout_multiplier = odds / 100
            else:
                decimal_odds = (100 / abs(odds)) + 1
                payout_multiplier = 100 / abs(odds)
        
        b = decimal_odds - 1
        p = win_prob
        q = 1 - win_prob
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0.0, min(0.25, kelly_fraction))
        
        bet_size = kelly_fraction * bankroll
        potential_profit = bet_size * payout_multiplier if kelly_fraction > 0 else 0.0
        
        return kelly_fraction, bet_size, potential_profit
        
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0, 0.0, 0.0

def is_spread_within_limits(spread):
    """Check if spread is within configured limits."""
    if MAX_SPREAD_LIMIT <= 0:
        return True
    return abs(spread) <= MAX_SPREAD_LIMIT

def is_moneyline_within_limits(moneyline_str):
    """Check if moneyline odds are within configured limits."""
    if pd.isna(moneyline_str) or not isinstance(moneyline_str, str):
        return False
        
    try:
        ml_str = str(moneyline_str).strip()
        if ml_str.startswith('+'):
            odds = float(ml_str[1:])
            if MAX_UNDERDOG_ML_ODDS <= 0:
                return True
            return odds <= MAX_UNDERDOG_ML_ODDS
        elif ml_str.startswith('-'):
            odds = float(ml_str[1:])
            if MAX_FAVORITE_ML_ODDS <= 0:
                return True
            return odds <= MAX_FAVORITE_ML_ODDS
        else:
            odds = float(ml_str)
            if odds > 0:
                if MAX_UNDERDOG_ML_ODDS <= 0:
                    return True
                return odds <= MAX_UNDERDOG_ML_ODDS
            else:
                if MAX_FAVORITE_ML_ODDS <= 0:
                    return True
                return abs(odds) <= MAX_FAVORITE_ML_ODDS
                
    except (ValueError, TypeError):
        return False

def parse_simulation_scores(all_scores_str):
    """Parse individual simulation scores from the 'all_scores' string."""
    if pd.isna(all_scores_str) or not all_scores_str:
        return []
    
    scores = []
    score_strings = [s.strip() for s in all_scores_str.split(',') if s.strip()]
    
    for score_str in score_strings:
        try:
            if ' - ' in score_str:
                away_part, home_part = score_str.split(' - ')
                away_score = int(away_part.split()[-1])
                home_score = int(home_part.split()[-1])
                scores.append((away_score, home_score))
        except (ValueError, IndexError):
            continue
    
    return scores

def analyze_spread_coverage(scores, spread, spread_edge_threshold=0.0):
    """Analyze what percentage of simulations would cover each side of the spread."""
    if not scores or pd.isna(spread):
        return 0.0, 0.0
    
    away_covers = 0
    home_covers = 0
    
    for away_score, home_score in scores:
        actual_diff = away_score - home_score
        
        if actual_diff > -(spread - spread_edge_threshold):
            away_covers += 1
            
        if actual_diff < -(spread + spread_edge_threshold):
            home_covers += 1
    
    total = len(scores)
    return away_covers / total, home_covers / total

def get_simulation_betting_recommendations(actual_results_df, use_combined_data=True):
    """Determine betting recommendations based on simulation results."""
    
    if use_combined_data:
        sim_df = get_combined_simulation_data("betting")
    else:
        script_dir = Path(__file__).parent
        db_path = script_dir / DATABASE_PATH if not Path(DATABASE_PATH).is_absolute() else Path(DATABASE_PATH)
        
        if not db_path.exists():
            st.error(f"❌ Database file not found: {DATABASE_PATH}")
            st.info(f"Looking for file at: {db_path.absolute()}")
            return pd.DataFrame()
        
        conn = sqlite3.connect(str(db_path))
        
        query = f"""
        SELECT 
            game_id,
            season_year,
            final_score,
            status
        FROM simulation_runs 
        WHERE status IN ('game_ended', 'completed') 
            AND successful_predictions <= {MAX_ITERATIONS_THRESHOLD}
            AND successful_predictions >= {MIN_ITERATIONS_THRESHOLD}
        ORDER BY game_id, season_year
        """
        
        raw_sim_df = pd.read_sql_query(query, conn)
        conn.close()
        
        filtered_sim_df = raw_sim_df[raw_sim_df['final_score'].apply(has_realistic_scores)]
        
        if filtered_sim_df.empty:
            sim_df = pd.DataFrame()
        else:
            sim_df = filtered_sim_df.groupby(['game_id', 'season_year']).agg({
                'status': 'count',
                'final_score': lambda x: ','.join(x)
            }).reset_index()
            sim_df.columns = ['game_id', 'season_year', 'completed_sims', 'all_scores']
    
    if sim_df.empty:
        return pd.DataFrame()
    
    betting_recs = []
    
    for _, sim_row in sim_df.iterrows():
        game_id = str(sim_row['game_id'])
        total_sims = sim_row['completed_sims']
        all_scores_str = sim_row['all_scores']
        
        if total_sims < MIN_SIMULATIONS_REQUIRED:
            continue
            
        scores = parse_simulation_scores(all_scores_str)
        
        if not scores:
            continue
            
        actual_game = actual_results_df[actual_results_df['game_id'].astype(str) == str(game_id)]
        
        if actual_game.empty:
            continue
            
        actual_row = actual_game.iloc[0]
        actual_spread = actual_row['final_spread']
        away_ml = actual_row['away_moneyline'] 
        home_ml = actual_row['home_moneyline']
        
        home_wins = sum(1 for away_score, home_score in scores if home_score > away_score)
        away_wins = sum(1 for away_score, home_score in scores if away_score > home_score)
        total_games = len(scores)
        
        home_win_pct = home_wins / total_games
        away_win_pct = away_wins / total_games
        
        away_cover_pct, home_cover_pct = analyze_spread_coverage(scores, actual_spread, SPREAD_EDGE_THRESHOLD)
        
        spread_recommendation = 'pass'
        spread_confidence = 0.0
        
        potential_spread_rec = 'pass'
        potential_spread_conf = 0.0
        
        if away_cover_pct >= SPREAD_CONFIDENCE_THRESHOLD:
            potential_spread_rec = 'away'
            potential_spread_conf = away_cover_pct
        elif home_cover_pct >= SPREAD_CONFIDENCE_THRESHOLD:
            potential_spread_rec = 'home'
            potential_spread_conf = home_cover_pct
        
        if potential_spread_rec != 'pass' and is_spread_within_limits(actual_spread):
            spread_recommendation = potential_spread_rec
            spread_confidence = potential_spread_conf
            
        ml_recommendation = 'pass'
        ml_confidence = 0.0
        ml_kelly_pct = 0.0
        ml_bet_size = 0.0
        ml_potential_profit = 0.0
        
        away_implied_prob = calculate_implied_probability(away_ml)
        home_implied_prob = calculate_implied_probability(home_ml)
        
        away_kelly_pct, away_bet_size, away_profit = calculate_kelly_bet_size(away_win_pct, away_ml)
        home_kelly_pct, home_bet_size, home_profit = calculate_kelly_bet_size(home_win_pct, home_ml)
        
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
            'away_cover_pct': away_cover_pct,
            'home_cover_pct': home_cover_pct,
            'spread_recommendation': spread_recommendation,
            'spread_confidence': spread_confidence,
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
    """Main function to analyze betting performance."""
    
    actual_results = load_actual_game_results()
    if actual_results.empty:
        st.error("❌ Cannot perform betting analysis - actual results not available")
        return None
    
    betting_recs = get_simulation_betting_recommendations(actual_results)
    if betting_recs.empty:
        st.warning("❌ Cannot extract betting recommendations from simulation data")
        return None
    
    betting_recs['game_id'] = betting_recs['game_id'].astype(str)
    actual_results_copy = actual_results.copy()
    actual_results_copy['game_id'] = actual_results_copy['game_id'].astype(str)
    
    merged_df = betting_recs.merge(actual_results_copy, on='game_id', how='inner')
    
    if merged_df.empty:
        st.warning("❌ No matching games found between simulations and actual results")
        return None
    
    betting_results = []
    
    for _, row in merged_df.iterrows():
        game_id = row['game_id']
        
        actual_away = row['away_score']
        actual_home = row['home_score']
        actual_diff = actual_away - actual_home
        actual_spread = row['final_spread']
        
        spread_rec = row['spread_recommendation']
        ml_rec = row['ml_recommendation']
        
        spread_bet_correct = None
        if spread_rec == 'away':
            spread_bet_correct = actual_diff > -actual_spread
        elif spread_rec == 'home':
            spread_bet_correct = actual_diff < -actual_spread
            
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
            'pred_away': row['predicted_away_score'],
            'pred_home': row['predicted_home_score'],
            'pred_diff': row['predicted_score_diff'],
            'pred_home_prob': row['predicted_home_win_prob'],
            'pred_away_prob': row['predicted_away_win_prob'],
            'actual_away': actual_away,
            'actual_home': actual_home,
            'actual_diff': actual_diff,
            'actual_spread': actual_spread,
            'away_cover_pct': row['away_cover_pct'],
            'home_cover_pct': row['home_cover_pct'],
            'spread_recommendation': spread_rec,
            'spread_bet_correct': spread_bet_correct,
            'spread_confidence': row['spread_confidence'],
            'away_implied_prob': row['away_implied_prob'],
            'home_implied_prob': row['home_implied_prob'],
            'ml_recommendation': ml_rec,
            'ml_bet_correct': ml_bet_correct,
            'ml_confidence': row['ml_confidence'],
            'ml_kelly_pct': row.get('ml_kelly_pct', 0.0),
            'ml_bet_size': row.get('ml_bet_size', 0.0),
            'ml_potential_profit': row.get('ml_potential_profit', 0.0),
            'score_diff_error': abs(row['predicted_score_diff'] - actual_diff),
            'away_score_error': abs(row['predicted_away_score'] - actual_away),
            'home_score_error': abs(row['predicted_home_score'] - actual_home),
        })
    
    results_df = pd.DataFrame(betting_results)
    
    if results_df.empty:
        st.warning("❌ No complete betting analysis data available")
        return None
    
    return results_df

# ================================================================================
# MAIN APP
# ================================================================================

st.title("🏀 NBA Betting Analysis Dashboard")
st.markdown("### Simulation Results vs Reality")

# Get the script directory for relative paths
script_dir = Path(__file__).parent

# Construct full database paths relative to script directory
db_path_full = script_dir / DATABASE_PATH if not Path(DATABASE_PATH).is_absolute() else Path(DATABASE_PATH)
hist_db_path_full = script_dir / HISTORICAL_DATABASE_PATH if HISTORICAL_DATABASE_PATH and not Path(HISTORICAL_DATABASE_PATH).is_absolute() else Path(HISTORICAL_DATABASE_PATH) if HISTORICAL_DATABASE_PATH else None

# Check if database exists
if not db_path_full.exists():
    st.error(f"❌ Database file not found: `{DATABASE_PATH}`")
    st.info(f"Looking for file at: `{db_path_full.absolute()}`")
    st.info("Please check the database path in the sidebar configuration.")
    st.stop()

# Load data
with st.spinner("Loading simulation data..."):
    db_paths = [str(db_path_full)]
    if hist_db_path_full and hist_db_path_full.exists():
        db_paths.append(str(hist_db_path_full))
    
    combined_summary = get_combined_simulation_data("summary", db_paths)

if combined_summary.empty:
    st.warning("⚠️ No simulation data found!")
    st.stop()

# Process summary data
combined_summary['success_rate'] = (combined_summary['completed_sims'] / combined_summary['total_sims'] * 100).round(1)

combined_summary[['med_score_diff', 'med_away_score', 'med_home_score']] = combined_summary['all_scores'].apply(
    lambda x: pd.Series(parse_scores_for_stats(x))
)

def format_done_column(completed_sims):
    if completed_sims > 0:
        return f"✅ {int(completed_sims)}"
    else:
        return "❌ 0"

combined_summary['done_formatted'] = combined_summary['completed_sims'].apply(format_done_column)

summary_df = combined_summary[[
    'game_id',
    'season_year',
    'total_sims',
    'done_formatted',
    'error_sims',
    'avg_predictions',
    'min_predictions',
    'max_predictions',
    'avg_duration_min',
    'med_score_diff',
    'med_away_score',
    'med_home_score',
    'success_rate',
    'all_scores',
    'final_scores',
    'source_databases'
]]

# Display summary metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Games", len(summary_df))
with col2:
    st.metric("Total Simulations", int(summary_df['total_sims'].sum()))
with col3:
    st.metric("Avg Simulations/Game", f"{summary_df['total_sims'].mean():.1f}")
with col4:
    st.metric("Avg Duration (min)", f"{summary_df['avg_duration_min'].mean():.1f}")

st.markdown("---")

# ================================================================================
# BETTING TABLE
# ================================================================================

st.header("📊 Detailed Game Results Table")

# Column selector
all_columns = [
    'Game', 'Season', 'Done', 'MedDiff', 'MedAway', 'MedHome', 
    'Actual Away', 'Actual Home', 'Actual Diff', 'Spread', 'ML',
    'Best Bet Cover', 'Best Bet ML', 'Recommended Cover', 'Recommended ML'
]

default_columns = [
    'Game', 'Done', 'MedDiff', 'Actual Diff', 'Spread', 'ML',
    'Best Bet Cover', 'Best Bet ML', 'Recommended Cover', 'Recommended ML'
]

# Initialize selected columns in session state if not present
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = default_columns

# Place the column selector on the right side
col_left, col_right = st.columns([3, 1])
with col_right:
    with st.popover("📋 Select Columns"):
        st.session_state.selected_columns = st.multiselect(
            "Choose columns to display",
            options=all_columns,
            default=st.session_state.selected_columns,
            help="Select which columns to show in the table"
        )

# If no columns selected, show a message
if not st.session_state.selected_columns:
    st.warning("⚠️ Please select at least one column to display")
    st.session_state.selected_columns = default_columns

selected_columns = st.session_state.selected_columns

actual_results = load_actual_game_results()
betting_recs = get_simulation_betting_recommendations(actual_results) if not actual_results.empty else pd.DataFrame()

# Create enhanced display DataFrame
display_rows = []

for _, row in summary_df.iterrows():
    game_id = str(row['game_id'])
    
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
        
        if away_cover_pct > home_cover_pct:
            if actual_spread >= 0:
                best_bet_cover = f"AWAY +{actual_spread} ({away_cover_pct:.0%})"
            else:
                best_bet_cover = f"AWAY {actual_spread} ({away_cover_pct:.0%})"
        else:
            if actual_spread >= 0:
                best_bet_cover = f"HOME -{actual_spread} ({home_cover_pct:.0%})"
            else:
                best_bet_cover = f"HOME +{abs(actual_spread)} ({home_cover_pct:.0%})"
        
        ml_rec = bet_row.get('ml_recommendation', 'pass')
        if ml_rec != 'pass':
            if ml_rec == 'away':
                best_bet_ml = f"AWAY {away_ml} ({away_win_pct:.0%})"
            else:
                best_bet_ml = f"HOME {home_ml} ({home_win_pct:.0%})"
        else:
            if away_win_pct > home_win_pct:
                best_bet_ml = f"AWAY {away_ml} ({away_win_pct:.0%}) - No Bet"
            else:
                best_bet_ml = f"HOME {home_ml} ({home_win_pct:.0%}) - No Bet"
        
        spread_rec = bet_row.get('spread_recommendation', 'pass')
        if spread_rec != 'pass' and actual_away_score != "N/A" and actual_home_score != "N/A":
            actual_diff = actual_away_score - actual_home_score
            
            if spread_rec == 'away':
                if actual_spread >= 0:
                    rec_text = f"AWAY +{actual_spread}"
                else:
                    rec_text = f"AWAY {actual_spread}"
                outcome = "✅" if actual_diff > -actual_spread else "❌"
            else:
                if actual_spread >= 0:
                    rec_text = f"HOME -{actual_spread}"
                else:
                    rec_text = f"HOME +{abs(actual_spread)}"
                outcome = "✅" if actual_diff < -actual_spread else "❌"
            
            recommended_cover = f"{rec_text} {outcome}"
        elif spread_rec != 'pass':
            if spread_rec == 'away':
                if actual_spread >= 0:
                    recommended_cover = f"AWAY +{actual_spread} ?"
                else:
                    recommended_cover = f"AWAY {actual_spread} ?"
            else:
                if actual_spread >= 0:
                    recommended_cover = f"HOME -{actual_spread} ?"
                else:
                    recommended_cover = f"HOME +{abs(actual_spread)} ?"
        
        ml_rec = bet_row.get('ml_recommendation', 'pass')
        if ml_rec != 'pass' and actual_away_score != "N/A" and actual_home_score != "N/A":
            if ml_rec == 'away':
                rec_text = f"AWAY {away_ml}"
                outcome = "✅" if actual_away_score > actual_home_score else "❌"
            else:
                rec_text = f"HOME {home_ml}"
                outcome = "✅" if actual_home_score > actual_away_score else "❌"
            
            recommended_ml = f"{rec_text} {outcome}"
        elif ml_rec != 'pass':
            if ml_rec == 'away':
                recommended_ml = f"AWAY {away_ml} ?"
            else:
                recommended_ml = f"HOME {home_ml} ?"

    display_rows.append({
        'Game': game_id,
        'Season': row['season_year'],
        'Done': row['done_formatted'],
        'MedDiff': f"{row['med_score_diff']:.1f}" if pd.notna(row['med_score_diff']) else "N/A",
        'MedAway': f"{row['med_away_score']:.1f}" if pd.notna(row['med_away_score']) else "N/A",
        'MedHome': f"{row['med_home_score']:.1f}" if pd.notna(row['med_home_score']) else "N/A",
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

# Filter to only show selected columns
display_df_filtered = display_df[selected_columns]

# Display the table with custom styling
st.dataframe(
    display_df_filtered,
    use_container_width=True,
    height=600,
    hide_index=True,
    column_config={
        "Game": st.column_config.TextColumn("Game ID", width="small"),
        "Season": st.column_config.TextColumn("Season", width="small"),
        "Done": st.column_config.TextColumn("Done", width="small"),
        "MedDiff": st.column_config.TextColumn("Med Diff", width="small"),
        "MedAway": st.column_config.TextColumn("Med Away", width="small"),
        "MedHome": st.column_config.TextColumn("Med Home", width="small"),
        "Actual Away": st.column_config.TextColumn("Act Away", width="small"),
        "Actual Home": st.column_config.TextColumn("Act Home", width="small"),
        "Actual Diff": st.column_config.TextColumn("Act Diff", width="small"),
        "Spread": st.column_config.TextColumn("Spread", width="small"),
        "ML": st.column_config.TextColumn("ML", width="small"),
        "Best Bet Cover": st.column_config.TextColumn("Best Cover", width="medium"),
        "Best Bet ML": st.column_config.TextColumn("Best ML", width="medium"),
        "Recommended Cover": st.column_config.TextColumn("Rec Cover", width="medium"),
        "Recommended ML": st.column_config.TextColumn("Rec ML", width="medium"),
    }
)

# ================================================================================
# BETTING ANALYSIS
# ================================================================================

st.markdown("---")
st.header("🎯 Betting Performance Analysis")

with st.spinner("Analyzing betting performance..."):
    results_df = analyze_betting_performance()

if results_df is not None and not results_df.empty:
    
    # Spread betting performance
    spread_bets = results_df[results_df['spread_recommendation'] != 'pass']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Spread Betting")
        
        if len(spread_bets) > 0:
            spread_correct = int(spread_bets['spread_bet_correct'].sum())
            spread_total = len(spread_bets)
            spread_win_rate = spread_correct / spread_total
            
            st.metric("Total Spread Bets", spread_total)
            st.metric("Correct", spread_correct)
            st.metric("Win Rate", f"{spread_win_rate:.1%}")
            st.metric("Average Confidence", f"{spread_bets['spread_confidence'].mean():.1%}")
            
            spread_away = spread_bets[spread_bets['spread_recommendation'] == 'away']
            spread_home = spread_bets[spread_bets['spread_recommendation'] == 'home']
            
            if len(spread_away) > 0:
                away_correct = int(spread_away['spread_bet_correct'].sum())
                avg_away_conf = spread_away['spread_confidence'].mean()
                st.write(f"**Away bets:** {away_correct}/{len(spread_away)} ({away_correct/len(spread_away):.1%}) - Avg {avg_away_conf:.1%} confidence")
            
            if len(spread_home) > 0:
                home_correct = int(spread_home['spread_bet_correct'].sum())
                avg_home_conf = spread_home['spread_confidence'].mean()
                st.write(f"**Home bets:** {home_correct}/{len(spread_home)} ({home_correct/len(spread_home):.1%}) - Avg {avg_home_conf:.1%} confidence")
                
            no_spread_bets = len(results_df[results_df['spread_recommendation'] == 'pass'])
            st.write(f"**Passed on bets:** {no_spread_bets} games")
            
        else:
            st.info(f"No bets met {SPREAD_CONFIDENCE_THRESHOLD:.0%} confidence threshold")
    
    with col2:
        st.subheader("💰 Moneyline Betting")
        
        ml_bets = results_df[results_df['ml_recommendation'] != 'pass']
        
        if len(ml_bets) > 0:
            ml_correct = int(ml_bets['ml_bet_correct'].sum())
            ml_total = len(ml_bets)
            ml_win_rate = ml_correct / ml_total
            
            st.metric("Total ML Bets", ml_total)
            st.metric("Correct", ml_correct)
            st.metric("Win Rate", f"{ml_win_rate:.1%}")
            st.metric("Average Edge", f"{ml_bets['ml_confidence'].mean():.1%}")
            
            total_bet_amount = ml_bets['ml_bet_size'].sum()
            total_winnings = ml_bets.apply(lambda row: 
                row['ml_potential_profit'] if row['ml_bet_correct'] else -row['ml_bet_size'], axis=1).sum()
            total_roi = (total_winnings / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
            
            st.write(f"**Total bet (per $100):** ${total_bet_amount:.2f}")
            st.write(f"**Total profit/loss:** ${total_winnings:.2f}")
            st.write(f"**ROI:** {total_roi:.1f}%")
            
            ml_away = ml_bets[ml_bets['ml_recommendation'] == 'away']
            ml_home = ml_bets[ml_bets['ml_recommendation'] == 'home']
            
            if len(ml_away) > 0:
                away_correct = int(ml_away['ml_bet_correct'].sum())
                avg_away_edge = ml_away['ml_confidence'].mean()
                avg_away_kelly = ml_away['ml_kelly_pct'].mean()
                st.write(f"**Away bets:** {away_correct}/{len(ml_away)} ({away_correct/len(ml_away):.1%}) - "
                          f"Avg {avg_away_edge:.1%} edge, {avg_away_kelly:.1%} Kelly")
            
            if len(ml_home) > 0:
                home_correct = int(ml_home['ml_bet_correct'].sum())
                avg_home_edge = ml_home['ml_confidence'].mean()
                avg_home_kelly = ml_home['ml_kelly_pct'].mean()
                st.write(f"**Home bets:** {home_correct}/{len(ml_home)} ({home_correct/len(ml_home):.1%}) - "
                          f"Avg {avg_home_edge:.1%} edge, {avg_home_kelly:.1%} Kelly")
                
            no_ml_bets = len(results_df[results_df['ml_recommendation'] == 'pass'])
            st.write(f"**Passed on bets:** {no_ml_bets} games")
            
        else:
            st.info(f"No bets met {ML_EDGE_THRESHOLD:.0%} edge threshold")
    
    # Prediction accuracy
    st.markdown("---")
    st.subheader("🎯 Prediction Accuracy")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Score Diff Error", f"{results_df['score_diff_error'].mean():.1f} pts")
    with col2:
        st.metric("Avg Away Score Error", f"{results_df['away_score_error'].mean():.1f} pts")
    with col3:
        st.metric("Avg Home Score Error", f"{results_df['home_score_error'].mean():.1f} pts")
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📈 Summary")
    
    total_games = len(results_df)
    games_with_spread_bets = len(results_df[results_df['spread_recommendation'] != 'pass'])
    games_with_ml_bets = len(results_df[results_df['ml_recommendation'] != 'pass'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games Analyzed", total_games)
    with col2:
        st.metric("Games with Spread Bets", f"{games_with_spread_bets} ({games_with_spread_bets/total_games:.1%})")
    with col3:
        st.metric("Games with ML Bets", f"{games_with_ml_bets} ({games_with_ml_bets/total_games:.1%})")
    with col4:
        st.metric("Avg Simulations/Game", f"{results_df['total_simulations'].mean():.1f}")
else:
    st.info("No betting analysis data available yet.")

st.markdown("---")
st.caption("🏀 NBA Betting Analysis Dashboard | Powered by Streamlit")

