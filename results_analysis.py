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
SPREAD_CONFIDENCE_THRESHOLD = 0.70  # 70% of simulations must support the bet
MIN_SIMULATIONS_REQUIRED = 8       # Minimum simulations needed for analysis
ML_EDGE_THRESHOLD = 0.15             # 10% minimum edge over implied odds for ML bets

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

def analyze_spread_coverage(scores, spread):
    """
    Analyze what percentage of simulations would cover each side of the spread.
    
    Args:
        scores: List of (away_score, home_score) tuples
        spread: Spread from away team's perspective (positive = away underdog)
    
    Returns:
        (away_cover_pct, home_cover_pct) - percentages as decimals
    """
    if not scores or pd.isna(spread):
        return 0.0, 0.0
    
    away_covers = 0
    home_covers = 0
    
    for away_score, home_score in scores:
        actual_diff = away_score - home_score  # Positive = away won
        
        # Away covers if: away_score + spread > home_score
        # Which means: actual_diff > -spread
        if actual_diff > -spread:
            away_covers += 1
        else:
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

def get_simulation_betting_recommendations(actual_results_df, db_path="enhanced_simulation_results.db"):
    """
    Determine betting recommendations based on simulation results using new logic:
    
    - Spread: Bet if X% of simulations show that bet would win
    - ML: Bet if simulation win% > implied probability from actual ML odds
    
    Returns DataFrame with detailed betting analysis for each game.
    """
    
    # Load raw simulation data to get individual scores
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Get raw simulation data with individual scores
    query = """
    SELECT 
        game_id,
        season_year,
        COUNT(*) as total_sims,
        SUM(CASE WHEN status = 'game_ended' THEN 1 ELSE 0 END) as completed_sims,
        GROUP_CONCAT(final_score) as all_scores
    FROM simulation_runs 
    WHERE status = 'game_ended'  -- Only include completed games
    GROUP BY game_id, season_year
    ORDER BY game_id, season_year
    """
    
    sim_df = pd.read_sql_query(query, conn)
    conn.close()
    
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
            print(f"⚠️  Skipping game {game_id}: only {total_sims} simulations (need {MIN_SIMULATIONS_REQUIRED}+)")
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
        
        # Analyze spread coverage
        away_cover_pct, home_cover_pct = analyze_spread_coverage(scores, actual_spread)
        
        # SPREAD BETTING LOGIC
        spread_recommendation = 'pass'
        spread_confidence = 0.0
        
        if away_cover_pct >= SPREAD_CONFIDENCE_THRESHOLD:
            spread_recommendation = 'away'
            spread_confidence = away_cover_pct
        elif home_cover_pct >= SPREAD_CONFIDENCE_THRESHOLD:
            spread_recommendation = 'home'
            spread_confidence = home_cover_pct
            
        # MONEYLINE BETTING LOGIC  
        ml_recommendation = 'pass'
        ml_confidence = 0.0
        
        # Calculate implied probabilities from actual odds
        away_implied_prob = calculate_implied_probability(away_ml)
        home_implied_prob = calculate_implied_probability(home_ml)
        
        # Bet if our simulation probability exceeds market implied probability + threshold
        if away_implied_prob is not None and away_win_pct > (away_implied_prob + ML_EDGE_THRESHOLD):
            ml_recommendation = 'away'
            ml_confidence = away_win_pct - away_implied_prob
        elif home_implied_prob is not None and home_win_pct > (home_implied_prob + ML_EDGE_THRESHOLD):
            ml_recommendation = 'home' 
            ml_confidence = home_win_pct - home_implied_prob
            
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
        })
    
    return pd.DataFrame(betting_recs)

def analyze_betting_performance():
    """
    Main function to analyze betting performance by comparing simulation predictions
    with actual game outcomes using the new statistical betting logic.
    """
    
    print(f"\n" + "="*100)
    print("🎯 BETTING ANALYSIS: Simulations vs Reality (New Logic)")
    print("="*100)
    print(f"📊 Configuration:")
    print(f"   Spread confidence threshold: {SPREAD_CONFIDENCE_THRESHOLD:.0%}")
    print(f"   ML edge threshold: {ML_EDGE_THRESHOLD:.0%} above implied odds")
    print(f"   Minimum simulations required: {MIN_SIMULATIONS_REQUIRED}")
    
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
        
        print(f"📊 SPREAD BETTING (≥{SPREAD_CONFIDENCE_THRESHOLD:.0%} simulation confidence):")
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
        print(f"   Passed on spread bets: {no_spread_bets} games (insufficient confidence)")
        
    else:
        print(f"📊 SPREAD BETTING: No bets met {SPREAD_CONFIDENCE_THRESHOLD:.0%} confidence threshold")
    
    # Moneyline betting performance
    ml_bets = results_df[results_df['ml_recommendation'] != 'pass']
    if len(ml_bets) > 0:
        ml_correct = int(ml_bets['ml_bet_correct'].sum())
        ml_total = len(ml_bets)
        ml_win_rate = ml_correct / ml_total
        
        print(f"\n💰 MONEYLINE BETTING (Simulation % > Market Implied % + {ML_EDGE_THRESHOLD:.0%}):")
        print(f"   Total ML bets: {ml_total}")
        print(f"   Correct: {ml_correct}")
        print(f"   Win rate: {ml_win_rate:.1%}")
        print(f"   Average edge: {ml_bets['ml_confidence'].mean():.1%} over market odds")
        
        # Show ML betting breakdown
        ml_away = ml_bets[ml_bets['ml_recommendation'] == 'away']
        ml_home = ml_bets[ml_bets['ml_recommendation'] == 'home']
        
        if len(ml_away) > 0:
            away_correct = int(ml_away['ml_bet_correct'].sum())
            avg_away_edge = ml_away['ml_confidence'].mean()
            print(f"   Away bets: {away_correct}/{len(ml_away)} ({away_correct/len(ml_away):.1%}) - Avg {avg_away_edge:.1%} edge")
        
        if len(ml_home) > 0:
            home_correct = int(ml_home['ml_bet_correct'].sum())
            avg_home_edge = ml_home['ml_confidence'].mean()
            print(f"   Home bets: {home_correct}/{len(ml_home)} ({home_correct/len(ml_home):.1%}) - Avg {avg_home_edge:.1%} edge")
            
        # Show games where we passed on ML bets
        no_ml_bets = len(results_df[results_df['ml_recommendation'] == 'pass'])
        print(f"   Passed on ML bets: {no_ml_bets} games (insufficient edge over market)")
        
    else:
        print(f"\n💰 MONEYLINE BETTING: No bets met {ML_EDGE_THRESHOLD:.0%} edge threshold over market odds")
    
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
        
        # Format betting recommendations
        spread_rec = "NO BET"
        ml_rec = "NO BET"
        spread_won = "N/A"
        ml_won = "N/A"
        
        if not betting_info.empty:
            bet_row = betting_info.iloc[0]
            
            # Spread recommendation
            if bet_row.get('spread_recommendation') == 'away':
                if actual_spread >= 0:
                    spread_rec = f"AWAY +{actual_spread}"
                else:
                    spread_rec = f"AWAY {actual_spread}"
            elif bet_row.get('spread_recommendation') == 'home':
                if actual_spread <= 0:
                    spread_rec = f"HOME {actual_spread}"
                else:
                    spread_rec = f"HOME -{actual_spread}"
            
            # ML recommendation  
            if bet_row.get('ml_recommendation') == 'away':
                ml_rec = f"AWAY {away_ml}"
            elif bet_row.get('ml_recommendation') == 'home':
                ml_rec = f"HOME {home_ml}"
            
            # Calculate betting outcomes (only if we made bets)
            if bet_row.get('spread_recommendation') != 'pass' and not actual_game.empty:
                actual_diff = actual_away_score - actual_home_score
                if bet_row.get('spread_recommendation') == 'away':
                    spread_won = "✅ WON" if actual_diff > -actual_spread else "❌ LOST"
                elif bet_row.get('spread_recommendation') == 'home':
                    spread_won = "✅ WON" if actual_diff < -actual_spread else "❌ LOST"
            
            if bet_row.get('ml_recommendation') != 'pass' and not actual_game.empty:
                if bet_row.get('ml_recommendation') == 'away':
                    ml_won = "✅ WON" if actual_away_score > actual_home_score else "❌ LOST"
                elif bet_row.get('ml_recommendation') == 'home':
                    ml_won = "✅ WON" if actual_home_score > actual_away_score else "❌ LOST"
        
        # Get simulation percentages for display
        spread_cover_display = "N/A"
        win_pct_display = "N/A"
        
        if not betting_info.empty:
            bet_row = betting_info.iloc[0]
            away_cover_pct = bet_row.get('away_cover_pct', 0)
            home_cover_pct = bet_row.get('home_cover_pct', 0)
            away_win_pct = bet_row.get('predicted_away_win_prob', 0) 
            home_win_pct = bet_row.get('predicted_home_win_prob', 0)
            
            # Choose the team that covers >50% of the time
            if away_cover_pct > 0.5:
                spread_cover_display = f"AWAY {away_cover_pct:.1%}"
            elif home_cover_pct > 0.5:
                spread_cover_display = f"HOME {home_cover_pct:.1%}"
            else:
                # Show the higher one even if neither is >50%
                if away_cover_pct > home_cover_pct:
                    spread_cover_display = f"AWAY {away_cover_pct:.1%}"
                else:
                    spread_cover_display = f"HOME {home_cover_pct:.1%}"
            
            # Choose the team that wins >50% of the time
            if away_win_pct > 0.5:
                win_pct_display = f"AWAY {away_win_pct:.1%}"
            elif home_win_pct > 0.5:
                win_pct_display = f"HOME {home_win_pct:.1%}"
            else:
                # Show the higher one even if neither is >50%
                if away_win_pct > home_win_pct:
                    win_pct_display = f"AWAY {away_win_pct:.1%}"
                else:
                    win_pct_display = f"HOME {home_win_pct:.1%}"

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
            'Spread': actual_spread,
            'ML': f"{away_ml}/{home_ml}" if away_ml != "N/A" and home_ml != "N/A" else "N/A",
            'Dominant Cover': spread_cover_display,
            'Dominant Win': win_pct_display,
            'Rec Spread': spread_rec,
            'Rec ML': ml_rec,
            'Spread Won': spread_won,
            'ML Won': ml_won
        })
    
    display_df = pd.DataFrame(display_rows)
    
    # Print main summary table
    print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
    
# Display the betting table first
display_betting_table(summary_df)

# # Run the betting analysis
# if summary_df is not None and not summary_df.empty:
#     betting_analysis_results = analyze_betting_performance()