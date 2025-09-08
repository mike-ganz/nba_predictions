"""
NBA Simulation Results Viewer - Jupyter Notebook Version
Easy-to-use functions for analyzing NBA simulation results in Jupyter notebooks.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

def load_results_df(db_path: str = "enhanced_simulation_results.db") -> pd.DataFrame:
    """Load all simulation results into a pandas DataFrame."""
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM simulation_runs ORDER BY created_at DESC", conn)
    conn.close()
    return df

def show_game_summary(db_path: str = "enhanced_simulation_results.db"):
    """Display a beautiful game summary table in Jupyter."""
    
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Get game summary data INCLUDING individual final scores for statistics
    query = """
    SELECT 
        game_id,
        season_year,
        COUNT(*) as total_sims,
        SUM(CASE WHEN status = 'game_ended' THEN 1 ELSE 0 END) as completed_sims,
        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_sims,
        ROUND(AVG(successful_predictions), 1) as avg_predictions,
        ROUND(MIN(successful_predictions), 0) as min_predictions,
        ROUND(MAX(successful_predictions), 0) as max_predictions,
        ROUND(AVG(duration_seconds)/60, 1) as avg_duration_min,
        GROUP_CONCAT(DISTINCT final_score) as final_scores,
        GROUP_CONCAT(final_score) as all_scores
    FROM simulation_runs 
    WHERE status = 'game_ended'  -- Only include completed games for statistics
    GROUP BY game_id, season_year
    ORDER BY game_id, season_year
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("⚠️ No simulation results found!")
        return
    
    # Calculate success rate
    df['success_rate'] = (df['completed_sims'] / df['total_sims'] * 100).round(1)
    
    # Parse scores and calculate new statistics
    def parse_scores_and_calculate_stats(all_scores_str):
        if pd.isna(all_scores_str) or not all_scores_str:
            return {
                'avg_score_diff': 'N/A',
                'median_score_diff': 'N/A', 
                'median_away_score': 'N/A',
                'median_home_score': 'N/A',
                'home_win_pct': 'N/A'
            }
        
        scores = [s.strip() for s in all_scores_str.split(',') if s.strip()]
        away_scores = []
        home_scores = []
        score_diffs = []
        home_wins = 0
        
        for score_str in scores:
            try:
                # Parse format: "AWAY_TEAM SCORE - HOME_TEAM SCORE"
                if ' - ' in score_str:
                    away_part, home_part = score_str.split(' - ')
                    away_score = int(away_part.split()[-1])  # Get last part (score)
                    home_score = int(home_part.split()[-1])  # Get last part (score)
                    
                    away_scores.append(away_score)
                    home_scores.append(home_score)
                    score_diffs.append(away_score - home_score)  # Positive means away won
                    
                    if home_score > away_score:
                        home_wins += 1
            except (ValueError, IndexError):
                continue  # Skip malformed scores
        
        if not away_scores:
            return {
                'avg_score_diff': 'N/A',
                'median_score_diff': 'N/A',
                'median_away_score': 'N/A', 
                'median_home_score': 'N/A',
                'home_win_pct': 'N/A'
            }
        
        return {
            'avg_score_diff': round(sum(score_diffs) / len(score_diffs), 1),
            'median_score_diff': round(pd.Series(score_diffs).median(), 1),
            'median_away_score': round(pd.Series(away_scores).median(), 1),
            'median_home_score': round(pd.Series(home_scores).median(), 1),
            'home_win_pct': round((home_wins / len(scores)) * 100, 1)
        }
    
    # Apply score parsing to each game
    score_stats = df['all_scores'].apply(parse_scores_and_calculate_stats)
    
    # Extract statistics into separate columns
    df['avg_score_diff'] = [stats['avg_score_diff'] for stats in score_stats]
    df['median_score_diff'] = [stats['median_score_diff'] for stats in score_stats]
    df['median_away_score'] = [stats['median_away_score'] for stats in score_stats]
    df['median_home_score'] = [stats['median_home_score'] for stats in score_stats]
    df['home_win_pct'] = [stats['home_win_pct'] for stats in score_stats]
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['Success Rate'] = display_df['success_rate'].astype(str) + '%'
    display_df['Home Win %'] = display_df['home_win_pct'].apply(lambda x: f'{x}%' if x != 'N/A' else 'N/A')
    display_df['Completed'] = '✅ ' + display_df['completed_sims'].astype(str) 
    display_df['Errors'] = display_df['error_sims'].apply(lambda x: f'⚠️ {x}' if x > 0 else str(x))
    
    # Clean up final scores
    display_df['final_scores'] = display_df['final_scores'].apply(
        lambda x: ', '.join([s for s in str(x).split(',') if s and s != 'None']) if pd.notna(x) else 'N/A'
    )
    
    # Format score statistics for display
    display_df['Avg Score Diff'] = display_df['avg_score_diff'].apply(lambda x: f'{x:+}' if x != 'N/A' else 'N/A')
    display_df['Med Score Diff'] = display_df['median_score_diff'].apply(lambda x: f'{x:+}' if x != 'N/A' else 'N/A')
    display_df['Med Away'] = display_df['median_away_score']
    display_df['Med Home'] = display_df['median_home_score']
    
    # Rename columns for display
    display_df = display_df[[
        'game_id', 'season_year', 'total_sims', 'Completed', 'Errors', 
        'Success Rate', 'avg_predictions', 'avg_duration_min', 
        'Avg Score Diff', 'Med Score Diff', 'Med Away', 'Med Home', 'Home Win %', 'final_scores'
    ]]
    
    display_df.columns = [
        'Game ID', 'Season', 'Total Sims', 'Completed', 'Errors', 
        'Success Rate', 'Avg Predictions', 'Avg Duration (min)', 
        'Avg Score Diff', 'Med Score Diff', 'Med Away', 'Med Home', 'Home Win %', 'Final Scores'
    ]
    
    print("🏀 NBA Simulation Results Summary")
    print("=" * 80)
    # display(display_df.style.set_table_attributes('style="font-size: 12px"'))
    
    # # Overall stats
    # total_sims = df['total_sims'].sum()
    # total_completed = df['completed_sims'].sum()
    # total_errors = df['error_sims'].sum()
    # overall_success = round(total_completed / total_sims * 100, 1) if total_sims > 0 else 0
    
    # print(f"\n📊 Overall Summary:")
    # print(f"   🎯 Total Simulations: {total_sims}")
    # print(f"   ✅ Completed: {total_completed} ({overall_success}%)")
    # print(f"   ⚠️  Errors: {total_errors} ({round(total_errors/total_sims*100, 1)}%)")
    
    # Calculate overall score statistics
    all_score_diffs = []
    all_away_scores = []
    all_home_scores = []
    total_home_wins = 0
    total_games_with_scores = 0
    
    for _, row in df.iterrows():
        if row['avg_score_diff'] != 'N/A':
            # Calculate individual game stats
            if pd.notna(row['all_scores']) and row['all_scores']:
                scores = [s.strip() for s in row['all_scores'].split(',') if s.strip()]
                for score_str in scores:
                    try:
                        if ' - ' in score_str:
                            away_part, home_part = score_str.split(' - ')
                            away_score = int(away_part.split()[-1])
                            home_score = int(home_part.split()[-1])
                            
                            all_score_diffs.append(away_score - home_score)
                            all_away_scores.append(away_score)
                            all_home_scores.append(home_score)
                            total_games_with_scores += 1
                            
                            if home_score > away_score:
                                total_home_wins += 1
                    except (ValueError, IndexError):
                        continue
    
    if all_score_diffs:
        avg_overall_diff = round(sum(all_score_diffs) / len(all_score_diffs), 1)
        median_overall_diff = round(pd.Series(all_score_diffs).median(), 1)
        median_overall_away = round(pd.Series(all_away_scores).median(), 1)
        median_overall_home = round(pd.Series(all_home_scores).median(), 1)
        overall_home_win_pct = round((total_home_wins / total_games_with_scores) * 100, 1)
        
        print(f"\n🏈 Score Statistics Across All Games:")
        print(f"   📈 Avg Score Difference: {avg_overall_diff:+} (Away - Home)")
        print(f"   📊 Median Score Difference: {median_overall_diff:+}")
        print(f"   🚗 Median Away Score: {median_overall_away}")
        print(f"   🏠 Median Home Score: {median_overall_home}")
        print(f"   🎯 Home Team Win Rate: {overall_home_win_pct}% ({total_home_wins}/{total_games_with_scores} games)")
    
    return display_df

def show_run_timeline(db_path: str = "enhanced_simulation_results.db"):
    """Show a timeline of simulation runs with success/failure indicators."""
    
    df = load_results_df(db_path)
    if df.empty:
        return
    
    # Convert created_at to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Create timeline plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot successful runs
    success_runs = df[df['status'] == 'game_ended']
    error_runs = df[df['status'] == 'error']
    
    if not success_runs.empty:
        ax.scatter(success_runs['created_at'], success_runs['game_id'], 
                  c='green', s=100, alpha=0.7, label='✅ Completed', marker='o')
    
    if not error_runs.empty:
        ax.scatter(error_runs['created_at'], error_runs['game_id'], 
                  c='red', s=100, alpha=0.7, label='❌ Error', marker='x')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Game ID')
    ax.set_title('🏀 NBA Simulation Run Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return fig

def show_performance_stats(db_path: str = "enhanced_simulation_results.db"):
    """Show performance statistics and charts."""
    
    df = load_results_df(db_path)
    if df.empty:
        return
    
    completed_runs = df[df['status'] == 'game_ended']
    
    if completed_runs.empty:
        print("⚠️ No completed runs to analyze!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Predictions per run
    axes[0,0].hist(completed_runs['successful_predictions'], bins=10, alpha=0.7, color='skyblue')
    axes[0,0].set_title('📊 Predictions per Completed Run')
    axes[0,0].set_xlabel('Number of Predictions')
    axes[0,0].set_ylabel('Frequency')
    
    # 2. Duration per run
    completed_runs['duration_minutes'] = completed_runs['duration_seconds'] / 60
    axes[0,1].hist(completed_runs['duration_minutes'], bins=10, alpha=0.7, color='lightgreen')
    axes[0,1].set_title('⏱️ Duration per Completed Run')
    axes[0,1].set_xlabel('Duration (minutes)')
    axes[0,1].set_ylabel('Frequency')
    
    # 3. Success rate by game
    game_stats = df.groupby('game_id').agg({
        'status': lambda x: (x == 'game_ended').sum() / len(x) * 100
    }).round(1)
    
    axes[1,0].bar(game_stats.index.astype(str), game_stats['status'], alpha=0.7, color='orange')
    axes[1,0].set_title('🎯 Success Rate by Game')
    axes[1,0].set_xlabel('Game ID')
    axes[1,0].set_ylabel('Success Rate (%)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Runs over time
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    daily_counts = df.groupby(['date', 'status']).size().unstack(fill_value=0)
    
    if 'game_ended' in daily_counts.columns:
        axes[1,1].plot(daily_counts.index, daily_counts['game_ended'], 
                      marker='o', label='✅ Completed', color='green')
    if 'error' in daily_counts.columns:
        axes[1,1].plot(daily_counts.index, daily_counts['error'], 
                      marker='x', label='❌ Errors', color='red')
    
    axes[1,1].set_title('📅 Runs Over Time')
    axes[1,1].set_xlabel('Date')
    axes[1,1].set_ylabel('Number of Runs')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary stats
    print("📈 Performance Summary:")
    print(f"   Average predictions per run: {completed_runs['successful_predictions'].mean():.1f}")
    print(f"   Average duration per run: {completed_runs['duration_minutes'].mean():.1f} minutes")
    print(f"   Fastest run: {completed_runs['duration_minutes'].min():.1f} minutes")
    print(f"   Longest run: {completed_runs['duration_minutes'].max():.1f} minutes")
    
    return fig

def quick_summary(db_path: str = "enhanced_simulation_results.db"):
    """Quick one-line summary of results."""
    df = load_results_df(db_path)
    if df.empty:
        print("No results found.")
        return
    
    total = len(df)
    completed = len(df[df['status'] == 'game_ended'])
    unique_games = df['game_id'].nunique()
    
    print(f"🏀 {total} total runs • ✅ {completed} completed • 🎯 {unique_games} unique games • 📊 {completed/total*100:.1f}% success rate")
    
    if completed > 0:
        latest = df[df['status'] == 'game_ended'].iloc[0]
        print(f"🏆 Latest: Game {latest['game_id']} - {latest['final_score']} ({latest['successful_predictions']} plays)")

# Convenience functions for common tasks
def get_successful_runs():
    """Get only successfully completed runs as a DataFrame."""
    df = load_results_df()
    return df[df['status'] == 'game_ended']

def get_run_details(run_id: str):
    """Get detailed information for a specific run."""
    df = load_results_df()
    run = df[df['run_id'] == run_id]
    if run.empty:
        print(f"❌ Run {run_id} not found")
        return None
    return run.iloc[0].to_dict()
