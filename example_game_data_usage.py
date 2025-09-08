"""
Example usage of the game-level data generated from transform_boxscore_to_games.py

This shows how you can use the new game_results_2024-2025.csv file in analysis scripts
like results_analysis.py for comparing actual game results with predictions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the game-level data
def load_game_results():
    """Load the transformed game-level results."""
    df = pd.read_csv(r"C:\Users\micha\nba_predictions\data\game_results_2024-2025.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

def analyze_game_results():
    """Analyze the actual game results from the 2024-2025 season."""
    df = load_game_results()
    
    print("GAME RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Total games: {len(df)}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Teams: {len(set(df['away_team'].tolist() + df['home_team'].tolist()))}")
    
    # Basic statistics
    print(f"\nScoring Statistics:")
    print(f"  Average total score: {df['total_score'].mean():.1f}")
    print(f"  Average score difference: {df['score_difference'].mean():.1f}")
    print(f"  Home team win rate: {df['home_win'].mean():.1%}")
    
    # Betting statistics
    print(f"\nBetting Statistics:")
    print(f"  Average spread: {df['final_spread'].mean():.1f}")
    print(f"  Average total line: {df['final_ou'].mean():.1f}")
    
    # Show games that went over/under
    df['over_under'] = df['total_score'] > df['final_ou']
    print(f"  Games going OVER: {df['over_under'].sum()} ({df['over_under'].mean():.1%})")
    
    # Show spread results
    df['home_covered'] = (df['home_score'] - df['away_score']) > df['final_spread']
    print(f"  Home team covered spread: {df['home_covered'].sum()} ({df['home_covered'].mean():.1%})")
    
    return df

def example_prediction_comparison():
    """Example of how to compare with prediction results."""
    # Load actual results
    actual_df = load_game_results()
    
    print("\nEXAMPLE: Comparing with prediction results")
    print("-" * 50)
    
    # This would be where you load your simulation/prediction results
    # For demonstration, let's create some sample predictions
    sample_predictions = actual_df.head(5).copy()
    sample_predictions['predicted_away_score'] = sample_predictions['away_score'] + 3  # Example
    sample_predictions['predicted_home_score'] = sample_predictions['home_score'] - 2  # Example
    
    print("Sample comparison (first 5 games):")
    for _, row in sample_predictions.iterrows():
        print(f"\nGame {row['game_id']}: {row['away_team']} @ {row['home_team']}")
        print(f"  Actual: {row['away_score']}-{row['home_score']}")
        print(f"  Predicted: {row['predicted_away_score']}-{row['predicted_home_score']}")
        
        actual_diff = abs(row['away_score'] - row['home_score'])
        pred_diff = abs(row['predicted_away_score'] - row['predicted_home_score'])
        print(f"  Score diff - Actual: {actual_diff}, Predicted: {pred_diff}")

def plot_scoring_trends(df=None):
    """Plot scoring trends over time."""
    if df is None:
        df = load_game_results()
    
    # Group by date and calculate daily averages
    daily_stats = df.groupby('date').agg({
        'total_score': 'mean',
        'score_difference': 'mean',
        'home_win': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NBA 2024-2025 Season Trends', fontsize=16)
    
    # Total scoring
    axes[0,0].plot(daily_stats['date'], daily_stats['total_score'], alpha=0.7)
    axes[0,0].axhline(df['total_score'].mean(), color='red', linestyle='--', alpha=0.7)
    axes[0,0].set_title('Average Total Score per Game')
    axes[0,0].set_ylabel('Total Score')
    
    # Score differences
    axes[0,1].plot(daily_stats['date'], daily_stats['score_difference'], alpha=0.7, color='orange')
    axes[0,1].axhline(df['score_difference'].mean(), color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_title('Average Score Difference')
    axes[0,1].set_ylabel('Score Difference')
    
    # Home win rate
    axes[1,0].plot(daily_stats['date'], daily_stats['home_win'], alpha=0.7, color='green')
    axes[1,0].axhline(df['home_win'].mean(), color='red', linestyle='--', alpha=0.7)
    axes[1,0].set_title('Home Team Win Rate')
    axes[1,0].set_ylabel('Win Rate')
    axes[1,0].set_ylim(0, 1)
    
    # Score distribution
    axes[1,1].hist(df['total_score'], bins=30, alpha=0.7, color='purple')
    axes[1,1].axvline(df['total_score'].mean(), color='red', linestyle='--', alpha=0.7)
    axes[1,1].set_title('Total Score Distribution')
    axes[1,1].set_xlabel('Total Score')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the analysis
    df = analyze_game_results()
    
    # Show prediction comparison example
    example_prediction_comparison()
    
    # Uncomment to show plots
    # plot_scoring_trends(df)
