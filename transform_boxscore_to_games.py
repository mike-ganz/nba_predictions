import pandas as pd
import numpy as np
from pathlib import Path

def transform_team_boxscores_to_games(input_file, output_file=None):
    """
    Transform team-level boxscore data to game-level data.
    
    Each game has 2 rows (home/away teams) in input, 
    creates 1 row per game in output with:
    - away_score, home_score
    - final_spread, final_ml, final_ou
    """
    
    print("Loading Excel file...")
    df = pd.read_excel(input_file)
    
    print(f"Original data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Number of unique games: {df['GAME-ID'].nunique()}")
    
    # Verify each game has exactly 2 teams
    games_team_counts = df.groupby('GAME-ID').size()
    unusual_games = games_team_counts[games_team_counts != 2]
    if len(unusual_games) > 0:
        print(f"Warning: {len(unusual_games)} games don't have exactly 2 teams:")
        print(unusual_games.head())
    
    # Group by game ID and transform
    print("Transforming data...")
    games_list = []
    
    for game_id, game_group in df.groupby('GAME-ID'):
        if len(game_group) != 2:
            print(f"Skipping game {game_id} - has {len(game_group)} teams instead of 2")
            continue
        
        # Separate home and away teams
        # Handle regular home/away games
        away_team = game_group[game_group['VENUE\n(R/H/N)'] == 'R']
        home_team = game_group[game_group['VENUE\n(R/H/N)'] == 'H']
        
        # Handle neutral site games (both teams marked as 'N')
        neutral_teams = game_group[game_group['VENUE\n(R/H/N)'] == 'N']
        
        if len(neutral_teams) == 2:
            # For neutral sites, use betting spread to assign home/away
            # Team with positive spread = away (underdog)
            # Team with negative spread = home (favorite)
            team1, team2 = neutral_teams.iloc[0], neutral_teams.iloc[1]
            if team1['CLOSING SPREAD'] > 0:
                away_team, home_team = pd.DataFrame([team1]), pd.DataFrame([team2])
            else:
                away_team, home_team = pd.DataFrame([team2]), pd.DataFrame([team1])
            print(f"Neutral site game {game_id}: {away_team.iloc[0]['TEAM']} @ {home_team.iloc[0]['TEAM']} (assigned by spread)")
        elif len(away_team) != 1 or len(home_team) != 1:
            print(f"Skipping game {game_id} - unusual venue configuration (R:{len(away_team)}, H:{len(home_team)}, N:{len(neutral_teams)})")
            continue
        
        away_team = away_team.iloc[0]
        home_team = home_team.iloc[0]
        
        # Extract game information
        is_neutral_site = len(neutral_teams) == 2
        game_data = {
            'game_id': game_id,
            'date': away_team['DATE'],  # Same for both teams
            'away_team': away_team['TEAM'],
            'home_team': home_team['TEAM'],
            'away_score': away_team['F'],  # F and PTS are identical
            'home_score': home_team['F'],
            'neutral_site': is_neutral_site,
        }
        
        # Extract betting information
        # Spread: Use the away team's perspective (positive means away is underdog)
        # Total: Same for both teams
        # Moneyline: Extract both home and away lines
        
        # For spread, away team shows the spread (e.g., +6.0 means away gets 6 points)
        if pd.notna(away_team['CLOSING SPREAD']):
            game_data['final_spread'] = away_team['CLOSING SPREAD']
        else:
            game_data['final_spread'] = None
            
        # Total is the same for both teams
        if pd.notna(away_team['CLOSING TOTAL']):
            game_data['final_ou'] = away_team['CLOSING TOTAL']
        else:
            game_data['final_ou'] = None
        
        # Moneyline: Store both home and away
        game_data['away_moneyline'] = away_team['MONEYLINE'] if pd.notna(away_team['MONEYLINE']) else None
        game_data['home_moneyline'] = home_team['MONEYLINE'] if pd.notna(home_team['MONEYLINE']) else None
        
        # For final_ml, let's use the away team's moneyline as the primary indicator
        game_data['final_ml'] = game_data['away_moneyline']
        
        # Add some additional useful fields
        game_data['total_score'] = game_data['away_score'] + game_data['home_score']
        game_data['score_difference'] = abs(game_data['home_score'] - game_data['away_score'])
        game_data['home_win'] = 1 if game_data['home_score'] > game_data['away_score'] else 0
        
        games_list.append(game_data)
    
    # Create final DataFrame
    games_df = pd.DataFrame(games_list)
    
    print(f"Transformed data: {len(games_df)} games")
    print(f"Date range: {games_df['date'].min()} to {games_df['date'].max()}")
    
    # Display sample of the data
    print("\nSample of transformed data:")
    print(games_df[['game_id', 'date', 'away_team', 'home_team', 'away_score', 
                    'home_score', 'final_spread', 'final_ml', 'final_ou', 'neutral_site']].head(10))
    
    # Save to file if specified
    if output_file:
        # Determine file format from extension
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.xlsx':
            games_df.to_excel(output_file, index=False)
            print(f"\nSaved to Excel file: {output_file}")
        elif output_path.suffix.lower() == '.csv':
            games_df.to_csv(output_file, index=False)
            print(f"\nSaved to CSV file: {output_file}")
        else:
            # Default to CSV
            output_file = str(output_path.with_suffix('.csv'))
            games_df.to_csv(output_file, index=False)
            print(f"\nSaved to CSV file: {output_file}")
    
    return games_df

def main():
    # File paths
    input_file = r"C:\Users\micha\nba_predictions\data\team_boxscores\historical\2024-2025_NBA_Box_Score_Team-Stats.xlsx"
    output_file = r"C:\Users\micha\nba_predictions\data\game_results_2024-2025.csv"
    
    # Transform the data
    games_df = transform_team_boxscores_to_games(input_file, output_file)
    
    # Show some statistics
    print("\n" + "="*80)
    print("TRANSFORMATION SUMMARY")
    print("="*80)
    print(f"Total games processed: {len(games_df)}")
    print(f"Date range: {games_df['date'].min()} to {games_df['date'].max()}")
    print(f"Teams involved: {sorted(set(games_df['away_team'].tolist() + games_df['home_team'].tolist()))}")
    print(f"\nHome team win rate: {games_df['home_win'].mean():.1%}")
    print(f"Average total score: {games_df['total_score'].mean():.1f}")
    print(f"Average score difference: {games_df['score_difference'].mean():.1f}")
    print(f"Neutral site games: {games_df['neutral_site'].sum()} / {len(games_df)} ({games_df['neutral_site'].mean():.1%})")
    
    # Show betting data completeness
    print(f"\nBetting data completeness:")
    print(f"  Final spread available: {games_df['final_spread'].notna().sum()} / {len(games_df)} ({games_df['final_spread'].notna().mean():.1%})")
    print(f"  Final O/U available: {games_df['final_ou'].notna().sum()} / {len(games_df)} ({games_df['final_ou'].notna().mean():.1%})")
    print(f"  Moneyline available: {games_df['final_ml'].notna().sum()} / {len(games_df)} ({games_df['final_ml'].notna().mean():.1%})")
    
    return games_df

if __name__ == "__main__":
    games_df = main()
