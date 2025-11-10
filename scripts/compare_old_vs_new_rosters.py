"""
Compare old (legacy) vs new (injury reconstruction) roster generation on historical games.
"""
import sys
import random
import pandas as pd
from datetime import datetime
from typing import List, Dict

from scripts.player_data_loader import (
    get_team_players_legacy,
    get_team_players_with_injuries,
    precompute_season_baselines
)
from transform_player_stats import load_player_data


def load_team_boxscores(season: str) -> pd.DataFrame:
    """Load team boxscore data to get game dates and teams."""
    import glob
    import os
    
    # Try to find team boxscore files for the season
    pattern = f"data/team_boxscores/historical/*{season}*.xlsx"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No team boxscore files found for {season}")
        return None
    
    # Load the most recent file
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading team boxscores from: {latest_file}")
    
    df = pd.read_excel(latest_file, sheet_name='NBA Team Box Score')
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    return df


def get_sample_games(team_boxscore_df: pd.DataFrame, n_games: int = 50) -> List[Dict]:
    """Get a random sample of games from the dataset."""
    # Get unique game dates
    unique_dates = team_boxscore_df['DATE'].unique()
    
    # Filter to games after the first 15 games of the season (to ensure lookback data)
    if len(unique_dates) > 20:
        unique_dates = unique_dates[15:]
    
    # Sample random dates
    sample_size = min(n_games, len(unique_dates))
    sampled_dates = random.sample(list(unique_dates), sample_size)
    
    # For each date, get the teams playing
    games = []
    for date in sampled_dates:
        date_games = team_boxscore_df[team_boxscore_df['DATE'] == date]
        
        # Get unique teams for this date
        teams = date_games['OWN \nTEAM'].unique()
        
        for team in teams:
            # Extract city name
            team_city = team.split()[0] if ' ' in team else team
            games.append({
                'date': date,
                'team': team,
                'team_city': team_city
            })
            
            # Limit to n_games total
            if len(games) >= n_games:
                break
        
        if len(games) >= n_games:
            break
    
    return games


def compare_rosters(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame
) -> Dict:
    """Compare legacy vs new roster for a single game."""
    
    # Get rosters using both methods
    legacy_roster = get_team_players_legacy(team_name, game_date, season, player_boxscore_df)
    new_roster = get_team_players_with_injuries(team_name, game_date, season, player_boxscore_df)
    
    # Compare
    legacy_names = set(p['player_name'] for p in legacy_roster)
    new_names = set(p['player_name'] for p in new_roster)
    
    # Count injury markers
    legacy_injured = sum(1 for p in legacy_roster if p['projected_minutes'] == 0.0)
    new_injured = sum(1 for p in new_roster if p['projected_minutes'] == 0.0)
    
    # Calculate minutes_missing_top2 equivalent
    legacy_top2_missing = 0.0
    new_top2_missing = 0.0
    
    if len(legacy_roster) >= 2:
        for p in legacy_roster[:2]:
            if p['projected_minutes'] == 0.0:
                legacy_top2_missing += p['baseline_minutes']
    
    if len(new_roster) >= 2:
        for p in new_roster[:2]:
            if p['projected_minutes'] == 0.0:
                new_top2_missing += p['baseline_minutes']
    
    # Check for star_out
    legacy_star_out = any(
        p['projected_minutes'] == 0.0 and p['baseline_usage_rate'] >= 0.27
        for p in legacy_roster
    )
    new_star_out = any(
        p['projected_minutes'] == 0.0 and p['baseline_usage_rate'] >= 0.27
        for p in new_roster
    )
    
    return {
        'legacy_count': len(legacy_roster),
        'new_count': len(new_roster),
        'only_in_legacy': legacy_names - new_names,
        'only_in_new': new_names - legacy_names,
        'legacy_injured_count': legacy_injured,
        'new_injured_count': new_injured,
        'legacy_top2_missing': legacy_top2_missing,
        'new_top2_missing': new_top2_missing,
        'legacy_star_out': legacy_star_out,
        'new_star_out': new_star_out
    }


def main():
    season = '2023-2024'
    n_games = 50
    
    print(f"Comparing old vs new roster generation on {n_games} random games from {season}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    team_boxscore_df = load_team_boxscores(season)
    if team_boxscore_df is None:
        print("Could not load team boxscores")
        return
    
    player_boxscore_df = load_player_data(season)
    if player_boxscore_df is None:
        print("Could not load player boxscores")
        return
    
    # Precompute baselines for performance
    print("Precomputing season baselines...")
    from scripts.player_data_loader import precompute_season_baselines
    precompute_season_baselines(season, player_boxscore_df)
    
    # Get sample games
    print(f"\nSampling {n_games} games...")
    sample_games = get_sample_games(team_boxscore_df, n_games)
    print(f"Selected {len(sample_games)} team-games to analyze")
    
    # Compare rosters
    print("\nComparing rosters...")
    results = []
    
    for i, game in enumerate(sample_games):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(sample_games)}")
        
        try:
            comparison = compare_rosters(
                game['team_city'],
                game['date'].strftime('%Y-%m-%d'),
                season,
                player_boxscore_df
            )
            
            results.append({
                'date': game['date'].strftime('%Y-%m-%d'),
                'team': game['team'],
                **comparison
            })
        except Exception as e:
            print(f"  Error processing {game['team']} on {game['date']}: {e}")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if not results:
        print("No results to analyze")
        return
    
    # Overall statistics
    total_games = len(results)
    
    print(f"\nGames analyzed: {total_games}")
    
    # Games with injuries marked
    legacy_with_injuries = sum(1 for r in results if r['legacy_injured_count'] > 0)
    new_with_injuries = sum(1 for r in results if r['new_injured_count'] > 0)
    
    print(f"\nGames with injured players marked:")
    print(f"  Legacy: {legacy_with_injuries} ({legacy_with_injuries/total_games*100:.1f}%)")
    print(f"  New:    {new_with_injuries} ({new_with_injuries/total_games*100:.1f}%)")
    
    # Average minutes_missing_top2
    avg_legacy_top2 = sum(r['legacy_top2_missing'] for r in results) / total_games
    avg_new_top2 = sum(r['new_top2_missing'] for r in results) / total_games
    
    print(f"\nAverage minutes_missing_top2:")
    print(f"  Legacy: {avg_legacy_top2:.2f}")
    print(f"  New:    {avg_new_top2:.2f}")
    
    # Games with star_out
    legacy_star_out_count = sum(1 for r in results if r['legacy_star_out'])
    new_star_out_count = sum(1 for r in results if r['new_star_out'])
    
    print(f"\nGames with star_out=1:")
    print(f"  Legacy: {legacy_star_out_count} ({legacy_star_out_count/total_games*100:.1f}%)")
    print(f"  New:    {new_star_out_count} ({new_star_out_count/total_games*100:.1f}%)")
    
    # Roster size differences
    avg_legacy_size = sum(r['legacy_count'] for r in results) / total_games
    avg_new_size = sum(r['new_count'] for r in results) / total_games
    
    print(f"\nAverage roster size:")
    print(f"  Legacy: {avg_legacy_size:.1f}")
    print(f"  New:    {avg_new_size:.1f}")
    
    # Show examples of games with differences
    print("\n" + "=" * 80)
    print("EXAMPLES OF GAMES WITH INJURY DIFFERENCES")
    print("=" * 80)
    
    examples_shown = 0
    for r in results:
        if r['new_injured_count'] > r['legacy_injured_count'] and examples_shown < 5:
            print(f"\n{r['date']} - {r['team']}")
            print(f"  Legacy: {r['legacy_injured_count']} injured, top2_missing={r['legacy_top2_missing']:.1f}")
            print(f"  New:    {r['new_injured_count']} injured, top2_missing={r['new_top2_missing']:.1f}")
            if r['only_in_new']:
                print(f"  New detected injuries: {', '.join(list(r['only_in_new'])[:3])}")
            examples_shown += 1
    
    if examples_shown == 0:
        print("\n(No significant differences found in sample)")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"\nThe new injury reconstruction method identified injuries in")
    print(f"{new_with_injuries - legacy_with_injuries} additional games")
    print(f"({(new_with_injuries - legacy_with_injuries)/total_games*100:.1f}% of sample)")
    print(f"\nThis increases the average minutes_missing_top2 feature by")
    print(f"{avg_new_top2 - avg_legacy_top2:.2f} minutes per game.")


if __name__ == '__main__':
    main()

