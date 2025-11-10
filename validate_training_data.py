"""
Validate the newly generated training data with injury reconstruction.
Compare key statistics between old and new training datasets.
"""
import json
import random
from collections import defaultdict


def load_jsonl(file_path):
    """Load JSONL file into a list of game records."""
    games = []
    with open(file_path, 'r') as f:
        for line in f:
            games.append(json.loads(line.strip()))
    return games


def analyze_injury_features(games):
    """Analyze injury-related features in the dataset."""
    stats = {
        'total_games': len(games),
        'games_with_home_injured': 0,
        'games_with_away_injured': 0,
        'games_with_home_star_out': 0,
        'games_with_away_star_out': 0,
        'home_minutes_missing_top2': [],
        'away_minutes_missing_top2': [],
    }
    
    for game in games:
        # Check for injured players (home)
        home_players = game.get('home_players', [])
        home_injured = sum(1 for p in home_players if p.get('projected_minutes') == 0.0)
        if home_injured > 0:
            stats['games_with_home_injured'] += 1
        
        # Check for star out (home)
        home_star = any(
            p.get('projected_minutes') == 0.0 and p.get('baseline_usage_rate', 0) >= 0.27
            for p in home_players
        )
        if home_star:
            stats['games_with_home_star_out'] += 1
        
        # Get minutes missing top2 (home)
        sorted_home = sorted(home_players, key=lambda p: p.get('baseline_minutes', 0), reverse=True)
        home_top2_missing = 0.0
        if len(sorted_home) >= 2:
            for p in sorted_home[:2]:
                if p.get('projected_minutes') == 0.0:
                    home_top2_missing += p.get('baseline_minutes', 0)
        stats['home_minutes_missing_top2'].append(home_top2_missing)
        
        # Check for injured players (away)
        away_players = game.get('away_players', [])
        away_injured = sum(1 for p in away_players if p.get('projected_minutes') == 0.0)
        if away_injured > 0:
            stats['games_with_away_injured'] += 1
        
        # Check for star out (away)
        away_star = any(
            p.get('projected_minutes') == 0.0 and p.get('baseline_usage_rate', 0) >= 0.27
            for p in away_players
        )
        if away_star:
            stats['games_with_away_star_out'] += 1
        
        # Get minutes missing top2 (away)
        sorted_away = sorted(away_players, key=lambda p: p.get('baseline_minutes', 0), reverse=True)
        away_top2_missing = 0.0
        if len(sorted_away) >= 2:
            for p in sorted_away[:2]:
                if p.get('projected_minutes') == 0.0:
                    away_top2_missing += p.get('baseline_minutes', 0)
        stats['away_minutes_missing_top2'].append(away_top2_missing)
    
    return stats


def check_schema_compliance(games, n_samples=10):
    """Check that games comply with expected schema."""
    issues = []
    
    sample_games = random.sample(games, min(n_samples, len(games)))
    
    for game in sample_games:
        # Check required fields
        required_fields = ['game_id', 'date', 'away_team', 'home_team']
        for field in required_fields:
            if field not in game:
                issues.append(f"Game {game.get('game_id', 'UNKNOWN')} missing field: {field}")
        
        # Check player data
        if 'home_players' in game:
            if len(game['home_players']) < 5:
                issues.append(f"Game {game['game_id']}: home team has <5 players")
        
        if 'away_players' in game:
            if len(game['away_players']) < 5:
                issues.append(f"Game {game['game_id']}: away team has <5 players")
    
    return issues


def spot_check_games(games, n_samples=10):
    """Spot check random games for roster correctness."""
    print("\nSPOT CHECK: Random game rosters")
    print("="*80)
    
    sample_games = random.sample(games, min(n_samples, len(games)))
    
    for game in sample_games:
        print(f"\n{game['date']} - {game['away_team']} @ {game['home_team']}")
        
        # Home roster
        home_players = game.get('home_players', [])
        home_injured = [p for p in home_players if p.get('projected_minutes') == 0.0]
        print(f"  Home: {len(home_players)} players, {len(home_injured)} injured")
        if home_injured:
            print(f"    Injured: {', '.join(p['player_name'] for p in home_injured[:3])}")
        
        # Away roster
        away_players = game.get('away_players', [])
        away_injured = [p for p in away_players if p.get('projected_minutes') == 0.0]
        print(f"  Away: {len(away_players)} players, {len(away_injured)} injured")
        if away_injured:
            print(f"    Injured: {', '.join(p['player_name'] for p in away_injured[:3])}")


def main():
    print("="*80)
    print("VALIDATING NEW TRAINING DATA WITH INJURY RECONSTRUCTION")
    print("="*80)
    
    # Load old and new training data
    print("\nLoading datasets...")
    try:
        old_games = load_jsonl('data/games_train_with_players_90.jsonl')
        print(f"  Old (legacy): {len(old_games)} games")
    except FileNotFoundError:
        print("  Old training data not found (this is okay)")
        old_games = []
    
    try:
        new_games = load_jsonl('data/games_train_with_players_90_new.jsonl')
        print(f"  New (injury reconstruction): {len(new_games)} games")
    except FileNotFoundError:
        print("ERROR: New training data not found!")
        return
    
    # Analyze injury features
    print("\n" + "="*80)
    print("INJURY FEATURE ANALYSIS")
    print("="*80)
    
    if old_games:
        print("\nOLD (Legacy) Data:")
        old_stats = analyze_injury_features(old_games)
        print(f"  Total games: {old_stats['total_games']}")
        print(f"  Games with home injuries: {old_stats['games_with_home_injured']} "
              f"({old_stats['games_with_home_injured']/old_stats['total_games']*100:.1f}%)")
        print(f"  Games with away injuries: {old_stats['games_with_away_injured']} "
              f"({old_stats['games_with_away_injured']/old_stats['total_games']*100:.1f}%)")
        print(f"  Games with home star out: {old_stats['games_with_home_star_out']} "
              f"({old_stats['games_with_home_star_out']/old_stats['total_games']*100:.1f}%)")
        print(f"  Games with away star out: {old_stats['games_with_away_star_out']} "
              f"({old_stats['games_with_away_star_out']/old_stats['total_games']*100:.1f}%)")
        print(f"  Avg home_minutes_missing_top2: {sum(old_stats['home_minutes_missing_top2'])/len(old_stats['home_minutes_missing_top2']):.2f}")
        print(f"  Avg away_minutes_missing_top2: {sum(old_stats['away_minutes_missing_top2'])/len(old_stats['away_minutes_missing_top2']):.2f}")
    
    print("\nNEW (Injury Reconstruction) Data:")
    new_stats = analyze_injury_features(new_games)
    print(f"  Total games: {new_stats['total_games']}")
    print(f"  Games with home injuries: {new_stats['games_with_home_injured']} "
          f"({new_stats['games_with_home_injured']/new_stats['total_games']*100:.1f}%)")
    print(f"  Games with away injuries: {new_stats['games_with_away_injured']} "
          f"({new_stats['games_with_away_injured']/new_stats['total_games']*100:.1f}%)")
    print(f"  Games with home star out: {new_stats['games_with_home_star_out']} "
          f"({new_stats['games_with_home_star_out']/new_stats['total_games']*100:.1f}%)")
    print(f"  Games with away star out: {new_stats['games_with_away_star_out']} "
          f"({new_stats['games_with_away_star_out']/new_stats['total_games']*100:.1f}%)")
    print(f"  Avg home_minutes_missing_top2: {sum(new_stats['home_minutes_missing_top2'])/len(new_stats['home_minutes_missing_top2']):.2f}")
    print(f"  Avg away_minutes_missing_top2: {sum(new_stats['away_minutes_missing_top2'])/len(new_stats['away_minutes_missing_top2']):.2f}")
    
    if old_games:
        print("\nCOMPARISON (New vs Old):")
        home_inj_diff = new_stats['games_with_home_injured'] - old_stats['games_with_home_injured']
        away_inj_diff = new_stats['games_with_away_injured'] - old_stats['games_with_away_injured']
        print(f"  Additional games with home injuries: {home_inj_diff} (+{home_inj_diff/old_stats['total_games']*100:.1f}%)")
        print(f"  Additional games with away injuries: {away_inj_diff} (+{away_inj_diff/old_stats['total_games']*100:.1f}%)")
        
        home_star_diff = new_stats['games_with_home_star_out'] - old_stats['games_with_home_star_out']
        away_star_diff = new_stats['games_with_away_star_out'] - old_stats['games_with_away_star_out']
        print(f"  Additional games with home star out: {home_star_diff}")
        print(f"  Additional games with away star out: {away_star_diff}")
    
    # Schema compliance
    print("\n" + "="*80)
    print("SCHEMA COMPLIANCE CHECK")
    print("="*80)
    
    issues = check_schema_compliance(new_games, n_samples=20)
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n  All sampled games comply with schema")
    
    # Spot check
    spot_check_games(new_games, n_samples=10)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if old_games:
        injury_increase = ((new_stats['games_with_home_injured'] + new_stats['games_with_away_injured']) - 
                          (old_stats['games_with_home_injured'] + old_stats['games_with_away_injured']))
        print(f"\nThe new injury reconstruction method detected injuries in")
        print(f"{injury_increase} additional game-team instances")
        print(f"({injury_increase/(old_stats['total_games']*2)*100:.1f}% of all team-games)")
    
    print(f"\nTotal training examples: {new_stats['total_games']}")
    print(f"Games with any injury: {(new_stats['games_with_home_injured'] + new_stats['games_with_away_injured'])} "
          f"(note: a game can have injuries on both teams)")
    
    if len(issues) == 0:
        print("\n  PASSED: All schema compliance checks")
    else:
        print(f"\n  WARNING: {len(issues)} schema compliance issues found")
    
    print("\nNew training data is ready for model training!")
    print("="*80)


if __name__ == '__main__':
    main()

