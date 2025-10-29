"""
Comprehensive QA script for game data validation.

Validates:
1. Data structure (JSON format, required fields)
2. Field presence (all expected fields exist)
3. Field non-emptiness (no missing/null critical values)
4. Value distributions (ranges, types, consistency)
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import statistics


class GameDataQA:
    """Validator for game JSONL data."""
    
    # Required top-level fields
    REQUIRED_FIELDS = {
        'game_id', 'season', 'date', 'teams', 'market', 'outcome'
    }
    
    # Optional but expected fields
    OPTIONAL_FIELDS = {'players', 'metadata'}
    
    # Required team fields (inside teams.A and teams.H)
    TEAM_FIELDS = {
        'team_id', 'team_name', 'off_rating', 'def_rating', 'pace', 'rest_days'
    }
    
    # Required market fields
    MARKET_FIELDS = {'spread_home', 'total'}
    
    # Required outcome fields
    OUTCOME_FIELDS = {'home_final', 'away_final'}
    
    # Required player fields (if players present)
    PLAYER_FIELDS = {
        'player_id', 'player_name', 'baseline_minutes', 
        'projected_minutes', 'baseline_ts_pct', 'baseline_usage_rate'
    }
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.games = []
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(list)
        
    def load_data(self) -> bool:
        """Load JSONL data from file."""
        if not self.file_path.exists():
            self.errors.append(f"File not found: {self.file_path}")
            return False
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        game = json.loads(line)
                        self.games.append(game)
                    except json.JSONDecodeError as e:
                        self.errors.append(f"Line {line_num}: Invalid JSON - {e}")
            
            if not self.games:
                self.errors.append("No valid games found in file")
                return False
            
            print(f"✓ Loaded {len(self.games)} games from {self.file_path.name}")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to read file: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate JSON structure and required fields."""
        print("\n" + "="*60)
        print("1. VALIDATING DATA STRUCTURE")
        print("="*60)
        
        missing_fields = defaultdict(int)
        
        for idx, game in enumerate(self.games):
            game_id = game.get('game_id', f'game_{idx}')
            
            # Check required fields
            for field in self.REQUIRED_FIELDS:
                if field not in game:
                    missing_fields[field] += 1
                    if missing_fields[field] <= 3:  # Only log first 3
                        self.errors.append(f"Game {game_id}: Missing required field '{field}'")
            
            # Check teams structure
            if 'teams' in game:
                teams = game['teams']
                if not isinstance(teams, dict):
                    self.errors.append(f"Game {game_id}: 'teams' is not a dict")
                else:
                    for side in ['A', 'H']:
                        if side not in teams:
                            self.errors.append(f"Game {game_id}: Missing teams.{side}")
                            continue
                        
                        team = teams[side]
                        if not isinstance(team, dict):
                            self.errors.append(f"Game {game_id}: teams.{side} is not a dict")
                            continue
                        
                        for field in self.TEAM_FIELDS:
                            if field not in team:
                                missing_fields[f'teams.{side}.{field}'] += 1
                                if missing_fields[f'teams.{side}.{field}'] <= 3:
                                    self.errors.append(f"Game {game_id}: Missing field 'teams.{side}.{field}'")
            
            # Check market structure
            if 'market' in game:
                market = game['market']
                if not isinstance(market, dict):
                    self.errors.append(f"Game {game_id}: 'market' is not a dict")
                else:
                    for field in self.MARKET_FIELDS:
                        if field not in market:
                            missing_fields[f'market.{field}'] += 1
            
            # Check outcome structure
            if 'outcome' in game:
                outcome = game['outcome']
                if not isinstance(outcome, dict):
                    self.errors.append(f"Game {game_id}: 'outcome' is not a dict")
                else:
                    for field in self.OUTCOME_FIELDS:
                        if field not in outcome:
                            missing_fields[f'outcome.{field}'] += 1
            
            # Check player structures (if present)
            if 'players' in game:
                players = game['players']
                if not isinstance(players, dict):
                    self.errors.append(f"Game {game_id}: 'players' is not a dict")
                    continue
                
                for side in ['A', 'H']:
                    if side not in players:
                        self.warnings.append(f"Game {game_id}: Missing players.{side}")
                        continue
                    
                    if not isinstance(players[side], list):
                        self.errors.append(f"Game {game_id}: players.{side} is not a list")
                        continue
                    
                    if len(players[side]) < 5:
                        self.warnings.append(f"Game {game_id}: players.{side} has only {len(players[side])} players (expected ≥5)")
                    
                    for player_idx, player in enumerate(players[side]):
                        for field in self.PLAYER_FIELDS:
                            if field not in player:
                                if missing_fields[f'player.{field}'] == 0:  # Only log once
                                    self.errors.append(f"Game {game_id}: Player {player_idx} missing field '{field}'")
                                missing_fields[f'player.{field}'] += 1
        
        # Summary
        if missing_fields:
            print(f"\n⚠ Found missing fields across games:")
            for field, count in sorted(missing_fields.items(), key=lambda x: -x[1])[:10]:
                print(f"  - {field}: missing in {count} games")
        
        if len([e for e in self.errors if 'Missing' in e]) == 0:
            print("✓ All required fields present")
            return True
        else:
            print(f"✗ Found {len([e for e in self.errors if 'Missing' in e])} missing field errors")
            return False
    
    def validate_non_emptiness(self) -> bool:
        """Validate that critical fields are not null/empty."""
        print("\n" + "="*60)
        print("2. VALIDATING FIELD NON-EMPTINESS")
        print("="*60)
        
        null_counts = defaultdict(int)
        
        for idx, game in enumerate(self.games):
            game_id = game.get('game_id', f'game_{idx}')
            
            # Check scores (must not be None)
            outcome = game.get('outcome', {})
            for field in ['away_final', 'home_final']:
                if outcome.get(field) is None:
                    self.errors.append(f"Game {game_id}: outcome.{field} is null")
                    null_counts[f'outcome.{field}'] += 1
            
            # Check market data (should not be None)
            market = game.get('market', {})
            for field in ['spread_home', 'total']:
                if market.get(field) is None:
                    self.warnings.append(f"Game {game_id}: market.{field} is null (missing market data)")
                    null_counts[f'market.{field}'] += 1
            
            # Check team stats
            teams = game.get('teams', {})
            for side in ['A', 'H']:
                team = teams.get(side, {})
                for field in ['off_rating', 'def_rating', 'pace', 'rest_days']:
                    if team.get(field) is None:
                        self.warnings.append(f"Game {game_id}: teams.{side}.{field} is null")
                        null_counts[f'teams.{side}.{field}'] += 1
            
            # Check player data (if present)
            if 'players' in game:
                players = game['players']
                for side in ['A', 'H']:
                    if side in players:
                        for player in players[side]:
                            for field in ['baseline_minutes', 'baseline_ts_pct', 'baseline_usage_rate']:
                                if player.get(field) is None:
                                    null_counts[f'player.{field}'] += 1
        
        # Summary
        if null_counts:
            print(f"\n⚠ Found null/empty values:")
            for field, count in sorted(null_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  - {field}: null in {count} games")
        
        critical_nulls = null_counts.get('outcome.away_final', 0) + null_counts.get('outcome.home_final', 0)
        if critical_nulls == 0:
            print("✓ All critical fields (scores) are non-null")
            return True
        else:
            print(f"✗ Found {critical_nulls} null scores")
            return False
    
    def validate_distributions(self) -> bool:
        """Validate value distributions and ranges."""
        print("\n" + "="*60)
        print("3. VALIDATING VALUE DISTRIBUTIONS")
        print("="*60)
        
        # Collect statistics
        for game in self.games:
            # Scores
            outcome = game.get('outcome', {})
            away_score = outcome.get('away_final')
            home_score = outcome.get('home_final')
            if away_score is not None and home_score is not None:
                self.stats['away_score'].append(away_score)
                self.stats['home_score'].append(home_score)
                self.stats['total_score'].append(away_score + home_score)
                self.stats['margin'].append(home_score - away_score)
            
            # Market data
            market = game.get('market', {})
            spread = market.get('spread_home')
            if spread is not None:
                self.stats['spread_home'].append(spread)
            total = market.get('total')
            if total is not None:
                self.stats['total'].append(total)
            
            # Team stats
            teams = game.get('teams', {})
            for side_name, side_key in [('away', 'A'), ('home', 'H')]:
                team = teams.get(side_key, {})
                for field in ['off_rating', 'def_rating', 'pace', 'rest_days']:
                    value = team.get(field)
                    if value is not None:
                        self.stats[f'{side_name}.{field}'].append(value)
            
            # Player stats
            if 'players' in game:
                players = game['players']
                for side in ['A', 'H']:
                    if side in players:
                        for player in players[side]:
                            for field in ['baseline_minutes', 'projected_minutes', 'baseline_ts_pct', 'baseline_usage_rate']:
                                value = player.get(field)
                                if value is not None:
                                    self.stats[f'player.{field}'].append(value)
        
        # Validate ranges
        issues = []
        
        # Scores (should be 60-180)
        for score_type in ['away_score', 'home_score']:
            if self.stats[score_type]:
                min_score = min(self.stats[score_type])
                max_score = max(self.stats[score_type])
                if min_score < 60 or max_score > 180:
                    issues.append(f"{score_type} range [{min_score}, {max_score}] is unusual (expected 60-180)")
        
        # Spreads (should be -30 to +30)
        if self.stats['spread_home']:
            min_spread = min(self.stats['spread_home'])
            max_spread = max(self.stats['spread_home'])
            if min_spread < -30 or max_spread > 30:
                issues.append(f"spread_home range [{min_spread}, {max_spread}] is unusual (expected -30 to +30)")
        
        # Offensive/defensive ratings (should be 90-130)
        for field in ['away.off_rating', 'home.off_rating', 'away.def_rating', 'home.def_rating']:
            if self.stats[field]:
                min_rating = min(self.stats[field])
                max_rating = max(self.stats[field])
                if min_rating < 80 or max_rating > 140:
                    issues.append(f"{field} range [{min_rating:.1f}, {max_rating:.1f}] is unusual (expected 90-130)")
        
        # Player minutes (should be 0-48)
        for field in ['player.baseline_minutes', 'player.projected_minutes']:
            if self.stats[field]:
                min_min = min(self.stats[field])
                max_min = max(self.stats[field])
                if min_min < 0 or max_min > 50:
                    issues.append(f"{field} range [{min_min:.1f}, {max_min:.1f}] is unusual (expected 0-48)")
        
        # Player TS% (should be 0.3-0.7)
        if self.stats['player.baseline_ts_pct']:
            min_ts = min(self.stats['player.baseline_ts_pct'])
            max_ts = max(self.stats['player.baseline_ts_pct'])
            if min_ts < 0.2 or max_ts > 0.8:
                issues.append(f"player.baseline_ts_pct range [{min_ts:.3f}, {max_ts:.3f}] is unusual (expected 0.3-0.7)")
        
        # Player usage rate (should be 0.05-0.40)
        if self.stats['player.baseline_usage_rate']:
            min_usg = min(self.stats['player.baseline_usage_rate'])
            max_usg = max(self.stats['player.baseline_usage_rate'])
            if min_usg < 0 or max_usg > 0.5:
                issues.append(f"player.baseline_usage_rate range [{min_usg:.3f}, {max_usg:.3f}] is unusual (expected 0.05-0.40)")
        
        # Print summary statistics
        print("\nKey Statistics:")
        print("-" * 60)
        
        stat_summary = [
            ('Scores (away)', 'away_score'),
            ('Scores (home)', 'home_score'),
            ('Total score', 'total_score'),
            ('Margin (home)', 'margin'),
            ('Spread (home)', 'spread_home'),
            ('Market total', 'total'),
            ('Off rating (away)', 'away.off_rating'),
            ('Off rating (home)', 'home.off_rating'),
            ('Def rating (away)', 'away.def_rating'),
            ('Def rating (home)', 'home.def_rating'),
            ('Pace (away)', 'away.pace'),
            ('Pace (home)', 'home.pace'),
            ('Rest days (away)', 'away.rest_days'),
            ('Rest days (home)', 'home.rest_days'),
        ]
        
        for label, key in stat_summary:
            if self.stats[key]:
                values = self.stats[key]
                print(f"{label:20s}: mean={statistics.mean(values):7.2f}, "
                      f"std={statistics.stdev(values) if len(values) > 1 else 0:6.2f}, "
                      f"min={min(values):7.2f}, max={max(values):7.2f}")
        
        # Player statistics
        if self.stats['player.baseline_minutes']:
            print("\nPlayer Statistics:")
            print("-" * 60)
            player_stats = [
                ('Minutes (baseline)', 'player.baseline_minutes'),
                ('Minutes (projected)', 'player.projected_minutes'),
                ('TS% (baseline)', 'player.baseline_ts_pct'),
                ('Usage (baseline)', 'player.baseline_usage_rate'),
            ]
            
            for label, key in player_stats:
                if self.stats[key]:
                    values = self.stats[key]
                    print(f"{label:20s}: mean={statistics.mean(values):6.3f}, "
                          f"std={statistics.stdev(values) if len(values) > 1 else 0:6.3f}, "
                          f"min={min(values):6.3f}, max={max(values):6.3f}")
        
        # Report issues
        if issues:
            print("\n⚠ Distribution warnings:")
            for issue in issues:
                print(f"  - {issue}")
                self.warnings.append(issue)
        else:
            print("\n✓ All value distributions are within expected ranges")
        
        return len(issues) == 0
    
    def validate_consistency(self) -> bool:
        """Validate internal consistency and relationships."""
        print("\n" + "="*60)
        print("4. VALIDATING DATA CONSISTENCY")
        print("="*60)
        
        consistency_issues = []
        
        for idx, game in enumerate(self.games):
            game_id = game.get('game_id', f'game_{idx}')
            
            # Check score vs margin
            outcome = game.get('outcome', {})
            away_score = outcome.get('away_final')
            home_score = outcome.get('home_final')
            if away_score is not None and home_score is not None:
                if away_score < 0 or home_score < 0:
                    consistency_issues.append(f"Game {game_id}: Negative score")
                
                # Check if total is close to sum of scores (if market data present)
                # Note: Large differences are normal, markets are predictions
                market = game.get('market', {})
                total = market.get('total')
                if total is not None:
                    actual_total = away_score + home_score
                    if abs(actual_total - total) > 50:  # Only flag extreme outliers
                        self.warnings.append(f"Game {game_id}: Market total {total} vs actual {actual_total} (diff={actual_total-total:.1f})")
            
            # Check player counts
            if 'players' in game:
                players = game['players']
                for side_name, side_key in [('away', 'A'), ('home', 'H')]:
                    if side_key in players:
                        num_players = len(players[side_key])
                        if num_players < 5:
                            consistency_issues.append(f"Game {game_id}: Only {num_players} {side_name} players (need ≥5)")
                        elif num_players > 15:
                            consistency_issues.append(f"Game {game_id}: {num_players} {side_name} players (unusually high)")
                        
                        # Check player minutes sum (overtime games can exceed 240)
                        total_minutes = sum(p.get('projected_minutes', 0) for p in players[side_key])
                        if total_minutes < 150 or total_minutes > 350:
                            self.warnings.append(f"Game {game_id}: {side_name} total minutes = {total_minutes:.1f} (expected ~240, may be OT)")
            
            # Check date format
            date = game.get('date')
            if date and not isinstance(date, str):
                consistency_issues.append(f"Game {game_id}: Date is not a string")
            elif date:
                # Basic date format check (YYYY-MM-DD)
                parts = date.split('-')
                if len(parts) != 3 or len(parts[0]) != 4:
                    consistency_issues.append(f"Game {game_id}: Date format is not YYYY-MM-DD: {date}")
        
        # Games with players vs without
        games_with_players = sum(1 for g in self.games if 'players' in g and g['players'])
        games_without_players = len(self.games) - games_with_players
        
        print(f"\nPlayer Data Coverage:")
        print(f"  - Games with players: {games_with_players} ({100*games_with_players/len(self.games):.1f}%)")
        print(f"  - Games without players: {games_without_players} ({100*games_without_players/len(self.games):.1f}%)")
        
        if games_without_players > 0:
            self.warnings.append(f"{games_without_players} games are missing player data")
        
        # Report issues
        if consistency_issues:
            print(f"\n⚠ Found {len(consistency_issues)} consistency issues:")
            for issue in consistency_issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(consistency_issues) > 10:
                print(f"  ... and {len(consistency_issues) - 10} more")
            self.errors.extend(consistency_issues)
            return False
        else:
            print("\n✓ All consistency checks passed")
            return True
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("\n" + "="*60)
        print(f"QA VALIDATION: {self.file_path.name}")
        print("="*60)
        
        if not self.load_data():
            return False
        
        results = [
            self.validate_structure(),
            self.validate_non_emptiness(),
            self.validate_distributions(),
            self.validate_consistency(),
        ]
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Total games: {len(self.games)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print(f"\n❌ VALIDATION FAILED with {len(self.errors)} errors")
            print("\nFirst 10 errors:")
            for error in self.errors[:10]:
                print(f"  - {error}")
            return False
        elif self.warnings:
            print(f"\n⚠️  VALIDATION PASSED with {len(self.warnings)} warnings")
            print("\nWarnings:")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            return True
        else:
            print("\n✅ VALIDATION PASSED - All checks successful!")
            return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QA validation for NBA game data")
    parser.add_argument("file", type=str, help="Path to JSONL file to validate")
    parser.add_argument("--verbose", action="store_true", help="Show all errors and warnings")
    
    args = parser.parse_args()
    
    qa = GameDataQA(args.file)
    success = qa.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

