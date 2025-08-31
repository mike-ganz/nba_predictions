#!/usr/bin/env python3
"""
Comprehensive Training Data Validator for NBA JSONL Files

This script validates JSONL training data files for both structural correctness 
and NBA-specific business logic consistency, including:

1. JSON format validation
2. OpenAI training format validation  
3. Time consistency validation (game clock progression)
4. Scoring logic validation (non-scoring plays should have null scoring_team)
5. Data structure validation (required fields, data types)
6. NBA rules validation (quarters, time formats, team names)

Usage:
    python validate_training_data.py <jsonl_file_path> [--sample N] [--verbose]

Examples:
    python validate_training_data.py data/training/output.jsonl
    python validate_training_data.py data/training/output.jsonl --sample 1000 --verbose
"""

import json
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter


class NBATrainingDataValidator:
    """Comprehensive validator for NBA training JSONL data."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = defaultdict(list)
        self.warnings = defaultdict(list)
        self.stats = defaultdict(int)
        
        # NBA-specific constants
        self.NON_SCORING_PLAY_KEYWORDS = [
            'REBOUND', 'SUB', 'MISS', 'TURNOVER', 'FOUL', 'TIMEOUT', 'VIOLATION',
            'EJECTION', 'TECHNICAL', 'FLAGRANT', 'JUMP BALL'
        ]
        
        # Common NBA team abbreviations for validation
        self.VALID_NBA_TEAMS = {
            'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        }
    
    def validate_jsonl_file(self, file_path: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate an entire JSONL file.
        
        Args:
            file_path: Path to JSONL file
            sample_size: If provided, only validate first N lines
            
        Returns:
            dict: Comprehensive validation report
        """
        print(f"🔍 Validating NBA training data: {file_path}")
        print("=" * 80)
        
        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}
        
        # Reset validation state
        self.errors.clear()
        self.warnings.clear()
        self.stats.clear()
        
        line_number = 0
        valid_examples = 0
        processed_examples = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    processed_examples += 1
                    
                    # Apply sample size limit
                    if sample_size and processed_examples > sample_size:
                        break
                    
                    # Validate this line
                    is_valid = self._validate_training_example(line, line_number)
                    if is_valid:
                        valid_examples += 1
                    
                    # Show progress
                    if processed_examples % 1000 == 0:
                        print(f"📊 Processed {processed_examples:,} examples...")
        
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
        
        # Generate comprehensive report
        return self._generate_validation_report(
            file_path, line_number, processed_examples, valid_examples
        )
    
    def _validate_training_example(self, line: str, line_number: int) -> bool:
        """Validate a single training example."""
        try:
            # 1. JSON format validation
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                self.errors['json_format'].append(f"Line {line_number}: Invalid JSON - {e}")
                return False
            
            # 2. OpenAI format validation
            if not self._validate_openai_format(data, line_number):
                return False
            
            # 3. NBA-specific business logic validation
            user_content = json.loads(data['messages'][0]['content'])
            assistant_content = json.loads(data['messages'][1]['content'])
            
            # Validate user context
            self._validate_user_context(user_content, line_number)
            
            # Validate assistant response (next_play)
            self._validate_next_play(assistant_content, line_number)
            
            # Cross-validate time consistency
            self._validate_time_consistency(user_content, assistant_content, line_number)
            
            # Cross-validate scoring logic
            self._validate_scoring_consistency(assistant_content, line_number)
            
            return True
            
        except Exception as e:
            self.errors['validation_error'].append(f"Line {line_number}: Validation failed - {e}")
            return False
    
    def _validate_openai_format(self, data: Dict[str, Any], line_number: int) -> bool:
        """Validate OpenAI fine-tuning format."""
        if 'messages' not in data:
            self.errors['openai_format'].append(f"Line {line_number}: Missing 'messages' field")
            return False
        
        messages = data['messages']
        if not isinstance(messages, list) or len(messages) != 2:
            self.errors['openai_format'].append(f"Line {line_number}: Expected exactly 2 messages")
            return False
        
        # Validate user message
        user_msg = messages[0]
        if user_msg.get('role') != 'user':
            self.errors['openai_format'].append(f"Line {line_number}: First message should have role 'user'")
            return False
        
        # Validate assistant message
        assistant_msg = messages[1]
        if assistant_msg.get('role') != 'assistant':
            self.errors['openai_format'].append(f"Line {line_number}: Second message should have role 'assistant'")
            return False
        
        # Validate message content is valid JSON
        try:
            json.loads(user_msg['content'])
            json.loads(assistant_msg['content'])
        except json.JSONDecodeError as e:
            self.errors['openai_format'].append(f"Line {line_number}: Message content is not valid JSON - {e}")
            return False
        
        return True
    
    def _validate_user_context(self, user_content: Dict[str, Any], line_number: int):
        """Validate user context structure and content."""
        # Required fields
        required_fields = ['away_team', 'home_team', 'players', 'recent_plays']
        for field in required_fields:
            if field not in user_content:
                self.errors['user_context'].append(f"Line {line_number}: Missing '{field}' in user context")
        
        # Validate teams
        for team_key in ['away_team', 'home_team']:
            if team_key in user_content:
                team = user_content[team_key]
                if not isinstance(team, dict):
                    self.errors['user_context'].append(f"Line {line_number}: {team_key} should be a dict")
                else:
                    # Check team name
                    if 'name' in team and team['name'] not in self.VALID_NBA_TEAMS:
                        self.warnings['team_validation'].append(f"Line {line_number}: Unknown team '{team['name']}'")
                    
                    # Check team stats
                    if 'stats' in team:
                        self._validate_team_stats(team['stats'], team_key, line_number)
        
        # Validate recent plays
        if 'recent_plays' in user_content:
            recent_plays = user_content['recent_plays']
            if not isinstance(recent_plays, list):
                self.errors['user_context'].append(f"Line {line_number}: recent_plays should be a list")
            else:
                for i, play in enumerate(recent_plays):
                    self._validate_play_structure(play, f"recent_plays[{i}]", line_number)
        
        # Update stats
        if 'recent_plays' in user_content:
            self.stats['recent_plays_count'] += len(user_content['recent_plays'])
    
    def _validate_team_stats(self, stats: Dict[str, Any], team_key: str, line_number: int):
        """Validate team statistics."""
        expected_stats = ['OEFF', 'DEFF', 'PACE', 'REST_DAYS']
        for stat in expected_stats:
            if stat in stats:
                value = stats[stat]
                if stat == 'REST_DAYS':
                    if not isinstance(value, int) or value < 0:
                        self.warnings['team_stats'].append(
                            f"Line {line_number}: {team_key} {stat} should be non-negative integer"
                        )
                else:
                    if not isinstance(value, (int, float)) or value is None:
                        self.warnings['team_stats'].append(
                            f"Line {line_number}: {team_key} {stat} should be a number"
                        )
    
    def _validate_play_structure(self, play: Dict[str, Any], context: str, line_number: int):
        """Validate play structure."""
        required_play_fields = ['quarter', 'time_remaining', 'description', 'score', 'scoring_team', 'points_scored']
        
        for field in required_play_fields:
            if field not in play:
                self.errors['play_structure'].append(f"Line {line_number}: Missing '{field}' in {context}")
        
        # Validate quarter
        if 'quarter' in play:
            quarter = play['quarter']
            if not isinstance(quarter, int) or quarter < 1 or quarter > 10:  # Allow up to 6OT
                self.warnings['play_validation'].append(
                    f"Line {line_number}: {context} quarter {quarter} seems invalid"
                )
        
        # Validate time format
        if 'time_remaining' in play:
            time_str = play['time_remaining']
            if not self._is_valid_time_format(time_str):
                self.errors['play_validation'].append(
                    f"Line {line_number}: {context} invalid time format '{time_str}'"
                )
        
        # Validate points_scored
        if 'points_scored' in play:
            points = play['points_scored']
            if not isinstance(points, int) or points < 0 or points > 4:
                self.warnings['play_validation'].append(
                    f"Line {line_number}: {context} points_scored {points} seems unusual"
                )
    
    def _validate_next_play(self, assistant_content: Dict[str, Any], line_number: int):
        """Validate next_play structure."""
        if 'next_play' not in assistant_content:
            self.errors['next_play'].append(f"Line {line_number}: Missing 'next_play' in assistant content")
            return
        
        next_play = assistant_content['next_play']
        self._validate_play_structure(next_play, 'next_play', line_number)
    
    def _validate_time_consistency(self, user_content: Dict[str, Any], assistant_content: Dict[str, Any], line_number: int):
        """
        Validate time consistency: next_play time should be later (smaller) than recent_plays last time
        within the same quarter.
        """
        if 'recent_plays' not in user_content or 'next_play' not in assistant_content:
            return
        
        recent_plays = user_content['recent_plays']
        next_play = assistant_content['next_play']
        
        if not recent_plays:
            return
        
        # Get last recent play
        last_recent_play = recent_plays[-1]
        
        # Check if same quarter
        last_quarter = last_recent_play.get('quarter')
        next_quarter = next_play.get('quarter')
        
        if last_quarter != next_quarter:
            # Different quarters - time can reset, this is valid
            return
        
        # Same quarter - validate time progression
        last_time = last_recent_play.get('time_remaining')
        next_time = next_play.get('time_remaining')
        
        if last_time and next_time:
            last_seconds = self._time_to_seconds(last_time)
            next_seconds = self._time_to_seconds(next_time)
            
            if last_seconds is not None and next_seconds is not None:
                # Time should be decreasing (next_time should be smaller)
                if next_seconds > last_seconds:
                    self.errors['time_consistency'].append(
                        f"Line {line_number}: Time inconsistency - last play {last_time}, next play {next_time} (same quarter {next_quarter})"
                    )
                elif next_seconds == last_seconds:
                    self.warnings['time_consistency'].append(
                        f"Line {line_number}: Same time for consecutive plays - {next_time} in quarter {next_quarter}"
                    )
    
    def _validate_scoring_consistency(self, assistant_content: Dict[str, Any], line_number: int):
        """
        Validate scoring consistency: plays with non-scoring keywords should have 
        scoring_team=null and points_scored=0.
        """
        if 'next_play' not in assistant_content:
            return
        
        next_play = assistant_content['next_play']
        description = next_play.get('description', '').upper()
        scoring_team = next_play.get('scoring_team')
        points_scored = next_play.get('points_scored', 0)
        
        # Check if description contains non-scoring keywords
        contains_non_scoring = any(keyword in description for keyword in self.NON_SCORING_PLAY_KEYWORDS)
        
        if contains_non_scoring:
            self.stats['non_scoring_plays'] += 1
            
            # Should have null scoring_team and 0 points
            if scoring_team is not None:
                self.errors['scoring_consistency'].append(
                    f"Line {line_number}: Non-scoring play '{description[:50]}...' has scoring_team='{scoring_team}' (should be null)"
                )
            
            if points_scored != 0:
                self.errors['scoring_consistency'].append(
                    f"Line {line_number}: Non-scoring play '{description[:50]}...' has points_scored={points_scored} (should be 0)"
                )
        else:
            self.stats['potentially_scoring_plays'] += 1
            
            # If points > 0, should have scoring team
            if points_scored > 0 and scoring_team is None:
                self.warnings['scoring_consistency'].append(
                    f"Line {line_number}: Scoring play with {points_scored} points has no scoring_team"
                )
    
    def _is_valid_time_format(self, time_str: str) -> bool:
        """Validate time format (MM:SS)."""
        if not isinstance(time_str, str):
            return False
        
        pattern = r'^\d{1,2}:\d{2}$'
        if not re.match(pattern, time_str):
            return False
        
        # Additional validation
        try:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            return 0 <= minutes <= 12 and 0 <= seconds <= 59
        except (ValueError, IndexError):
            return False
    
    def _time_to_seconds(self, time_str: str) -> Optional[int]:
        """Convert time string to total seconds for comparison."""
        if not self._is_valid_time_format(time_str):
            return None
        
        try:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        except (ValueError, IndexError):
            return None
    
    def _generate_validation_report(self, file_path: str, total_lines: int, 
                                  processed_examples: int, valid_examples: int) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        total_errors = sum(len(errors) for errors in self.errors.values())
        total_warnings = sum(len(warnings) for warnings in self.warnings.values())
        
        report = {
            'file_info': {
                'path': file_path,
                'total_lines': total_lines,
                'processed_examples': processed_examples,
                'valid_examples': valid_examples,
                'invalid_examples': processed_examples - valid_examples,
                'success_rate': valid_examples / processed_examples if processed_examples > 0 else 0
            },
            'validation_summary': {
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'error_categories': len(self.errors),
                'warning_categories': len(self.warnings)
            },
            'business_logic_stats': {
                'non_scoring_plays': self.stats.get('non_scoring_plays', 0),
                'potentially_scoring_plays': self.stats.get('potentially_scoring_plays', 0),
                'avg_recent_plays': self.stats.get('recent_plays_count', 0) / processed_examples if processed_examples > 0 else 0
            },
            'errors': dict(self.errors),
            'warnings': dict(self.warnings)
        }
        
        return report
    
    def print_validation_report(self, report: Dict[str, Any]):
        """Print a formatted validation report."""
        print("\n" + "=" * 80)
        print("📊 VALIDATION REPORT")
        print("=" * 80)
        
        # File info
        file_info = report['file_info']
        print(f"📁 File: {file_info['path']}")
        print(f"📋 Total lines: {file_info['total_lines']:,}")
        print(f"✅ Valid examples: {file_info['valid_examples']:,}")
        print(f"❌ Invalid examples: {file_info['invalid_examples']:,}")
        print(f"📈 Success rate: {file_info['success_rate']:.1%}")
        
        # Summary
        summary = report['validation_summary']
        print(f"\n🔍 Validation Summary:")
        print(f"   • Total errors: {summary['total_errors']:,}")
        print(f"   • Total warnings: {summary['total_warnings']:,}")
        
        # Business logic stats
        stats = report['business_logic_stats']
        print(f"\n🏀 NBA Business Logic:")
        print(f"   • Non-scoring plays: {stats['non_scoring_plays']:,}")
        print(f"   • Potentially scoring plays: {stats['potentially_scoring_plays']:,}")
        print(f"   • Avg recent plays per example: {stats['avg_recent_plays']:.1f}")
        
        # Errors by category
        if report['errors']:
            print(f"\n❌ ERRORS BY CATEGORY:")
            for category, errors in report['errors'].items():
                print(f"\n   {category.upper()} ({len(errors)} errors):")
                for error in errors[:5]:  # Show first 5 errors per category
                    print(f"      • {error}")
                if len(errors) > 5:
                    print(f"      ... and {len(errors) - 5} more")
        
        # Warnings by category
        if report['warnings'] and self.verbose:
            print(f"\n⚠️ WARNINGS BY CATEGORY:")
            for category, warnings in report['warnings'].items():
                print(f"\n   {category.upper()} ({len(warnings)} warnings):")
                for warning in warnings[:3]:  # Show first 3 warnings per category
                    print(f"      • {warning}")
                if len(warnings) > 3:
                    print(f"      ... and {len(warnings) - 3} more")
        
        # Final verdict
        print("\n" + "=" * 80)
        if summary['total_errors'] == 0:
            print("🎉 VALIDATION PASSED!")
            print("✅ All examples are structurally valid and follow NBA business logic")
            if summary['total_warnings'] > 0:
                print(f"⚠️  Found {summary['total_warnings']} warnings (use --verbose to see details)")
        else:
            print("❌ VALIDATION FAILED!")
            print(f"🔧 Fix {summary['total_errors']} errors before using this data")
        print("=" * 80)


def main():
    """Main function to run validation from command line."""
    parser = argparse.ArgumentParser(
        description="Validate NBA training JSONL data for OpenAI fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate entire file
    python validate_training_data.py data/training/output.jsonl
    
    # Validate first 1000 examples with verbose output
    python validate_training_data.py data/training/output.jsonl --sample 1000 --verbose
    
    # Quick validation of first 100 examples
    python validate_training_data.py data/training/output.jsonl --sample 100
        """
    )
    
    parser.add_argument(
        'jsonl_file',
        type=str,
        help='Path to JSONL file to validate'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        help='Only validate first N examples (for quick testing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed warnings and additional information'
    )
    
    try:
        args = parser.parse_args()
    except SystemExit:
        return
    
    try:
        # Create validator and run validation
        validator = NBATrainingDataValidator(verbose=args.verbose)
        report = validator.validate_jsonl_file(args.jsonl_file, args.sample)
        
        if 'error' in report:
            print(f"❌ Error: {report['error']}", file=sys.stderr)
            sys.exit(1)
        
        # Print report
        validator.print_validation_report(report)
        
        # Exit with appropriate code
        if report['validation_summary']['total_errors'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
