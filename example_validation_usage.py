#!/usr/bin/env python3
"""
Example usage of NBA training data validation.

This shows how to use the validation tool programmatically in your code.
"""

from validate_training_data import NBATrainingDataValidator
import json


def example_validation_usage():
    """Example of using the validator programmatically."""
    
    # Create validator instance
    validator = NBATrainingDataValidator(verbose=True)
    
    # Example 1: Validate a sample file
    print("=" * 60)
    print("EXAMPLE 1: Validating JSONL file")
    print("=" * 60)
    
    sample_file = "data/training/ULTRA_OPTIMIZED_sample_1000_part_001.jsonl"
    
    # Run validation on first 20 examples
    report = validator.validate_jsonl_file(sample_file, sample_size=20)
    
    if 'error' in report:
        print(f"❌ Error: {report['error']}")
        return
    
    # Print summary
    file_info = report['file_info']
    print(f"✅ Successfully validated {file_info['processed_examples']:,} examples")
    print(f"📈 Success rate: {file_info['success_rate']:.1%}")
    
    summary = report['validation_summary']
    print(f"❌ Found {summary['total_errors']} errors")
    print(f"⚠️  Found {summary['total_warnings']} warnings")
    
    # Show specific business logic results
    stats = report['business_logic_stats']
    print(f"🏀 Non-scoring plays detected: {stats['non_scoring_plays']}")
    print(f"🏀 Potentially scoring plays: {stats['potentially_scoring_plays']}")
    
    # Example 2: Show specific error types found
    print(f"\n" + "=" * 60)
    print("EXAMPLE 2: Common validation issues found")
    print("=" * 60)
    
    if report['errors']:
        for error_type, errors in report['errors'].items():
            print(f"\n{error_type.upper()} ({len(errors)} errors):")
            for error in errors[:3]:  # Show first 3 examples
                print(f"  • {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more")
    
    # Example 3: Create validation summary for monitoring
    print(f"\n" + "=" * 60)
    print("EXAMPLE 3: Monitoring summary")
    print("=" * 60)
    
    monitoring_summary = {
        'file_path': sample_file,
        'validation_timestamp': '2024-01-01T12:00:00Z',  # Would be actual timestamp
        'total_examples': file_info['processed_examples'],
        'success_rate': file_info['success_rate'],
        'critical_errors': summary['total_errors'],
        'warnings': summary['total_warnings'],
        'business_logic_passed': {
            'scoring_consistency': 'scoring_consistency' not in report['errors'],
            'time_consistency': 'time_consistency' not in report['errors']
        }
    }
    
    print("Monitoring Summary JSON:")
    print(json.dumps(monitoring_summary, indent=2))
    
    print(f"\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS:")
    print("=" * 60)
    
    if summary['total_errors'] > 0:
        print("❌ CRITICAL: Fix scoring consistency errors before training")
        print("   - Non-scoring plays should have scoring_team=null and points_scored=0")
        print("   - Check the scoring logic in your data generation pipeline")
    
    if summary['total_warnings'] > 0:
        print("⚠️  REVIEW: Time consistency warnings found")
        print("   - Multiple plays with identical timestamps may indicate data quality issues")
        print("   - Consider investigating the source data or play-by-play processing")
    
    print(f"\n✅ Validation complete! Ready for production use.")


def example_business_rules():
    """Example showing the business rules being validated."""
    
    print("=" * 60)
    print("NBA BUSINESS RULES VALIDATED")
    print("=" * 60)
    
    rules = [
        {
            "rule": "Time Consistency",
            "description": "next_play time_remaining should be later (smaller) than recent_plays last time within same quarter",
            "example_violation": "Recent play: 10:30, Next play: 10:45 (in same quarter)",
            "validation": "✅ Implemented"
        },
        {
            "rule": "Scoring Consistency", 
            "description": "Non-scoring plays should have scoring_team=null and points_scored=0",
            "keywords": "REBOUND, SUB, MISS, TURNOVER, FOUL, TIMEOUT, VIOLATION",
            "example_violation": "Play 'MISS Johnson 3PT' has scoring_team='LAL' and points_scored=3",
            "validation": "✅ Implemented"
        },
        {
            "rule": "Data Structure",
            "description": "Proper OpenAI fine-tuning format with required fields",
            "requirements": "messages[user/assistant], JSON content, required NBA fields",
            "validation": "✅ Implemented"
        },
        {
            "rule": "NBA Data Validation",
            "description": "Valid team abbreviations, quarters, time formats, point values",
            "checks": "Team names, quarter range (1-10), time format MM:SS, points (0-4)",
            "validation": "✅ Implemented"
        }
    ]
    
    for i, rule in enumerate(rules, 1):
        print(f"\n{i}. {rule['rule']}")
        print(f"   Description: {rule['description']}")
        if 'keywords' in rule:
            print(f"   Keywords: {rule['keywords']}")
        if 'example_violation' in rule:
            print(f"   Example violation: {rule['example_violation']}")
        if 'requirements' in rule:
            print(f"   Requirements: {rule['requirements']}")
        if 'checks' in rule:
            print(f"   Checks: {rule['checks']}")
        print(f"   Status: {rule['validation']}")


if __name__ == "__main__":
    print("🏀 NBA Training Data Validation Examples")
    print("=" * 60)
    
    # Show business rules
    example_business_rules()
    
    print(f"\n")
    
    # Run validation example  
    example_validation_usage()
