#!/usr/bin/env python3
"""
Systematic test to understand what's actually happening with period detection.
This will simulate the exact scenario from the user's logs and trace execution.
"""

import sys
from game.response_validator import NBAResponseValidator, ValidationResult

def test_real_validation_scenario():
    """
    Test the exact scenario from user logs to see what data format reaches the validator
    and whether our period detection logic actually triggers.
    """
    
    print("🔍 SYSTEMATIC PERIOD VALIDATION DEBUG")
    print("=" * 60)
    
    # Create validator in the same mode the orchestrator likely uses
    validator = NBAResponseValidator('fast')
    
    print("📋 Testing Scenario: User's logs from iteration 102")
    print("   Current play: Q1 [00:00] unknown")
    print("   Recent plays contain: Q1 [00:00] period")
    print()
    
    # Test different possible data formats the validator might receive
    test_formats = [
        {
            "name": "Format 1: Verbose with full fields",
            "next_play": {
                "quarter": 1,
                "time_remaining": "00:00",
                "description": "unknown",
                "score": "DEN 31 - NYK 39"
            },
            "recent_plays": [
                {"quarter": 1, "time_remaining": "00:02", "description": "Braun personal foul"},
                {"quarter": 1, "time_remaining": "00:02", "description": "Achiuwa made free throw"}, 
                {"quarter": 1, "time_remaining": "00:00", "description": "Westbrook turnover"},
                {"quarter": 1, "time_remaining": "00:00", "description": "period"},
                {"quarter": 1, "time_remaining": "00:00", "description": "unknown"}
            ]
        },
        {
            "name": "Format 2: Compact with event_code",
            "next_play": {
                "quarter": 1,
                "time_remaining": "00:00",
                "description": "unknown",
                "event_code": "unknown"
            },
            "recent_plays": [
                {"quarter": 1, "time_remaining": "00:02", "description": "Braun personal foul", "event_code": "p_foul"},
                {"quarter": 1, "time_remaining": "00:02", "description": "Achiuwa made free throw", "event_code": "ftm"}, 
                {"quarter": 1, "time_remaining": "00:00", "description": "Westbrook turnover", "event_code": "tov"},
                {"quarter": 1, "time_remaining": "00:00", "description": "period", "event_code": "period"},  # KEY!
                {"quarter": 1, "time_remaining": "00:00", "description": "unknown", "event_code": "unknown"}
            ]
        },
        {
            "name": "Format 3: Minimal fields only",
            "next_play": {
                "time_remaining": "00:00",
                "description": "unknown"
            },
            "recent_plays": [
                {"time_remaining": "00:00", "description": "period"},  # Minimal but should work
                {"time_remaining": "00:00", "description": "unknown"}
            ]
        }
    ]
    
    for i, test_format in enumerate(test_formats, 1):
        print(f"\n🧪 TEST {i}: {test_format['name']}")
        print("-" * 40)
        
        # Reset validator state for each test
        validator_test = NBAResponseValidator('fast')
        validator_test.last_time_remaining = "00:00"  # Simulate we've seen 00:00 before
        validator_test.consecutive_same_time = 1  # First repeat
        
        try:
            print(f"📥 Input next_play: {test_format['next_play']}")
            print(f"📥 Input recent_plays: {len(test_format['recent_plays'])} plays")
            
            # Check for period in recent plays manually first
            period_plays = [p for p in test_format['recent_plays'] if 'period' in str(p.get('description', '')).lower()]
            print(f"🔍 Period plays found manually: {len(period_plays)}")
            if period_plays:
                print(f"   Period play: {period_plays[0]}")
            
            # Now test the actual validation logic
            result = validator_test._check_consecutive_same_time(
                test_format['next_play'], 
                test_format['recent_plays']
            )
            
            print(f"✅ Validation result: {result}")
            
            if result == ValidationResult.QUARTER_TRANSITION:
                print("🚀 SUCCESS: Period detection worked!")
                target = validator_test.get_quarter_transition_target()
                print(f"   Target quarter: Q{target}")
            elif result == ValidationResult.VALID:
                print("⚪ NEUTRAL: Validation passed but no transition")
            else:
                print(f"❓ OTHER: Got {result}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print("🎯 DIAGNOSIS:")
    print("   - Which format(s) triggered period detection?")
    print("   - What data format does the real validator actually receive?") 
    print("   - Is the validation code path even being hit?")
    print("=" * 60)

if __name__ == "__main__":
    test_real_validation_scenario()
