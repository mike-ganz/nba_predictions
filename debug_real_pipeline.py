#!/usr/bin/env python3
"""
Debug the actual prediction pipeline to see why period detection isn't working
in the real orchestrator run.
"""

import os
from game.response_validator import NBAResponseValidator, ValidationResult
from game.prediction_client import BasePredictionClient
from predict_next_play import predict_rolling_sequence, OptimizedGameContext, CompactGameContext

def debug_validation_call_path():
    """
    Investigate if the validation is even being called and with what data format.
    """
    
    print("🔍 DEBUGGING ACTUAL VALIDATION CALL PATH")
    print("=" * 60)
    
    # Let's monkey patch the validation method to see what data it actually receives
    original_check_consecutive_same_time = NBAResponseValidator._check_consecutive_same_time
    
    def debug_check_consecutive_same_time(self, next_play, current_recent_plays):
        print(f"\n🚨 VALIDATION DEBUG - _check_consecutive_same_time CALLED!")
        print(f"📥 next_play keys: {list(next_play.keys()) if isinstance(next_play, dict) else type(next_play)}")
        print(f"📥 next_play: {next_play}")
        print(f"📥 recent_plays count: {len(current_recent_plays)}")
        
        # Check for period plays manually
        period_in_current = 'period' in str(next_play.get('description', '')).lower()
        period_in_recent = any('period' in str(p.get('description', '')).lower() for p in current_recent_plays)
        
        print(f"🔍 Period in current play: {period_in_current}")
        print(f"🔍 Period in recent plays: {period_in_recent}")
        
        if period_in_recent:
            period_plays = [p for p in current_recent_plays if 'period' in str(p.get('description', '')).lower()]
            print(f"🎯 Found {len(period_plays)} period plays in recent_plays:")
            for i, p in enumerate(period_plays):
                print(f"   {i+1}. {p}")
        
        # Call original method and see result
        result = original_check_consecutive_same_time(self, next_play, current_recent_plays)
        print(f"✅ Original method returned: {result}")
        
        return result
    
    # Apply monkey patch
    NBAResponseValidator._check_consecutive_same_time = debug_check_consecutive_same_time
    
    print("🔧 Applied debug monkey patch to _check_consecutive_same_time")
    print("🎯 Now when you run the orchestrator, you'll see exactly what data the validator receives!")
    
    return "Debug patch applied. Run your orchestrator command now to see the validation data."

def debug_play_format_conversion():
    """
    Check how plays get converted from compact format to the format that reaches the validator.
    """
    
    print("\n🔍 DEBUGGING PLAY FORMAT CONVERSION")
    print("=" * 50)
    
    # Sample compact play data (what the model returns)
    sample_compact_play = [1, 0, [31, 39], ["A", -1], "period", 0]
    
    print(f"📦 Sample compact play (model output): {sample_compact_play}")
    
    # Let's see how this gets converted for validation
    try:
        from predict_next_play import format_play_description
        
        # Mock game context for format_play_description
        mock_context = {
            "A": "DEN", 
            "H": "NYK",
            "ap": [["Player1", 0, 0, 0, 0, 0, 0]],  # Mock away players
            "hp": [["Player2", 0, 0, 0, 0, 0, 0]]   # Mock home players
        }
        
        # This is how compact plays get formatted for display
        formatted = format_play_description(sample_compact_play, mock_context)
        print(f"📝 Formatted description: '{formatted}'")
        
        # Check if "period" survives the formatting
        if 'period' in formatted.lower():
            print("✅ 'period' preserved in formatted description")
        else:
            print("❌ 'period' LOST during formatting!")
            print("   This could be why period detection fails!")
        
    except Exception as e:
        print(f"❌ Error testing format conversion: {e}")
        import traceback
        traceback.print_exc()

def create_validation_test_script():
    """
    Create a script that can be run during actual orchestrator execution to debug validation.
    """
    
    script_content = '''
import sys
import os
sys.path.append(os.getcwd())

from game.response_validator import NBAResponseValidator

# Save the original method
original_validate = NBAResponseValidator.validate_response

def debug_validate_response(self, response_text, context=None):
    print(f"\\n🚨 VALIDATOR.VALIDATE_RESPONSE CALLED!")
    print(f"📥 Response text length: {len(response_text) if response_text else 0}")
    print(f"📥 Response preview: {response_text[:200] if response_text else 'None'}...")
    print(f"📥 Context keys: {list(context.keys()) if context else 'None'}")
    
    if context and 'recent_plays' in context:
        recent_plays = context['recent_plays']
        print(f"📥 Recent plays count: {len(recent_plays)}")
        period_plays = [p for p in recent_plays if 'period' in str(p.get('description', '')).lower()]
        if period_plays:
            print(f"🎯 PERIOD PLAYS FOUND IN CONTEXT: {len(period_plays)}")
            for p in period_plays:
                print(f"   {p}")
    
    # Call original method
    result = original_validate(self, response_text, context)
    print(f"✅ Validation result: {result}")
    
    return result

# Apply patch
NBAResponseValidator.validate_response = debug_validate_response
print("🔧 Validation debug patch applied!")
'''
    
    with open('apply_validation_debug.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n📝 Created apply_validation_debug.py")
    print("💡 Usage: Import this at the start of your orchestrator run to debug validation")

if __name__ == "__main__":
    debug_validation_call_path()
    debug_play_format_conversion() 
    create_validation_test_script()
    
    print(f"\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("1. The monkey patch is applied - run your orchestrator now")
    print("2. Watch for '🚨 VALIDATION DEBUG' messages in the output")
    print("3. See exactly what data format the validator receives")
    print("4. Check if period plays are preserved or lost in conversion")
    print("=" * 60)
