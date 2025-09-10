#!/usr/bin/env python3
"""Test GameContextBuilder directly for game 22400530"""

import sys
sys.path.append(".")

try:
    from game_context_builder import GameContextBuilder
    
    print("🏀 TESTING GAME CONTEXT BUILDER")
    print("=" * 50)
    
    # Create the builder for 2024-2025
    builder = GameContextBuilder("2024-2025")
    
    print(f"📊 Data system available: {hasattr(builder, 'play_by_play_data') and builder.play_by_play_data is not None}")
    
    if hasattr(builder, 'play_by_play_data') and builder.play_by_play_data is not None:
        print(f"📋 Total play-by-play records: {len(builder.play_by_play_data)}")
        
        # Check available games
        available_games = builder.get_available_games()
        print(f"🎯 Total available games: {len(available_games) if available_games else 0}")
        
        # Check if our target game is available
        target_game = "22400530"
        if available_games and target_game in available_games:
            print(f"✅ Target game {target_game} is available!")
            
            # Try to build context for this game
            try:
                context = builder.build_game_context(target_game)
                if context:
                    print(f"✅ Successfully built context for game {target_game}")
                    print(f"📝 Context preview: {str(context)[:200]}...")
                    
                    # Check if context contains Washington/Chicago
                    context_str = str(context).lower()
                    has_washington = "washington" in context_str or "was" in context_str
                    has_chicago = "chicago" in context_str or "chi" in context_str
                    
                    print(f"🏀 Contains Washington data: {has_washington}")
                    print(f"🏀 Contains Chicago data: {has_chicago}")
                    
                else:
                    print(f"❌ Failed to build context for game {target_game}")
                    
            except Exception as e:
                print(f"❌ Error building context: {e}")
                
        else:
            print(f"❌ Target game {target_game} not in available games")
            if available_games:
                print(f"📋 First 10 available games:")
                for game in sorted(available_games)[:10]:
                    print(f"   {game}")
    else:
        print("❌ No play-by-play data loaded")
        
except Exception as e:
    print(f"❌ GameContextBuilder test failed: {e}")
    import traceback
    traceback.print_exc()
