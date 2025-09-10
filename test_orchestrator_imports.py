#!/usr/bin/env python3
"""Test orchestrator imports"""

print("🧪 Testing orchestrator imports...")

try:
    from config.settings import config, set_season_year
    print("✅ config.settings imports successful")
except Exception as e:
    print(f"❌ config.settings import failed: {e}")

try:
    from game_context_builder import GameContextBuilder
    print("✅ GameContextBuilder import successful")
except Exception as e:
    print(f"❌ GameContextBuilder import failed: {e}")

print("🎯 All critical imports tested!")
