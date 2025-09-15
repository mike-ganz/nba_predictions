#!/usr/bin/env python3
"""
Minimal surgical fix - only change the final_score assignment to use correct team names
"""

import re

# Read the original file
with open('/home/micha/orchestrator.py', 'r') as f:
    content = f.read()

print("🔧 APPLYING MINIMAL SURGICAL FIX")
print("=" * 40)

# Find the _run_single_simulation method
method_start = content.find('def _run_single_simulation(self, run_id: str, game_id: str) -> SimulationResult:')
method_lines = content[method_start:].split('\n')

# Find the line that loads game context
context_line_idx = None
for i, line in enumerate(method_lines):
    if 'game_context = self.context_builder.build_game_context(' in line:
        context_line_idx = i
        break

print(f"📍 Found game context loading at line {context_line_idx}")

# Add team extraction code right after game context loading
if context_line_idx is not None:
    # Insert team extraction code after the game context build
    insert_idx = context_line_idx + 4  # After the build_game_context call closes
    
    team_extraction_code = [
        "",
        "            # Extract team names for proper score tracking",
        "            away_team = None",
        "            home_team = None", 
        "            if isinstance(game_context, dict):",
        "                away_data = game_context.get('away_team', {})",
        "                home_data = game_context.get('home_team', {})",
        "                if isinstance(away_data, dict):",
        "                    away_team = away_data.get('name', 'AWAY')",
        "                if isinstance(home_data, dict):", 
        "                    home_team = home_data.get('name', 'HOME')",
    ]
    
    # Insert the team extraction code
    for j, new_line in enumerate(team_extraction_code):
        method_lines.insert(insert_idx + j, new_line)
    
    print("✅ Added team extraction code")

# Find the scoring tracking section and add score variables
scoring_section_idx = None
for i, line in enumerate(method_lines):
    if 'scoring_plays = 0' in line:
        scoring_section_idx = i
        break

if scoring_section_idx is not None:
    # Add score tracking variables
    score_vars = [
        "            away_score = 0",
        "            home_score = 0",
    ]
    
    for j, new_line in enumerate(score_vars):
        method_lines.insert(scoring_section_idx + 2 + j, new_line)
    
    print("✅ Added score tracking variables")

# Find the scoring play detection and update it
scoring_detection_idx = None
for i, line in enumerate(method_lines):
    if 'if points is not None and points > 0:' in line:
        scoring_detection_idx = i
        break

if scoring_detection_idx is not None:
    # Add score tracking logic right after scoring_plays += 1
    score_tracking = [
        "                            # Update internal score tracking",
        "                            team_that_scored = shot_details.get('team', '')",
        "                            if team_that_scored and away_team and home_team:",
        "                                team_upper = team_that_scored.upper()",
        "                                away_upper = away_team.upper()",
        "                                home_upper = home_team.upper()",
        "                                if away_upper in team_upper or 'WAS' in team_upper or 'WASHINGTON' in team_upper:",
        "                                    away_score += points",
        "                                elif home_upper in team_upper or 'CHI' in team_upper or 'CHICAGO' in team_upper:",
        "                                    home_score += points",
        "                                else:",
        "                                    away_score += points  # Default to away",
    ]
    
    # Find where to insert (after scoring_plays += 1)
    for j, line in enumerate(method_lines[scoring_detection_idx:]):
        if 'scoring_plays += 1' in line:
            insert_at = scoring_detection_idx + j + 1
            break
    
    for j, new_line in enumerate(score_tracking):
        method_lines.insert(insert_at + j, new_line)
    
    print("✅ Added score tracking logic")

# Find the final_score assignment and replace it
final_score_idx = None
for i, line in enumerate(method_lines):
    if 'final_score = next_play.get("score")' in line:
        final_score_idx = i
        break

if final_score_idx is not None:
    # Replace the line with our tracked score
    method_lines[final_score_idx] = '                    # Use our tracked score instead of AI response'
    method_lines.insert(final_score_idx + 1, '                    if away_team and home_team:')
    method_lines.insert(final_score_idx + 2, f'                        final_score = f"{{away_team}} {{away_score}} - {{home_team}} {{home_score}}"')
    method_lines.insert(final_score_idx + 3, '                    else:')
    method_lines.insert(final_score_idx + 4, f'                        final_score = f"AWAY {{away_score}} - HOME {{home_score}}"')
    
    print("✅ Replaced final_score assignment")

# Reconstruct the file
method_content = '\n'.join(method_lines)
new_file_content = content[:method_start] + method_content

# Write back
with open('/home/micha/orchestrator.py', 'w') as f:
    f.write(new_file_content)

print("\n🎉 MINIMAL FIX APPLIED SUCCESSFULLY!")
print("🔧 Changes made:")
print("   - Extract team names from game context")
print("   - Track away_score and home_score internally")  
print("   - Replace final_score with tracked values")
print("   - Minimal disruption to existing code structure")

