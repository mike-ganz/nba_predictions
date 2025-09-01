#!/usr/bin/env python3
"""
NBA Play Prediction Script

Simple script to test predictions using a fine-tuned OpenAI model.
Takes team and player data and predicts the next play using the trained model.
"""

import json
from dotenv import load_dotenv
import os
from typing import Dict, Any, Tuple
from openai import OpenAI

# Fine-tuned model IDs
MODEL_1_ID = "ft:gpt-4.1-nano-2025-04-14:personal:first-n-plays:CAieHHyW"  # First model
MODEL_2_ID = "ft:gpt-4.1-nano-2025-04-14:personal:part-1:CAoc9Uw6"  # Second model

def init_openai_client() -> OpenAI:
    """Initialize OpenAI client with API key."""
    load_dotenv()  # this loads the .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not found. "
            "Please set your OpenAI API key as an environment variable."
        )
    
    return OpenAI(api_key=api_key)

def predict_rolling_sequence(game_context: Dict[str, Any], n_iterations: int = 5) -> Dict[str, Any]:
    """
    Rolling prediction pipeline:
    1. Send game_context to MODEL_1 to get next_plays (initial sequence)
    2. Add next_plays to game_context as recent_plays
    3. Send to MODEL_2 to get next_play
    4. Add next_play to END of recent_plays, remove FIRST play (sliding window)
    5. Repeat step 3-4 N times
    
    Args:
        game_context: Dictionary containing team and player data
        n_iterations: Number of times to repeat the rolling prediction
        
    Returns:
        Dict containing all stages and iterations
    """
    print("🤖 Initializing OpenAI client...")
    client = init_openai_client()
    
    # Results storage
    results = {
        "stage1_response": None,
        "stage2_responses": [],
        "iterations": []
    }
    
    # === STAGE 1: Get initial predictions ===
    print("\n🎯 STAGE 1: Getting initial next_plays from first model...")
    context_json = json.dumps(game_context, separators=(',', ':'))
    print(f"📡 Sending to model: {MODEL_1_ID}")
    print(f"📊 Context size: {len(context_json)} characters")
    
    try:
        # Stage 1 API call
        stage1_response = client.chat.completions.create(
            model=MODEL_1_ID,
            messages=[
                {
                    "role": "user",
                    "content": context_json
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        stage1_content = stage1_response.choices[0].message.content
        print(f"📊 Stage 1 tokens: {stage1_response.usage.completion_tokens} / 1500")
        print("✅ Stage 1 completed!")
        
        results["stage1_response"] = stage1_content
        
        # Parse Stage 1 response
        try:
            stage1_json = json.loads(stage1_content)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Stage 1 response as JSON: {e}")
            return {"error": f"Stage 1 JSON parse error: {e}"}
        
        # Initialize working context with recent_plays
        working_context = game_context.copy()
        if "next_plays" in stage1_json:
            working_context["recent_plays"] = stage1_json["next_plays"]
            print(f"✅ Added {len(stage1_json['next_plays'])} recent_plays to context")
        else:
            print("⚠️ Warning: No 'next_plays' found in Stage 1 response")
            working_context["recent_plays"] = []
        
        # === ROLLING ITERATIONS ===
        print(f"\n🔄 Starting {n_iterations} rolling iterations...")
        
        for iteration in range(n_iterations):
            print(f"\n--- ITERATION {iteration + 1}/{n_iterations} ---")
            
            # Convert current context to JSON
            iteration_json = json.dumps(working_context, separators=(',', ':'))
            print(f"📡 Sending to model: {MODEL_2_ID}")
            print(f"📊 Context size: {len(iteration_json)} characters")
            print(f"📋 Current recent_plays count: {len(working_context.get('recent_plays', []))}")
            
            # Stage 2 API call
            stage2_response = client.chat.completions.create(
                model=MODEL_2_ID,
                messages=[
                    {
                        "role": "user",
                        "content": iteration_json
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            stage2_content = stage2_response.choices[0].message.content
            print(f"📊 Stage 2 tokens: {stage2_response.usage.completion_tokens} / 1500")
            
            # Store this iteration's response
            results["stage2_responses"].append(stage2_content)
            
            # Parse the response to get the next_play
            try:
                stage2_json = json.loads(stage2_content)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse iteration {iteration + 1} response as JSON: {e}")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "error": f"JSON parse error: {e}",
                    "raw_response": stage2_content
                })
                continue
            
            # Extract the next_play
            if "next_play" in stage2_json:
                next_play = stage2_json["next_play"]
                print(f"✅ Got next_play: {next_play.get('description', 'No description')}")
                
                # Update the sliding window: add to end, remove from beginning
                if "recent_plays" in working_context:
                    # Add new play to the END
                    working_context["recent_plays"].append(next_play)
                    # Remove first play (maintain window size)
                    if len(working_context["recent_plays"]) > 10:
                        removed_play = working_context["recent_plays"].pop(0)
                        print(f"🔄 Sliding window: Added new play, removed: {removed_play.get('description', 'No description')}")
                    else:
                        print(f"📈 Window growing: Now {len(working_context['recent_plays'])} plays")
                
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "next_play": next_play,
                    "raw_response": stage2_content,
                    "recent_plays_count": len(working_context.get("recent_plays", []))
                })
                
            else:
                print(f"⚠️ Warning: No 'next_play' found in iteration {iteration + 1} response")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "error": "No 'next_play' in response",
                    "raw_response": stage2_content
                })
        
        print(f"\n✅ Rolling sequence completed! {n_iterations} iterations done.")
        return results
        
    except Exception as e:
        print(f"❌ Error in rolling sequence: {e}")
        raise

def main():
    """Main function to test the prediction system."""
    
    print("🏀 NBA Play Prediction Test")
    print("=" * 50)
    
    # Example team and player data (from user's input)
    game_context = {"away_team":{"name":"ORL","stats":{"OEFF":113.7,"DEFF":112.7,"PACE":96.5,"REST_DAYS":2},"players":[{"name":"Franz Wagner","profile":{"offense":2.34,"defense":1.42,"shot_selection":0.69,"efficiency":-0.67,"MPG":33,"usage":26}},{"name":"Paolo Banchero","profile":{"offense":4.14,"defense":1.02,"shot_selection":1.28,"efficiency":-1.04,"MPG":35,"usage":30}},{"name":"Jonathan Isaac","profile":{"offense":-0.71,"defense":1.02,"shot_selection":0.3,"efficiency":-0.5,"MPG":16,"usage":17}},{"name":"Gary Harris","profile":{"offense":-0.41,"defense":0.52,"shot_selection":-1.29,"efficiency":-0.33,"MPG":24,"usage":12}},{"name":"Jalen Suggs","profile":{"offense":1.44,"defense":2.5,"shot_selection":-0.34,"efficiency":-1.05,"MPG":27,"usage":20}},{"name":"Wendell Carter Jr.","profile":{"offense":0.42,"defense":0.76,"shot_selection":0.46,"efficiency":-0.96,"MPG":25,"usage":18}},{"name":"Joe Ingles","profile":{"offense":0.18,"defense":-0.52,"shot_selection":-1.26,"efficiency":-0.82,"MPG":17,"usage":11}},{"name":"Markelle Fultz","profile":{"offense":-0.09,"defense":0.42,"shot_selection":0.52,"efficiency":0.67,"MPG":21,"usage":19}},{"name":"Cole Anthony","profile":{"offense":1.16,"defense":0.87,"shot_selection":0.48,"efficiency":-0.2,"MPG":22,"usage":24}},{"name":"Moritz Wagner","profile":{"offense":0.68,"defense":0.16,"shot_selection":1.33,"efficiency":-1.71,"MPG":18,"usage":23}},{"name":"Caleb Houstan","profile":{"offense":-1.09,"defense":-1.29,"shot_selection":-2.17,"efficiency":-0.26,"MPG":14,"usage":12}}]},"home_team":{"name":"CLE","stats":{"OEFF":114.9,"DEFF":112.5,"PACE":97.2,"REST_DAYS":2},"players":[{"name":"Max Strus","profile":{"offense":1.48,"defense":1.24,"shot_selection":-1.23,"efficiency":-0.44,"MPG":32,"usage":17}},{"name":"Evan Mobley","profile":{"offense":1.49,"defense":2.77,"shot_selection":1.39,"efficiency":-1.13,"MPG":31,"usage":21}},{"name":"Jarrett Allen","profile":{"offense":1.82,"defense":1.41,"shot_selection":2.04,"efficiency":-1.77,"MPG":32,"usage":20}},{"name":"Donovan Mitchell","profile":{"offense":3.79,"defense":2.55,"shot_selection":0.2,"efficiency":-1.04,"MPG":35,"usage":31}},{"name":"Darius Garland","profile":{"offense":3.38,"defense":0.87,"shot_selection":-0.12,"efficiency":-0.88,"MPG":33,"usage":25}},{"name":"Caris LeVert","profile":{"offense":1.86,"defense":1.08,"shot_selection":0.15,"efficiency":-0.01,"MPG":29,"usage":23}},{"name":"Georges Niang","profile":{"offense":0.07,"defense":0.16,"shot_selection":-1.34,"efficiency":-0.52,"MPG":22,"usage":18}},{"name":"Isaac Okoro","profile":{"offense":0.17,"defense":0.95,"shot_selection":0.19,"efficiency":-0.68,"MPG":27,"usage":14}}]}}

    print("📋 Game Context Summary:")
    print(f"   Away Team: {game_context['away_team']['name']} ({len(game_context['away_team']['players'])} players)")
    print(f"   Home Team: {game_context['home_team']['name']} ({len(game_context['home_team']['players'])} players)")
    print()
    
    try:
        # Configure rolling sequence parameters
        n_iterations = 50  # Change this to control how many rolling predictions
        
        print(f"\n🚀 Starting rolling prediction sequence (N={n_iterations})")
        
        # Run the rolling sequence
        results = predict_rolling_sequence(game_context, n_iterations=n_iterations)
        
        print("\n" + "=" * 80)
        print("🎯 ROLLING SEQUENCE RESULTS")
        print("=" * 80)
        
        # Show Stage 1 results
        if results.get("stage1_response"):
            print("\n📋 STAGE 1 (Initial next_plays):")
            try:
                stage1_json = json.loads(results["stage1_response"])
                print(f"Generated {len(stage1_json.get('next_plays', []))} initial plays")
                # Show first few plays
                for i, play in enumerate(stage1_json.get('next_plays', [])[:3]):
                    print(f"  {i+1}. {play.get('time_remaining', 'N/A')} - {play.get('description', 'No description')}")
                if len(stage1_json.get('next_plays', [])) > 3:
                    print(f"  ... and {len(stage1_json.get('next_plays', [])) - 3} more plays")
            except json.JSONDecodeError:
                print("❌ Could not parse Stage 1 response")
        
        # Show rolling iterations
        print(f"\n🔄 ROLLING ITERATIONS ({len(results.get('iterations', []))} completed):")
        for iteration_data in results.get("iterations", []):
            iteration_num = iteration_data.get("iteration", "?")
            print(f"\n--- Iteration {iteration_num} ---")
            
            if "error" in iteration_data:
                print(f"❌ Error: {iteration_data['error']}")
            elif "next_play" in iteration_data:
                next_play = iteration_data["next_play"]
                time_remaining = next_play.get("time_remaining", "N/A")
                description = next_play.get("description", "No description")
                score = next_play.get("score", "N/A")
                recent_plays_count = iteration_data.get("recent_plays_count", "?")
                
                print(f"⏰ Time: {time_remaining}")
                print(f"🏀 Play: {description}")
                print(f"📊 Score: {score}")
                print(f"📋 Recent plays window: {recent_plays_count} plays")
        
        print("\n" + "=" * 80)
        print("✅ ROLLING SEQUENCE COMPLETE!")
        print("=" * 80)
        
        # Summary
        total_predictions = len(results.get("iterations", []))
        successful_predictions = len([i for i in results.get("iterations", []) if "next_play" in i])
        print(f"📊 Summary: {successful_predictions}/{total_predictions} successful predictions")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure OPENAI_API_KEY environment variable is set")
        print("   • Verify the fine-tuned model ID is correct")
        print("   • Check your OpenAI account has access to the model")

if __name__ == "__main__":
    main()
