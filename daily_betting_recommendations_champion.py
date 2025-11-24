#!/usr/bin/env python
"""Daily NBA Betting Recommendations Pipeline - Rest-Aware Champion v2

═══════════════════════════════════════════════════════════════════════════════
MODEL: Rest-Aware Champion v2 (XGBoost, Unified Injury Handling, No Redundancy)
Artifact: artifacts/champion_rest_schedule
═══════════════════════════════════════════════════════════════════════════════

PERFORMANCE (CURRENT CHAMPION - OPTION B, NOV 2025):
  • Training: ~57.5% ATS on 3,560 games (2021-2024 seasons)
  • Training MAE: ~10.20 points, RMSE: ~13.08 points
  • 2024-2025 backtest: ~52.2% ATS (very similar to prior champion)
  • 2025-2026 to date: ~57.3% ATS overall, with stable late-season performance
  • Unified injury handling: All pipelines now consistent

FEATURES (14 total: 12 core + rest_days):
  ✓ home_oeff, away_oeff           - Offensive efficiency (league-relative)
  ✓ home_deff, away_deff           - Defensive efficiency (league-relative)
  ✓ home_tpar, away_tpar           - 3-point attempt rate (league-relative)
  ✓ home_drr, away_drr             - Defensive rebound rate (league-relative)
  ✓ home_astr, away_astr           - Assist rate (league-relative)
  ✓ home_tor, away_tor             - Turnover rate (league-relative)
  ✓ home_tov_edge                  - Turnover differential (away TOr - home TOr)
  ✓ home_orb_edge                  - Offensive rebound differential
  ✓ home_rest_days, away_rest_days - Days of rest before the game (schedule-aware)
  
  ✗ Excludes home_ftr, away_ftr    - Free throw rate (regime-specific)
  ✗ Excludes role indicators       - Favorite/underdog features (overfit risk)
  ✗ Excludes away_tov_edge         - Redundant (perfect inverse of home_tov_edge)
  ✗ Excludes away_orb_edge         - Highly correlated with home_orb_edge
  
TRAINING DATA (CURRENT CHAMPION):
  • 3,560 games (2021-2024 seasons) with unified injury handling
  • 83.2% of games have detected injuries (vs 0% in legacy pipeline)
  • Rest_days now included as model features (home_rest_days, away_rest_days)
  • Retrained: November 2025 as rest-aware Champion (Option B)
  
HYPERPARAMETER TUNING:
  Method: Model-specific RandomizedSearchCV (100 iterations, 5-fold CV)
  Optimized for: RMSE (not ATS, as direct ATS optimization led to overfitting)
  Note: Using same hyperparameters as previous champion (proven effective)
  
  Key hyperparameters:
    • n_estimators: 210 (vs 100 in original)
    • max_depth: 2 (shallow trees for generalization)
    • learning_rate: 0.00994 (~0.01)
    • subsample: 0.714 (row sampling)
    • colsample_bytree: 0.847 (vs 0.70 in original)
      → Higher colsample compensates for removed redundancy
      → With 12 features, 0.847 means ~10 features per tree
      → Ensures turnover info (home_tov_edge) included in ~85% of trees
    • reg_alpha: 1.90 (L1 regularization)
    • reg_lambda: 13.06 (L2 regularization)

KEY IMPROVEMENTS (Nov 9, 2025):
  1. Unified Injury Handling Across All Pipelines
     • Day-of, backlook, and training now handle injuries identically
     • Eliminates train-test distribution mismatch
     • Injury features (minutes_missing_top2, star_out) now meaningful
     • Roster reconstruction: 10-game lookback, 10-min injury threshold
  
  2. Consistent Player Availability Features
     • Training data: 83.2% of games have detected injuries
     • Legacy data: 0% injury detection (players omitted entirely)
     • Model now sees same injury patterns in training and prediction
  
  3. Better Feature Utilization
     • Injury features should have higher importance in new model
     • More reliable predictions for games with significant injuries
     • Improved generalization from consistent data patterns
  
  4. Maintained Performance with Better Foundation
     • Same hyperparameters as proven champion model
     • Retrained on larger dataset (5,271 vs 3,560 games)
     • Expected similar ATS performance with more consistent predictions

MODEL EVOLUTION:
  Original Champion (Pre-Nov 2025):
    • Trained on 3,560 games (2021-2024)
    • Legacy injury handling (incomplete roster, no unified injuries)
    • 61.83% ATS on early 2025-26 season (before unified injury fix)
  
  Unified-Injury Champion v1 (Nov 9, 2025, archived):
    • Artifact: artifacts/champion_corrected_rest_days
    • Trained on 5,271 games (2021-2025) with unified injury handling
    • 12-feature set (no rest_days, no FTR, no redundancy)
    • 2025-26 to date: ~56.8% ATS overall
  
  Rest-Aware Champion v2 (Nov 2025, CURRENT):
    • Artifact: artifacts/champion_rest_schedule
    • Same XGBoost architecture and redundancy controls as v1
    • Adds home_rest_days and away_rest_days as features (rest-aware)
    • Trained on 3,560 games (2021-2024) to avoid 24-25 regime shift issues
    • 2024-25 backtest: ~52.2% ATS (similar to v1)
    • 2025-26 to date: ~57.3% ATS with improved late-season stability

═══════════════════════════════════════════════════════════════════════════════

This script automates the complete workflow for generating and emailing
daily NBA betting recommendations:
1. Scrapes current odds and injury data
2. Prepares today's games with features
3. Generates margin predictions using Rest-Aware Champion v2 model
4. Analyzes predictions for betting recommendations
5. Emails results via SendGrid API

Usage:
    python daily_betting_recommendations_champion.py --api-key YOUR_SENDGRID_API_KEY
    
    # Dry run (skip email)
    python daily_betting_recommendations_champion.py --dry-run
    
    # Custom date (Eastern time)
    python daily_betting_recommendations_champion.py --date 2025-11-10 --api-key YOUR_KEY
"""

import argparse
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Canonical model metadata for this pipeline. Keep these in sync with the
# artifact actually used in the predict step so logs, comments, and email
# subject lines always reflect reality.
MODEL_NAME = "Rest-Aware Champion v2"
MODEL_SHORT_NAME = "Champion"
MODEL_ARTIFACT_PATH = "artifacts/champion_rest_schedule"


def get_eastern_date() -> str:
    """Get current date in Eastern timezone."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
    except ImportError:
        # Fallback for Python < 3.9
        import pytz
        tz = pytz.timezone("America/New_York")
    
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d")


def run_step(step_name: str, command: List[str], check: bool = True) -> bool:
    """Run a pipeline step and handle errors."""
    logger.info("=" * 70)
    logger.info(f"STEP: {step_name}")
    logger.info("=" * 70)
    logger.info(f"Running: {' '.join(command)}")
    logger.info("")
    
    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✓ {step_name} completed successfully")
            logger.info("")
            return True
        else:
            logger.error(f"✗ {step_name} failed with return code {result.returncode}")
            logger.info("")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} failed: {e}")
        logger.info("")
        return False
    except Exception as e:
        logger.error(f"✗ {step_name} failed with unexpected error: {e}")
        logger.info("")
        return False


def analyze_predictions(predictions_file: Path) -> Tuple[pd.DataFrame, List[Dict]]:
    """Analyze predictions and generate betting recommendations.
    
    Returns:
        Tuple of (full dataframe, list of recommendation dicts)
    """
    logger.info("=" * 70)
    logger.info("STEP: Analyzing Predictions")
    logger.info("=" * 70)
    
    df = pd.read_csv(predictions_file)
    logger.info(f"Loaded {len(df)} game predictions")
    
    recommendations = []
    
    for idx, row in df.iterrows():
        pred_margin = row['pred_margin_mu']
        baseline_margin = row['baseline_margin']
        market_spread = row['market_spread_home']
        
        # Determine recommendation
        if pred_margin > baseline_margin:
            recommended_team = "home"
            recommended_team_name = row['home_team']
            edge = pred_margin - baseline_margin
        elif pred_margin < baseline_margin:
            recommended_team = "away"
            recommended_team_name = row['away_team']
            edge = baseline_margin - pred_margin
        else:
            recommended_team = "push"
            recommended_team_name = "None"
            edge = 0.0
        
        recommendation = {
            'game_id': row['game_id'],
            'date': row['date'],
            'game_time': row.get('game_time'),
            'away_team': row['away_team'],
            'home_team': row['home_team'],
            'market_spread_home': market_spread,
            'baseline_margin': baseline_margin,
            'pred_margin_mu': pred_margin,
            'recommended_side': recommended_team,
            'recommended_team': recommended_team_name,
            'edge': edge
        }
        
        recommendations.append(recommendation)
        
        logger.info(f"  {row['away_team']} @ {row['home_team']}")
        logger.info(f"    Market spread: {market_spread:.1f} (home)")
        logger.info(f"    Predicted margin: {pred_margin:.2f}")
        logger.info(f"    Baseline margin: {baseline_margin:.2f}")
        logger.info(f"    → Recommendation: {recommended_team_name} ({recommended_team.upper()})")
        logger.info(f"    → Edge: {edge:.2f} points")
        logger.info("")
    
    logger.info(f"✓ Generated {len(recommendations)} recommendations")
    logger.info("")
    
    return df, recommendations


def parse_game_time_for_sorting(game_time: str) -> int:
    """
    Convert game time string (e.g., '7:30 PM', '10:00 AM') to minutes since midnight for sorting.
    Returns a large number if parsing fails to push unparseable times to the end.
    """
    if not game_time:
        return 9999  # Push null times to the end
    
    try:
        # Parse time like "7:30 PM" or "10:00 AM"
        import re
        match = re.match(r'(\d+):(\d+)\s*(AM|PM)', game_time.strip(), re.IGNORECASE)
        if not match:
            return 9999
        
        hours = int(match.group(1))
        minutes = int(match.group(2))
        period = match.group(3).upper()
        
        # Convert to 24-hour format
        if period == 'PM' and hours != 12:
            hours += 12
        elif period == 'AM' and hours == 12:
            hours = 0
        
        # Return minutes since midnight
        return hours * 60 + minutes
    except:
        return 9999


def get_spread_bucket(spread: float) -> int:
    """
    Calculate spread bucket based on absolute spread.
    
    Formula: =if(ABS(F2)>=8,1,IF(ABS(F2)>=6,2,IF(ABS(F2)>=4,2,3)))
    
    Bucket 1: |spread| >= 8
    Bucket 2: 4 <= |spread| < 8
    Bucket 3: |spread| < 4
    """
    abs_spread = abs(spread)
    if abs_spread >= 10:
        return 1
    elif abs_spread >= 5:
        return 2
    else:
        return 3


def get_confidence_level(rec: Dict) -> str:
    """
    Determine the confidence level for a recommendation.

    Rules are applied in priority order: Extra High → High → Medium → Low.

    Extra High:
        - All road favorites (away favorites, any spread bucket)

    High:
        - All home underdogs
        - Home favorites in buckets 1 or 3

    Medium:
        - Road underdogs in bucket 2
        - Road favorites in bucket 2

    Low:
        - All other options
    """
    spread = rec['market_spread_home']
    pick_side = rec['recommended_side']
    bucket = get_spread_bucket(spread)
    
    # Home or Away favorite/underdog (relative to market spread, which is home-centric)
    is_home = pick_side == 'home'
    is_away = pick_side == 'away'

    is_home_favorite = is_home and spread < 0
    is_home_underdog = is_home and spread > 0
    is_away_favorite = is_away and spread > 0
    is_away_underdog = is_away and spread < 0

    if is_home_underdog:
        if spread <= 4:
            return 'high'
        # elif spread > 5:
        #     return 'medium'
        else:
            return 'low'
    elif is_home_favorite and (spread >= -5 or spread < -12):
        return 'medium'
    # elif is_away_underdog:
    #     if -9 <= spread <= -5:
    #         return 'medium'
    #     else:
    #         return 'low'
    elif is_away_favorite:
        if spread < 5:
            return 'high'
        else:
            return 'medium'
    else:
        return 'low'
    
    # # In NBA spreads: negative = home favorite, positive = away favorite (relative to home)
    # # Home favorite: spread < 0
    # # Home underdog: spread > 0
    # # Away favorite: spread > 0
    # # Away underdog: spread < 0

    # # Extra High: all road favorites (away favorites), any bucket
    # if is_away_favorite and bucket == 3:
    #     return 'extra_high'

    # # High:
    # #   - All home underdogs
    # #   - Home favorites in buckets 1 or 3
    # if (
    #     is_home_underdog
    #     or (is_home_favorite and bucket == 3)
    #     or (is_away_favorite and bucket in {1,2})
    # ):
    #     return 'high'

    # # Medium:
    # #   - Road underdogs in bucket 2
    # #   - Road favorites in bucket 2 (note: overlaps with Extra High, but Extra High
    # #     takes precedence because rules are applied in order)
    # if (
    #     (is_home_favorite and bucket == 1)
    #     or (is_away_underdog and bucket == 2)
    # ):
    #     return 'medium'

    # # All other cases are low confidence
    # return 'low'


# Configuration for how confidence levels should appear in the email.
#
# To add more levels (e.g., 5 instead of 3), you can:
#   1) Update `get_confidence_level` to return new labels (e.g. "very_high")
#   2) Add entries here with matching "key" values and desired titles/styles.
# The email renderer will automatically create one table per configured level,
# in the order defined below, and will also handle any extra confidence labels
# not listed here using a generic style.
CONFIDENCE_LEVELS = [
#     {
#         "key": "extra_high",
#         "title": "🚀 Extra High Confidence Picks (bet the farm)",
#         "table_class": "high-confidence-table",
#         "badge_class": "high-confidence-pick",
#         "no_picks_message": "No extra high confidence picks today.",
#     },
    {
        "key": "high",
        "title": "🔥 High Confidence Picks",
        "table_class": "high-confidence-table",
        "badge_class": "high-confidence-pick",
        "no_picks_message": "No high confidence picks today.",
    },
    {
        "key": "medium",
        "title": "⚡ Medium Confidence Picks",
        "table_class": "medium-confidence-table",
        "badge_class": "medium-confidence-pick",
        "no_picks_message": "No medium confidence picks today.",
    },
    {
        "key": "low",
        "title": "📊 Low Confidence Picks",
        "table_class": "low-confidence-table",
        "badge_class": "low-confidence-pick",
        "no_picks_message": "No low confidence picks today.",
    },
]


def format_email_body(date_str: str, recommendations: List[Dict]) -> str:
    """Format recommendations as HTML email body with clean table design."""
    
    # Filter out pushes
    filtered_recs = [r for r in recommendations if r['recommended_side'] != 'push']
    
    # Group picks by confidence label (supports N distinct levels)
    picks_by_confidence: Dict[str, List[Dict]] = {}
    for rec in filtered_recs:
        confidence = get_confidence_level(rec)
        picks_by_confidence.setdefault(confidence, []).append(rec)

    # Sort each confidence bucket by game time (chronological order)
    for recs in picks_by_confidence.values():
        recs.sort(key=lambda r: parse_game_time_for_sorting(r.get('game_time', '')))
    
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #1f2937;
                max-width: 1000px;
                margin: 0 auto;
                padding: 40px 20px;
                background-color: #f9fafb;
            }}
            .container {{
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                overflow: hidden;
                max-width: 800px;
                margin: 0 auto;
            }}
            .header {{
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                color: #ffffff;
                padding: 32px 40px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0 0 8px 0;
                font-size: 28px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }}
            .header .date {{
                font-size: 16px;
                opacity: 0.95;
                font-weight: 400;
            }}
            .content {{
                padding: 10px;
            }}
            .intro {{
                font-size: 16px;
                color: #6b7280;
                margin-bottom: 32px;
                text-align: center;
            }}
            .section-title {{
                font-size: 14px;
                font-weight: 700;
                color: #059669;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 32px 0 16px 0;
                padding-left: 4px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 0 0 24px 0;
                font-size: 15px;
                border-radius: 8px;
                overflow: hidden;
            }}
            .high-confidence-table {{
                border: 3px solid #10b981;
                box-shadow: 0 0 0 1px #10b981;
            }}
            .medium-confidence-table {{
                border: 2px solid #3b82f6;
                box-shadow: 0 0 0 1px #3b82f6;
            }}
            .low-confidence-table {{
                border: 1px solid #9ca3af;
            }}
            thead {{
                background-color: #f3f4f6;
            }}
            .high-confidence-table thead {{
                background-color: #d1fae5;
            }}
            .medium-confidence-table thead {{
                background-color: #dbeafe;
            }}
            .low-confidence-table thead {{
                background-color: #f3f4f6;
            }}
            th {{
                padding: 14px 16px;
                text-align: left;
                font-weight: 600;
                color: #374151;
                border-bottom: 2px solid #e5e7eb;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .high-confidence-table th {{
                color: #065f46;
                border-bottom: 2px solid #10b981;
            }}
            .medium-confidence-table th {{
                color: #1e40af;
                border-bottom: 2px solid #3b82f6;
            }}
            .low-confidence-table th {{
                color: #4b5563;
                border-bottom: 1px solid #9ca3af;
            }}
            td {{
                padding: 16px;
                border-bottom: 1px solid #f3f4f6;
                color: #1f2937;
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            tbody tr {{
                transition: background-color 0.2s ease;
            }}
            tbody tr:hover {{
                background-color: #f9fafb;
            }}
            .high-confidence-table tbody tr:hover {{
                background-color: #ecfdf5;
            }}
            .medium-confidence-table tbody tr:hover {{
                background-color: #eff6ff;
            }}
            .low-confidence-table tbody tr:hover {{
                background-color: #f9fafb;
            }}
            .matchup {{
                font-weight: 600;
                color: #111827;
                font-size: 15px;
            }}
            .vs {{
                color: #9ca3af;
                font-weight: 400;
                margin: 0 6px;
            }}
            .favored {{
                color: #4b5563;
            }}
            .team-badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 14px;
            }}
            .pick {{
                background-color: #dbeafe;
                color: #1e40af;
            }}
            .high-confidence-pick {{
                background-color: #d1fae5;
                color: #065f46;
            }}
            .medium-confidence-pick {{
                background-color: #dbeafe;
                color: #1e40af;
            }}
            .low-confidence-pick {{
                background-color: #e5e7eb;
                color: #4b5563;
            }}
            .no-games {{
                text-align: center;
                padding: 60px 20px;
                color: #9ca3af;
                font-size: 16px;
            }}
            .no-value-picks {{
                text-align: center;
                padding: 24px 20px;
                color: #6b7280;
                font-size: 14px;
                background-color: #f9fafb;
                border-radius: 8px;
                margin-bottom: 24px;
            }}
            .footer {{
                margin-top: 32px;
                padding-top: 24px;
                border-top: 1px solid #e5e7eb;
                text-align: center;
                font-size: 12px;
                color: #9ca3af;
            }}
            .footer p {{
                margin: 4px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏀 NBA Betting Recommendations</h1>
                <div class="date">{date_str}</div>
            </div>
            
            <div class="content">
    """
    
    if not filtered_recs:
        html += """
                <div class="no-games">
                    <p>No games with betting recommendations today.</p>
                </div>
        """
    else:
        # Render sections for each configured confidence level (in order)
        configured_keys = set()
        for level in CONFIDENCE_LEVELS:
            key = level["key"]
            title = level["title"]
            table_class = level.get("table_class", "")
            badge_class = level.get("badge_class", "pick")
            no_picks_message = level.get("no_picks_message", f"No {key} picks today.")
            configured_keys.add(key)

            html += f"""
                <div class="section-title">{title}</div>
            """

            picks = picks_by_confidence.get(key, [])

            if picks:
                table_class_attr = f' class="{table_class}"' if table_class else ""
                html += f"""
                <table{table_class_attr}>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Matchup</th>
                            <th>Favored</th>
                            <th>Model Pick</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                for rec in picks:
                    # Format game time
                    game_time_display = rec.get('game_time', 'TBD')
                    if not game_time_display:
                        game_time_display = 'TBD'

                    # Format matchup
                    matchup = f"{rec['away_team']} <span class='vs'>@</span> {rec['home_team']}"

                    # Determine who is favored
                    spread = rec['market_spread_home']
                    if spread < 0:
                        # Home team favored
                        favored = f"{rec['home_team']} (Home) by {abs(spread):.1f}"
                    else:
                        # Away team favored
                        favored = f"{rec['away_team']} (Away) by {abs(spread):.1f}"

                    # Format pick
                    pick_team = rec['recommended_team']
                    pick_location = rec['recommended_side'].capitalize()
                    pick = f"{pick_team} ({pick_location})"

                    html += f"""
                        <tr>
                            <td>{game_time_display}</td>
                            <td class="matchup">{matchup}</td>
                            <td class="favored">{favored}</td>
                            <td><span class="team-badge {badge_class}">{pick}</span></td>
                        </tr>
                    """

                html += """
                    </tbody>
                </table>
                """
            else:
                html += f"""
                <div class="no-value-picks">
                    <p>{no_picks_message}</p>
                </div>
                """

        # Render any additional confidence levels not explicitly configured,
        # using a generic style. This lets you extend `get_confidence_level`
        # to return new labels without having to touch the email layout.
        extra_keys = [k for k in picks_by_confidence.keys() if k not in configured_keys]
        for key in sorted(extra_keys):
            picks = picks_by_confidence[key]
            if not picks:
                continue

            # Human-friendly title from the raw key
            label_title = key.replace("_", " ").title()
            title = f"📌 {label_title} Picks"
            no_picks_message = f"No {label_title.lower()} picks today."

            html += f"""
                <div class="section-title">{title}</div>
            """

            table_class_attr = ""  # generic table styling
            html += f"""
                <table{table_class_attr}>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Matchup</th>
                            <th>Favored</th>
                            <th>Model Pick</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for rec in picks:
                # Format game time
                game_time_display = rec.get('game_time', 'TBD')
                if not game_time_display:
                    game_time_display = 'TBD'

                # Format matchup
                matchup = f"{rec['away_team']} <span class='vs'>@</span> {rec['home_team']}"

                # Determine who is favored
                spread = rec['market_spread_home']
                if spread < 0:
                    favored = f"{rec['home_team']} (Home) by {abs(spread):.1f}"
                else:
                    favored = f"{rec['away_team']} (Away) by {abs(spread):.1f}"

                # Format pick
                pick_team = rec['recommended_team']
                pick_location = rec['recommended_side'].capitalize()
                pick = f"{pick_team} ({pick_location})"

                html += f"""
                        <tr>
                            <td>{game_time_display}</td>
                            <td class="matchup">{matchup}</td>
                            <td class="favored">{favored}</td>
                            <td><span class="team-badge pick">{pick}</span></td>
                        </tr>
                """

            html += """
                    </tbody>
                </table>
            """
    
    html += """
                <div class="footer">
                    <p>Automatically generated by NBA Predictions Pipeline</p>
                    <p>""" + datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z") + """</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(api_key: str, from_email: str, to_email: str, subject: str, html_content: str, from_alias: str = None) -> bool:
    """Send email via SendGrid API."""
    logger.info("=" * 70)
    logger.info("STEP: Sending Email")
    logger.info("=" * 70)
    
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, From
        
        # Build from_email with optional alias
        if from_alias:
            from_email_obj = From(from_email, from_alias)
        else:
            from_email_obj = from_email
        
        message = Mail(
            from_email=from_email_obj,
            to_emails=to_email,
            subject=subject,
            html_content=html_content
        )
        
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        
        logger.info(f"✓ Email sent successfully!")
        logger.info(f"  Status code: {response.status_code}")
        if from_alias:
            logger.info(f"  From: {from_alias} <{from_email}>")
        else:
            logger.info(f"  From: {from_email}")
        logger.info(f"  To: {to_email}")
        logger.info("")
        
        return True
        
    except ImportError:
        logger.error("✗ SendGrid library not installed")
        logger.error("  Install with: pip install sendgrid")
        logger.info("")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to send email: {e}")
        logger.info("")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Daily NBA betting recommendations pipeline (Rest-Aware Champion v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format (defaults to today in Eastern time)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="SendGrid API key (can also use SENDGRID_API_KEY in .env)"
    )
    parser.add_argument(
        "--from-email",
        type=str,
        default=None,
        help="Sender email address (can also use SENDGRID_FROM_EMAIL in .env)"
    )
    parser.add_argument(
        "--from-alias",
        type=str,
        default=None,
        help="Sender display name (can also use SENDGRID_FROM_ALIAS in .env)"
    )
    parser.add_argument(
        "--to-email",
        type=str,
        default=None,
        help="Recipient email address (can also use SENDGRID_TO_EMAIL in .env)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but skip sending email"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping step (use existing data)"
    )
    
    args = parser.parse_args()
    
    # Get date
    if args.date:
        date_str = args.date
        logger.info(f"Using specified date: {date_str}")
    else:
        date_str = get_eastern_date()
        logger.info(f"Using today's date (Eastern time): {date_str}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("NBA DAILY BETTING RECOMMENDATIONS PIPELINE")
    logger.info(f"Model: {MODEL_NAME} ({MODEL_SHORT_NAME}, 14 features including rest_days)")
    logger.info(f"Model artifact: {MODEL_ARTIFACT_PATH}")
    logger.info("=" * 70)
    logger.info("")
    
    # Define file paths
    data_file = f"data/games_future_{date_str}_norm.jsonl"
    predictions_file = f"predictions/predictions_champion_{date_str}.csv"
    
    # STEP 1: Prepare today's games (with scraping)
    if not args.skip_scrape:
        cmd = [sys.executable, "prepare_todays_games.py", "--scrape-all"]
        if args.date:
            cmd.extend(["--date", date_str])
        
        success = run_step("Prepare Today's Games", cmd)
        if not success:
            logger.error("Pipeline failed at preparation step")
            sys.exit(1)
    else:
        logger.info("Skipping scrape step as requested")
        logger.info("")
    
    # Verify data file exists
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Cannot proceed without game data")
        sys.exit(1)
    
    # STEP 2: Generate predictions using Rest-Aware Champion v2 model
    # The predict_margin.py script will automatically use the correct features
    # based on the model's config (rest-aware Champion v2: 14 features, no FTR, no role indicators, no redundancy)
    cmd = [
        sys.executable,
        "predict_margin.py",
        "--data", data_file,
        "--model", MODEL_ARTIFACT_PATH,
        "--output", predictions_file
    ]
    
    success = run_step(f"Generate Predictions ({MODEL_NAME})", cmd)
    if not success:
        logger.error("Pipeline failed at prediction step")
        sys.exit(1)
    
    # STEP 3: Analyze predictions
    try:
        df, recommendations = analyze_predictions(Path(predictions_file))
    except Exception as e:
        logger.error(f"Failed to analyze predictions: {e}")
        sys.exit(1)
    
    # STEP 4: Send email
    if args.dry_run:
        logger.info("=" * 70)
        logger.info("DRY RUN - Email Preview")
        logger.info("=" * 70)
        logger.info("")
        
        html_content = format_email_body(date_str, recommendations)
        
        # Save preview to file
        preview_file = Path("predictions") / f"email_preview_champion_{date_str}.html"
        preview_file.write_text(html_content, encoding='utf-8')
        logger.info(f"Email preview saved to: {preview_file}")
        logger.info("")
        logger.info("✓ Pipeline completed successfully (dry run)")
        
    else:
        # Get configuration from args, .env, or defaults
        api_key = args.api_key or os.getenv("SENDGRID_API_KEY")
        from_email = args.from_email or os.getenv("SENDGRID_FROM_EMAIL")
        from_alias = args.from_alias or os.getenv("SENDGRID_FROM_ALIAS")
        to_email = args.to_email or os.getenv("SENDGRID_TO_EMAIL")
        
        # Validate configuration
        if not api_key:
            logger.error("SendGrid API key not provided")
            logger.error("Add to .env file: SENDGRID_API_KEY=your_key")
            logger.error("Or use: --api-key YOUR_KEY")
            sys.exit(1)
        
        if not from_email:
            logger.error("Sender email not provided")
            logger.error("Add to .env file: SENDGRID_FROM_EMAIL=your@email.com")
            logger.error("Or use: --from-email your@email.com")
            sys.exit(1)
        
        if not to_email:
            logger.error("Recipient email not provided")
            logger.error("Add to .env file: SENDGRID_TO_EMAIL=recipient@email.com")
            logger.error("Or use: --to-email recipient@email.com")
            sys.exit(1)
        
        # Format and send email
        subject = f"NBA Betting Recommendations - {date_str}"
        html_content = format_email_body(date_str, recommendations)
        
        success = send_email(api_key, from_email, to_email, subject, html_content, from_alias)
        
        if success:
            logger.info("=" * 70)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info("")
        else:
            logger.error("Pipeline completed but email failed to send")
            # Save email to file as backup
            backup_file = Path("predictions") / f"email_backup_champion_{date_str}.html"
            backup_file.write_text(html_content, encoding='utf-8')
            logger.info(f"Email content saved to: {backup_file}")
            sys.exit(1)


if __name__ == "__main__":
    main()

