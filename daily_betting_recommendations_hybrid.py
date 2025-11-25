#!/usr/bin/env python
"""Daily NBA Betting Recommendations Pipeline - Hybrid Model Selection

═══════════════════════════════════════════════════════════════════════════════
STRATEGY: Automatic Regime-Based Model Selection
  • CHAMPION model during normal variance periods
  • CONTEXT model during high variance periods (>30% above baseline for 5+ days)
═══════════════════════════════════════════════════════════════════════════════

PERFORMANCE (HYBRID STRATEGY - NOV 2025):
  • Backtested: 56.63% ATS (+2.41% vs pure Champion)
  • Regime switch date: Nov 14, 2025 (detected high variance)
  • Champion portion: 57.39% ATS (176 games)
  • Context portion: 54.79% ATS (73 games)

CHAMPION MODEL (Normal Variance):
  • Artifact: artifacts/champion_rest_schedule
  • Features: 14 (rest-aware, individually tuned)
  • Best for: Stable league conditions

CONTEXT MODEL (High Variance):
  • Artifact: artifacts/experiments/exp6_context
  • Features: 18 (14 base + 4 league volatility context)
  • Best for: High variance periods (trades peak for stability)

REGIME DETECTION:
  • Metric: DEFF (Defensive Efficiency) standard deviation ratio
  • Threshold: 1.30 (30% above historical baseline)
  • Confirmation: 5 consecutive days above threshold
  • Min games: 250 (to avoid early-season noise)

═══════════════════════════════════════════════════════════════════════════════

Usage:
    python daily_betting_recommendations_hybrid.py --api-key YOUR_SENDGRID_API_KEY
    
    # Dry run (skip email)
    python daily_betting_recommendations_hybrid.py --dry-run
    
    # Custom date (Eastern time)
    python daily_betting_recommendations_hybrid.py --date 2025-11-25 --api-key YOUR_KEY
"""

import argparse
import subprocess
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import os
from dotenv import load_dotenv

# Import hybrid model selection
from hybrid_model_selector import VarianceRegimeDetector, RegimeConfig
from regime_monitor import RegimeMonitor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configurations
CHAMPION_CONFIG = {
    "name": "Rest-Aware Champion v2",
    "short_name": "Champion",
    "artifact_path": "artifacts/champion_rest_schedule",
    "features": 14,
    "description": "Optimized for normal variance periods"
}

CONTEXT_CONFIG = {
    "name": "Context-Aware Model",
    "short_name": "Context",
    "artifact_path": "artifacts/experiments/exp6_context",
    "features": 18,
    "description": "Optimized for high variance periods"
}

# Regime detection configuration (optimized via sensitivity analysis)
REGIME_CONFIG = RegimeConfig(
    min_games_for_detection=250,
    high_variance_threshold=1.30,
    normal_variance_threshold=1.20,
    consecutive_days_required=5,
)


def get_eastern_date() -> str:
    """Get current date in Eastern timezone."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz = pytz.timezone("America/New_York")
    
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d")


def load_regime_state() -> dict:
    """Load the previous regime state from file."""
    state_file = Path("analysis/regime_state.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {'last_regime': None, 'last_check_date': None, 'last_model': None}


def save_regime_state(regime: str, date: str, model: str, details: dict):
    """Save the current regime state to file."""
    state_file = Path("analysis/regime_state.json")
    state_file.parent.mkdir(exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump({
            'last_regime': regime,
            'last_check_date': date,
            'last_model': model,
            'variance_ratio': details.get('ratio_deff'),
            'consecutive_days': details.get('consecutive_high_days', 0),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)


def detect_current_regime(date_str: str) -> Tuple[str, str, dict, bool]:
    """
    Detect the current variance regime and determine which model to use.
    
    Uses RegimeMonitor which maintains history for proper consecutive day tracking.
    
    Returns:
        Tuple of (regime, model_name, details, regime_changed)
    """
    # Use RegimeMonitor which maintains history
    monitor = RegimeMonitor(REGIME_CONFIG)
    status = monitor.check_regime(date_str, '2025-2026')
    
    # Extract info from status
    regime = status.regime
    model_name = status.model_recommended
    
    details = {
        'ratio_deff': status.variance_ratio,
        'std_deff': status.std_deff,
        'game_count': status.games_analyzed,
        'consecutive_high_days': status.consecutive_high_days,
    }
    
    regime_changed = status.changed_from_previous
    
    # Also save to regime_state.json for compatibility
    save_regime_state(regime, date_str, model_name, details)
    
    return regime, model_name, details, regime_changed


def print_regime_status(regime: str, model_name: str, details: dict, regime_changed: bool):
    """Print regime status with appropriate formatting."""
    
    ratio = details.get('ratio_deff', 0)
    ratio_str = f"{ratio:.3f}" if ratio else "N/A"
    consecutive = details.get('consecutive_high_days', 0)
    
    logger.info("=" * 70)
    logger.info("REGIME STATUS CHECK")
    logger.info("=" * 70)
    
    if regime_changed:
        logger.info("")
        logger.info("*** " + "=" * 60 + " ***")
        logger.info("*** REGIME CHANGE DETECTED - MODEL SWITCHING ***")
        logger.info("*** " + "=" * 60 + " ***")
        logger.info("")
    
    if regime == 'high':
        logger.info("[!] HIGH VARIANCE REGIME - Using CONTEXT model")
        logger.info(f"   Variance Ratio: {ratio_str} (threshold: 1.30)")
        logger.info(f"   Consecutive High Days: {consecutive}")
        logger.info(f"   Model: {CONTEXT_CONFIG['name']}")
        logger.info(f"   Features: {CONTEXT_CONFIG['features']}")
    else:
        logger.info(f"[OK] {regime.upper()} VARIANCE REGIME - Using CHAMPION model")
        logger.info(f"   Variance Ratio: {ratio_str}")
        logger.info(f"   Model: {CHAMPION_CONFIG['name']}")
        logger.info(f"   Features: {CHAMPION_CONFIG['features']}")
    
    logger.info("")


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
            logger.info(f"[OK] {step_name} completed successfully")
            logger.info("")
            return True
        else:
            logger.error(f"[FAIL] {step_name} failed with return code {result.returncode}")
            logger.info("")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] {step_name} failed: {e}")
        logger.info("")
        return False
    except Exception as e:
        logger.error(f"[FAIL] {step_name} failed with unexpected error: {e}")
        logger.info("")
        return False


def prepare_context_data(date_str: str) -> bool:
    """Prepare context-enhanced data for the Context model."""
    logger.info("=" * 70)
    logger.info("STEP: Preparing Context-Enhanced Data")
    logger.info("=" * 70)
    
    try:
        # Import normalizer
        from league_normalizer import normalize_game_jsonl
        
        input_file = f"data/games_future_{date_str}.jsonl"
        output_file = f"data/games_future_{date_str}_context.jsonl"
        
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return False
        
        normalize_game_jsonl(
            input_file,
            output_file,
            method='center',
            include_context=True
        )
        
        logger.info(f"[OK] Context data saved to: {output_file}")
        logger.info("")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Failed to prepare context data: {e}")
        logger.info("")
        return False


def analyze_predictions(predictions_file: Path) -> Tuple[pd.DataFrame, List[Dict]]:
    """Analyze predictions and generate betting recommendations."""
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
        logger.info(f"    Recommendation: {recommended_team_name} ({recommended_team.upper()})")
        logger.info(f"    Edge: {edge:.2f} points")
        logger.info("")
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    logger.info("")
    
    return df, recommendations


def parse_game_time_for_sorting(game_time: str) -> int:
    """Convert game time string to minutes since midnight for sorting."""
    if not game_time:
        return 9999
    
    try:
        import re
        match = re.match(r'(\d+):(\d+)\s*(AM|PM)', game_time.strip(), re.IGNORECASE)
        if not match:
            return 9999
        
        hours = int(match.group(1))
        minutes = int(match.group(2))
        period = match.group(3).upper()
        
        if period == 'PM' and hours != 12:
            hours += 12
        elif period == 'AM' and hours == 12:
            hours = 0
        
        return hours * 60 + minutes
    except:
        return 9999


def get_confidence_level_champion(rec: Dict) -> str:
    """Determine confidence level for Champion model recommendations."""
    spread = rec['market_spread_home']
    pick_side = rec['recommended_side']
    
    is_home = pick_side == 'home'
    is_away = pick_side == 'away'

    is_home_favorite = is_home and spread < 0
    is_home_underdog = is_home and spread > 0
    is_away_favorite = is_away and spread > 0
    is_away_underdog = is_away and spread < 0

    if is_home_underdog:
        if spread <= 4:
            return 'high'
        else:
            return 'low'
    elif is_home_favorite and (spread >= -5 or spread < -12):
        return 'medium'
    elif is_away_favorite:
        if spread < 5:
            return 'high'
        else:
            return 'medium'
    else:
        return 'low'


def get_confidence_level_context(rec: Dict) -> str:
    """
    Determine confidence level for Context model recommendations.
    
    Medium confidence:
      - Home dogs: spread 2.5 to 4 (inclusive)
      - Home favorites: spread < -12
      - Road dogs: spread -5 to -9 (inclusive)
      - Road favorites: none
    
    Everything else is low confidence.
    """
    spread = rec['market_spread_home']
    pick_side = rec['recommended_side']
    
    is_home = pick_side == 'home'
    is_away = pick_side == 'away'

    # Home underdog: spread > 0 (away team favored)
    # Home favorite: spread < 0 (home team favored)
    is_home_underdog = is_home and spread > 0
    is_home_favorite = is_home and spread < 0
    is_away_underdog = is_away and spread < 0  # Away team is underdog when home is favored
    is_away_favorite = is_away and spread > 0  # Away team is favorite when spread > 0

    # Home dogs: 2.5 to 4 (inclusive) -> medium
    if is_home_underdog and 2.5 <= spread <= 4:
        return 'medium'
    
    # Home favorites: < -12 -> medium
    if is_home_favorite and spread < -12:
        return 'medium'
    
    # Road dogs: -5 to -9 (inclusive) -> medium
    # When picking away team and spread is negative (home favored), away is underdog
    if is_away_underdog and -9 <= spread <= -5:
        return 'medium'
    
    # Road favorites: none are medium
    # Everything else is low
    return 'low'


def get_confidence_level(rec: Dict, model_name: str = 'champion') -> str:
    """
    Determine confidence level based on which model is being used.
    
    Args:
        rec: Recommendation dictionary
        model_name: 'champion' or 'context'
    
    Returns:
        Confidence level: 'high', 'medium', or 'low'
    """
    if model_name == 'context':
        return get_confidence_level_context(rec)
    else:
        return get_confidence_level_champion(rec)


CONFIDENCE_LEVELS = [
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


def format_email_body(date_str: str, recommendations: List[Dict], 
                      model_name: str, regime: str, regime_details: dict,
                      regime_changed: bool) -> str:
    """Format recommendations as HTML email body with regime status."""
    
    # Filter out pushes
    filtered_recs = [r for r in recommendations if r['recommended_side'] != 'push']
    
    # Group picks by confidence (using model-specific logic)
    picks_by_confidence: Dict[str, List[Dict]] = {}
    for rec in filtered_recs:
        confidence = get_confidence_level(rec, model_name)
        picks_by_confidence.setdefault(confidence, []).append(rec)

    # Sort each bucket by game time
    for recs in picks_by_confidence.values():
        recs.sort(key=lambda r: parse_game_time_for_sorting(r.get('game_time', '')))
    
    # Model info
    if model_name == 'context':
        model_config = CONTEXT_CONFIG
        model_badge_color = "#dc2626"  # Red for high variance
        regime_status = "🚨 HIGH VARIANCE"
    else:
        model_config = CHAMPION_CONFIG
        model_badge_color = "#059669"  # Green for normal
        regime_status = "✓ NORMAL"
    
    ratio = regime_details.get('ratio_deff', 0)
    ratio_str = f"{ratio:.2f}" if ratio else "N/A"
    
    # Regime change alert HTML
    regime_alert_html = ""
    if regime_changed:
        regime_alert_html = f"""
            <div style="background-color: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 16px; margin-bottom: 24px; text-align: center;">
                <div style="font-size: 18px; font-weight: 700; color: #92400e; margin-bottom: 8px;">
                    ⚠️ REGIME CHANGE DETECTED
                </div>
                <div style="font-size: 14px; color: #78350f;">
                    Model has switched to <strong>{model_config['short_name']}</strong> due to {regime} variance conditions.
                </div>
            </div>
        """
    
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
            .model-badge {{
                display: inline-block;
                margin-top: 12px;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                background-color: {model_badge_color};
                color: white;
            }}
            .content {{
                padding: 10px;
            }}
            .regime-info {{
                background-color: #f3f4f6;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 24px;
                text-align: center;
            }}
            .regime-info .label {{
                font-size: 12px;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .regime-info .value {{
                font-size: 16px;
                font-weight: 600;
                color: #1f2937;
                margin-top: 4px;
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
                <div class="model-badge">{regime_status} • {model_config['short_name']} Model</div>
            </div>
            
            <div class="content">
                {regime_alert_html}
                
                <div class="regime-info">
                    <table style="width: 100%; border: none; margin: 0;">
                        <tr>
                            <td style="text-align: center; border: none; padding: 8px;">
                                <div class="label">Model</div>
                                <div class="value">{model_config['name']}</div>
                            </td>
                            <td style="text-align: center; border: none; padding: 8px;">
                                <div class="label">Variance Ratio</div>
                                <div class="value">{ratio_str}</div>
                            </td>
                            <td style="text-align: center; border: none; padding: 8px;">
                                <div class="label">Features</div>
                                <div class="value">{model_config['features']}</div>
                            </td>
                        </tr>
                    </table>
                </div>
    """
    
    if not filtered_recs:
        html += """
                <div class="no-games">
                    <p>No games with betting recommendations today.</p>
                </div>
        """
    else:
        # Render sections for each confidence level
        for level in CONFIDENCE_LEVELS:
            key = level["key"]
            title = level["title"]
            table_class = level.get("table_class", "")
            badge_class = level.get("badge_class", "pick")
            no_picks_message = level.get("no_picks_message", f"No {key} picks today.")

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
                    game_time_display = rec.get('game_time', 'TBD') or 'TBD'
                    matchup = f"{rec['away_team']} <span class='vs'>@</span> {rec['home_team']}"
                    
                    spread = rec['market_spread_home']
                    if spread < 0:
                        favored = f"{rec['home_team']} (Home) by {abs(spread):.1f}"
                    else:
                        favored = f"{rec['away_team']} (Away) by {abs(spread):.1f}"

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
    
    html += """
                <div class="footer">
                    <p>Hybrid Model Selection Pipeline</p>
                    <p>""" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(api_key: str, from_email: str, to_email: str, subject: str, 
               html_content: str, from_alias: str = None) -> bool:
    """Send email via SendGrid API."""
    logger.info("=" * 70)
    logger.info("STEP: Sending Email")
    logger.info("=" * 70)
    
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, From
        
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
        
        logger.info(f"[OK] Email sent successfully!")
        logger.info(f"  Status code: {response.status_code}")
        logger.info(f"  To: {to_email}")
        logger.info("")
        
        return True
        
    except ImportError:
        logger.error("[FAIL] SendGrid library not installed")
        logger.error("  Install with: pip install sendgrid")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Daily NBA betting recommendations - Hybrid Model Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Date in YYYY-MM-DD format (defaults to today Eastern)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="SendGrid API key")
    parser.add_argument("--from-email", type=str, default=None,
                        help="Sender email address")
    parser.add_argument("--from-alias", type=str, default=None,
                        help="Sender display name")
    parser.add_argument("--to-email", type=str, default=None,
                        help="Recipient email address")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline but skip sending email")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping step (use existing data)")
    
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
    logger.info("NBA DAILY BETTING RECOMMENDATIONS - HYBRID MODEL SELECTION")
    logger.info("=" * 70)
    logger.info("")
    
    # STEP 1: Detect current regime
    regime, model_name, regime_details, regime_changed = detect_current_regime(date_str)
    print_regime_status(regime, model_name, regime_details, regime_changed)
    
    # Get model config
    if model_name == 'context':
        model_config = CONTEXT_CONFIG
    else:
        model_config = CHAMPION_CONFIG
    
    # Define file paths
    data_file = f"data/games_future_{date_str}_norm.jsonl"
    context_data_file = f"data/games_future_{date_str}_context.jsonl"
    predictions_file = f"predictions/predictions_hybrid_{date_str}.csv"
    
    # STEP 2: Prepare today's games (with scraping)
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
    
    # STEP 3: Prepare context data if using Context model
    if model_name == 'context':
        success = prepare_context_data(date_str)
        if not success:
            logger.warning("Failed to prepare context data, falling back to Champion model")
            model_name = 'champion'
            model_config = CHAMPION_CONFIG
    
    # STEP 4: Generate predictions
    if model_name == 'context' and Path(context_data_file).exists():
        actual_data_file = context_data_file
    else:
        actual_data_file = data_file
    
    cmd = [
        sys.executable,
        "predict_margin.py",
        "--data", actual_data_file,
        "--model", model_config['artifact_path'],
        "--output", predictions_file
    ]
    
    success = run_step(f"Generate Predictions ({model_config['name']})", cmd)
    if not success:
        logger.error("Pipeline failed at prediction step")
        sys.exit(1)
    
    # STEP 5: Analyze predictions
    try:
        df, recommendations = analyze_predictions(Path(predictions_file))
    except Exception as e:
        logger.error(f"Failed to analyze predictions: {e}")
        sys.exit(1)
    
    # STEP 6: Send email
    if args.dry_run:
        logger.info("=" * 70)
        logger.info("DRY RUN - Email Preview")
        logger.info("=" * 70)
        
        html_content = format_email_body(
            date_str, recommendations, model_name, regime, regime_details, regime_changed
        )
        
        preview_file = Path("predictions") / f"email_preview_hybrid_{date_str}.html"
        preview_file.write_text(html_content, encoding='utf-8')
        logger.info(f"Email preview saved to: {preview_file}")
        logger.info("")
        logger.info("Pipeline completed successfully (dry run)")
        
    else:
        # Get configuration
        api_key = args.api_key or os.getenv("SENDGRID_API_KEY")
        from_email = args.from_email or os.getenv("SENDGRID_FROM_EMAIL")
        from_alias = args.from_alias or os.getenv("SENDGRID_FROM_ALIAS")
        to_email = args.to_email or os.getenv("SENDGRID_TO_EMAIL")
        
        # Validate
        if not api_key:
            logger.error("SendGrid API key not provided")
            sys.exit(1)
        if not from_email:
            logger.error("Sender email not provided")
            sys.exit(1)
        if not to_email:
            logger.error("Recipient email not provided")
            sys.exit(1)
        
        # Format subject with regime indicator
        if regime == 'high':
            subject = f"🚨 NBA Picks - {date_str} (HIGH VARIANCE)"
        else:
            subject = f"NBA Betting Recommendations - {date_str}"
        
        html_content = format_email_body(
            date_str, recommendations, model_name, regime, regime_details, regime_changed
        )
        
        success = send_email(api_key, from_email, to_email, subject, html_content, from_alias)
        
        if success:
            logger.info("=" * 70)
            logger.info("[OK] PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"  Model used: {model_config['name']}")
            logger.info(f"  Regime: {regime}")
            logger.info("=" * 70)
        else:
            backup_file = Path("predictions") / f"email_backup_hybrid_{date_str}.html"
            backup_file.write_text(html_content, encoding='utf-8')
            logger.info(f"Email content saved to: {backup_file}")
            sys.exit(1)


if __name__ == "__main__":
    main()

