#!/usr/bin/env python
"""Daily NBA Betting Recommendations Pipeline

This script automates the complete workflow for generating and emailing
daily NBA betting recommendations:
1. Scrapes current odds and injury data
2. Prepares today's games with features
3. Generates margin predictions
4. Analyzes predictions for betting recommendations
5. Emails results via SendGrid API

Usage:
    python daily_betting_recommendations.py --api-key YOUR_SENDGRID_API_KEY
    
    # Dry run (skip email)
    python daily_betting_recommendations.py --dry-run
    
    # Custom date (Eastern time)
    python daily_betting_recommendations.py --date 2025-11-10 --api-key YOUR_KEY
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


def format_email_body(date_str: str, recommendations: List[Dict]) -> str:
    """Format recommendations as HTML email body with clean table design."""
    
    # Filter out pushes
    filtered_recs = [r for r in recommendations if r['recommended_side'] != 'push']
    
    # Separate Best Value picks from regular picks
    # Best Value criteria:
    # 1. Away favored, pick home to cover, spread > 0 and < 4
    # 2. Home favored, pick away to cover, spread >= 4 and < 8
    upset_picks = []
    regular_picks = []
    
    for rec in filtered_recs:
        spread = rec['market_spread_home']
        pick_side = rec['recommended_side']
        
        is_best_value = False
        
        # Case 1: Away team favored (positive spread), model picks home
        # Spread must be > 0 and < 4
        if spread > 0 and spread < 4 and pick_side == 'home':
            is_best_value = True
        
        # Case 2: Home team favored (negative spread), model picks away
        # Spread must be <= -4 and > -8 (i.e., abs(spread) >= 4 and < 8)
        elif spread <= -4 and spread > -8 and pick_side == 'away':
            is_best_value = True
        
        if is_best_value:
            upset_picks.append(rec)
        else:
            regular_picks.append(rec)
    
    # Sort each category by edge
    upset_picks.sort(key=lambda r: r['edge'], reverse=True)
    regular_picks.sort(key=lambda r: r['edge'], reverse=True)
    
    # Combine: upsets first, then regular
    sorted_recs = upset_picks + regular_picks
    
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
                padding: 40px;
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
            .upset-table {{
                border: 3px solid #10b981;
                box-shadow: 0 0 0 1px #10b981;
            }}
            thead {{
                background-color: #f3f4f6;
            }}
            .upset-table thead {{
                background-color: #d1fae5;
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
            .upset-table th {{
                color: #065f46;
                border-bottom: 2px solid #10b981;
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
            .upset-table tbody tr:hover {{
                background-color: #ecfdf5;
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
            .upset-pick {{
                background-color: #d1fae5;
                color: #065f46;
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
                <p class="intro">Today's model predictions and recommended picks</p>
    """
    
    if not sorted_recs:
        html += """
                <div class="no-games">
                    <p>No games with betting recommendations today.</p>
                </div>
        """
    else:
        # Render upset picks first if they exist
        html += """
                <div class="section-title">⭐ Best Value Picks</div>
        """
        
        if upset_picks:
            html += """
                <table class="upset-table">
                    <thead>
                        <tr>
                            <th>Matchup</th>
                            <th>Favored</th>
                            <th>Model Pick</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for rec in upset_picks:
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
                            <td class="matchup">{matchup}</td>
                            <td class="favored">{favored}</td>
                            <td><span class="team-badge upset-pick">{pick}</span></td>
                        </tr>
                """
            
            html += """
                    </tbody>
                </table>
            """
        else:
            html += """
                <div class="no-value-picks">
                    <p>No games meet the Best Value Pick criteria today.</p>
                    <p style="font-size: 13px; margin-top: 8px; color: #9ca3af;">
                        (Away favored 0-4 pts picking home, or Home favored 4-8 pts picking away)
                    </p>
                </div>
            """
        
        # Render regular picks if they exist
        if regular_picks:
            if upset_picks:
                html += """
                <div class="section-title">Additional Picks</div>
                """
            else:
                html += """
                <div class="section-title">Today's Picks</div>
                """
            
            html += """
                <table>
                    <thead>
                        <tr>
                            <th>Matchup</th>
                            <th>Favored</th>
                            <th>Model Pick</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for rec in regular_picks:
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
                    <p>Model: margin_xgboost_optimized_with2425 | """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z") + """</p>
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
        description="Daily NBA betting recommendations pipeline",
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
    logger.info("=" * 70)
    logger.info("")
    
    # Define file paths
    data_file = f"data/games_future_{date_str}_norm.jsonl"
    predictions_file = f"predictions/predictions_{date_str}.csv"
    
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
    
    # STEP 2: Generate predictions
    cmd = [
        sys.executable,
        "predict_margin.py",
        "--data", data_file,
        "--model", "artifacts/margin_xgboost_optimized_with2425",
        "--output", predictions_file
    ]
    
    success = run_step("Generate Predictions", cmd)
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
        preview_file = Path("predictions") / f"email_preview_{date_str}.html"
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
            backup_file = Path("predictions") / f"email_backup_{date_str}.html"
            backup_file.write_text(html_content, encoding='utf-8')
            logger.info(f"Email content saved to: {backup_file}")
            sys.exit(1)


if __name__ == "__main__":
    main()

