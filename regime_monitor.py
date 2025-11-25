"""
Regime Monitor: Real-time detection and alerting for variance regime changes.

This module provides:
1. Daily regime status check
2. Logging of regime history
3. Alerts when regime changes occur
4. Integration with prediction pipeline

Usage:
    python regime_monitor.py                    # Check today's regime
    python regime_monitor.py --date 2025-11-15  # Check specific date
    python regime_monitor.py --history          # Show regime history
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml

from hybrid_model_selector import VarianceRegimeDetector, RegimeConfig


# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "regime_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RegimeStatus:
    """Current regime status with all relevant metrics."""
    date: str
    season: str
    regime: str  # 'unknown', 'normal', 'elevated', 'high' (final regime after hysteresis)
    raw_regime: str  # Raw regime before hysteresis (for consecutive day tracking)
    model_recommended: str  # 'champion' or 'context'
    variance_ratio: Optional[float]
    std_deff: Optional[float]
    games_analyzed: int
    consecutive_high_days: int
    threshold: float
    changed_from_previous: bool
    previous_regime: Optional[str]


class RegimeMonitor:
    """
    Monitors variance regime and provides alerts on changes.
    """
    
    # Optimal parameters from sensitivity analysis
    DEFAULT_CONFIG = RegimeConfig(
        min_games_for_detection=250,
        high_variance_threshold=1.30,
        normal_variance_threshold=1.20,
        consecutive_days_required=5,
    )
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or self.DEFAULT_CONFIG
        self.detector = VarianceRegimeDetector(self.config)
        self.history_file = Path("analysis/regime_history.json")
        self._load_history()
    
    def _load_history(self):
        """Load regime history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self._history = json.load(f)
        else:
            self._history = []
    
    def _save_history(self):
        """Save regime history to file."""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self._history, f, indent=2, default=str)
    
    def _get_history_for_detection(self) -> List[Dict]:
        """Convert history to format expected by detector."""
        return [
            {
                'raw_regime': h.get('raw_regime', h.get('regime')),  # Fall back to regime for old entries
                'final_regime': h.get('regime'),
                'date': h.get('date'),
            }
            for h in self._history[-10:]  # Last 10 days
        ]
    
    def check_regime(self, date: str = None, season: str = '2025-2026') -> RegimeStatus:
        """
        Check the variance regime for a given date.
        
        Args:
            date: Date to check (default: today)
            season: NBA season
            
        Returns:
            RegimeStatus with current regime info
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Get detection history
        detection_history = self._get_history_for_detection()
        
        # Detect regime
        regime, details = self.detector.detect_regime(season, date, detection_history)
        
        # Determine recommended model
        model_recommended = 'context' if regime == 'high' else 'champion'
        
        # Check if regime changed
        previous_regime = self._history[-1]['regime'] if self._history else None
        changed = regime != previous_regime
        
        status = RegimeStatus(
            date=date,
            season=season,
            regime=regime,
            raw_regime=details.get('raw_regime', regime),
            model_recommended=model_recommended,
            variance_ratio=details.get('ratio_deff'),
            std_deff=details.get('std_deff'),
            games_analyzed=details.get('game_count', 0),
            consecutive_high_days=details.get('consecutive_high_days', 0),
            threshold=self.config.high_variance_threshold,
            changed_from_previous=changed,
            previous_regime=previous_regime,
        )
        
        return status
    
    def log_status(self, status: RegimeStatus):
        """Log the current regime status."""
        ratio_str = f"{status.variance_ratio:.3f}" if status.variance_ratio else "N/A"
        
        # Log to file
        if status.changed_from_previous:
            logger.warning(
                f"REGIME CHANGE: {status.previous_regime} -> {status.regime} | "
                f"Model: {status.model_recommended} | "
                f"Ratio: {ratio_str}"
            )
        else:
            logger.info(
                f"Regime: {status.regime} | "
                f"Model: {status.model_recommended} | "
                f"Ratio: {ratio_str} | "
                f"Games: {status.games_analyzed}"
            )
        
        # Update history
        self._history.append(asdict(status))
        self._save_history()
    
    def get_alert_message(self, status: RegimeStatus) -> Optional[str]:
        """
        Generate alert message if regime changed.
        
        Returns:
            Alert message string, or None if no alert needed
        """
        if not status.changed_from_previous:
            return None
        
        ratio_str = f"{status.variance_ratio:.3f}" if status.variance_ratio else "N/A"
        std_str = f"{status.std_deff:.2f}" if status.std_deff else "N/A"
        
        if status.regime == 'high':
            return f"""
🚨 REGIME ALERT: HIGH VARIANCE DETECTED

Date: {status.date}
Previous Regime: {status.previous_regime}
New Regime: {status.regime}

Metrics:
  - Variance Ratio: {ratio_str} (threshold: {status.threshold})
  - Std Dev DEFF: {std_str}
  - Consecutive High Days: {status.consecutive_high_days}
  - Games Analyzed: {status.games_analyzed}

ACTION REQUIRED:
  Switch from CHAMPION model to CONTEXT model for predictions.
  
  Context model performs better during high-variance periods.
"""
        elif status.previous_regime == 'high':
            return f"""
✅ REGIME ALERT: RETURNING TO NORMAL

Date: {status.date}
Previous Regime: {status.previous_regime}
New Regime: {status.regime}

Metrics:
  - Variance Ratio: {ratio_str} (threshold: {status.threshold})
  - Std Dev DEFF: {std_str}

ACTION REQUIRED:
  Switch from CONTEXT model back to CHAMPION model for predictions.
"""
        else:
            return f"""
ℹ️ REGIME CHANGE: {status.previous_regime} -> {status.regime}

Date: {status.date}
Variance Ratio: {ratio_str}

No action required - still using CHAMPION model.
"""
    
    def print_history(self, days: int = 30):
        """Print recent regime history."""
        print("\n" + "="*70)
        print("  REGIME HISTORY (Last {} days)".format(days))
        print("="*70)
        print(f"\n{'Date':<12} {'Regime':<10} {'Model':<10} {'Ratio':<8} {'StdDEFF':<8} {'Games':<6}")
        print("-"*70)
        
        for entry in self._history[-days:]:
            ratio = entry.get('variance_ratio')
            ratio_str = f"{ratio:.3f}" if ratio else "N/A"
            std_deff = entry.get('std_deff')
            std_str = f"{std_deff:.2f}" if std_deff else "N/A"
            
            marker = " *" if entry.get('changed_from_previous') else ""
            
            print(f"{entry['date']:<12} {entry['regime']:<10} {entry['model_recommended']:<10} "
                  f"{ratio_str:<8} {std_str:<8} {entry['games_analyzed']:<6}{marker}")
        
        print("\n* = Regime changed from previous day")
    
    def get_current_recommendation(self) -> Dict:
        """
        Get the current model recommendation.
        
        Returns:
            Dict with 'model', 'regime', 'confidence', and 'reason'
        """
        status = self.check_regime()
        ratio_str = f"{status.variance_ratio:.3f}" if status.variance_ratio else "N/A"
        
        if status.regime == 'unknown':
            return {
                'model': 'champion',
                'regime': 'unknown',
                'confidence': 'low',
                'reason': f'Insufficient data ({status.games_analyzed} games, need {self.config.min_games_for_detection})',
            }
        elif status.regime == 'high':
            return {
                'model': 'context',
                'regime': 'high',
                'confidence': 'high',
                'reason': f'Variance ratio {ratio_str} exceeds threshold {status.threshold} for {status.consecutive_high_days} days',
            }
        else:
            return {
                'model': 'champion',
                'regime': status.regime,
                'confidence': 'high',
                'reason': f'Variance ratio {ratio_str} below threshold {status.threshold}',
            }


def main():
    """Main entry point for regime monitoring."""
    parser = argparse.ArgumentParser(description='Monitor NBA variance regime')
    parser.add_argument('--date', type=str, help='Date to check (YYYY-MM-DD)')
    parser.add_argument('--history', action='store_true', help='Show regime history')
    parser.add_argument('--days', type=int, default=30, help='Days of history to show')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    monitor = RegimeMonitor()
    
    if args.history:
        monitor.print_history(args.days)
        return
    
    # Check regime
    status = monitor.check_regime(args.date)
    
    if not args.quiet:
        ratio_str = f"{status.variance_ratio:.3f}" if status.variance_ratio else "N/A"
        std_str = f"{status.std_deff:.2f}" if status.std_deff else "N/A"
        
        print("\n" + "="*70)
        print("  REGIME STATUS CHECK")
        print("="*70)
        print(f"""
    Date:               {status.date}
    Season:             {status.season}
    
    Current Regime:     {status.regime.upper()}
    Recommended Model:  {status.model_recommended.upper()}
    
    Variance Ratio:     {ratio_str}
    Std Dev DEFF:       {std_str}
    Games Analyzed:     {status.games_analyzed}
    Consecutive High:   {status.consecutive_high_days} days
    Threshold:          {status.threshold}
""")
    
    # Log status
    monitor.log_status(status)
    
    # Check for alerts
    alert = monitor.get_alert_message(status)
    if alert:
        print(alert)
    
    # Print recommendation
    rec = monitor.get_current_recommendation()
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION: Use {rec['model'].upper()} model")
    print(f"  Confidence: {rec['confidence'].upper()}")
    print(f"  Reason: {rec['reason']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

