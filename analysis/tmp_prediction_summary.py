"""Quick summary statistics for prediction JSON output."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(label: str, values: list[float]) -> None:
    count = len(values)
    mean = statistics.fmean(values)
    minimum = min(values)
    maximum = max(values)
    stdev = statistics.pstdev(values) if count > 1 else 0.0
    print(f"{label}: n={count} mean={mean:.4f} std={stdev:.4f} min={minimum:.4f} max={maximum:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prediction JSON output")
    parser.add_argument("predictions", type=Path, help="Path to predictions JSON file")
    args = parser.parse_args()

    data = json.loads(args.predictions.read_text())
    if not data:
        raise SystemExit("No records found in predictions file")

    win_probs = [float(rec["win_prob_home"]) for rec in data]
    exp_margin = [float(rec["expected_margin"]) for rec in data]
    exp_home = [float(rec["expected_home"]) for rec in data]
    exp_away = [float(rec["expected_away"]) for rec in data]

    summarize("win_prob_home", win_probs)
    summarize("expected_margin", exp_margin)
    summarize("expected_home", exp_home)
    summarize("expected_away", exp_away)


if __name__ == "__main__":
    main()


