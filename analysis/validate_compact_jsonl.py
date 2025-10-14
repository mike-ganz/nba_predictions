#!/usr/bin/env python3
"""
Quick validator for compact training JSONL.

Metrics:
- shot_zone coverage on shooting events (made2/miss2/made3/miss3)
- assist rate on made shots (made2/made3 with non-null assister)

Usage:
  python analysis/validate_compact_jsonl.py --file "data/training/your_file.jsonl" --max-lines 100000
"""

import argparse
import json
from typing import Optional


SHOOT_EVENTS = {"made2", "miss2", "made3", "miss3"}
MADE_EVENTS = {"made2", "made3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate compact JSONL shot_zone and assist coverage")
    parser.add_argument("--file", required=True, help="Path to compact JSONL file")
    parser.add_argument("--max-lines", type=int, default=200000, help="Max lines to scan for quick validation")
    return parser.parse_args()


def extract_compact_from_gemini_wrapper(obj: dict) -> Optional[dict]:
    """If the record is a Gemini wrapper, extract the compact JSON from parts[].text."""
    try:
        contents = obj.get("contents")
        if not isinstance(contents, list) or not contents:
            return None
        # Find first part with text that looks like JSON starting with '{'
        for part in contents[0].get("parts", []):
            text = part.get("text")
            if isinstance(text, str) and text.strip().startswith("{"):
                return json.loads(text)
    except Exception:
        return None
    return None


def update_metrics(record: dict, totals: dict) -> None:
    plays = record.get("p") or []
    for play in plays:
        if not isinstance(play, list) or len(play) < 10:
            continue
        event = str(play[6]) if len(play) > 6 else None
        shot_zone = play[7] if len(play) > 7 else None
        assist_by = play[8] if len(play) > 8 else None

        if event in SHOOT_EVENTS:
            totals["shoot_events"] += 1
            if shot_zone is not None:
                totals["shoot_zone_non_null"] += 1

            if event in MADE_EVENTS:
                totals["made_shots"] += 1
                if assist_by is not None:
                    totals["assisted_makes"] += 1


def format_pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{(100.0 * num / den):.1f}%"


def main() -> None:
    args = parse_args()

    totals = {
        "lines": 0,
        "parsed": 0,
        "shoot_events": 0,
        "shoot_zone_non_null": 0,
        "made_shots": 0,
        "assisted_makes": 0,
    }

    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            totals["lines"] += 1
            if totals["lines"] > args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            # Direct compact or Gemini wrapper
            record = obj if ("A" in obj and "H" in obj) else extract_compact_from_gemini_wrapper(obj)
            if not isinstance(record, dict):
                continue

            totals["parsed"] += 1
            update_metrics(record, totals)

    print("Compact JSONL validation summary")
    print("-" * 34)
    print(f"Lines scanned: {totals['lines']}")
    print(f"Records parsed: {totals['parsed']}")
    print(f"Shooting events: {totals['shoot_events']}")
    print(f"shot_zone coverage: {totals['shoot_zone_non_null']} / {totals['shoot_events']} (" +
          f"{format_pct(totals['shoot_zone_non_null'], totals['shoot_events'])})")
    print(f"Made shots: {totals['made_shots']}")
    print(f"Assist rate on makes: {totals['assisted_makes']} / {totals['made_shots']} (" +
          f"{format_pct(totals['assisted_makes'], totals['made_shots'])})")


if __name__ == "__main__":
    main()


