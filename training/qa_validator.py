#!/usr/bin/env python3
"""
Training data QA validator.

Validates JSONL training files produced for Gemini and OpenAI formats.
Checks both input (user content) and output (model/assistant content), for:
- Structure and required fields
- Field counts and tuple lengths
- Value ranges/types and presence/absence rules
- Consistency across compact and verbose schemas

Usage:
  python -m training.qa_validator --path data/training/*.jsonl
  python -m training.qa_validator --path data/training/file.jsonl --max 2000

Exit code is non-zero if hard errors are found (invalid lines).
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# -------------------------------
# Helpers
# -------------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _in_range(x: Any, lo: float, hi: float) -> bool:
    if not _is_number(x):
        return False
    return (x >= lo) and (x <= hi)


def _parse_line_wrapped(line: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse a single JSONL line supporting different wrappers:
    - Gemini: {"contents":[{"role":"user","parts":[{"text":"<json>"}]},{"role":"model","parts":[{"text":"<json>"}]}]}
    - OpenAI: {"messages":[{"role":"user","content":"<json>"},{"role":"assistant","content":"<json>"}]}
    Returns: (user_json, model_json as dict or list)
    """
    try:
        outer = json.loads(line)
    except json.JSONDecodeError:
        return None, None

    # Gemini wrapper
    if isinstance(outer, dict) and 'contents' in outer and isinstance(outer['contents'], list) and len(outer['contents']) == 2:
        try:
            user_part = outer['contents'][0]['parts'][0]['text']
            model_part = outer['contents'][1]['parts'][0]['text']
            user_json = json.loads(user_part)
            model_json = json.loads(model_part)
            return user_json, model_json
        except Exception:
            return None, None

    # OpenAI wrapper
    if isinstance(outer, dict) and 'messages' in outer and isinstance(outer['messages'], list) and len(outer['messages']) == 2:
        try:
            user_part = outer['messages'][0]['content']
            model_part = outer['messages'][1]['content']
            user_json = json.loads(user_part)
            model_json = json.loads(model_part)
            return user_json, model_json
        except Exception:
            return None, None

    # Fallback: maybe already raw user JSON only
    if isinstance(outer, dict) and ('A' in outer or 'away_team' in outer):
        return outer, None

    return None, None


# -------------------------------
# Validation rules
# -------------------------------

@dataclass
class Counts:
    files: int = 0
    lines: int = 0
    ok: int = 0
    input_errors: int = 0
    output_errors: int = 0
    warnings: int = 0
    actor_team_events: int = 0  # count of actor index == -1
    shot_events: int = 0
    shot_zone_missing: int = 0
    details: List[str] = field(default_factory=list)
    # thresholds evaluation
    threshold_failures: List[str] = field(default_factory=list)


COMPACT_REQUIRED_KEYS = {'A', 'H', 'as', 'hs', 'ap', 'hp'}
ALLOWED_EVENTS = {
    'made2','miss2','made3','miss3','mft','xft','d_reb','o_reb','tov',
    's_foul','p_foul','o_foul','sub','timeout','period','jumpball','viol','tech','stl','unknown'
}


def validate_compact_input(doc: Dict[str, Any], counts: Counts, line_no: int) -> None:
    # Required keys present
    missing = [k for k in COMPACT_REQUIRED_KEYS if k not in doc]
    if missing:
        counts.input_errors += 1
        counts.details.append(f"L{line_no}: missing compact keys {missing}")
        return

    # Team abbreviations
    import re as _re
    for k in ('A', 'H'):
        val = doc.get(k)
        if not isinstance(val, str) or not val:
            counts.input_errors += 1
            counts.details.append(f"L{line_no}: '{k}' must be non-empty string")
        else:
            if not _re.match(r'^[A-Z]{2,4}$', val):
                counts.warnings += 1
                counts.details.append(f"L{line_no}: '{k}' '{val}' not a standard abbrev (2-4 uppercase)")

    # Team stats arrays (len >= 4, reasonable ranges)
    for k in ('as', 'hs'):
        arr = doc.get(k)
        if not isinstance(arr, list) or len(arr) < 10:
            counts.input_errors += 1
            counts.details.append(f"L{line_no}: '{k}' must be list len==10")
            continue
        # OEFF/DEFF/PACE plausible ranges
        if not _in_range(arr[0], 60, 140): counts.warnings += 1
        if not _in_range(arr[1], 60, 140): counts.warnings += 1
        if not _in_range(arr[2], 80, 115): counts.warnings += 1
        # Rates 3PAr, FTr, ORr, DRr, ASTr, TOr in [0,1]
        for idx in (3,4,5,6,7,8):
            if not _in_range(arr[idx], 0, 1):
                counts.warnings += 1
                counts.details.append(f"L{line_no}: '{k}[{idx}]' out of [0,1]")
        # REST_DAYS >= 0
        if not _is_number(arr[9]) or arr[9] < 0:
            counts.warnings += 1

    # Player arrays (ap/hp): list of lists, first item name str
    for team_key in ('ap', 'hp'):
        players = doc.get(team_key)
        if not isinstance(players, list) or len(players) == 0:
            counts.input_errors += 1
            counts.details.append(f"L{line_no}: '{team_key}' must be non-empty list")
            continue
        names_seen = set()
        for i, p in enumerate(players):
            if not (isinstance(p, list) and len(p) == 20 and isinstance(p[0], str) and p[0]):
                counts.input_errors += 1
                counts.details.append(f"L{line_no}: '{team_key}[{i}]' invalid player array (len must be 20)")
                continue
            name = p[0]
            if name in names_seen:
                counts.warnings += 1
                counts.details.append(f"L{line_no}: duplicate player name in {team_key}: {name}")
            names_seen.add(name)
            # Types and ranges
            mpg = p[1]
            usage = p[2]
            if not (_is_number(mpg) and mpg >= 0):
                counts.warnings += 1
            if not _in_range(usage, 0, 1):
                counts.warnings += 1
            # Percentages pairs all in [0,1]
            for idx in range(8, 19):
                if not _in_range(p[idx], 0, 1):
                    counts.warnings += 1
            fouls = p[19]
            if not (_is_number(fouls) and 0 <= int(fouls) <= 6):
                counts.warnings += 1
        # Optional ap_count/hp_count consistency
        meta_key = 'ap_count' if team_key == 'ap' else 'hp_count'
        if meta_key in doc and isinstance(doc.get(meta_key), int):
            if doc.get(meta_key) != len(players):
                counts.warnings += 1
                counts.details.append(f"L{line_no}: {meta_key} != len({team_key})")

    # Optional: lineup list L
    if 'L' in doc:
        L = doc.get('L')
        if not isinstance(L, list):
            counts.input_errors += 1
            counts.details.append(f"L{line_no}: 'L' must be list of lineups")
        else:
            for i, lu in enumerate(L):
                if not (isinstance(lu, dict) and 'A' in lu and 'H' in lu):
                    counts.input_errors += 1
                    counts.details.append(f"L{line_no}: L[{i}] invalid lineup format")
                else:
                    if not (isinstance(lu['A'], list) and isinstance(lu['H'], list) and len(lu['A']) == 5 and len(lu['H']) == 5):
                        counts.input_errors += 1
                        counts.details.append(f"L{line_no}: L[{i}] must have 5 A + 5 H indices")

    # Plays array p (when present in remaining_plays mode)
    if 'p' in doc and doc['p'] is not None:
        p_arr = doc.get('p')
        if not isinstance(p_arr, list):
            counts.input_errors += 1
            counts.details.append(f"L{line_no}: 'p' must be list of play tuples")
        else:
            for i, t in enumerate(p_arr):
                if not (isinstance(t, list) and len(t) == 9):
                    counts.input_errors += 1
                    counts.details.append(f"L{line_no}: p[{i}] must be 9-element tuple list")
                    continue
                q, ts, score, margin, actor, actor_fouls, ev, shot_zone, lineup_id = t
                if not (_is_number(q) and 1 <= int(q) <= 10):
                    counts.warnings += 1
                if not (_is_number(ts) and 0 <= int(ts) <= 720):
                    counts.warnings += 1
                if not (isinstance(score, list) and len(score) == 2 and all(_is_number(s) for s in score)):
                    counts.input_errors += 1
                    counts.details.append(f"L{line_no}: p[{i}].score invalid")
                if not isinstance(actor, list) or len(actor) != 2 or actor[0] not in ('A','H') or not isinstance(actor[1], int):
                    counts.input_errors += 1
                    counts.details.append(f"L{line_no}: p[{i}].actor invalid")
                if not isinstance(actor_fouls, int) or actor_fouls < 0:
                    counts.warnings += 1
                if not isinstance(ev, str) or ev not in ALLOWED_EVENTS:
                    counts.input_errors += 1
                    counts.details.append(f"L{line_no}: p[{i}].event_code '{ev}' invalid")
                if ev in ('made2','miss2','made3','miss3'):
                    counts.shot_events += 1
                    # shot_zone should be one of rim/mid/c3/nc3
                    if shot_zone not in ('rim','mid','c3','nc3'):
                        counts.shot_zone_missing += 1
                else:
                    if shot_zone is not None:
                        counts.warnings += 1
                if not isinstance(lineup_id, int) or lineup_id < 0:
                    counts.warnings += 1
                # margin consistency
                try:
                    if isinstance(score, list) and len(score) == 2 and _is_number(margin):
                        if int(margin) != int(score[0]) - int(score[1]):
                            counts.warnings += 1
                            counts.details.append(f"L{line_no}: p[{i}].margin != away-home")
                except Exception:
                    pass
            # lineup_id bounds when L present
            if isinstance(doc.get('L'), list):
                L_len = len(doc.get('L'))
                for i, t in enumerate(p_arr):
                    if isinstance(t, list) and len(t) == 9:
                        lid = t[8]
                        if isinstance(lid, int) and (lid < 0 or lid >= L_len):
                            counts.warnings += 1
                            counts.details.append(f"L{line_no}: p[{i}].lineup_id out of bounds (0..{L_len-1})")


def validate_verbose_input(doc: Dict[str, Any], counts: Counts, line_no: int) -> None:
    # Minimal checks for verbose schema
    away = doc.get('away_team', {})
    home = doc.get('home_team', {})
    if 'name' not in away or 'name' not in home:
        counts.input_errors += 1
        counts.details.append(f"L{line_no}: verbose must contain away_team.name and home_team.name")


def validate_output(model: Any, counts: Counts, line_no: int) -> None:
    if model is None:
        return
    # first_N_plays (compact): {"y": [ [9-element], ... ]}
    if isinstance(model, dict) and 'y' in model and isinstance(model['y'], list) and model['y'] and isinstance(model['y'][0], list):
        for i, t in enumerate(model['y']):
            if not (isinstance(t, list) and len(t) == 9):
                counts.output_errors += 1
                counts.details.append(f"L{line_no}: output.y[{i}] must be 9-element tuple list")
                continue
            _, _, _, _, actor, _, ev, shot_zone, _ = t
            if isinstance(actor, list) and len(actor) == 2 and actor[1] == -1:
                counts.actor_team_events += 1
            if ev in ('made2','miss2','made3','miss3'):
                counts.shot_events += 1
                if shot_zone not in ('rim','mid','c3','nc3'):
                    counts.shot_zone_missing += 1
        return

    # Compact remaining_plays: {"y": [q, ts, [a,h], margin, actor, actor_fouls, event, shot_zone, lineup_id]}
    if isinstance(model, dict) and 'y' in model:
        t = model['y']
        if not (isinstance(t, list) and len(t) == 9):
            counts.output_errors += 1
            counts.details.append(f"L{line_no}: output 'y' must be 9-element tuple list")
            return
        q, ts, score, margin, actor, actor_fouls, ev, shot_zone, lineup_id = t
        if not (_is_number(q) and 1 <= int(q) <= 10):
            counts.output_errors += 1
            counts.details.append(f"L{line_no}: output.q invalid")
        if not (_is_number(ts) and 0 <= int(ts) <= 720):
            counts.output_errors += 1
            counts.details.append(f"L{line_no}: output.ts invalid")
        if not (isinstance(score, list) and len(score) == 2 and all(_is_number(s) for s in score)):
            counts.output_errors += 1
            counts.details.append(f"L{line_no}: output.score invalid")
        if not (isinstance(actor, list) and len(actor) == 2 and actor[0] in ('A','H') and isinstance(actor[1], int)):
            counts.output_errors += 1
            counts.details.append(f"L{line_no}: output.actor invalid")
        else:
            if actor[1] == -1:
                counts.actor_team_events += 1
        if not isinstance(actor_fouls, int) or actor_fouls < 0:
            counts.warnings += 1
        # Shot-zone presence logic
        if not isinstance(ev, str) or ev not in ALLOWED_EVENTS:
            counts.output_errors += 1
            counts.details.append(f"L{line_no}: output.event_code '{ev}' invalid")
        if ev in ('made2','miss2','made3','miss3'):
            counts.shot_events += 1
            if shot_zone not in ('rim','mid','c3','nc3'):
                counts.shot_zone_missing += 1
        else:
            if shot_zone is not None:
                counts.warnings += 1
        if not isinstance(lineup_id, int) or lineup_id < 0:
            counts.warnings += 1
        return

    # Verbose output: simple sanity check (array of plays or single play dict)
    if isinstance(model, list) or isinstance(model, dict):
        # Light validation only
        return

    counts.output_errors += 1
    counts.details.append(f"L{line_no}: unrecognized output format")


def validate_line(line: str, counts: Counts, line_no: int) -> None:
    user_json, model_json = _parse_line_wrapped(line)
    if user_json is None:
        counts.input_errors += 1
        counts.details.append(f"L{line_no}: could not parse user content")
        return

    # Compact vs verbose input
    if isinstance(user_json, dict) and 'A' in user_json and 'H' in user_json:
        validate_compact_input(user_json, counts, line_no)
    elif isinstance(user_json, dict) and 'away_team' in user_json and 'home_team' in user_json:
        validate_verbose_input(user_json, counts, line_no)
    else:
        counts.input_errors += 1
        counts.details.append(f"L{line_no}: input schema not recognized")

    # Output validation
    validate_output(model_json, counts, line_no)

    counts.ok += 1


def scan_file(path: str, max_lines: Optional[int]) -> Counts:
    counts = Counts()
    counts.files = 1
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if max_lines and i > max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                counts.lines += 1
                validate_line(line, counts, i)
    except FileNotFoundError:
        print(f"File not found: {path}")
    return counts


def aggregate_totals(results: List[Counts]) -> Counts:
    total = Counts()
    total.files = len(results)
    for c in results:
        total.lines += c.lines
        total.ok += c.ok
        total.input_errors += c.input_errors
        total.output_errors += c.output_errors
        total.warnings += c.warnings
        total.actor_team_events += c.actor_team_events
        total.shot_events += c.shot_events
        total.shot_zone_missing += c.shot_zone_missing
        total.details.extend(c.details)
    return total


def print_summary(path_list: List[str], total: Counts) -> None:
    print("QA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Files: {len(path_list)}")
    print(f"Lines: {total.lines}")
    print(f"OK: {total.ok}")
    print(f"Input errors: {total.input_errors}")
    print(f"Output errors: {total.output_errors}")
    print(f"Warnings: {total.warnings}")
    if total.shot_events > 0:
        pct = 100.0 * (total.shot_zone_missing / max(1, total.shot_events))
        print(f"Shot events: {total.shot_events} | Shot-zone missing: {total.shot_zone_missing} ({pct:.1f}%)")
    if total.ok > 0:
        print(f"Actor team-events (-1 index) rate: {100.0 * total.actor_team_events / total.ok:.1f}%")
    if total.threshold_failures:
        print("\nThreshold failures:")
        for t in total.threshold_failures:
            print(f"  - {t}")
    if total.details:
        print("\nFirst 50 issues:")
        for msg in total.details[:50]:
            print(f"  - {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate training JSONL files")
    parser.add_argument('--path', required=True, nargs='+', help='Glob(s) or file path(s)')
    parser.add_argument('--max', type=int, default=None, help='Max lines per file to scan')
    # thresholds
    parser.add_argument('--max-team-event-rate', type=float, default=None, help='Fail if team-event actor rate (%) exceeds this')
    parser.add_argument('--max-shotzone-missing', type=float, default=None, help='Fail if shot-zone missing rate (%) exceeds this')
    parser.add_argument('--max-error-rate', type=float, default=None, help='Fail if (input+output) error rate (%) exceeds this')
    args = parser.parse_args()

    # Expand globs
    paths: List[str] = []
    for p in args.path:
        if any(ch in p for ch in ['*', '?', '[']):
            paths.extend(glob.glob(p))
        else:
            paths.append(p)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No files found to validate.")
        return 1

    results: List[Counts] = []
    for path in paths:
        print(f"Validating: {path}")
        results.append(scan_file(path, args.max))

    total = aggregate_totals(results)
    # thresholds
    hard_fail = (total.input_errors > 0 or total.output_errors > 0)
    lines = max(1, total.lines)
    ok = max(1, total.ok)
    # team-event rate
    if args.max_team_event_rate is not None:
        rate = 100.0 * total.actor_team_events / ok
        if rate > args.max_team_event_rate:
            total.threshold_failures.append(f"team-event rate {rate:.2f}% > {args.max_team_event_rate:.2f}%")
            hard_fail = True
    # shot-zone missing
    if args.max_shotzone_missing is not None and total.shot_events > 0:
        miss_rate = 100.0 * total.shot_zone_missing / total.shot_events
        if miss_rate > args.max_shotzone_missing:
            total.threshold_failures.append(f"shot-zone missing {miss_rate:.2f}% > {args.max_shotzone_missing:.2f}%")
            hard_fail = True
    # error rate
    if args.max_error_rate is not None:
        err_rate = 100.0 * (total.input_errors + total.output_errors) / lines
        if err_rate > args.max_error_rate:
            total.threshold_failures.append(f"error rate {err_rate:.2f}% > {args.max_error_rate:.2f}%")
            hard_fail = True

    print_summary(paths, total)

    return 0 if not hard_fail else 2


if __name__ == '__main__':
    sys.exit(main())


