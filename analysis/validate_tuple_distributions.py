#!/usr/bin/env python3
"""
Validate distributions for compact 9-element play tuples in JSONL training data.

Checks both input context plays (p) and assistant labels (y) for:
- tuple length consistency (expect 9)
- quarter, time_seconds ranges
- score arrays, margin stats
- actor team distribution and index resolution rate
- actor_fouls distribution (zero/non-zero, max)
- event code distribution
- shot_zone distribution
- lineup_id distribution and validity vs context L

Usage:
  python analysis/validate_tuple_distributions.py --file data/training/your_file.jsonl --max-lines 200000
"""

import argparse
import json
from collections import Counter, defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate distributions for compact 9-tuple plays (p and y)")
    parser.add_argument("--file", required=True, help="Path to compact JSONL file")
    parser.add_argument("--max-lines", type=int, default=200000, help="Max lines to process")
    return parser.parse_args()


def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def as_str(val):
    if val is None:
        return "None"
    return str(val)


def update_tuple_stats(t, L, agg, allow_new_lineup: bool = False):
    # Expect t: [q, tsec, [a,h], margin, actor, actor_fouls, event, zone, lineup_id]
    agg['len'][len(t)] += 1
    if len(t) != 9:
        return
    q = t[0]; ts = t[1]; sc = t[2]; mar = t[3]; actor = t[4]; af = t[5]; ev = t[6]; zone = t[7]; lid = t[8]

    # quarter
    try:
        qv = int(q)
        agg['quarter'][qv] += 1
        agg['q_min'] = qv if agg['q_min'] is None else min(agg['q_min'], qv)
        agg['q_max'] = qv if agg['q_max'] is None else max(agg['q_max'], qv)
    except Exception:
        agg['quarter'][as_str(q)] += 1

    # time_seconds
    try:
        ts_int = int(ts)
        agg['t_min'] = ts_int if agg['t_min'] is None else min(agg['t_min'], ts_int)
        agg['t_max'] = ts_int if agg['t_max'] is None else max(agg['t_max'], ts_int)
    except Exception:
        pass

    # score and margin
    if isinstance(sc, list) and len(sc) == 2:
        try:
            agg['away_sum'] += int(sc[0]); agg['away_cnt'] += 1
            agg['home_sum'] += int(sc[1]); agg['home_cnt'] += 1
        except Exception:
            pass
    try:
        mar_int = int(mar)
        agg['margin_min'] = mar_int if agg['margin_min'] is None else min(agg['margin_min'], mar_int)
        agg['margin_max'] = mar_int if agg['margin_max'] is None else max(agg['margin_max'], mar_int)
    except Exception:
        pass

    # actor team and index
    if isinstance(actor, list) and actor:
        agg['actor_team'][as_str(actor[0])] += 1
        if len(actor) > 1 and isinstance(actor[1], int):
            if actor[1] >= 0:
                agg['actor_idx_ge0'] += 1
            else:
                agg['actor_idx_neg1'] += 1
        else:
            agg['actor_idx_neg1'] += 1
    else:
        agg['actor_team'][as_str(actor)] += 1
        agg['actor_idx_neg1'] += 1

    # actor_fouls
    try:
        afi = int(af)
        agg['af_vals'][afi] += 1
        agg['af_zero'] += 1 if afi == 0 else 0
        agg['af_nonzero'] += 1 if afi != 0 else 0
        agg['af_max'] = afi if agg['af_max'] is None else max(agg['af_max'], afi)
    except Exception:
        agg['af_vals'][as_str(af)] += 1

    # event
    agg['events'][as_str(ev)] += 1
    # shot zone
    agg['zones'][as_str(zone)] += 1

    # lineup_id
    try:
        lid_int = int(lid)
        if lid_int == 0:
            agg['lid_zero'] += 1
        else:
            agg['lid_nonzero'] += 1
        # Determine validity bounds
        L_len = len(L) if isinstance(L, list) else -1
        if allow_new_lineup and L_len >= 0 and lid_int == L_len:
            # Treat as explicit "new lineup" signal, not invalid
            agg['lid_new'] += 1
        else:
            if not isinstance(L, list) or lid_int < 0 or lid_int >= L_len:
                agg['lid_invalid'] += 1
        agg['lid_vals'][lid_int] += 1
    except Exception:
        agg['lid_invalid'] += 1
        agg['lid_vals'][as_str(lid)] += 1


def new_agg():
    return {
        'len': Counter(),
        'quarter': Counter(), 'q_min': None, 'q_max': None,
        't_min': None, 't_max': None,
        'away_sum': 0, 'away_cnt': 0, 'home_sum': 0, 'home_cnt': 0,
        'margin_min': None, 'margin_max': None,
        'actor_team': Counter(), 'actor_idx_ge0': 0, 'actor_idx_neg1': 0,
        'af_vals': Counter(), 'af_zero': 0, 'af_nonzero': 0, 'af_max': None,
        'events': Counter(), 'zones': Counter(),
        'lid_vals': Counter(), 'lid_zero': 0, 'lid_nonzero': 0, 'lid_invalid': 0, 'lid_new': 0,
        'L_len_min': None, 'L_len_max': None,
        'total': 0
    }


def finalize(agg):
    out = {}
    out['tuple_lengths'] = dict(agg['len'])
    out['quarter'] = {'min': agg['q_min'], 'max': agg['q_max'], 'counts': dict(agg['quarter'])}
    out['time_seconds'] = {'min': agg['t_min'], 'max': agg['t_max']}
    out['score_means'] = {
        'away_mean': (agg['away_sum'] / agg['away_cnt']) if agg['away_cnt'] else None,
        'home_mean': (agg['home_sum'] / agg['home_cnt']) if agg['home_cnt'] else None
    }
    out['margin'] = {'min': agg['margin_min'], 'max': agg['margin_max']}
    total_idx = (agg['actor_idx_ge0'] + agg['actor_idx_neg1'])
    out['actor'] = {
        'team_counts': dict(agg['actor_team']),
        'idx_ge0_pct': (100.0 * agg['actor_idx_ge0'] / total_idx) if total_idx else 0.0,
        'idx_neg1': agg['actor_idx_neg1']
    }
    out['actor_fouls'] = {
        'nonzero_pct': (100.0 * agg['af_nonzero'] / (agg['af_zero'] + agg['af_nonzero'])) if (agg['af_zero'] + agg['af_nonzero']) else 0.0,
        'max': agg['af_max'],
        'top_vals': dict(agg['af_vals'].most_common(10))
    }
    out['events_top'] = dict(agg['events'].most_common(20))
    out['zones_counts'] = dict(agg['zones'])
    lid_total = (agg['lid_zero'] + agg['lid_nonzero'])
    out['lineup_ids'] = {
        'zero_pct': (100.0 * agg['lid_zero'] / lid_total) if lid_total else 0.0,
        'invalid_count': agg['lid_invalid'],
        'new_lineup_count': agg['lid_new'],
        'unique_ids': len(agg['lid_vals']),
        'top_ids': dict(agg['lid_vals'].most_common(15))
    }
    out['L_len'] = {'min': agg['L_len_min'], 'max': agg['L_len_max']}
    out['total_tuples'] = agg['total']
    return out


def main():
    args = parse_args()
    infile = args.file
    max_lines = args.max_lines

    p_agg = new_agg()
    y_agg = new_agg()

    records = 0
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            if records >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            rec = safe_json_load(line)
            if not isinstance(rec, dict):
                continue
            records += 1
            msgs = rec.get('messages') or []
            if not isinstance(msgs, list) or len(msgs) < 2:
                continue
            ctx = safe_json_load(msgs[0].get('content', '{}')) or {}
            lab = safe_json_load(msgs[1].get('content', '{}')) or {}

            L = ctx.get('L') or []
            if isinstance(L, list):
                p_agg['L_len_min'] = len(L) if p_agg['L_len_min'] is None else min(p_agg['L_len_min'], len(L))
                p_agg['L_len_max'] = len(L) if p_agg['L_len_max'] is None else max(p_agg['L_len_max'], len(L))
                y_agg['L_len_min'] = len(L) if y_agg['L_len_min'] is None else min(y_agg['L_len_min'], len(L))
                y_agg['L_len_max'] = len(L) if y_agg['L_len_max'] is None else max(y_agg['L_len_max'], len(L))

            plays = ctx.get('p') or []
            for t in plays:
                if isinstance(t, list):
                    update_tuple_stats(t, L, p_agg, allow_new_lineup=False)
                    p_agg['total'] += 1

            y = lab.get('y', lab)
            if isinstance(y, list) and y and isinstance(y[0], list):
                for t in y:
                    update_tuple_stats(t, L, y_agg, allow_new_lineup=True)
                    y_agg['total'] += 1
            elif isinstance(y, list):
                update_tuple_stats(y, L, y_agg, allow_new_lineup=True)
                y_agg['total'] += 1

    result = {
        'file': infile,
        'records': records,
        'input_p': finalize(p_agg),
        'output_y': finalize(y_agg)
    }

    # Quick Q1 foul presence check for first_N_plays-style files
    try:
        if 'first_N_plays' in infile:
            q1_foul_rows = 0
            total_q1_rows = 0
            with open(infile, 'r', encoding='utf-8') as ff:
                for line in ff:
                    rec = safe_json_load(line)
                    if not isinstance(rec, dict):
                        continue
                    msgs = rec.get('messages') or []
                    if not isinstance(msgs, list) or len(msgs) < 2:
                        continue
                    # Try to pick the raw Q1 rows from the label tuples
                    lab = safe_json_load(msgs[1].get('content', '{}')) or {}
                    y = lab.get('y', lab)
                    tuples = []
                    if isinstance(y, list) and y and isinstance(y[0], list):
                        tuples = y
                    elif isinstance(y, list):
                        tuples = [y]
                    for t in tuples:
                        if isinstance(t, list) and len(t) == 9 and int(t[0]) == 1:
                            total_q1_rows += 1
                            ev = str(t[6])
                            if ev in ('s_foul', 'p_foul', 'o_foul'):
                                q1_foul_rows += 1
            result['firstN_q1_foul_presence'] = {
                'q1_foul_rows': q1_foul_rows,
                'q1_total_rows': total_q1_rows,
                'q1_foul_pct': (100.0 * q1_foul_rows / total_q1_rows) if total_q1_rows else 0.0
            }
    except Exception:
        pass

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


