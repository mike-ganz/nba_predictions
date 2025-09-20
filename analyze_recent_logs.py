#!/usr/bin/env python3
import os
import re
from datetime import datetime, timedelta

LOG = os.path.expanduser('~/big_run_multithreaded.log')

ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - ')

def read_recent(path: str, minutes: int = 5):
    if not os.path.exists(path):
        print(f"Log not found: {path}")
        return []
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    # Read all (file typically manageable); if large, optimize by seeking tail
    lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    recent = []
    include = False
    for line in lines:
        m = ts_re.match(line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                # Assume log timestamps are UTC; if local time, adjust if needed
                if ts >= cutoff:
                    include = True
                else:
                    include = False
            except Exception:
                pass
        if include:
            recent.append(line.rstrip('\n'))
    return recent

def analyze_recent(lines):
    if not lines:
        print('No recent lines found in the last window.')
        return
    s1_starts = sum(1 for l in lines if 'STAGE 1: Getting initial next_plays' in l)
    s1_json_err = sum(1 for l in lines if 'JSON parse error' in l and 'Stage 1' in l or 'Failed to parse Stage 1' in l)
    s1_missing = sum(1 for l in lines if "Missing 'next_plays' field" in l)
    s1_conv_err = sum(1 for l in lines if 'Compact format conversion error' in l or 'conversion_exception' in l)
    s1_errors = sum(1 for l in lines if l.startswith('Error in Stage 1:'))

    s2_raw = sum(1 for l in lines if '=== DEBUG Stage 2 RAW RESPONSE' in l)
    s1_raw_idxs = [i for i,l in enumerate(lines) if '=== DEBUG Stage 1 RAW RESPONSE' in l]

    print('--- Recent Stage 1 summary (last window) ---')
    print(f'Stage 1 attempts: {s1_starts}')
    print(f'Stage 1 errors (JSON parse): {s1_json_err}')
    print(f"Stage 1 'missing next_plays' reasons: {s1_missing}")
    print(f'Stage 1 compact conversion errors: {s1_conv_err}')
    print(f'Stage 1 terminal error lines: {s1_errors}')
    print(f'Stage 2 raw responses seen: {s2_raw}')

    # Print up to 3 Stage 1 raw response blocks
    if s1_raw_idxs:
        print('\n--- Sample Stage 1 raw responses ---')
        printed = 0
        for idx in s1_raw_idxs[:3]:
            print('\n=== SAMPLE STAGE 1 RAW RESPONSE ===')
            j = idx
            while j < len(lines):
                print(lines[j])
                if '=== END RAW RESPONSE ===' in lines[j]:
                    break
                j += 1
            printed += 1
        if printed == 0:
            print('(no complete Stage 1 raw blocks found)')
    else:
        print('\n(no Stage 1 raw responses captured in this window)')

if __name__ == '__main__':
    recent = read_recent(LOG, minutes=5)
    analyze_recent(recent)


