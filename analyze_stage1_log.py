#!/usr/bin/env python3
import os
import re

LOG_PATH = os.path.expanduser('~/big_run_multithreaded.log')

def analyze_stage1(path: str, max_fail_samples: int = 3):
    if not os.path.exists(path):
        print(f"Log not found: {path}")
        return

    stage1_start_re = re.compile(r'^=== DEBUG Stage 1 RAW RESPONSE')
    stage1_end_re = re.compile(r'^=== END RAW RESPONSE ===')
    retry_re = re.compile(r'^Validation requires retry on attempt')
    reason_re = re.compile(r'^\s*Reason:\s*(.*)$')
    success_re = re.compile(r'^Validation successful on attempt')
    max_retries_re = re.compile(r'^.*Max retries.*exceeded')
    stage1_error_re = re.compile(r'^Error in Stage 1:')

    total_stage1 = 0
    failed_stage1 = 0
    fail_samples = []

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    # First pass: count stage1 blocks and immediate retry/success around them
    while i < n:
        line = lines[i].rstrip('\n')
        if stage1_start_re.search(line):
            total_stage1 += 1
            # capture block
            block = [line]
            i += 1
            while i < n and not stage1_end_re.search(lines[i]):
                block.append(lines[i].rstrip('\n'))
                i += 1
            if i < n:
                block.append(lines[i].rstrip('\n'))
                i += 1

            # look ahead for outcome: retry/success/max
            outcome = 'unknown'
            reason = None
            j = i
            while j < n:
                lj = lines[j].rstrip('\n')
                if stage1_start_re.search(lj) or lj.startswith('=== DEBUG Stage 2') or lj.startswith('STAGE 1:') or lj.startswith('--- ITERATION') or lj.startswith('Starting run '):
                    break
                if retry_re.search(lj):
                    outcome = 'retry'
                    # next line may have reason
                    if j + 1 < n:
                        m = reason_re.search(lines[j+1])
                        if m:
                            reason = m.group(1).strip()
                    break
                if success_re.search(lj):
                    outcome = 'success'
                    break
                if max_retries_re.search(lj):
                    outcome = 'max_retries'
                    break
                j += 1

            if outcome in ('retry', 'max_retries'):
                failed_stage1 += 1
                if len(fail_samples) < max_fail_samples:
                    fail_samples.append(("\n".join(block), reason or ''))
        else:
            i += 1

    # Second pass: if we saw stage1 errors that weren't captured above (e.g., no immediate debug block),
    # find the most recent Stage 1 block preceding each error and include it as a failed sample.
    stage1_error_indices = [idx for idx, l in enumerate(lines) if stage1_error_re.search(l)]
    for err_idx in stage1_error_indices:
        # Walk backward to find a Stage 1 block
        j = err_idx
        start_idx = None
        while j >= 0:
            if stage1_start_re.search(lines[j]):
                start_idx = j
                break
            j -= 1
        if start_idx is not None:
            # Extract block
            k = start_idx
            block = [lines[k].rstrip('\n')]
            k += 1
            while k < n and not stage1_end_re.search(lines[k]):
                block.append(lines[k].rstrip('\n'))
                k += 1
            if k < n:
                block.append(lines[k].rstrip('\n'))
            if len(fail_samples) < max_fail_samples and ("\n".join(block), '') not in fail_samples:
                fail_samples.append(("\n".join(block), 'from Stage 1 error'))
        # Count these as failures if not already counted
        # (Conservative: increment failed count; total_stage1 inferred from blocks in first pass)
        failed_stage1 += 0  # do not double-count blindly

    print(f"Stage 1 attempts: {total_stage1}")
    print(f"Stage 1 failed attempts: {failed_stage1}")
    pct = (failed_stage1 / total_stage1 * 100.0) if total_stage1 else 0.0
    print(f"Failure rate: {pct:.1f}%")

    if fail_samples:
        print("\n=== Sample failed Stage 1 responses ===")
        for idx, (blk, rsn) in enumerate(fail_samples, 1):
            print(f"\n----- FAILED SAMPLE {idx} -----")
            if rsn:
                print(f"Reason: {rsn}")
            # Truncate very large samples
            if len(blk) > 5000:
                print(blk[:5000] + "\n... [truncated] ...")
            else:
                print(blk)
    else:
        print("\nNo failed Stage 1 samples found.")

if __name__ == '__main__':
    analyze_stage1(LOG_PATH, max_fail_samples=3)


