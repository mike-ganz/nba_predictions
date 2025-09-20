#!/usr/bin/env python3
import os

LOG_PATH = os.path.expanduser('~/big_run_multithreaded.log')

def extract_stage1_blocks(path: str, max_blocks: int = 3):
    if not os.path.exists(path):
        print(f"Log not found: {path}")
        return []

    blocks = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        in_block = False
        current = []
        for line in f:
            if '=== DEBUG Stage 1 RAW RESPONSE' in line:
                in_block = True
                current = [line.rstrip('\n')]
                continue
            if in_block:
                current.append(line.rstrip('\n'))
                if '=== END RAW RESPONSE ===' in line:
                    blocks.append('\n'.join(current))
                    in_block = False
                    if len(blocks) >= max_blocks:
                        break
    return blocks

if __name__ == '__main__':
    blocks = extract_stage1_blocks(LOG_PATH, max_blocks=5)
    if not blocks:
        print('No Stage 1 raw response blocks found.')
    else:
        for i, b in enumerate(blocks, 1):
            print(f"\n===== STAGE 1 SAMPLE {i} =====")
            # Cap very long blocks to keep output manageable
            if len(b) > 4000:
                print(b[:4000] + "\n... [truncated] ...")
            else:
                print(b)
        print("\n-- end of samples --")


