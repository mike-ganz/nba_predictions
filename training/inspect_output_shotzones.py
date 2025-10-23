#!/usr/bin/env python3
import sys
import json

SHOT_EVENTS = {"made2","miss2","made3","miss3"}

def iter_examples(path):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                outer = json.loads(line)
            except Exception:
                continue
            # unwrap Gemini/OpenAI wrappers
            model_json = None
            if isinstance(outer, dict) and 'contents' in outer:
                try:
                    mtxt = outer['contents'][1]['parts'][0]['text']
                    model_json = json.loads(mtxt)
                except Exception:
                    # sometimes model content is a raw tuple list
                    try:
                        model_json = json.loads(outer['contents'][1]['parts'][0]['text'])
                    except Exception:
                        model_json = None
            elif isinstance(outer, dict) and 'messages' in outer:
                try:
                    mtxt = outer['messages'][1]['content']
                    model_json = json.loads(mtxt)
                except Exception:
                    model_json = None
            else:
                model_json = outer if isinstance(outer, dict) else None
            yield i, model_json

def main():
    if len(sys.argv) < 2:
        print("Usage: python training/inspect_output_shotzones.py <file.jsonl> [max_samples=10]")
        return 1
    path = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    total_shots = 0
    missing_shots = 0
    samples_missing = []
    samples_present = []

    for line_no, model in iter_examples(path):
        if not isinstance(model, dict) or 'y' not in model:
            # first_N_plays may be wrapped as {"y": [...]}; handle below
            if isinstance(model, list) and model and isinstance(model[0], list):
                # unexpected shape; skip
                continue
            continue
        y = model['y']
        if isinstance(y, list) and y and isinstance(y[0], list):
            # first_N_plays
            for t in y:
                if not (isinstance(t, list) and len(t) >= 9):
                    continue
                ev = t[6]
                sz = t[7]
                if ev in SHOT_EVENTS:
                    total_shots += 1
                    if sz not in ("rim","mid","c3","nc3"):
                        missing_shots += 1
                        if len(samples_missing) < max_samples:
                            samples_missing.append((line_no, t))
                    else:
                        if len(samples_present) < max_samples:
                            samples_present.append((line_no, t))
        else:
            # remaining_plays single tuple
            if not (isinstance(y, list) and len(y) >= 9):
                continue
            ev = y[6]
            sz = y[7]
            if ev in SHOT_EVENTS:
                total_shots += 1
                if sz not in ("rim","mid","c3","nc3"):
                    missing_shots += 1
                    if len(samples_missing) < max_samples:
                        samples_missing.append((line_no, y))
                else:
                    if len(samples_present) < max_samples:
                        samples_present.append((line_no, y))

    print(f"Total output shot plays: {total_shots}")
    print(f"Missing shot_zone: {missing_shots} ({(100.0*missing_shots/max(1,total_shots)):.2f}%)")
    print("\nExamples with missing shot_zone:")
    for ln, tup in samples_missing:
        print(f"  L{ln}: {tup}")
    print("\nExamples with present shot_zone:")
    for ln, tup in samples_present:
        print(f"  L{ln}: {tup}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())


