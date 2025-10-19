#!/usr/bin/env python3
"""
Streaming converter: OpenAI/Together chat-completions JSONL → Gemini GenerateContent JSONL

Input line schema (OpenAI/Together fine-tune):
  {"messages": [
      {"role": "user", "content": "{...context json...}"},
      {"role": "assistant", "content": "{...label json...}"}
  ], ...}

Output line schema (Gemini fine-tune):
  {"contents": [
      {"role": "user",  "parts": [{"text": "{...context json...}"}]},
      {"role": "model", "parts": [{"text": "{...label json...}"}]}
  ]}

Notes:
- Processes line-by-line to keep memory usage low for large files.
- Copies through unknown top-level fields if needed? No — we emit only the minimal Gemini fields per README spec.
- Skips invalid lines with a warning rather than stopping the whole job.
"""

import argparse
import io
import json
import os
import sys


def convert_record_to_gemini(record: dict) -> dict | None:
    """Convert a single OpenAI/Together record to Gemini GenerateContent format.

    Returns None if the record doesn't match expected schema.
    """
    try:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return None

        # Find first user and first assistant messages
        user_msg = None
        assistant_msg = None
        for msg in messages:
            role = msg.get("role")
            if role == "user" and user_msg is None:
                user_msg = msg
            elif role in ("assistant", "model") and assistant_msg is None:
                assistant_msg = msg
            if user_msg is not None and assistant_msg is not None:
                break

        if user_msg is None or assistant_msg is None:
            return None

        user_text = user_msg.get("content")
        assistant_text = assistant_msg.get("content")
        if not isinstance(user_text, str) or not isinstance(assistant_text, str):
            return None

        # Emit minimal Gemini GenerateContent structure
        gemini = {
            "contents": [
                {"role": "user", "parts": [{"text": user_text}]},
                {"role": "model", "parts": [{"text": assistant_text}]}
            ]
        }
        return gemini
    except Exception:
        return None


def stream_convert(input_path: str, output_path: str) -> tuple[int, int]:
    """Convert input JSONL to output JSONL, line-by-line.

    Returns (converted_count, skipped_count).
    """
    converted = 0
    skipped = 0

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with io.open(input_path, "r", encoding="utf-8") as fin, \
         io.open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                skipped += 1
                if skipped <= 10:
                    print(f"Warning: Skipping line {line_num}: invalid JSON", file=sys.stderr)
                continue

            gem = convert_record_to_gemini(rec)
            if gem is None:
                skipped += 1
                if skipped <= 10:
                    print(f"Warning: Skipping line {line_num}: unexpected schema (no user/assistant messages)", file=sys.stderr)
                continue

            fout.write(json.dumps(gem, separators=(",", ":")) + "\n")
            converted += 1

    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenAI/Together JSONL to Gemini GenerateContent JSONL (streaming)")
    parser.add_argument("--input", required=True, help="Path to input JSONL (OpenAI/Together messages format)")
    parser.add_argument("--output", help="Path to output JSONL (Gemini GenerateContent format)")
    args = parser.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        base, _ = os.path.splitext(os.path.basename(in_path))
        out_path = os.path.join(os.path.dirname(in_path), f"{base}_gemini.jsonl")

    converted, skipped = stream_convert(in_path, out_path)
    print(f"Converted: {converted} | Skipped: {skipped}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()


