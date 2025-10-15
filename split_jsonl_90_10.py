#!/usr/bin/env python3
"""
Split a JSONL file into train/validation sets (default 90/10).

Usage:
  python split_jsonl_90_10.py --input "C:\\path\\to\\file.jsonl" [--train-ratio 0.9]

Outputs:
  <base>_train.jsonl and <base>_val.jsonl in the same directory.
"""

import os
import argparse
from training.together_client import split_train_validation


def main():
    parser = argparse.ArgumentParser(description="Split JSONL into train/val")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train ratio (default: 0.9)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    train_path, val_path = split_train_validation(input_path, train_ratio=args.train_ratio)
    print(f"Train file: {train_path}")
    print(f"Val file:   {val_path}")


if __name__ == "__main__":
    raise SystemExit(main())


