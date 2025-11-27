"""Prepare context-enhanced data for the Context model.

This script normalizes game data and adds league context features
required by the Context model.
"""
import argparse
from league_normalizer import normalize_game_jsonl


def main():
    parser = argparse.ArgumentParser(description="Prepare context-enhanced data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()
    
    normalize_game_jsonl(
        args.input,
        args.output,
        method="center",
        include_context=True
    )
    print(f"Context data prepared: {args.output}")


if __name__ == "__main__":
    main()

