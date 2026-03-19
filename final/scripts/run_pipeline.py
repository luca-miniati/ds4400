#!/usr/bin/env python3
"""CLI entry point for IRC to PHH conversion pipeline."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Convert IRC poker hand history to PHH (PokerKit) format."
    )
    parser.add_argument(
        "irc_root",
        nargs="?",
        default="data/IRCdata",
        help="Root directory containing IRC data (default: data/IRCdata)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/phh",
        help="Output directory; creates json/ and phh/ subdirs (default: data/phh)",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit number of hands to process (for testing)",
    )
    args = parser.parse_args()

    irc_root = Path(args.irc_root)
    if not irc_root.exists():
        print(f"Error: IRC root {irc_root} does not exist")
        return 1

    parsed, converted, failed = run_pipeline(
        irc_root,
        args.output,
        limit=args.limit,
        verbose=True,
    )

    print(f"\nSummary: {parsed} parsed, {converted} converted to PHH, {failed} failed")
    return 0 if parsed > 0 else 1  # Success if we parsed anything (JSON is always exported)


if __name__ == "__main__":
    exit(main())
