#!/usr/bin/env python3
"""Extract ML features from PHH files. Output: CSV or Parquet with decision points and 28 features."""

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract fold/call/raise decision points and 28 features from PHH hand histories."
    )
    ap.add_argument(
        "phh_dir",
        type=Path,
        nargs="?",
        default=REPO / "data" / "phh" / "phh",
        help="Directory containing hands_*.phhs files (default: data/phh/phh)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO / "data" / "out" / "decisions.csv",
        help="Output path (default: data/out/decisions.csv)",
    )
    ap.add_argument(
        "-j",
        "--json-dir",
        type=Path,
        default=REPO / "data" / "phh" / "json",
        help="JSON directory for hole cards at showdown (default: data/phh/json)",
    )
    ap.add_argument(
        "--no-json",
        action="store_true",
        help="Do not load hole cards from JSON",
    )
    ap.add_argument(
        "-l",
        "--limit-hands",
        type=int,
        default=None,
        help="Limit number of hands to process",
    )
    ap.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Limit number of PHH files to process",
    )
    ap.add_argument(
        "-b",
        "--bb",
        type=int,
        default=10,
        help="Big blind size for pot/stack normalization (default: 10)",
    )
    args = ap.parse_args()

    sys_path = REPO
    if str(sys_path) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(sys_path))

    from src.feature_extraction import run_extraction

    json_dir = None if args.no_json else args.json_dir
    n_decisions, n_hands = run_extraction(
        args.phh_dir,
        args.output,
        json_dir=json_dir,
        bb=args.bb,
        limit_hands=args.limit_hands,
        limit_files=args.limit_files,
    )

    print(f"Wrote {n_decisions} decision points from ~{n_hands} hands to {args.output}")


if __name__ == "__main__":
    main()
