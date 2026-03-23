#!/usr/bin/env python3
"""Print summary statistics for an extracted decisions dataset (CSV or Parquet)."""

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Describe extracted decisions dataset: label distribution, feature stats, missingness."
    )
    ap.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=REPO / "data" / "ml" / "decisions.csv",
        help="Path to decisions CSV or Parquet (default: data/ml/decisions.csv)",
    )
    ap.add_argument(
        "-n",
        "--head",
        type=int,
        default=None,
        help="Show first N rows",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        print("Run extract_features.py first.")
        return

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required. Install with: pip install pandas")
        return

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print(f"Dataset: {path}")
    print(f"Rows: {len(df)}")
    print()

    if "label" in df.columns:
        print("Label distribution:")
        print(df["label"].value_counts())
        print()

    # Feature columns (exclude hand_id, player, street, label)
    meta = {"hand_id", "player", "street", "label"}
    feat_cols = [c for c in df.columns if c not in meta]
    if feat_cols:
        print("Feature summary (numeric):")
        print(df[feat_cols].describe())
        print()
        print("Missing values:")
        missing = df[feat_cols].isna().sum()
        missing = missing[missing > 0]
        if len(missing):
            print(missing)
        else:
            print("None")
        print()
        # -1 is used for unknown draw features
        draw_cols = [c for c in ["flush_draw", "straight_draw"] if c in df.columns]
        if draw_cols:
            unknown_draw = (df[draw_cols] == -1).sum()
            if (unknown_draw > 0).any():
                print("Unknown draw features (-1):")
                print(unknown_draw)
                print()

    if args.head:
        print(f"First {args.head} rows:")
        print(df.head(args.head).to_string())


if __name__ == "__main__":
    main()
