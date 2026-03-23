#!/usr/bin/env python3
"""Train Luca's models: Logistic Regression and Gradient Boosted Trees."""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "flush_draw",
    "straight_draw",
    "n_suited_board",
    "highest_run_board",
    "board_pair",
    "board_trips",
    "monotone_board",
    "two_tone_board",
    "rainbow_board",
    "highest_board_rank",
    "pot_bb",
    "effective_stack_bb",
    "spr",
    "pot_odds",
    "facing_bet_bb",
    "facing_bet_pct_pot",
    "all_in",
    "commitment_pct",
    "mdf",
    "opp_aggressive_count",
    "opp_passive_count",
    "last_action_1",
    "last_action_2",
    "villain_checkraise",
    "villain_donk",
    "hero_position",
    "villain_position",
    "preflop_aggressor_is_hero",
    "street_index",
]
LABEL_COL = "label"
CLASSES = ["call", "fold", "raise"]


def load_and_prepare(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV and return (X, y)."""
    df = pd.read_csv(data_path)
    # Ensure label order for consistency
    df[LABEL_COL] = pd.Categorical(df[LABEL_COL], categories=CLASSES, ordered=True)
    df = df.dropna(subset=[LABEL_COL])
    # Impute -1 (unknown) in draw/rank features with 0
    for col in ["flush_draw", "straight_draw", "highest_board_rank"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, 0)
    X = df[FEATURE_COLS].fillna(0).values
    y = df[LABEL_COL].cat.codes.values
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Logistic Regression and Gradient Boosted Trees")
    ap.add_argument(
        "data",
        type=Path,
        nargs="?",
        default=REPO / "data" / "ml" / "decisions.csv",
        help="Path to decisions CSV",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=REPO / "data" / "ml" / "models",
        help="Directory to save models and metrics",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test set (default: 0.2)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    ap.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit rows for faster testing",
    )
    ap.add_argument(
        "-L",
        "--log-file",
        type=Path,
        default=None,
        help="Write output to file (default: output-dir/train.log)",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file if args.log_file is not None else output_dir / "train.log"

    @contextmanager
    def log_output():
        with open(log_path, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            try:
                yield
            finally:
                sys.stdout = old_stdout
        print(f"Log written to {log_path}", file=sys.__stdout__)

    with log_output():
        _run_training(args, output_dir)


def _run_training(args, output_dir: Path) -> None:
    import joblib

    print("Loading data...")
    X, y = load_and_prepare(args.data)
    if args.limit:
        X, y = X[: args.limit], y[: args.limit]
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=args.seed,
    )
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    f1_macro = f1_score(y_test, y_pred_lr, average="macro")
    f1_weighted = f1_score(y_test, y_pred_lr, average="weighted")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print("  Confusion matrix:")
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=[0, 1, 2])
    print(cm_lr)

    results["logistic_regression"] = {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm_lr.tolist(),
    }

    import joblib

    joblib.dump(lr, output_dir / "logistic_regression.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    print(f"  Saved to {output_dir / 'logistic_regression.joblib'}")

    # --- Gradient Boosted Trees ---
    print("\n--- Gradient Boosted Trees ---")
    gbt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=args.seed,
    )
    gbt.fit(X_train, y_train)  # GBT often works better without scaling
    y_pred_gbt = gbt.predict(X_test)

    f1_macro_gbt = f1_score(y_test, y_pred_gbt, average="macro")
    f1_weighted_gbt = f1_score(y_test, y_pred_gbt, average="weighted")
    print(f"  F1 (macro):    {f1_macro_gbt:.4f}")
    print(f"  F1 (weighted): {f1_weighted_gbt:.4f}")
    print("  Confusion matrix:")
    cm_gbt = confusion_matrix(y_test, y_pred_gbt, labels=[0, 1, 2])
    print(cm_gbt)

    results["gradient_boosted_trees"] = {
        "f1_macro": float(f1_macro_gbt),
        "f1_weighted": float(f1_weighted_gbt),
        "confusion_matrix": cm_gbt.tolist(),
    }

    joblib.dump(gbt, output_dir / "gradient_boosted_trees.joblib")
    print(f"  Saved to {output_dir / 'gradient_boosted_trees.joblib'}")

    # --- Summary ---
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Classification reports ---")
    print("Logistic Regression:")
    print(classification_report(y_test, y_pred_lr, target_names=CLASSES))
    print("Gradient Boosted Trees:")
    print(classification_report(y_test, y_pred_gbt, target_names=CLASSES))

    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
