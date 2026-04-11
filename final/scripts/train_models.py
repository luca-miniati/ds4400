#!/usr/bin/env python3
"""Train models: LR, GBT, Multinomial LR, Random Forest, and RNN."""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    y = np.array(df[LABEL_COL].cat.codes.values)
    return X, y


# ── RNN (LSTM) model ──────────────────────────────────────────────────

class PokerLSTM(nn.Module):
    """LSTM classifier for tabular poker decision data.

    Each sample's 29 features are treated as a length-29 sequence with 1
    feature per timestep. The feature ordering (draw → board → pot → action
    → game state) provides a natural progression that the LSTM can exploit.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 29, 1)
        _, (h_n, _) = self.lstm(x)       # h_n: (num_layers, batch, hidden)
        out = self.dropout(h_n[-1])       # last layer hidden state
        return self.fc(out)               # (batch, num_classes)


def _train_rnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
    hidden_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 512,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 5,
) -> np.ndarray:
    """Train an LSTM and return predictions on X_test."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    n_features = X_train.shape[1]

    # Reshape to (N, seq_len=n_features, input_size=1) for LSTM
    def to_tensors(X, y=None):
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, 29, 1)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.long)
            return Xt, yt
        return Xt

    X_tr, y_tr = to_tensors(X_train, y_train)
    X_te, y_te = to_tensors(X_test, y_test)

    # Use 10% of training data as validation for early stopping
    n_val = int(0.1 * len(X_tr))
    indices = torch.randperm(len(X_tr))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = TensorDataset(X_tr[train_idx], y_tr[train_idx])
    val_ds = TensorDataset(X_tr[val_idx], y_tr[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size)

    model = PokerLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model and predict
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train all models (LR, GBT, Multinomial LR, Random Forest, RNN)")
    ap.add_argument(
        "data",
        type=Path,
        nargs="?",
        default=REPO / "data" / "out" / "decisions.csv",
        help="Path to decisions CSV",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=REPO / "data" / "out",
        help="Directory to save models and metrics",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion for test set (default: 0.2)",
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
        help="Write output to file (default: out/train.log)",
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

    print("\n--- Gradient Boosted Trees ---")
    gbt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=args.seed,
    )
    gbt.fit(X_train, y_train)
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

    # ── Chigo's models ─────────────────────────────────────────────────

    print("\n--- Multinomial Logistic Regression (L1 / saga) ---")
    mlr = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=1.0,
        max_iter=2000,
        random_state=args.seed,
    )
    mlr.fit(X_train_scaled, y_train)
    y_pred_mlr = mlr.predict(X_test_scaled)

    f1_macro_mlr = f1_score(y_test, y_pred_mlr, average="macro")
    f1_weighted_mlr = f1_score(y_test, y_pred_mlr, average="weighted")
    print(f"  F1 (macro):    {f1_macro_mlr:.4f}")
    print(f"  F1 (weighted): {f1_weighted_mlr:.4f}")
    print("  Confusion matrix:")
    cm_mlr = confusion_matrix(y_test, y_pred_mlr, labels=[0, 1, 2])
    print(cm_mlr)

    results["multinomial_logistic_regression"] = {
        "f1_macro": float(f1_macro_mlr),
        "f1_weighted": float(f1_weighted_mlr),
        "confusion_matrix": cm_mlr.tolist(),
    }

    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    f1_macro_rf = f1_score(y_test, y_pred_rf, average="macro")
    f1_weighted_rf = f1_score(y_test, y_pred_rf, average="weighted")
    print(f"  F1 (macro):    {f1_macro_rf:.4f}")
    print(f"  F1 (weighted): {f1_weighted_rf:.4f}")
    print("  Confusion matrix:")
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[0, 1, 2])
    print(cm_rf)

    results["random_forest"] = {
        "f1_macro": float(f1_macro_rf),
        "f1_weighted": float(f1_weighted_rf),
        "confusion_matrix": cm_rf.tolist(),
    }

    print("\n--- RNN (LSTM) ---")
    y_pred_rnn = _train_rnn(
        X_train_scaled, y_train, X_test_scaled, y_test, seed=args.seed
    )

    f1_macro_rnn = f1_score(y_test, y_pred_rnn, average="macro")
    f1_weighted_rnn = f1_score(y_test, y_pred_rnn, average="weighted")
    print(f"  F1 (macro):    {f1_macro_rnn:.4f}")
    print(f"  F1 (weighted): {f1_weighted_rnn:.4f}")
    print("  Confusion matrix:")
    cm_rnn = confusion_matrix(y_test, y_pred_rnn, labels=[0, 1, 2])
    print(cm_rnn)

    results["rnn_lstm"] = {
        "f1_macro": float(f1_macro_rnn),
        "f1_weighted": float(f1_weighted_rnn),
        "confusion_matrix": cm_rnn.tolist(),
    }

    # ── Save all metrics & reports ─────────────────────────────────────

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Classification reports ---")
    print("Logistic Regression:")
    print(classification_report(y_test, y_pred_lr, target_names=CLASSES))
    print("Gradient Boosted Trees:")
    print(classification_report(y_test, y_pred_gbt, target_names=CLASSES))
    print("Multinomial Logistic Regression (L1):")
    print(classification_report(y_test, y_pred_mlr, target_names=CLASSES))
    print("Random Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=CLASSES))
    print("RNN (LSTM):")
    print(classification_report(y_test, y_pred_rnn, target_names=CLASSES))

    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
