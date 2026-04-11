#!/usr/bin/env python3
"""Generate dataset and model evaluation visualizations, saving to data/out/figures/."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

REPO = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "flush_draw", "straight_draw", "n_suited_board", "highest_run_board",
    "board_pair", "board_trips", "monotone_board", "two_tone_board",
    "rainbow_board", "highest_board_rank", "pot_bb", "effective_stack_bb",
    "spr", "pot_odds", "facing_bet_bb", "facing_bet_pct_pot", "all_in",
    "commitment_pct", "mdf", "opp_aggressive_count", "opp_passive_count",
    "last_action_1", "last_action_2", "villain_checkraise", "villain_donk",
    "hero_position", "villain_position", "preflop_aggressor_is_hero",
    "street_index",
]
LABEL_COL = "label"
CLASSES = ["call", "fold", "raise"]
COLORS = {"call": "#4C72B0", "fold": "#DD8452", "raise": "#55A868"}
CLASS_COLORS = [COLORS[c] for c in CLASSES]


def load_and_prepare(data_path: Path, limit: int | None = None):
    df = pd.read_csv(data_path)
    df[LABEL_COL] = pd.Categorical(df[LABEL_COL], categories=CLASSES, ordered=True)
    df = df.dropna(subset=[LABEL_COL])
    for col in ["flush_draw", "straight_draw", "highest_board_rank"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, 0)
    if limit:
        df = df.iloc[:limit]
    X = df[FEATURE_COLS].fillna(0).values
    y = np.array(df[LABEL_COL].cat.codes.values)
    return df, X, y


# --- Dataset visualizations ---

def plot_label_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df[LABEL_COL].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASSES, counts.values, color=CLASS_COLORS)
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(df) * 0.003,
            f"{v:,}\n({100 * v / len(df):.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    _save(fig, out_dir / "label_distribution.png")


def plot_feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    features_to_plot = [
        ("pot_bb", "Pot (BB)"),
        ("effective_stack_bb", "Effective Stack (BB)"),
        ("spr", "Stack-to-Pot Ratio"),
        ("pot_odds", "Pot Odds"),
        ("facing_bet_pct_pot", "Facing Bet (% Pot)"),
        ("commitment_pct", "Commitment %"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title) in zip(axes.flat, features_to_plot):
        for label in CLASSES:
            vals = df.loc[df[LABEL_COL] == label, col].dropna()
            p99 = vals.quantile(0.99)
            ax.hist(vals.clip(upper=p99), bins=40, alpha=0.55,
                    label=label, color=COLORS[label], density=True)
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("Key Feature Distributions by Label (clipped at 99th percentile)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df[FEATURE_COLS].replace(-1, np.nan).corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Pearson r")
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURE_COLS, fontsize=7)
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir / "correlation_heatmap.png")


def plot_street_label_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    if "street" not in df.columns:
        return
    street_order = ["preflop", "flop", "turn", "river"]
    grp = (
        df.groupby("street")[LABEL_COL]
        .value_counts(normalize=True)
        .unstack()
        .reindex(street_order)[CLASSES]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    grp.plot(kind="bar", stacked=True, ax=ax, color=CLASS_COLORS, width=0.6)
    ax.set_xlabel("Street")
    ax.set_ylabel("Proportion")
    ax.set_title("Label Distribution by Street")
    ax.set_xticklabels(street_order, rotation=0)
    ax.legend(title="Action", loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir / "street_label_distribution.png")


# --- RNN model (same architecture as train_models.py) ---

class PokerLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))


def _train_rnn_for_viz(X_train_s, y_train, X_test_s, seed=42):
    """Train LSTM and return predicted probabilities on test set."""
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_tr = torch.tensor(X_train_s, dtype=torch.float32).unsqueeze(-1)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test_s, dtype=torch.float32).unsqueeze(-1)

    n_val = int(0.1 * len(X_tr))
    idx = torch.randperm(len(X_tr))
    train_ds = TensorDataset(X_tr[idx[n_val:]], y_tr[idx[n_val:]])
    val_ds = TensorDataset(X_tr[idx[:n_val]], y_tr[idx[:n_val]])
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = PokerLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, 31):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        vl = sum(criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)
                 for xb, yb in val_loader) / len(val_ds)
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 5:
                break

    model.load_state_dict(best_state)
    model.eval()
    # Get predictions and probabilities
    test_loader = DataLoader(TensorDataset(X_te, torch.zeros(len(X_te), dtype=torch.long)),
                             batch_size=512)
    all_probs, all_preds = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs)


# --- Model visualizations ---

def _train_models(X: np.ndarray, y: np.ndarray, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    lr.fit(X_train_s, y_train)

    print("  Training Multinomial LR (L1/saga)...")
    mlr = LogisticRegression(solver="saga", l1_ratio=1.0, C=1.0,
                             max_iter=2000, random_state=seed)
    mlr.fit(X_train_s, y_train)

    print("  Training Gradient Boosted Trees...")
    gbt = GradientBoostingClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed,
    )
    gbt.fit(X_train, y_train)

    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    print("  Training RNN (LSTM)...")
    rnn_preds, rnn_probs = _train_rnn_for_viz(X_train_s, y_train, X_test_s, seed=seed)

    return lr, mlr, gbt, rf, rnn_preds, rnn_probs, X_train_s, X_test_s, X_train, X_test, y_test


def plot_roc_curves(lr, mlr, gbt, rf, rnn_probs, X_test_s, X_test, y_test, out_dir: Path) -> None:
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flat
    model_infos = [
        ("Logistic Regression", lr.predict_proba(X_test_s)),
        ("Multinomial LR (L1)", mlr.predict_proba(X_test_s)),
        ("Gradient Boosted Trees", gbt.predict_proba(X_test)),
        ("Random Forest", rf.predict_proba(X_test)),
        ("RNN (LSTM)", rnn_probs),
    ]
    for ax, (name, y_score) in zip(axes_flat, model_infos):
        for i, (cls, color) in enumerate(zip(CLASSES, CLASS_COLORS)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {name}")
        ax.legend(loc="lower right", fontsize=8)
    # Hide unused 6th subplot
    axes_flat[5].set_visible(False)
    fig.suptitle("One-vs-Rest ROC Curves", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir / "roc_curves.png")


def plot_confusion_matrices(lr, mlr, gbt, rf, rnn_preds, X_test_s, X_test, y_test, out_dir: Path) -> None:
    model_infos = [
        ("Logistic Regression", lr.predict(X_test_s)),
        ("Multinomial LR (L1)", mlr.predict(X_test_s)),
        ("Gradient Boosted Trees", gbt.predict(X_test)),
        ("Random Forest", rf.predict(X_test)),
        ("RNN (LSTM)", rnn_preds),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes_flat = axes.flat
    for ax, (name, y_pred) in zip(axes_flat, model_infos):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, label="Row-norm")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
        for r in range(3):
            for c in range(3):
                ax.text(
                    c, r, f"{cm[r, c]:,}\n({cm_norm[r, c]:.2f})",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[r, c] > 0.6 else "black",
                )
    axes_flat[5].set_visible(False)
    fig.suptitle("Confusion Matrices (all models)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir / "confusion_matrices.png")


def plot_feature_importance(lr, gbt, rf, out_dir: Path) -> None:
    gbt_imp = gbt.feature_importances_
    rf_imp = rf.feature_importances_
    lr_imp = np.abs(lr.coef_).mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (name, imp, color) in zip(axes, [
        ("Gradient Boosted Trees\n(Gini Importance)", gbt_imp, "#55A868"),
        ("Random Forest\n(Gini Importance)", rf_imp, "#C44E52"),
        ("Logistic Regression\n(Mean |Coef| across classes)", lr_imp, "#4C72B0"),
    ]):
        idx = np.argsort(imp)
        ax.barh([FEATURE_COLS[i] for i in idx], imp[idx], color=color, alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title(name)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Feature Importances", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir / "feature_importance.png")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate dataset and model visualizations."
    )
    ap.add_argument(
        "data",
        type=Path,
        nargs="?",
        default=REPO / "data" / "out" / "decisions.csv",
        help="Path to decisions CSV (default: data/out/decisions.csv)",
    )
    ap.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=REPO / "data" / "out" / "figures",
        help="Directory to save figures (default: data/out/figures/)",
    )
    ap.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Limit rows (useful for faster test runs)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    ap.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model training; only generate dataset plots",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, X, y = load_and_prepare(args.data, limit=args.limit)
    print(f"  {len(df):,} rows, {X.shape[1]} features")

    print("\nDataset visualizations:")
    plot_label_distribution(df, out_dir)
    plot_feature_distributions(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_street_label_distribution(df, out_dir)

    if not args.skip_models:
        print("\nTraining all 5 models for evaluation plots...")
        (lr, mlr, gbt, rf, rnn_preds, rnn_probs,
         X_train_s, X_test_s, X_train, X_test, y_test) = _train_models(X, y, seed=args.seed)
        print("\nModel visualizations:")
        plot_roc_curves(lr, mlr, gbt, rf, rnn_probs, X_test_s, X_test, y_test, out_dir)
        plot_confusion_matrices(lr, mlr, gbt, rf, rnn_preds, X_test_s, X_test, y_test, out_dir)
        plot_feature_importance(lr, gbt, rf, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
