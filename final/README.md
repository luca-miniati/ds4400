# DS4400 Final Project: Modeling Player Actions in Low-Stakes Poker

**Team:** Luca Miniati, Chigozirim Ike

Predicting player actions (fold, call, raise) in No-Limit Texas Hold'em using game-state features. This repository includes the data processing pipeline and supports downstream ML modeling (Logistic Regression, Random Forest, Gradient Boosted Trees, RNNs).

See [proposal.md](docs/proposal.md) for the full project proposal.

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the final project report.

---

## Setup

```bash
python -m venv .venv
.venv/bin/activate
pip install -r requirements.txt
```

---

## Project Structure

```
final/
├── src/                  # Core library
│   ├── irc_parser/       # IRC format parsers (hdb, hroster, pdb)
│   ├── phh_converter/    # IRC → PokerKit PHH conversion
│   ├── feature_extraction/  # ML feature extraction from PHH
│   ├── json_export.py    # JSON export for ML
│   └── pipeline.py       # Pipeline orchestration
├── scripts/              # Utility scripts
│   ├── count_hole_cards.py
│   ├── extract_features.py   # Extract 28 features to CSV/Parquet
│   ├── describe_features.py  # Summarize extracted dataset
│   ├── show_phh_sample.py
│   └── validate_position_mapping.py
├── tests/                # Unit tests
├── docs/                 # Internal documentation
├── run_pipeline.py       # CLI entry point
├── proposal.md           # Project proposal
└── data/
    ├── IRCdata/          # Input: IRC hand history (place here)
    ├── phh/              # Output: generated JSON and PHH
    └── ml/               # ML-ready dataset (from extract_features)
```

---

## Data Processing Pipeline

The pipeline converts IRC poker hand history data into two output formats:

1. **JSON** — Parsed hand data (players, board, actions, stacks, hole cards) for ML feature extraction
2. **PHH** — PokerKit hand history format for simulation and validation

### Quick Start

```bash
# Run pipeline (place IRC data in data/IRCdata/)
.venv/bin/python run_pipeline.py data/IRCdata -o data/phh

# Limit hands for testing
.venv/bin/python run_pipeline.py data/IRCdata -o data/phh -l 5000
```

### Output Layout

The pipeline creates `json/` and `phh/` subdirectories under the output path:

```
data/phh/
├── json/           # Batched JSON (100 hands per file)
│   ├── hands_0000.json
│   ├── hands_0001.json
│   └── ...
└── phh/            # Batched PHH (1000 hands per file)
    ├── hands_0000.phhs
    ├── hands_0001.phhs
    └── ...
```

### Data Sources (IRC Format)

| File      | Description                                      |
|-----------|--------------------------------------------------|
| **hdb**   | Hand database: board cards, pot structure        |
| **hroster** | Player roster per hand                         |
| **pdb**   | Per-player actions, stacks, hole cards at showdown |

The pipeline discovers `holdem1`, `holdem2`, `holdem3` (NLH) paths under the IRC root, merges by `hand_id`, and reconstructs the action sequence for PokerKit validation.

### Limitations

IRC stores per-player action summaries (e.g., `Bc` = blind+call) rather than a full chronological log. Reconstructing the exact sequence for PokerKit validation is heuristic-based; some hands fail conversion. **The JSON export is the recommended output** for ML—it contains all parsed data including hole cards at showdown.

### Scripts

| Script                     | Purpose                                              |
|----------------------------|------------------------------------------------------|
| `count_hole_cards.py`      | Compare hole card coverage in JSON vs PHH output     |
| `extract_features.py`      | Extract 29 features + labels to CSV/Parquet for ML   |
| `describe_features.py`     | Summarize extracted dataset (labels, feature stats)  |
| `train_models.py`          | Train all models (LR, Multinomial LR, GBT, Random Forest, RNN) |
| `show_phh_sample.py`       | Print sample hands from PHH files in readable format |
| `validate_position_mapping.py` | Validate IRC roster vs PDB position alignment    |

---

## Feature Extraction (ML Dataset)

The feature extraction pipeline turns PHH hand histories into an ML-ready dataset with **29 features** and **fold/call/raise** labels.

### Quick Start

```bash
# Extract from PHH to CSV (default: data/phh/phh -> data/ml/decisions.csv)
.venv/bin/python scripts/extract_features.py

# Limit for testing
.venv/bin/python scripts/extract_features.py -l 1000 --limit-files 2

# Output Parquet instead
.venv/bin/python scripts/extract_features.py -o data/ml/decisions.parquet -f parquet

# Describe the dataset
.venv/bin/python scripts/describe_features.py data/ml/decisions.csv
```

### Features (29 total)

- **Draw** (2): flush draw, straight draw (require hole cards; -1 when unknown)
- **Board texture** (8): suited count, runs, pair/trips, monotone/two-tone/rainbow, highest rank
- **Pot/betting** (9): pot (BB), effective stack, SPR, pot odds, facing bet, all-in, commitment %, MDF
- **Action** (6): opponent aggressive/passive counts, last two actions, check-raise, donk
- **Game state** (4): hero/villain position, preflop aggressor, street

Hole cards are only available at showdown (from JSON). Draw features use -1 when unknown. Use `-j` to point at the JSON directory for showdown hole cards.

---

## Model Training

Train all 5 models (Logistic Regression, Multinomial LR, Gradient Boosted Trees, Random Forest, RNN/LSTM):

```bash
.venv/bin/python scripts/train_models.py

# Faster test run (subset)
.venv/bin/python scripts/train_models.py -l 50000
```

Metrics are saved to `data/out/metrics.json` (F1 scores and confusion matrices) and classification reports to `data/out/train.log`.

---

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

---

## Data

Place IRC data under `data/IRCdata/` with the expected structure (e.g. `holdem1/YYYYMM/hdb`, `hroster`, `pdb/`). A full archive is available from [IRC Poker Database](https://poker.cs.ualberta.ca/IRC/IRCdata.tgz).
