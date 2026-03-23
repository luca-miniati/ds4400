#!/usr/bin/env bash
# Extract full ML dataset from PHH files.
# Output: data/ml/decisions.csv (or .parquet)
# Run from project root: ./scripts/extract_ml_dataset.sh

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV_PYTHON="${REPO}/.venv/bin/python"
PHH_DIR="${REPO}/data/phh/phh"
JSON_DIR="${REPO}/data/phh/json"
OUTPUT_CSV="${REPO}/data/ml/decisions.csv"
OUTPUT_PARQUET="${REPO}/data/ml/decisions.parquet"

if [[ ! -d "$PHH_DIR" ]]; then
  echo "Error: PHH directory not found: $PHH_DIR"
  exit 1
fi

mkdir -p "${REPO}/data/ml"

echo "Extracting ML dataset from PHH files..."
echo "  PHH dir: $PHH_DIR"
echo "  JSON dir: $JSON_DIR (for hole cards at showdown)"
echo ""

# Extract to CSV (default). Use -f parquet for Parquet.
"$VENV_PYTHON" scripts/extract_features.py \
  "$PHH_DIR" \
  -o "$OUTPUT_CSV" \
  -j "$JSON_DIR" \
  -b 10

echo ""
echo "Running describe_features..."
"$VENV_PYTHON" scripts/describe_features.py "$OUTPUT_CSV"

echo ""
echo "Done. Dataset written to: $OUTPUT_CSV"
