"""Export decision points with features to CSV or Parquet."""

import csv
import json
from pathlib import Path
from typing import Iterator

from .decision_points import DecisionPoint, extract_decision_points_from_phh, load_phh_directory
from .features import FEATURE_NAMES, compute_features


def _load_hole_cards_from_json(
    json_dir: Path,
    hand_ids: set[str],
) -> dict[str, dict[str, list[str]]]:
    """
    Load hole cards from JSON files. Returns hand_id -> {player_name: [card1, card2]}.
    Only includes showdown hands where hole cards are shown.
    """
    result: dict[str, dict[str, list[str]]] = {}
    for jf in sorted(json_dir.glob("hands_*.json")):
        with open(jf) as f:
            data = json.load(f)
        for hid, hand in data.items():
            if hid not in hand_ids:
                continue
            for row in hand.get("pdb_rows", []):
                hole = row.get("hole_cards")
                if hole and len(hole) >= 2:
                    player = row.get("player", "")
                    if hid not in result:
                        result[hid] = {}
                    result[hid][player] = hole
    return result


def _normalize_card(c: str) -> str:
    """Normalize card to 'Ts' format."""
    c = str(c).strip().upper()
    if len(c) >= 2:
        rank = c[0] if c[0] != "1" else "T"
        suit = c[-1].lower()
        return rank + suit
    return c


def extract_features_for_decisions(
    decisions: list[DecisionPoint],
    bb: int = 10,
    hole_cards_map: dict[str, dict[str, list[str]]] | None = None,
) -> Iterator[dict]:
    """Yield one dict per decision with hand_id, player, street, label, and all features."""
    hole_cards_map = hole_cards_map or {}
    for dp in decisions:
        player_holes = hole_cards_map.get(dp.hand_id, {}).get(dp.player_name)
        cards = None
        if player_holes:
            cards = [_normalize_card(c) for c in player_holes]
        feats = compute_features(dp, bb=bb, hole_cards=cards)
        row = {
            "hand_id": dp.hand_id,
            "player": dp.player_name,
            "street": dp.street_name,
            "label": dp.label,
            **feats,
        }
        yield row


def export_decisions_to_csv(
    decisions: list[DecisionPoint],
    output_path: Path,
    bb: int = 10,
    hole_cards_map: dict[str, dict[str, list[str]]] | None = None,
) -> None:
    """Write decisions with features to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["hand_id", "player", "street", "label"] + FEATURE_NAMES
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in extract_features_for_decisions(decisions, bb, hole_cards_map):
            w.writerow(row)


def run_extraction(
    phh_dir: Path,
    output_path: Path,
    json_dir: Path | None = None,
    *,
    bb: int = 10,
    limit_hands: int | None = None,
    limit_files: int | None = None,
    format: str = "csv",
) -> tuple[int, int]:
    """
    Full extraction: load PHH, optionally merge hole cards from JSON, export.
    Returns (n_decisions, n_hands).
    """
    decisions = load_phh_directory(
        Path(phh_dir),
        limit_hands=limit_hands,
        limit_files=limit_files,
    )
    n_decisions = len(decisions)
    hand_ids = {dp.hand_id for dp in decisions}
    hole_cards_map = {}
    if json_dir and Path(json_dir).exists():
        hole_cards_map = _load_hole_cards_from_json(Path(json_dir), hand_ids)

    output_path = Path(output_path)
    export_decisions_to_csv(decisions, output_path, bb, hole_cards_map)

    n_hands = len(hand_ids)
    return n_decisions, n_hands
