"""Parser for IRC pdb (player database) files."""

import re
from dataclasses import dataclass
from pathlib import Path

CARD_PATTERN = re.compile(r"^[2-9TJQKAtjqka][schd]$", re.I)


@dataclass
class PdbRow:
    """A single row from a pdb file for one player in one hand."""

    player: str
    hand_id: str
    n_players: int
    position: int  # 1-indexed seat/position
    preflop_actions: str
    flop_actions: str
    turn_actions: str
    river_actions: str
    stack: int
    bet1: int  # First numeric column (often preflop/flop related)
    bet2: int  # Second numeric column
    hole_cards: str | None = None  # e.g., "5s Qs" when shown at showdown


def _parse_pdb_line_simple(line: str) -> PdbRow | None:
    """Parse a pdb line. Format: player hand_id n_players position pf flop turn river stack bet1 bet2 [card1 card2]."""
    parts = line.split()
    if len(parts) < 11:
        return None
    try:
        player = parts[0]
        hand_id = parts[1]
        n_players = int(parts[2])
        position = int(parts[3])
        preflop = parts[4]
        flop = parts[5]
        turn = parts[6]
        river = parts[7]
        stack = int(parts[8]) if parts[8] != "-" else 0
        bet1 = int(parts[9]) if parts[9] != "-" else 0
        bet2 = int(parts[10]) if parts[10] != "-" else 0
    except (ValueError, IndexError):
        return None
    hole = None
    if len(parts) >= 13 and CARD_PATTERN.match(parts[11]) and CARD_PATTERN.match(parts[12]):
        hole = f"{parts[11]} {parts[12]}"
    return PdbRow(
        player=player,
        hand_id=hand_id,
        n_players=n_players,
        position=position,
        preflop_actions=preflop,
        flop_actions=flop,
        turn_actions=turn,
        river_actions=river,
        stack=stack,
        bet1=bet1,
        bet2=bet2,
        hole_cards=hole,
    )


def parse_pdb_file(path: Path) -> list[PdbRow]:
    """Parse a single pdb file and return list of PdbRows."""
    player = path.stem.replace("pdb.", "")
    rows = []
    with open(path) as f:
        for line in f:
            row = _parse_pdb_line_simple(line)
            if row:
                rows.append(row)
    return rows


def parse_pdb_directory(pdb_dir: Path) -> dict[str, list[PdbRow]]:
    """
    Parse all pdb files in a directory.

    Returns dict mapping hand_id to list of PdbRows (one per player in that hand).
    """
    hand_to_rows: dict[str, list[PdbRow]] = {}
    for pdb_file in pdb_dir.glob("pdb.*"):
        if pdb_file.is_file():
            for row in parse_pdb_file(pdb_file):
                hand_to_rows.setdefault(row.hand_id, []).append(row)
    return hand_to_rows
