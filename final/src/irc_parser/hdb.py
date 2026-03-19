"""Parser for IRC hdb (hand database) files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HdbRecord:
    """A single hand record from the hdb file."""

    hand_id: str
    hand_num: int
    n_players: int
    pot_preflop: str  # e.g., "2/45"
    pot_flop: str
    pot_turn: str
    pot_river: str
    board_cards: str  # space-separated, e.g., "2h 4s 4c 2d 5c"


def parse_hdb(path: Path) -> dict[str, HdbRecord]:
    """Parse an hdb file and return a dict mapping hand_id to HdbRecord."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            # Format: hand_id ? hand_num n_players pot_pf pot_f pot_t pot_r board...
            hand_id = parts[0]
            hand_num = int(parts[2])
            n_players = int(parts[3])
            pot_preflop = parts[4]
            pot_flop = parts[5]
            pot_turn = parts[6]
            pot_river = parts[7]
            board_cards = " ".join(parts[8:])
            records[hand_id] = HdbRecord(
                hand_id=hand_id,
                hand_num=hand_num,
                n_players=n_players,
                pot_preflop=pot_preflop,
                pot_flop=pot_flop,
                pot_turn=pot_turn,
                pot_river=pot_river,
                board_cards=board_cards,
            )
    return records
