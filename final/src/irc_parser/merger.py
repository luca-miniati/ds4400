"""Merge parsed hdb, hroster, and pdb data into unified IRCHand objects."""

from dataclasses import dataclass, field

from .hdb import HdbRecord, parse_hdb
from .hroster import parse_hroster
from .pdb import PdbRow, parse_pdb_directory


@dataclass
class IRCHand:
    """Unified hand data from hdb, hroster, and pdb."""

    hand_id: str
    n_players: int
    players: list[str]  # Ordered: SB first, button last
    board_cards: str
    pot_preflop: str
    pot_flop: str
    pot_turn: str
    pot_river: str
    pdb_rows: list[PdbRow] = field(default_factory=list)  # One per player, may be partial

    def get_player_stacks(self) -> list[int]:
        """Get starting stacks in player order. Uses pdb stack or inf if unknown."""
        player_to_stack: dict[str, int] = {}
        for row in self.pdb_rows:
            player_to_stack[row.player] = row.stack
        return [player_to_stack.get(p, 10_000_000) for p in self.players]  # default inf-like

    def get_hole_cards(self, player: str) -> str | None:
        """Get hole cards for a player if shown at showdown."""
        for row in self.pdb_rows:
            if row.player == player and row.hole_cards:
                return row.hole_cards
        return None

    def get_pdb_row(self, player: str) -> PdbRow | None:
        """Get pdb row for a player."""
        for row in self.pdb_rows:
            if row.player == player:
                return row
        return None


def _players_in_position_order(
    n_players: int, roster: list[str], pdb_rows: list["PdbRow"]
) -> list[str]:
    """Reorder players so players[i] = player at position i+1 (from pdb). Falls back to roster."""
    if not pdb_rows or n_players < 2:
        return roster
    pos_to_player: dict[int, str] = {}
    for r in pdb_rows:
        if 1 <= r.position <= n_players and r.player in roster:
            pos_to_player[r.position] = r.player
    # Only reorder when we have complete position coverage; otherwise roster fill is unreliable
    if len(pos_to_player) != n_players or set(pos_to_player.keys()) != set(range(1, n_players + 1)):
        return roster
    return [pos_to_player[pos] for pos in range(1, n_players + 1)]


def merge_hands(
    hdb_path,
    hroster_path,
    pdb_dir,
    hdb_parser=parse_hdb,
    hroster_parser=parse_hroster,
    pdb_parser=parse_pdb_directory,
) -> dict[str, IRCHand]:
    """
    Merge hdb, hroster, and pdb into IRCHand objects by hand_id.

    Only returns hands that have hdb and hroster entries. Pdb rows are optional
    (hand may have partial pdb data).
    """
    from pathlib import Path

    hdb_path = Path(hdb_path)
    hroster_path = Path(hroster_path)
    pdb_dir = Path(pdb_dir)

    hdb_records = hdb_parser(hdb_path)
    rosters = hroster_parser(hroster_path)
    pdb_by_hand = pdb_parser(pdb_dir)

    hands: dict[str, IRCHand] = {}
    for hand_id, hdb in hdb_records.items():
        if hand_id not in rosters:
            continue
        n_players, players = rosters[hand_id]
        if n_players != hdb.n_players or len(players) != n_players:
            continue
        pdb_rows = pdb_by_hand.get(hand_id, [])
        # Reorder players to match pdb position order (1=SB, 2=BB, ...) for PHH conversion
        players_ordered = _players_in_position_order(n_players, players, pdb_rows)
        hands[hand_id] = IRCHand(
            hand_id=hand_id,
            n_players=n_players,
            players=players_ordered,
            board_cards=hdb.board_cards,
            pot_preflop=hdb.pot_preflop,
            pot_flop=hdb.pot_flop,
            pot_turn=hdb.pot_turn,
            pot_river=hdb.pot_river,
            pdb_rows=pdb_rows,
        )
    return hands
