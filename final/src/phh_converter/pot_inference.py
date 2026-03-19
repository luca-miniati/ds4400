"""Infer bet amounts from pot structure and pdb bet columns."""

from ..irc_parser.pdb import PdbRow  # noqa: I001



def parse_pot(pot_str: str) -> tuple[int, int]:
    """
    Parse pot string like '2/45' or '3/90'.

    Returns (min_bet_or_sb_units, total_chips).
    Returns (0, 0) for '-' or empty/invalid strings.
    """
    if not pot_str or pot_str.strip() == "-":
        return (0, 0)
    try:
        parts = pot_str.strip().split("/")
        if len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                return (int(a), int(b))
    except (ValueError, IndexError):
        pass
    return (0, 0)


def infer_bet_amounts(
    pot_preflop: str,
    pot_flop: str,
    pot_turn: str,
    pot_river: str,
    pdb_rows: list[PdbRow],
    sb: int,
    bb: int,
) -> dict[tuple[str, int, int], int]:
    """
    Infer bet/raise amounts from pot strings and pdb data.

    Returns mapping (street, player_idx, action_idx) -> chips for BET_OR_RAISE actions.
    street is one of 'preflop', 'flop', 'turn', 'river'.
    player_idx is 0-based (position - 1).
    action_idx is the 0-based index of that player's bet/raise on that street.

    Uses pot deltas and bet1/bet2 columns to infer. Returns empty dict when
    inference is not possible; caller should use a default (e.g. bb).
    """
    result: dict[tuple[str, int, int], int] = {}

    _, total_pf = parse_pot(pot_preflop)
    _, total_flop = parse_pot(pot_flop)
    _, total_turn = parse_pot(pot_turn)
    _, total_river = parse_pot(pot_river)

    pos_to_row = {r.position: r for r in pdb_rows}

    def _count_bet_raises(action_str: str) -> int:
        count = 0
        for c in action_str.replace("B", "").replace("-", ""):
            if c.lower() in "bra":
                count += 1
        return count

    def _infer_street_amounts(
        street: str,
        pot_before: int,
        pot_after: int,
        pos_order: list[int],
    ) -> None:
        delta = pot_after - pot_before
        if delta <= 0:
            return
        bet_counts: list[tuple[int, int]] = []  # (player_idx, count)
        for pos in pos_order:
            row = pos_to_row.get(pos)
            if not row:
                continue
            attr = f"{street}_actions"
            acts = getattr(row, attr, "") or ""
            n = _count_bet_raises(acts)
            if n > 0:
                bet_counts.append((pos - 1, n))
        total_bets = sum(c for _, c in bet_counts)
        if total_bets == 0:
            return
        amt_per_bet = delta // total_bets
        if amt_per_bet < bb:
            amt_per_bet = bb
        for player_idx, count in bet_counts:
            for action_idx in range(count):
                result[(street, player_idx, action_idx)] = amt_per_bet

    if not pdb_rows:
        return result
    n = max(r.position for r in pdb_rows)
    positions = list(range(1, n + 1))
    _infer_street_amounts("preflop", 0, total_pf, positions)
    _infer_street_amounts("flop", total_pf, total_flop, positions)
    _infer_street_amounts("turn", total_flop, total_turn, positions)
    _infer_street_amounts("river", total_turn, total_river, positions)

    return result
