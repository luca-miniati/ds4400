"""
Betting-round interleaver for IRC action strings.

Converts per-player action strings (e.g., "Bc", "br", "f") into a chronological
sequence of (player_idx, action_type, amount?) in correct poker order.
"""

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    FOLD = "f"
    CHECK_OR_CALL = "cc"
    BET_OR_RAISE = "cbr"


@dataclass
class Action:
    """A single poker action."""

    player_idx: int  # 0-based
    action_type: ActionType
    amount: int | None = None  # For cbr


def _parse_char(c: str) -> ActionType | None:
    """Map IRC action char to ActionType. B and - are skipped (caller filters).
    IRC: k=check, K=kicked(fold), c=call, f=fold, Q=quit(fold), b=bet, r=raise, A=all-in."""
    if c in "fFqQK":  # fold, quit, kicked (K uppercase = kicked)
        return ActionType.FOLD
    if c in "kcC":  # k=check, c/C=call (lowercase k distinct from K)
        return ActionType.CHECK_OR_CALL
    if c in "brRaA":  # bet, raise, all-in
        return ActionType.BET_OR_RAISE
    return None


def _get_action_chars(row_actions: str) -> list[tuple[int, ActionType]]:
    """Parse action string into list of (position_in_string, ActionType)."""
    result = []
    for i, c in enumerate(row_actions.replace("B", "").replace("-", "")):
        at = _parse_char(c)
        if at is not None:
            result.append((i, at))
    return result


def interleave_street(
    n_players: int,
    position_order: list[int],  # Order of action; position 1-based
    player_actions: dict[int, str],  # position -> action string
    *,
    pos_to_player_idx: dict[int, int] | None = None,
) -> list[tuple[int, ActionType]]:
    """
    Interleave per-player actions into correct chronological order.

    Uses round-robin: each player acts in turn. When a player raises (cbr),
    betting reopens and we continue from the first player after the raiser.
    Folded players are skipped.

    Args:
        n_players: Total players at start of street
        position_order: Order of action (1-based positions), e.g. [1,2,3] or [1,3,2]
        player_actions: Map position -> action string (e.g. "Bc", "br")
        pos_to_player_idx: Map position -> actual player index in hand.players.
            If None, uses position-1 (assumes roster order matches position order).

    Returns:
        List of (player_idx_0based, ActionType) in order. Amounts filled separately.
    """
    # Map position -> player index (roster order may differ from position order)
    if pos_to_player_idx is not None:
        pos_to_idx = {p: pos_to_player_idx.get(p, p - 1) for p in range(1, n_players + 1)}
    else:
        pos_to_idx = {p: p - 1 for p in range(1, n_players + 1)}
    queues: dict[int, list[ActionType]] = {}
    for pos in position_order:
        acts = player_actions.get(pos, "").replace("B", "").replace("-", "")
        queue = []
        for c in acts:
            at = _parse_char(c)
            if at is not None:
                queue.append(at)
        queues[pos] = queue

    # Position order as 0-based indices for output
    order = [pos_to_idx[p] for p in position_order]

    result: list[tuple[int, ActionType]] = []
    active = set(order)
    ptr = [0] * len(order)  # ptr[i] = next action index for order[i]
    first_to_act = 0
    last_raise_idx = -1

    max_iter = 100
    for _ in range(max_iter):
        acted_this_round = False
        i = first_to_act
        for _ in range(len(order)):
            idx = order[i]
            if idx not in active:
                i = (i + 1) % len(order)
                continue
            if ptr[i] >= len(queues[position_order[i]]):
                i = (i + 1) % len(order)
                continue
            at = queues[position_order[i]][ptr[i]]
            ptr[i] += 1
            result.append((idx, at))
            acted_this_round = True
            if at == ActionType.FOLD:
                active.discard(idx)
                if len(active) <= 1:
                    return result
            elif at == ActionType.BET_OR_RAISE:
                last_raise_idx = i
                first_to_act = (i + 1) % len(order)
            i = (i + 1) % len(order)
        if not acted_this_round:
            break
        if last_raise_idx < 0:
            break
        first_to_act = (last_raise_idx + 1) % len(order)
        last_raise_idx = -1

    return result


def get_preflop_order(n_players: int) -> list[int]:
    """
    Preflop action order. Positions 1-based: 1=SB, 2=BB, 3=UTG, 4=..., n=BTN.
    First to act = left of BB = UTG (position 3). Order: [3, 4, ..., n, 1, 2].
    For 2-way: SB, BB = [1, 2]. For 3-way: BTN=UTG, SB, BB = [3, 1, 2].
    """
    if n_players == 2:
        return [1, 2]
    if n_players == 3:
        return [3, 1, 2]  # BTN first (UTG), then SB, then BB
    # 4+: UTG (3), then 4...n, then SB (1), BB (2)
    return list(range(3, n_players + 1)) + [1, 2]


def get_postflop_order(n_players: int) -> list[int]:
    """Postflop: first to act = left of BTN = SB. Order 1, 2, 3, ..."""
    return list(range(1, n_players + 1))
