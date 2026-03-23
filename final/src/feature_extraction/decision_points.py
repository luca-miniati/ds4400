"""
Extract fold/call/raise decision points from PHH hand histories.

A decision point is a moment when a player faces a bet and must choose
fold, call, or raise. We exclude check/bet decisions (facing a check).
"""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

from pokerkit import HandHistory


@dataclass
class DecisionPoint:
    """A single decision point: player faces a bet and chooses fold/call/raise."""

    hand_id: str
    player_idx: int
    player_name: str
    street_index: int  # 0=preflop, 1=flop, 2=turn, 3=river
    street_name: str
    # State at decision
    total_pot: int
    hero_stack: int
    hero_bet: int
    facing_bet: int  # Amount to call (checking_or_calling_amount)
    min_raise_to: int
    max_raise_to: int
    stacks: tuple[int, ...]
    bets: tuple[int, ...]
    board_cards: tuple[str, ...]
    n_players: int
    folded: tuple[bool, ...]
    # Action history up to this point: list of (player_idx, action_type, amount?)
    # action_type: 'f', 'c', 'r' (fold, check/call, bet/raise)
    action_history: list[tuple[int, str, int | None]]
    # Label: what the player did
    label: str  # 'fold', 'call', 'raise'


# Action string patterns: "pN f", "pN cc", "pN cbr X"
_ACTION_PATTERN = re.compile(r"^p(\d+)\s+(f|cc|cbr(?:\s+(\d+))?)$")


def _parse_action(action_str: str) -> tuple[int, str, int | None] | None:
    """Parse action string to (player_idx, action_type, amount?)."""
    if not action_str or not isinstance(action_str, str):
        return None
    m = _ACTION_PATTERN.match(action_str.strip())
    if not m:
        return None
    player_idx = int(m.group(1)) - 1  # PHH uses 1-based p1, p2, ...
    action_type = m.group(2)
    amount = int(m.group(3)) if m.group(3) else None
    # Normalize: cc -> c, cbr -> r, f -> f
    if action_type == "cc":
        label = "c"  # check or call
    elif action_type.startswith("cbr"):
        label = "r"  # bet or raise
    elif action_type == "f":
        label = "f"
    else:
        return None
    return (player_idx, label, amount)


def _get_street_name(street_index: int) -> str:
    names = ("preflop", "flop", "turn", "river")
    return names[street_index] if 0 <= street_index < 4 else f"street_{street_index}"


def extract_decision_points_from_phh(
    phh_path: Path,
    *,
    limit_hands: int | None = None,
) -> tuple[list[DecisionPoint], int]:
    """
    Extract all fold/call/raise decision points from a PHH file.

    Only includes moments when the player faces an actual bet (call amount > 0).
    Returns (decisions, n_hands_processed).
    """
    with open(phh_path, "rb") as f:
        hands = list(HandHistory.load_all(f))

    decisions: list[DecisionPoint] = []
    hand_count = 0

    for hh in hands:
        if limit_hands is not None and hand_count >= limit_hands:
            break

        hand_id = getattr(hh, "hand", None) or str(hand_count)
        players = getattr(hh, "players", [])
        n_players = len(players)

        action_history: list[tuple[int, str, int | None]] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for state, action in hh.state_actions:
                if action and isinstance(action, str):
                    parsed = _parse_action(action)
                    if parsed:
                        pidx, atype, amt = parsed
                        action_history.append((pidx, atype, amt))

                        # Only count as decision point when facing a bet (call > 0)
                        if (
                            state.actor_index is not None
                            and state.can_fold()
                            and state.checking_or_calling_amount is not None
                            and state.checking_or_calling_amount > 0
                        ):
                            actor = state.actor_index
                            if actor >= len(players):
                                continue

                            # Map action to label
                            if atype == "f":
                                label = "fold"
                            elif atype == "c":
                                label = "call"
                            elif atype == "r":
                                label = "raise"
                            else:
                                continue

                            # Board cards as list of strings e.g. ["Td", "4c", "7h"]
                            board = list(state.board_cards) if state.board_cards else []
                            board_strs = []
                            for c in board:
                                card = c[0] if isinstance(c, (list, tuple)) and c else c
                                r = getattr(card.rank, "value", str(card.rank))
                                s = getattr(card.suit, "value", str(card.suit))
                                board_strs.append(f"{r}{s}".replace("10", "T"))

                            decisions.append(
                                DecisionPoint(
                                    hand_id=str(hand_id),
                                    player_idx=actor,
                                    player_name=players[actor],
                                    street_index=state.street_index,
                                    street_name=_get_street_name(state.street_index),
                                    total_pot=state.total_pot_amount,
                                    hero_stack=state.stacks[actor],
                                    hero_bet=state.bets[actor],
                                    facing_bet=state.checking_or_calling_amount,
                                    min_raise_to=(
                                        getattr(
                                            state,
                                            "min_completion_betting_or_raising_to_amount",
                                            None,
                                        )
                                        or state.checking_or_calling_amount * 2
                                    ),
                                    max_raise_to=(
                                        getattr(
                                            state,
                                            "max_completion_betting_or_raising_to_amount",
                                            None,
                                        )
                                        or state.stacks[actor]
                                    ),
                                    stacks=tuple(state.stacks),
                                    bets=tuple(state.bets),
                                    board_cards=tuple(board_strs),
                                    n_players=n_players,
                                    folded=tuple(not s for s in state.statuses),
                                    action_history=list(action_history[:-1]),
                                    label=label,
                                )
                            )

        hand_count += 1

    return decisions, hand_count


def load_phh_directory(
    phh_dir: Path,
    *,
    limit_hands: int | None = None,
    limit_files: int | None = None,
) -> list[DecisionPoint]:
    """Load decision points from all PHH files in a directory."""
    phh_files = sorted(phh_dir.glob("hands_*.phhs"))
    if limit_files is not None:
        phh_files = phh_files[:limit_files]

    all_decisions: list[DecisionPoint] = []
    hands_remaining = limit_hands

    for p in phh_files:
        if hands_remaining is not None and hands_remaining <= 0:
            break
        pts, n_hands = extract_decision_points_from_phh(p, limit_hands=hands_remaining)
        all_decisions.extend(pts)
        if limit_hands is not None:
            hands_remaining = max(0, hands_remaining - n_hands)

    return all_decisions
