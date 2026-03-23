"""
Compute the 28 proposal features for each decision point.

Features: draw (2), board texture (8), pot/betting (9), action (5), game state (4).
Hole cards are only available at showdown from IRC/JSON; draw features use NaN when unknown.
"""

from __future__ import annotations

import math
from collections import Counter

from .decision_points import DecisionPoint

# Rank order for straights (A can be high or low)
RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER)}


def _parse_card(s: str) -> tuple[str, str]:
    """Parse card string 'Td' -> (rank, suit)."""
    if len(s) >= 2:
        r = s[0].upper()
        suit = s[1].lower()
        if r == "1" and len(s) >= 3:
            r, suit = "T", s[2].lower()
        return (r, suit)
    return ("?", "?")


def _board_ranks_suits(board: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Return (ranks, suits) from board cards."""
    ranks, suits = [], []
    for c in board:
        r, s = _parse_card(c)
        ranks.append(r)
        suits.append(s)
    return ranks, suits


# --- Draw features (require hole cards; use -1 when unknown) ---
def _flush_draw_indicator(hole_cards: list[str] | None, board: tuple[str, ...]) -> float:
    """1 if hero has flush draw, 0 otherwise. -1 if hole cards unknown."""
    if not hole_cards or len(hole_cards) < 2:
        return -1.0
    _, hole_suits = zip(*[_parse_card(c) for c in hole_cards])
    _, board_suits = _board_ranks_suits(board)
    suits = list(hole_suits) + list(board_suits)
    for s in set(suits):
        if suits.count(s) >= 4:
            return 1.0
    return 0.0


def _straight_draw_indicator(hole_cards: list[str] | None, board: tuple[str, ...]) -> float:
    """1 if hero has OESD or gutshot, 0 otherwise. -1 if hole cards unknown."""
    if not hole_cards or len(hole_cards) < 2:
        return -1.0
    all_ranks = [RANK_VALUE.get(_parse_card(c)[0], -1) for c in hole_cards + list(board)]
    all_ranks = sorted(set(r for r in all_ranks if r >= 0))
    if len(all_ranks) < 4:
        return 0.0
    for i in range(len(all_ranks) - 3):
        run = all_ranks[i : i + 4]
        if run[-1] - run[0] <= 4:
            return 1.0
    return 0.0


# --- Board texture ---
def _n_suited_on_board(board: tuple[str, ...]) -> int:
    """Max count of same suit on board."""
    if not board:
        return 0
    _, suits = _board_ranks_suits(board)
    return max(Counter(suits).values()) if suits else 0


def _highest_run_connected(board: tuple[str, ...]) -> int:
    """Longest run of connected ranks on board."""
    if not board:
        return 0
    ranks, _ = _board_ranks_suits(board)
    vals = sorted(set(RANK_VALUE.get(r, -1) for r in ranks if r in RANK_VALUE))
    if not vals:
        return 0
    best = 1
    run = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _board_pair_indicator(board: tuple[str, ...]) -> int:
    """1 if board has a pair, 0 otherwise."""
    if not board:
        return 0
    ranks, _ = _board_ranks_suits(board)
    return 1 if len(ranks) != len(set(ranks)) else 0


def _board_trips_indicator(board: tuple[str, ...]) -> int:
    """1 if board has trips, 0 otherwise."""
    if not board:
        return 0
    ranks, _ = _board_ranks_suits(board)
    return 1 if max(Counter(ranks).values()) >= 3 else 0


def _monotone_board(board: tuple[str, ...]) -> int:
    """1 if all board same suit, 0 otherwise."""
    if len(board) < 3:
        return 0
    _, suits = _board_ranks_suits(board)
    return 1 if len(set(suits)) == 1 else 0


def _two_tone_board(board: tuple[str, ...]) -> int:
    """1 if board has exactly two suits, 0 otherwise."""
    if len(board) < 3:
        return 0
    _, suits = _board_ranks_suits(board)
    return 1 if len(set(suits)) == 2 else 0


def _rainbow_board(board: tuple[str, ...]) -> int:
    """1 if all board different suits, 0 otherwise."""
    if len(board) < 3:
        return 0
    _, suits = _board_ranks_suits(board)
    return 1 if len(set(suits)) == len(suits) else 0


def _highest_board_rank(board: tuple[str, ...]) -> int:
    """Highest rank on board (0-12, 12=A)."""
    if not board:
        return -1
    ranks, _ = _board_ranks_suits(board)
    vals = [RANK_VALUE.get(r, -1) for r in ranks if r in RANK_VALUE]
    return max(vals) if vals else -1


# --- Opponent/villain from action history ---
def _last_aggressor_idx(history: list[tuple[int, str, int | None]], exclude_idx: int) -> int | None:
    """Index of last player to bet/raise in history, excluding hero."""
    for i in range(len(history) - 1, -1, -1):
        pidx, atype, _ = history[i]
        if atype == "r" and pidx != exclude_idx:
            return pidx
    return None


def _opponent_aggressive_count(history: list[tuple[int, str, int | None]], villain_idx: int) -> int:
    """Count of bet/raise by villain in history."""
    return sum(1 for pidx, atype, _ in history if pidx == villain_idx and atype == "r")


def _opponent_passive_count(history: list[tuple[int, str, int | None]], villain_idx: int) -> int:
    """Count of check/call by villain in history."""
    return sum(1 for pidx, atype, _ in history if pidx == villain_idx and atype == "c")


def _last_two_actions(history: list[tuple[int, str, int | None]], hero_idx: int) -> tuple[str, str]:
    """Last two action types in order (e.g. ('c','r') = check then raise). Most recent last."""
    acts = [a for _, a, _ in history]
    if len(acts) >= 2:
        return (acts[-2], acts[-1])
    if len(acts) == 1:
        return ("", acts[0])
    return ("", "")


def _villain_checkraise_indicator(
    history: list[tuple[int, str, int | None]],
    villain_idx: int,
) -> int:
    """1 if villain's last aggressive action was a check-raise (check/call then raise in same street)."""
    # Simplified: we'd need street boundaries. Use: any sequence c,r from villain.
    for i in range(len(history) - 1):
        if history[i][0] == villain_idx and history[i][1] == "c":
            if history[i + 1][0] == villain_idx and history[i + 1][1] == "r":
                return 1
    return 0


def _villain_donk_indicator(
    history: list[tuple[int, str, int | None]],
    villain_idx: int,
    street_index: int,
) -> int:
    """1 if villain donk-bet (first aggressor on this street when they weren't preflop aggressor)."""
    if street_index == 0:
        return 0
    # First action on street: find first bet/raise. If it's villain and preflop aggressor wasn't villain, donk.
    # We don't have street boundaries in flat history - skip for now, return 0
    return 0


def _preflop_aggressor(history: list[tuple[int, str, int | None]]) -> int | None:
    """Player index of preflop aggressor (first to raise)."""
    for pidx, atype, _ in history:
        if atype == "r":
            return pidx
    return None


# --- Feature computation ---
FEATURE_NAMES = [
    "flush_draw",
    "straight_draw",
    "n_suited_board",
    "highest_run_board",
    "board_pair",
    "board_trips",
    "monotone_board",
    "two_tone_board",
    "rainbow_board",
    "highest_board_rank",
    "pot_bb",
    "effective_stack_bb",
    "spr",
    "pot_odds",
    "facing_bet_bb",
    "facing_bet_pct_pot",
    "all_in",
    "commitment_pct",
    "mdf",
    "opp_aggressive_count",
    "opp_passive_count",
    "last_action_1",
    "last_action_2",
    "villain_checkraise",
    "villain_donk",
    "hero_position",
    "villain_position",
    "preflop_aggressor_is_hero",
    "street_index",
]


def compute_features(
    dp: DecisionPoint,
    bb: int = 10,
    *,
    hole_cards: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute the 28 proposal features for a decision point.

    hole_cards: optional list like ["As", "Kd"] when known (showdown). Else draw features = -1.
    """
    board = dp.board_cards
    hero_idx = dp.player_idx
    villain_idx = _last_aggressor_idx(dp.action_history, hero_idx)
    if villain_idx is None:
        villain_idx = (hero_idx + 1) % dp.n_players

    # Effective stack = min(hero stack, largest opponent stack still in hand)
    opp_stacks = [
        dp.stacks[i]
        for i in range(dp.n_players)
        if i != hero_idx and not dp.folded[i]
    ]
    effective_stack = min(dp.hero_stack, max(opp_stacks)) if opp_stacks else dp.hero_stack
    pot_bb = dp.total_pot / bb if bb else 0
    eff_stack_bb = effective_stack / bb if bb else 0
    spr = effective_stack / dp.total_pot if dp.total_pot else 999
    pot_odds = dp.facing_bet / (dp.total_pot + dp.facing_bet) if (dp.total_pot + dp.facing_bet) else 0
    facing_bet_bb = dp.facing_bet / bb if bb else 0
    facing_bet_pct = (dp.facing_bet / dp.total_pot * 100) if dp.total_pot else 0
    # Commitment: total in pot for hero / starting stack. We use (total_pot share) as proxy.
    total_in_for_hero = dp.hero_bet + dp.facing_bet
    starting_hero = dp.hero_stack + dp.hero_bet
    commitment_pct = (total_in_for_hero / starting_hero * 100) if starting_hero else 0
    max_raise = dp.max_raise_to if dp.max_raise_to is not None else 0
    all_in = 1.0 if dp.hero_stack <= 0 or max_raise >= dp.hero_stack + dp.hero_bet else 0.0
    # MDF = pot / (pot + bet) for caller
    mdf = (dp.total_pot / (dp.total_pot + dp.facing_bet) * 100) if (dp.total_pot + dp.facing_bet) else 50

    preflop_agg = _preflop_aggressor(dp.action_history)
    preflop_agg_is_hero = 1.0 if preflop_agg == hero_idx else 0.0

    # One-hot last two actions: use simple encoding (c=0, r=1, f=2) -> 2 features
    last1, last2 = _last_two_actions(dp.action_history, hero_idx)
    _amap = {"": -1, "c": 0, "r": 1, "f": 2}
    last_action_1 = float(_amap.get(last1, -1))
    last_action_2 = float(_amap.get(last2, -1))

    return {
        "flush_draw": _flush_draw_indicator(hole_cards, board),
        "straight_draw": _straight_draw_indicator(hole_cards, board),
        "n_suited_board": float(_n_suited_on_board(board)),
        "highest_run_board": float(_highest_run_connected(board)),
        "board_pair": float(_board_pair_indicator(board)),
        "board_trips": float(_board_trips_indicator(board)),
        "monotone_board": float(_monotone_board(board)),
        "two_tone_board": float(_two_tone_board(board)),
        "rainbow_board": float(_rainbow_board(board)),
        "highest_board_rank": float(_highest_board_rank(board)),
        "pot_bb": pot_bb,
        "effective_stack_bb": eff_stack_bb,
        "spr": min(spr, 100),
        "pot_odds": min(pot_odds, 1.0),
        "facing_bet_bb": facing_bet_bb,
        "facing_bet_pct_pot": facing_bet_pct,
        "all_in": all_in,
        "commitment_pct": min(commitment_pct, 100),
        "mdf": min(mdf, 100),
        "opp_aggressive_count": float(_opponent_aggressive_count(dp.action_history, villain_idx)),
        "opp_passive_count": float(_opponent_passive_count(dp.action_history, villain_idx)),
        "last_action_1": last_action_1,
        "last_action_2": last_action_2,
        "villain_checkraise": float(_villain_checkraise_indicator(dp.action_history, villain_idx)),
        "villain_donk": float(_villain_donk_indicator(dp.action_history, villain_idx, dp.street_index)),
        "hero_position": float(hero_idx),
        "villain_position": float(villain_idx),
        "preflop_aggressor_is_hero": preflop_agg_is_hero,
        "street_index": float(dp.street_index),
    }
