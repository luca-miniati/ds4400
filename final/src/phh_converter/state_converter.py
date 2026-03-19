"""
State-driven PHH conversion: drive PokerKit's state machine directly.

When an action fails, we raise a clear error indicating which action broke.
"""

from ..irc_parser.merger import IRCHand
from .betting_round import (
    interleave_street,
    ActionType,
    get_preflop_order,
    get_postflop_order,
)
from .pot_inference import infer_bet_amounts

DEFAULT_SB = 5
DEFAULT_BB = 10


def _cards_to_phh(cards_str: str | None) -> str:
    """Convert IRC card format '5s Qs' to PHH format '5sQs'. Use ?? for unknown."""
    if not cards_str:
        return "??"
    result = []
    for part in str(cards_str).split():
        part = part.strip().upper().replace("10", "T")
        if len(part) >= 2:
            result.append(part[0] + part[1].lower())
        else:
            result.append(part)
    return "".join(result)


def _board_to_phh(s: str) -> str:
    """Normalize board cards: rank upper, suit lower."""
    out = []
    s = (s or "").replace(" ", "").replace("10", "T")
    for i in range(0, len(s), 2):
        if i + 1 < len(s):
            out.append(s[i].upper() + s[i + 1].lower())
    return "".join(out)


class ConversionError(Exception):
    """Raised when conversion fails, with context about which action broke."""

    def __init__(self, message: str, street: str = "", player_idx: int | None = None, action_type: str = ""):
        self.street = street
        self.player_idx = player_idx
        self.action_type = action_type
        super().__init__(message)


def convert_to_hand_history_state_driven(
    hand: IRCHand,
    sb: int = DEFAULT_SB,
    bb: int = DEFAULT_BB,
    *,
    raise_on_error: bool = True,
) -> "HandHistory | None":
    """
    Convert IRCHand to PokerKit HandHistory by driving the state machine.

    Applies actions one-by-one via state.fold(), state.check_or_call(),
    state.complete_bet_or_raise_to(). When an action fails, raises
    ConversionError with the exact action that broke (unless raise_on_error=False,
    in which case returns None).
    """
    try:
        from pokerkit import (
            NoLimitTexasHoldem,
            Automation,
            HandHistory,
        )
    except ImportError:
        raise ImportError("pokerkit is required. Install with: pip install pokerkit")

    n = hand.n_players
    if n < 2:
        return None

    pos_to_row = {r.position: r for r in hand.pdb_rows}
    if not pos_to_row:
        return None

    # Only convert when we have complete PDB coverage (all positions 1..n).
    # Partial PDB forces roster-based fill which is often wrong.
    positions_covered = {r.position for r in hand.pdb_rows if 1 <= r.position <= n}
    if len(hand.pdb_rows) != n or positions_covered != set(range(1, n + 1)):
        return None

    stacks = hand.get_player_stacks()
    inf = 10_000_000
    stacks = [s if 0 < s < inf else 2000 for s in stacks]
    stacks_tuple = tuple(stacks)

    antes = (0,) * n
    # Blinds: for 2 players PHH order is BB first, SB second; blinds = (BB, SB)
    # For 3+ players: (SB, BB, 0, ...)
    if n == 2:
        blinds = (bb, sb)
    else:
        blinds = (sb, bb) + (0,) * (n - 2)

    # Include HOLE_CARDS_SHOWING_OR_MUCKING so state enters showdown phase; we drive it manually
    # with known cards from pdb (automation would auto-muck since we dealt "??")
    automations = (
        Automation.ANTE_POSTING,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.BET_COLLECTION,
        Automation.CARD_BURNING,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
    )

    game = NoLimitTexasHoldem(
        automations,
        True,
        0,
        blinds,
        bb,
    )
    state = game(stacks_tuple, n)

    # Post blinds if not already done by create_state
    while state.can_post_blind_or_straddle():
        state.post_blind_or_straddle()

    amounts = infer_bet_amounts(
        hand.pot_preflop,
        hand.pot_flop,
        hand.pot_turn,
        hand.pot_river,
        hand.pdb_rows,
        sb,
        bb,
    )

    def get_amount(street: str, player_idx: int, bet_idx: int) -> int:
        # infer_bet_amounts keys by (street, pos-1, bet_idx); we use player_idx (pos-1)
        return amounts.get((street, player_idx, bet_idx), bb * 2)

    _empty = type("_", (), {"preflop_actions": "", "flop_actions": "", "turn_actions": "", "river_actions": ""})()

    def has_any_actions(get_actions) -> bool:
        for pos in range(1, n + 1):
            s = get_actions(pos_to_row.get(pos) or _empty) or ""
            s = (s or "").replace("B", "").replace("-", "").strip()
            if s:
                return True
        return False

    def apply_street_actions(street: str, position_order: list[int], get_actions) -> bool:
        """Apply street actions to state. Returns False if hand ended."""
        player_actions = {pos: get_actions(pos_to_row.get(pos) or _empty) for pos in position_order}
        if not any(player_actions.values()):
            return True

        # Don't pass pos_to_player_idx - use default pos->idx (roster matches position)
        # Passing pos_to_player_idx causes mismatches with PokerKit's expected order
        interleaved = interleave_street(n, position_order, player_actions)
        bet_idx: dict[int, int] = {}
        active_count = n

        for player_idx, action_type in interleaved:
            # Run automations until it's this player's turn
            while state.status:
                if state.actor_index == player_idx:
                    break
                # Run one automation step
                if state.can_post_ante():
                    state.post_ante()
                elif state.can_collect_bets():
                    state.collect_bets()
                elif state.can_post_blind_or_straddle():
                    state.post_blind_or_straddle()
                elif state.can_burn_card():
                    state.burn_card("??")
                elif state.can_deal_hole():
                    # Shouldn't reach here mid-street
                    break
                elif state.can_deal_board():
                    break
                else:
                    break

            if state.actor_index != player_idx:
                raise ConversionError(
                    f"Expected player {player_idx} to act but state has actor_index={state.actor_index}",
                    street=street,
                    player_idx=player_idx,
                    action_type=action_type.name,
                )

            try:
                if action_type == ActionType.FOLD:
                    state.fold()
                    active_count -= 1
                    if active_count <= 1:
                        return False
                elif action_type == ActionType.CHECK_OR_CALL:
                    state.check_or_call()
                elif action_type == ActionType.BET_OR_RAISE:
                    idx = bet_idx.get(player_idx, 0)
                    amt = get_amount(street, player_idx, idx)
                    # Clamp to valid range (min raise, max = stack/cap)
                    if state.can_complete_bet_or_raise_to():
                        min_amt = state.min_completion_betting_or_raising_to_amount
                        max_amt = state.max_completion_betting_or_raising_to_amount
                        amt = max(min_amt, min(amt, max_amt))
                    state.complete_bet_or_raise_to(amt)
                    bet_idx[player_idx] = idx + 1
            except ValueError as e:
                raise ConversionError(
                    f"Action failed: {e}",
                    street=street,
                    player_idx=player_idx,
                    action_type=action_type.name,
                ) from e

        return True

    try:
        # 1. Run automations until we need to deal hole cards
        while state.can_post_ante():
            state.post_ante()
        while state.can_post_blind_or_straddle():
            state.post_blind_or_straddle()

        # 2. Deal hole cards (one card per call; state manages order)
        while state.can_deal_hole():
            state.deal_hole("??")

        # 3. Preflop
        pf_order = get_preflop_order(n)
        if not apply_street_actions("preflop", pf_order, lambda r: r.preflop_actions):
            # Hand ended (fold to one player)
            hh = HandHistory.from_game_state(game, state, players=hand.players, hand=hand.hand_id)
            return hh

        # 4. Flop
        board = _board_to_phh(hand.board_cards or "")
        post_order = get_postflop_order(n)
        if len(board) >= 6 and has_any_actions(lambda r: r.flop_actions):
            while state.can_burn_card():
                state.burn_card("??")
            if state.can_deal_board():
                state.deal_board(board[:6])
            if not apply_street_actions("flop", post_order, lambda r: r.flop_actions):
                hh = HandHistory.from_game_state(game, state, players=hand.players, hand=hand.hand_id)
                return hh

        # 5. Turn
        if len(board) >= 8 and has_any_actions(lambda r: r.turn_actions):
            while state.can_burn_card():
                state.burn_card("??")
            if state.can_deal_board():
                state.deal_board(board[6:8])
            if not apply_street_actions("turn", post_order, lambda r: r.turn_actions):
                hh = HandHistory.from_game_state(game, state, players=hand.players, hand=hand.hand_id)
                return hh

        # 6. River
        if len(board) >= 10 and has_any_actions(lambda r: r.river_actions):
            while state.can_burn_card():
                state.burn_card("??")
            if state.can_deal_board():
                state.deal_board(board[8:10])
            apply_street_actions("river", post_order, lambda r: r.river_actions)

        # Build player index -> hole cards from pdb (for showdown)
        idx_to_hole_cards: dict[int, str] = {}
        for row in hand.pdb_rows:
            if row.hole_cards and row.player in hand.players:
                idx = hand.players.index(row.player)
                idx_to_hole_cards[idx] = _cards_to_phh(row.hole_cards)

        # Run remaining automations; manually show hole cards at showdown
        while state.status:
            if state.can_collect_bets():
                state.collect_bets()
            elif state.can_burn_card():
                state.burn_card("??")
            elif (idx := state.showdown_index) is not None and state.can_show_or_muck_hole_cards(
                cards := (idx_to_hole_cards.get(idx) if idx in idx_to_hole_cards else False),
                idx,
            ):
                # Show known cards from pdb or muck (False); can_* must be called with same args
                state.show_or_muck_hole_cards(cards if cards else False, idx)
            elif state.can_deal_board():
                break
            elif state.can_kill_hand():
                state.kill_hand()
            elif state.can_push_chips():
                state.push_chips()
            elif state.can_pull_chips():
                state.pull_chips()
            else:
                break

        hh = HandHistory.from_game_state(game, state, players=hand.players, hand=hand.hand_id)
        return hh

    except ConversionError:
        if raise_on_error:
            raise
        return None
    except Exception as e:
        if raise_on_error:
            raise ConversionError(str(e)) from e
        return None
