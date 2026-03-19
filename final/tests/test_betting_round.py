"""Tests for betting-round interleaver."""

import pytest

from src.phh_converter.betting_round import (
    interleave_street,
    ActionType,
    get_preflop_order,
    get_postflop_order,
)


class TestGetPreflopOrder:
    def test_two_players(self):
        assert get_preflop_order(2) == [1, 2]

    def test_three_players_btn_first(self):
        """3-way: BTN (UTG) acts first, then SB, then BB."""
        order = get_preflop_order(3)
        assert order == [3, 1, 2]

    def test_four_plus_players(self):
        """4-way: UTG (pos 3) first, then BTN (4), SB (1), BB (2)."""
        assert get_preflop_order(4) == [3, 4, 1, 2]
        assert get_preflop_order(5) == [3, 4, 5, 1, 2]
        assert get_preflop_order(6) == [3, 4, 5, 6, 1, 2]


class TestGetPostflopOrder:
    def test_sequential(self):
        assert get_postflop_order(3) == [1, 2, 3]
        assert get_postflop_order(4) == [1, 2, 3, 4]


class TestInterleaveStreet:
    def test_single_fold(self):
        """BTN folds, SB folds, hand ends."""
        actions = interleave_street(
            n_players=3,
            position_order=[3, 1, 2],
            player_actions={1: "BQ", 2: "Bc", 3: "f"},
        )
        assert len(actions) >= 1
        assert actions[0] == (2, ActionType.FOLD)
        assert actions[1] == (0, ActionType.FOLD)

    def test_fold_then_raise_then_call(self):
        """3-way: BTN raises, SB folds, BB calls."""
        actions = interleave_street(
            n_players=3,
            position_order=[3, 1, 2],  # BTN first
            player_actions={1: "BQ", 2: "Bc", 3: "r"},
        )
        assert actions[0] == (2, ActionType.BET_OR_RAISE)  # BTN=index 2
        assert actions[1] == (0, ActionType.FOLD)  # SB=index 0
        assert actions[2] == (1, ActionType.CHECK_OR_CALL)  # BB=index 1

    def test_check_down(self):
        """All check on a street."""
        actions = interleave_street(
            n_players=2,
            position_order=[1, 2],
            player_actions={1: "k", 2: "k"},
        )
        assert actions == [(0, ActionType.CHECK_OR_CALL), (1, ActionType.CHECK_OR_CALL)]

    def test_bet_call(self):
        """Player 1 bets, player 2 calls."""
        actions = interleave_street(
            n_players=2,
            position_order=[1, 2],
            player_actions={1: "b", 2: "c"},
        )
        assert actions == [(0, ActionType.BET_OR_RAISE), (1, ActionType.CHECK_OR_CALL)]

    def test_raise_reopens_betting(self):
        """Bet, raise, call - raise reopens so bettor can act again."""
        actions = interleave_street(
            n_players=2,
            position_order=[1, 2],
            player_actions={1: "br", 2: "rc"},  # 1 bets, 2 raises, 1 reraises, 2 calls
        )
        assert actions[0] == (0, ActionType.BET_OR_RAISE)
        assert actions[1] == (1, ActionType.BET_OR_RAISE)
        assert actions[2] == (0, ActionType.BET_OR_RAISE)
        assert actions[3] == (1, ActionType.CHECK_OR_CALL)

    def test_ignores_blind_char(self):
        """B in action string is ignored (blinds auto-posted)."""
        actions = interleave_street(
            n_players=2,
            position_order=[1, 2],
            player_actions={1: "Bc", 2: "Bk"},
        )
        assert actions == [(0, ActionType.CHECK_OR_CALL), (1, ActionType.CHECK_OR_CALL)]
