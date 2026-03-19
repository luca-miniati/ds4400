"""Tests for PHH (PokerKit) conversion - state-driven approach."""

import pytest

from src.phh_converter import convert_to_hand_history, ConversionError
from src.phh_converter.state_converter import convert_to_hand_history_state_driven
from src.irc_parser.merger import IRCHand
from src.irc_parser.pdb import PdbRow


def _make_hand(
    hand_id: str = "1",
    n_players: int = 2,
    players: list[str] | None = None,
    board_cards: str = "2h 4s 4c 2d 5c",
    pot_preflop: str = "2/25",
    pot_flop: str = "2/25",
    pot_turn: str = "2/25",
    pot_river: str = "2/25",
    pdb_rows: list | None = None,
) -> IRCHand:
    if players is None:
        players = [f"P{i+1}" for i in range(n_players)]
    if pdb_rows is None:
        pdb_rows = []
    return IRCHand(
        hand_id=hand_id,
        n_players=n_players,
        players=players,
        board_cards=board_cards,
        pot_preflop=pot_preflop,
        pot_flop=pot_flop,
        pot_turn=pot_turn,
        pot_river=pot_river,
        pdb_rows=pdb_rows,
    )


def _assert_valid_phh(hh) -> None:
    """Consume HandHistory iterator to validate."""
    assert hh is not None
    list(hh)


# ---------------------------------------------------------------------------
# Basic / smoke
# ---------------------------------------------------------------------------


class TestBasic:
    def test_hand_without_pdb_returns_none(self):
        hand = _make_hand(pdb_rows=[])
        assert convert_to_hand_history(hand) is None

    def test_single_player_returns_none(self):
        hand = _make_hand(n_players=1, pdb_rows=[])
        assert convert_to_hand_history(hand) is None


# ---------------------------------------------------------------------------
# Two-player hands
# ---------------------------------------------------------------------------


class TestTwoPlayer:
    """2-handed: positions 1=SB, 2=BB; order [1, 2]."""

    def test_sb_fold(self):
        """SB folds to BB."""
        hand = _make_hand(
            n_players=2,
            players=["Alice", "Bob"],
            pdb_rows=[
                PdbRow("Alice", "1", 2, 1, "BQ", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("Bob", "1", 2, 2, "Bk", "-", "-", "-", 1000, 10, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)
        assert hh.players == ["Alice", "Bob"]

    def test_bb_fold(self):
        """BB folds to SB raise."""
        hand = _make_hand(
            n_players=2,
            pot_preflop="2/25",
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "Br", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bf", "-", "-", "-", 1000, 10, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)

    def test_check_down_preflop(self):
        """Both check preflop."""
        hand = _make_hand(
            n_players=2,
            pot_preflop="2/15",
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "Bk", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bk", "-", "-", "-", 1000, 10, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)


# ---------------------------------------------------------------------------
# Three-player hands
# ---------------------------------------------------------------------------


class TestThreePlayer:
    """3-handed: 1=SB, 2=BB, 3=BTN; preflop order [3, 1, 2]."""

    def test_btn_raise_sb_fold_bb_call(self):
        hand = _make_hand(
            n_players=3,
            players=["SB", "BB", "BTN"],
            pot_preflop="2/35",
            pdb_rows=[
                PdbRow("SB", "1", 3, 1, "BQf", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 3, 2, "BkQ", "-", "-", "-", 1000, 10, 0, None),
                PdbRow("BTN", "1", 3, 3, "BQk", "-", "-", "-", 1000, 0, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)
        assert hh.players == ["SB", "BB", "BTN"]

    def test_btn_raise_sb_fold_bb_call_explicit(self):
        """Explicit: BTN raise, SB fold, BB call."""
        hand = _make_hand(
            n_players=3,
            players=["SB", "BB", "BTN"],
            pot_preflop="2/35",
            pdb_rows=[
                PdbRow("SB", "1", 3, 1, "Bf", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 3, 2, "Bc", "-", "-", "-", 1000, 10, 0, None),
                PdbRow("BTN", "1", 3, 3, "Br", "-", "-", "-", 1000, 0, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)

    def test_all_fold_to_btn(self):
        """BTN raises, both SB and BB fold."""
        hand = _make_hand(
            n_players=3,
            pot_preflop="2/25",
            pdb_rows=[
                PdbRow("SB", "1", 3, 1, "Bf", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 3, 2, "Bf", "-", "-", "-", 1000, 10, 0, None),
                PdbRow("BTN", "1", 3, 3, "Br", "-", "-", "-", 1000, 0, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)

    def test_three_way_check_preflop(self):
        hand = _make_hand(
            n_players=3,
            pot_preflop="2/15",
            pdb_rows=[
                PdbRow("SB", "1", 3, 1, "Bk", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 3, 2, "Bk", "-", "-", "-", 1000, 10, 0, None),
                PdbRow("BTN", "1", 3, 3, "Bk", "-", "-", "-", 1000, 0, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)


# ---------------------------------------------------------------------------
# Four-player hands
# ---------------------------------------------------------------------------


class TestFourPlayer:
    """4-handed: 1=SB, 2=BB, 3=UTG, 4=BTN; preflop order [3, 4, 1, 2]."""

    def test_utg_fold_btn_raise_sb_fold_bb_call(self):
        """UTG folds, BTN raises, SB folds, BB calls."""
        hand = _make_hand(
            n_players=4,
            players=["SB", "BB", "UTG", "BTN"],
            pot_preflop="2/45",
            pdb_rows=[
                PdbRow("SB", "1", 4, 1, "Bf", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 4, 2, "Bc", "-", "-", "-", 1000, 10, 0, None),
                PdbRow("UTG", "1", 4, 3, "f", "-", "-", "-", 1000, 0, 0, None),
                PdbRow("BTN", "1", 4, 4, "Br", "-", "-", "-", 1000, 0, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)

    def test_all_fold_to_btn_preflop(self):
        hand = _make_hand(
            n_players=4,
            pot_preflop="2/25",
            pdb_rows=[
                PdbRow("SB", "1", 4, 1, "Bf", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 4, 2, "Bf", "-", "-", "-", 1000, 10, 0, None),
                PdbRow("UTG", "1", 4, 3, "f", "-", "-", "-", 1000, 0, 0, None),
                PdbRow("BTN", "1", 4, 4, "Br", "-", "-", "-", 1000, 0, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)


# ---------------------------------------------------------------------------
# Multi-street hands (flop, turn, river)
# ---------------------------------------------------------------------------


class TestMultiStreet:
    def test_preflop_fold_no_flop(self):
        """Hand ends preflop - no flop dealt."""
        hand = _make_hand(
            n_players=2,
            pot_preflop="2/15",
            pot_flop="2/15",
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "BQ", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bk", "-", "-", "-", 1000, 10, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)

    def test_check_to_flop_then_fold(self):
        """Preflop check, flop bet, fold."""
        hand = _make_hand(
            n_players=2,
            board_cards="2h 4s 4c 2d 5c",
            pot_preflop="2/15",
            pot_flop="2/35",
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "Bk", "b", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bk", "f", "-", "-", 1000, 10, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)

    def test_check_down_all_streets(self):
        """Check through preflop, flop, turn, river."""
        hand = _make_hand(
            n_players=2,
            board_cards="2h 4s 4c 2d 5c",
            pot_preflop="2/15",
            pot_flop="2/15",
            pot_turn="2/15",
            pot_river="2/15",
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "Bk", "k", "k", "k", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bk", "k", "k", "k", 1000, 10, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)

    def test_three_way_to_showdown_with_hole_cards(self):
        """Three players see showdown, hole cards shown."""
        hand = _make_hand(
            n_players=3,
            players=["SB", "BB", "BTN"],
            board_cards="2h 4s 4c 2d 5c",
            pot_preflop="2/15",
            pot_flop="2/15",
            pot_turn="2/15",
            pot_river="2/15",
            pdb_rows=[
                PdbRow("SB", "1", 3, 1, "Bk", "k", "k", "k", 1000, 5, 0, "As Ks"),
                PdbRow("BB", "1", 3, 2, "Bk", "k", "k", "k", 1000, 10, 0, "Ad Kd"),
                PdbRow("BTN", "1", 3, 3, "Bk", "k", "k", "k", 1000, 0, 0, "Ac Kh"),
            ],
        )
        hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)


# ---------------------------------------------------------------------------
# ConversionError behavior
# ---------------------------------------------------------------------------


class TestConversionError:
    def test_raise_on_error_surfaces_exact_action(self):
        """ConversionError includes street, player_idx, action_type when action fails."""
        # Use hand with invalid action: raise when already matched (IRC data error)
        # This hand has extra raise in action string that PokerKit rejects
        from unittest.mock import patch

        hand = _make_hand(
            n_players=2,
            players=["SB", "BB"],
            pot_preflop="2/20",
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "Bk", "br", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bk", "ck", "-", "-", 1000, 10, 0, None),
            ],
        )
        # Force a bet then "raise" when we should check - triggers "already acted"
        # Simpler: patch to make complete_bet_or_raise_to raise
        with patch(
            "pokerkit.state.State.complete_bet_or_raise_to",
            side_effect=ValueError("Test inject"),
        ):
            with pytest.raises(ConversionError) as exc_info:
                convert_to_hand_history_state_driven(hand, raise_on_error=True)
        err = exc_info.value
        assert err.street != ""
        assert err.action_type == "BET_OR_RAISE"

    def test_raise_on_error_false_returns_none(self):
        """Invalid actions yield None when raise_on_error=False."""
        from unittest.mock import patch

        hand = _make_hand(
            n_players=2,
            players=["SB", "BB"],
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "Bk", "b", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bk", "c", "-", "-", 1000, 10, 0, None),
            ],
        )
        with patch(
            "pokerkit.state.State.complete_bet_or_raise_to",
            side_effect=ValueError("Test inject"),
        ):
            hh = convert_to_hand_history(hand, raise_on_error=False)
        assert hh is None


# ---------------------------------------------------------------------------
# Min-raise clamping (inferred amount too low)
# ---------------------------------------------------------------------------


class TestMinRaiseClamping:
    """Ensure we clamp to min raise when infer_bet_amounts returns too low."""

    def test_inferred_raise_below_min_clamped(self):
        """Pot inference returns 20 but min raise is 40 - we clamp and succeed."""
        hand = _make_hand(
            n_players=2,
            pot_preflop="2/35",
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "Br", "-", "-", "-", 1000, 5, 0, None),
                PdbRow("BB", "1", 2, 2, "Bc", "-", "-", "-", 1000, 10, 0, None),
            ],
        )
        # infer_bet_amounts might return 20 for a small pot; min raise is 2*BB=20 for first raise
        # Actually for SB vs BB, min raise from SB is to 20 (BB). So 20 is valid. Use a case
        # where we'd get a too-low value - e.g. delta split yields < min.
        from unittest.mock import patch

        with patch("src.phh_converter.state_converter.infer_bet_amounts") as mock_infer:
            mock_infer.return_value = {("preflop", 0, 0): 1}  # Way too low
            hh = convert_to_hand_history(hand)
        _assert_valid_phh(hh)


# ---------------------------------------------------------------------------
# Custom blinds
# ---------------------------------------------------------------------------


class TestCustomBlinds:
    def test_sb_bb_10_20(self):
        hand = _make_hand(
            n_players=2,
            pdb_rows=[
                PdbRow("SB", "1", 2, 1, "BQ", "-", "-", "-", 2000, 10, 0, None),
                PdbRow("BB", "1", 2, 2, "Bk", "-", "-", "-", 2000, 20, 0, None),
            ],
        )
        hh = convert_to_hand_history(hand, sb=10, bb=20)
        _assert_valid_phh(hh)


# ---------------------------------------------------------------------------
# Conversion rate (integration)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConversionRate:
    """Run conversion on full IRC dataset - target >= 82% of eligible hands."""

    def test_conversion_rate_meets_target(self):
        """
        At least 82% of hands with complete PDB should convert.

        We only attempt conversion for hands with complete PDB coverage.
        Use: pytest -m slow to run this test on the full dataset.
        """
        from pathlib import Path

        from src.irc_parser import merge_hands

        locations = []
        for hdb_path in Path("data/IRCdata").rglob("hdb"):
            if hdb_path.is_file():
                parent = hdb_path.parent
                parts = parent.parts
                if "holdem1" not in parts and "holdem2" not in parts and "holdem3" not in parts:
                    continue
                hroster = parent / "hroster"
                pdb_dir = parent / "pdb"
                if hroster.exists() and pdb_dir.exists():
                    locations.append((hdb_path, hroster, pdb_dir))

        if not locations:
            pytest.skip("No IRC data found")

        total_eligible = 0
        converted = 0
        for hdb_path, hroster_path, pdb_dir in locations:
            hands = merge_hands(hdb_path, hroster_path, pdb_dir)
            for hand in hands.values():
                if len(hand.pdb_rows) == 0:
                    continue
                positions_covered = {r.position for r in hand.pdb_rows if 1 <= r.position <= hand.n_players}
                if len(hand.pdb_rows) != hand.n_players or positions_covered != set(range(1, hand.n_players + 1)):
                    continue
                total_eligible += 1
                hh = convert_to_hand_history(hand, raise_on_error=False)
                if hh is not None:
                    converted += 1

        if total_eligible == 0:
            pytest.skip("No eligible hands with complete PDB")

        rate = 100 * converted / total_eligible
        assert rate >= 82, (
            f"Conversion rate {rate:.1f}% is below 82%. "
            f"Converted {converted}/{total_eligible} eligible hands."
        )
