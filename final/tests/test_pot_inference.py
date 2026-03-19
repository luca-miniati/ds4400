"""Tests for pot inference."""

import pytest

from src.phh_converter.pot_inference import parse_pot, infer_bet_amounts
from src.irc_parser.pdb import PdbRow


class TestParsePot:
    def test_standard_format(self):
        min_bet, total = parse_pot("2/45")
        assert min_bet == 2
        assert total == 45

    def test_larger_numbers(self):
        min_bet, total = parse_pot("4/165")
        assert min_bet == 4
        assert total == 165

    def test_empty_returns_zero(self):
        assert parse_pot("") == (0, 0)
        assert parse_pot("   ") == (0, 0)

    def test_dash_returns_zero(self):
        assert parse_pot("-") == (0, 0)
        assert parse_pot(" - ") == (0, 0)

    def test_zero_pot(self):
        min_bet, total = parse_pot("0/0")
        assert min_bet == 0
        assert total == 0


class TestInferBetAmounts:
    def test_simple_preflop_one_raise(self):
        """Preflop: SB, BB, one raise. Pot goes 0 -> 45."""
        rows = [
            PdbRow("A", "1", 3, 1, "BQ", "-", "-", "-", 1000, 5, 0, None),
            PdbRow("B", "1", 3, 2, "Bc", "-", "-", "-", 1000, 30, 0, None),
            PdbRow("C", "1", 3, 3, "r", "-", "-", "-", 1000, 30, 0, None),
        ]
        amounts = infer_bet_amounts(
            "2/45", "2/45", "2/45", "2/45", rows, sb=5, bb=10
        )
        # One bet/raise from player 2 (pos 3, idx 2). Pot delta 45 - 0 = 45.
        # Blinds 5+10=15, so 30 from actions. One raise.
        assert ("preflop", 2, 0) in amounts
        assert amounts[("preflop", 2, 0)] >= 10

    def test_multi_street_flop_bet(self):
        """Flop has betting: pot goes 45 -> 105."""
        rows = [
            PdbRow("A", "1", 2, 1, "Bc", "b", "-", "-", 1000, 35, 0, None),
            PdbRow("B", "1", 2, 2, "Bk", "c", "-", "-", 1000, 35, 0, None),
        ]
        amounts = infer_bet_amounts(
            "2/45", "2/105", "2/105", "2/105", rows, sb=5, bb=10
        )
        # Flop delta = 60. One bet from player 0, one call from player 1.
        assert ("flop", 0, 0) in amounts
        assert amounts[("flop", 0, 0)] >= 10

    def test_empty_pot_strings(self):
        """Hand with no postflop action (fold preflop)."""
        rows = [
            PdbRow("A", "1", 2, 1, "Bf", "-", "-", "-", 1000, 5, 0, None),
            PdbRow("B", "1", 2, 2, "Bk", "-", "-", "-", 1000, 10, 0, None),
        ]
        amounts = infer_bet_amounts(
            "2/15", "-", "-", "-", rows, sb=5, bb=10
        )
        # No bet/raise actions on later streets
        assert all(k[0] == "preflop" for k in amounts.keys())

    def test_empty_pdb_rows(self):
        amounts = infer_bet_amounts("2/45", "2/105", "2/145", "2/185", [], sb=5, bb=10)
        assert amounts == {}
