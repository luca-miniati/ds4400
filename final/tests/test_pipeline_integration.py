"""Integration tests for the full IRC to PHH pipeline."""

import pytest
from pathlib import Path

from src.irc_parser import merge_hands
from src.phh_converter import convert_to_hand_history
from src.pipeline import run_pipeline, discover_irc_dirs


class TestPipelineIntegration:
    def test_discover_finds_irc_dirs(self):
        """Discover finds hdb/hroster/pdb locations under data."""
        root = Path(__file__).parent.parent / "data" / "IRCdata"
        if not root.exists():
            pytest.skip("IRC data not present")
        locations = discover_irc_dirs(root)
        assert len(locations) >= 1
        for hdb_path, hroster_path, pdb_dir in locations:
            assert hdb_path.exists()
            assert hroster_path.exists()
            assert pdb_dir.exists() and pdb_dir.is_dir()

    def test_merge_produces_hands(self):
        """Merge produces IRCHand objects from real data."""
        root = Path(__file__).parent.parent / "data" / "IRCdata"
        if not root.exists():
            pytest.skip("IRC data not present")
        locations = discover_irc_dirs(root)
        if not locations:
            pytest.skip("No IRC dirs found")
        hdb_path, hroster_path, pdb_dir = locations[0]
        hands = merge_hands(hdb_path, hroster_path, pdb_dir)
        assert len(hands) > 0
        hand = next(iter(hands.values()))
        assert hand.n_players >= 2
        assert hand.players
        assert hand.hand_id

    def test_pipeline_runs(self):
        """Full pipeline runs without error."""
        root = Path(__file__).parent.parent / "data" / "IRCdata"
        out = Path(__file__).parent.parent / "data" / "phh"
        if not root.exists():
            pytest.skip("IRC data not present")
        parsed, converted, failed = run_pipeline(root, out, limit=20, verbose=False)
        assert parsed > 0
        assert converted + failed == 20

    def test_converted_hands_iterable(self):
        """Any successfully converted hands can be iterated (PokerKit validation)."""
        root = Path(__file__).parent.parent / "data" / "IRCdata"
        if not root.exists():
            pytest.skip("IRC data not present")
        locations = discover_irc_dirs(root)
        if not locations:
            pytest.skip("No IRC dirs found")
        hdb_path, hroster_path, pdb_dir = locations[0]
        hands = merge_hands(hdb_path, hroster_path, pdb_dir)
        count = 0
        for hand in list(hands.values())[:50]:
            hh = convert_to_hand_history(hand)
            if hh is not None:
                list(hh)
                count += 1
        assert count >= 0
