"""Tests for feature extraction module."""

import pytest
from pathlib import Path

from src.feature_extraction.decision_points import (
    DecisionPoint,
    extract_decision_points_from_phh,
    load_phh_directory,
)
from src.feature_extraction.features import compute_features, FEATURE_NAMES
from src.feature_extraction.export import (
    export_decisions_to_csv,
    extract_features_for_decisions,
)

REPO = Path(__file__).resolve().parent.parent
PHH_DIR = REPO / "data" / "phh" / "phh"


class TestDecisionPoints:
    def test_extract_from_phh_returns_tuple(self):
        if not PHH_DIR.exists():
            pytest.skip("PHH data not found")
        phh_files = sorted(PHH_DIR.glob("hands_*.phhs"))
        if not phh_files:
            pytest.skip("No PHH files")
        decisions, n_hands = extract_decision_points_from_phh(
            phh_files[0], limit_hands=5
        )
        assert isinstance(decisions, list)
        assert n_hands <= 5
        assert n_hands >= 1
        if decisions:
            dp = decisions[0]
            assert isinstance(dp, DecisionPoint)
            assert dp.hand_id
            assert dp.label in ("fold", "call", "raise")
            assert dp.street_name in ("preflop", "flop", "turn", "river")

    def test_load_phh_directory(self):
        if not PHH_DIR.exists():
            pytest.skip("PHH data not found")
        decisions = load_phh_directory(PHH_DIR, limit_files=1, limit_hands=10)
        assert isinstance(decisions, list)


class TestFeatures:
    def test_compute_features_returns_dict(self):
        if not PHH_DIR.exists():
            pytest.skip("PHH data not found")
        phh_files = sorted(PHH_DIR.glob("hands_*.phhs"))
        if not phh_files:
            pytest.skip("No PHH files")
        decisions, _ = extract_decision_points_from_phh(
            phh_files[0], limit_hands=3
        )
        if not decisions:
            pytest.skip("No decisions in sample")
        feats = compute_features(decisions[0], bb=10)
        assert isinstance(feats, dict)
        for name in FEATURE_NAMES:
            assert name in feats
        assert feats["street_index"] >= 0
        assert feats["pot_bb"] >= 0

    def test_draw_features_unknown_without_hole_cards(self):
        if not PHH_DIR.exists():
            pytest.skip("PHH data not found")
        phh_files = sorted(PHH_DIR.glob("hands_*.phhs"))
        if not phh_files:
            pytest.skip("No PHH files")
        decisions, _ = extract_decision_points_from_phh(
            phh_files[0], limit_hands=2
        )
        for dp in decisions:
            if dp.street_index > 0 and len(dp.board_cards) >= 3:
                feats = compute_features(dp, bb=10, hole_cards=None)
                assert feats["flush_draw"] in (-1.0, 0.0, 1.0)
                assert feats["straight_draw"] in (-1.0, 0.0, 1.0)
                break
        else:
            pytest.skip("No postflop decision in sample")


class TestExport:
    def test_export_csv_creates_file(self, tmp_path):
        if not PHH_DIR.exists():
            pytest.skip("PHH data not found")
        phh_files = sorted(PHH_DIR.glob("hands_*.phhs"))
        if not phh_files:
            pytest.skip("No PHH files")
        decisions, _ = extract_decision_points_from_phh(
            phh_files[0], limit_hands=3
        )
        if not decisions:
            pytest.skip("No decisions")
        out = tmp_path / "decisions.csv"
        export_decisions_to_csv(decisions, out, bb=10)
        assert out.exists()
        content = out.read_text()
        assert "hand_id" in content
        assert "label" in content
        assert "fold" in content or "call" in content or "raise" in content
