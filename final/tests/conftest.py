"""Pytest configuration."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (e.g. runs on full IRC dataset)",
    )
