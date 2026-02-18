"""Shared pytest configuration and fixtures."""

import sys
import os
import pytest
import numpy as np
from unittest import mock


def pytest_configure(config):
    """Configure pytest environment before tests run."""
    # Mock lunarsky module before any imports
    # This prevents network calls during import
    mock_lunarsky = mock.MagicMock()
    mock_lunarsky.MoonLocation = mock.MagicMock()
    sys.modules["lunarsky"] = mock_lunarsky
    sys.modules["lunarsky.spice_utils"] = mock.MagicMock()
    sys.modules["lunarsky.moon"] = mock.MagicMock()

    # Set environment variable to prevent astropy from blocking on
    # download locks during parallel test execution
    os.environ["ASTROPY_DOWNLOAD_CACHE_LOCK_ATTEMPTS"] = "0"


@pytest.fixture(scope="session")
def random_seed():
    """Set a random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_frequencies():
    """Standard set of frequencies for testing."""
    return np.array([50.0, 75.0, 100.0, 125.0, 150.0])


@pytest.fixture
def sample_times():
    """Standard set of observation times (Julian dates) for testing."""
    return np.linspace(2459000.0, 2459001.0, 24)


@pytest.fixture
def sample_location():
    """Standard observer location for testing (Berkeley, CA)."""
    return {
        "lon": -122.2585,
        "lat": 37.8719,
        "alt": 100.0,
    }
