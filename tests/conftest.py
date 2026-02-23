"""Shared pytest configuration and fixtures."""

import numpy as np
import pytest


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
