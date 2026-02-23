"""Tests for the beam module."""
import pytest

import jax.numpy as jnp

from mistsim.beam import Beam

def test_lmax_warning():
    """Test that a warning is raised when lmax is not None"""
    freqs = jnp.array([50.0, 100.0])
    data = jnp.ones((freqs.size, 181, 360))
    sampling = "mwss"

    with pytest.warns(
        FutureWarning, match="Lmax is now automatically determined"
    ):
        Beam(freqs, data, sampling, lmax=1000)

    # ensure that no warning is raised when lmax is None
    with pytest.warns(None) as record:
        Beam(freqs, data, sampling, lmax=None)
    assert len(record) == 0
