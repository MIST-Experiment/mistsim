"""Tests for the beam module."""
import jax.numpy as jnp
import pytest
import warnings

from mistsim.beam import Beam


def test_lmax_warning():
    """Test that a warning is raised when lmax is not None"""
    freqs = jnp.array([50.0, 100.0])
    data = jnp.ones((freqs.size, 181, 360))
    sampling = "mwss"

    with pytest.warns(
        FutureWarning, match="Lmax is now automatically determined"
    ):
        Beam(data, freqs, sampling, lmax=1000)

    # ensure that no warning is raised when lmax is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # treat warnings as errors
        Beam(data, freqs, sampling, lmax=None)
