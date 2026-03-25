"""Tests for the beam module."""
import warnings

import croissant as cro
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.parametrize("beam_az_rot", [0.0, 10.0, 64.0, 90.0])
def test_beam_az_rot_conversion(beam_az_rot):
    """mistsim beam_az_rot (astro az) maps to croissant beam_rot - 90."""
    nside = 16
    npix = 12 * nside**2
    data = jnp.ones((1, npix))
    freq = jnp.array([100.0])
    beam = Beam(data, freq, sampling="healpix", beam_az_rot=beam_az_rot)
    expected = beam_az_rot - 90.0
    np.testing.assert_allclose(
        float(beam.beam_rot), expected, atol=1e-12
    )


def test_beam_az_rot_matches_croissant():
    """beam_az_rot=0 (North) gives same alm as cro.Beam(beam_rot=-90)."""
    nside = 16
    npix = 12 * nside**2
    phi = jnp.linspace(0, 2 * jnp.pi, npix, endpoint=False)
    data = (1.0 + 0.3 * jnp.cos(phi))[None]
    freq = jnp.array([100.0])

    ms_beam = Beam(data, freq, sampling="healpix", beam_az_rot=0.0)
    cro_beam = cro.Beam(
        data, freq, sampling="healpix", beam_rot=-90.0, niter=0
    )
    np.testing.assert_allclose(
        np.array(ms_beam.compute_alm()),
        np.array(cro_beam.compute_alm()),
        atol=1e-12,
    )
