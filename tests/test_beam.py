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


def _mwss_grid():
    """Degrees of colatitude and longitude on the 1-deg MWSS grid."""
    theta = np.linspace(0.0, 180.0, 181)
    phi = np.arange(360.0)
    return theta, phi


def _ground_sector_mask(lo=80.0, hi=100.0, theta_h=60.0):
    """Ground mask blocking compass azimuth [lo, hi] below theta_h.

    mistsim's ground grid is the beam grid at beam_az_rot = 0, so
    compass azimuth A sits at phi = -A.
    """
    theta, phi = _mwss_grid()
    azimuth = np.mod(-phi, 360.0)
    in_sector = (azimuth >= lo) & (azimuth <= hi)
    return np.where((theta[:, None] > theta_h) & in_sector[None, :], 0.0, 1.0)


def _asymmetric_lobe():
    """A beam with a lobe along its x axis (phi = 0)."""
    theta, phi = _mwss_grid()
    cos_phi = np.cos(np.deg2rad(phi))[None, :]
    sin_theta = np.sin(np.deg2rad(theta))[:, None]
    return jnp.asarray((1.0 + 0.8 * cos_phi * sin_theta)[None])


def test_horizon_frame_defaults_to_topocentric():
    data = jnp.ones((1, 181, 360))
    beam = Beam(data, jnp.array([50.0]))
    assert beam.horizon_frame == "topocentric"


def test_horizon_frame_rejects_unknown_value():
    data = jnp.ones((1, 181, 360))
    with pytest.raises(ValueError, match="horizon_frame"):
        Beam(data, jnp.array([50.0]), horizon_frame="ground")


@pytest.mark.parametrize("beam_az_rot", [0.0, 40.0, 233.0])
def test_default_mask_stays_on_the_ground(beam_az_rot):
    """A ground-fixed sector lands at the same compass azimuth.

    A beam-grid column phi_b points to compass azimuth
    A = beam_az_rot - phi_b, which is ground column -A, so the weights
    applied in the beam frame must equal the ground mask read there.
    """
    _, phi = _mwss_grid()
    mask = _ground_sector_mask()
    beam = Beam(
        jnp.ones((1, 181, 360)),
        jnp.array([50.0]),
        horizon=jnp.asarray(mask),
        beam_az_rot=beam_az_rot,
    )
    applied = np.asarray(beam.horizon_in_beam_frame)
    azimuth = np.mod(beam_az_rot - phi, 360.0)
    ground_col = np.mod(-azimuth, 360.0).astype(int)
    np.testing.assert_array_equal(applied, mask[:, ground_col])
    # the blocked columns are the East sector, whatever the rotation
    blocked = azimuth[(applied == 0).any(axis=0)]
    assert blocked.min() >= 80.0 and blocked.max() <= 100.0
    assert blocked.size == 21


def test_beam_frame_mask_rotates_with_the_beam():
    """horizon_frame="beam" applies the mask as given, at any rotation."""
    mask = _ground_sector_mask()
    for beam_az_rot in (0.0, 40.0):
        beam = Beam(
            jnp.ones((1, 181, 360)),
            jnp.array([50.0]),
            horizon=jnp.asarray(mask),
            beam_az_rot=beam_az_rot,
            horizon_frame="beam",
        )
        np.testing.assert_array_equal(
            np.asarray(beam.horizon_in_beam_frame), mask
        )


def test_frames_agree_at_zero_rotation_mwss():
    """At beam_az_rot = 0 the new default changes nothing."""
    data = _asymmetric_lobe()
    mask = jnp.asarray(_ground_sector_mask())
    topo = Beam(data, jnp.array([50.0]), horizon=mask)
    old = Beam(data, jnp.array([50.0]), horizon=mask, horizon_frame="beam")
    np.testing.assert_array_equal(
        np.asarray(topo.horizon_in_beam_frame),
        np.asarray(old.horizon_in_beam_frame),
    )
    np.testing.assert_array_equal(
        np.asarray(topo.compute_fgnd()), np.asarray(old.compute_fgnd())
    )


def test_frames_agree_at_zero_rotation_healpix():
    """The 90-deg grid shift is exact on every HEALPix ring."""
    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(0)
    mask = jnp.asarray(rng.uniform(size=npix))
    data = jnp.ones((1, npix))
    topo = Beam(data, jnp.array([50.0]), sampling="healpix", horizon=mask)
    old = Beam(
        data,
        jnp.array([50.0]),
        sampling="healpix",
        horizon=mask,
        horizon_frame="beam",
    )
    np.testing.assert_allclose(
        np.asarray(topo.horizon_in_beam_frame),
        np.asarray(old.horizon_in_beam_frame),
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("beam_az_rot", [0.0, 40.0])
def test_theta_only_mask_is_frame_independent(beam_az_rot):
    """The pipeline's theta-only masks behave the same in both frames."""
    theta, _ = _mwss_grid()
    mask = jnp.asarray((theta <= 80.0)[:, None].astype(float))
    kw = dict(horizon=mask, beam_az_rot=beam_az_rot)
    topo = Beam(_asymmetric_lobe(), jnp.array([50.0]), **kw)
    old = Beam(
        _asymmetric_lobe(), jnp.array([50.0]), horizon_frame="beam", **kw
    )
    np.testing.assert_array_equal(
        np.asarray(topo.compute_fgnd()), np.asarray(old.compute_fgnd())
    )


def test_topocentric_fgnd_matches_manual_counter_rotation():
    """An asymmetric beam sees the ground mask a caller would build.

    With a lobe on the beam's x axis, the blocked fraction depends on
    where the lobe points relative to the ground-fixed sector. The
    default (topocentric) result must equal the beam-frame result with
    the mask counter-rotated by hand.
    """
    _, phi = _mwss_grid()
    data = _asymmetric_lobe()
    mask = _ground_sector_mask()
    fgnd = []
    for beam_az_rot in (0.0, 90.0):
        azimuth = np.mod(beam_az_rot - phi, 360.0)
        manual = mask[:, np.mod(-azimuth, 360.0).astype(int)]
        topo = Beam(
            data,
            jnp.array([50.0]),
            horizon=jnp.asarray(mask),
            beam_az_rot=beam_az_rot,
        )
        by_hand = Beam(
            data,
            jnp.array([50.0]),
            horizon=jnp.asarray(manual),
            beam_az_rot=beam_az_rot,
            horizon_frame="beam",
        )
        f_topo = float(np.asarray(topo.compute_fgnd()).ravel()[0])
        f_hand = float(np.asarray(by_hand.compute_fgnd()).ravel()[0])
        np.testing.assert_allclose(f_topo, f_hand, rtol=1e-12)
        fgnd.append(f_topo)
    # the lobe points East at beam_az_rot = 90, into the sector
    assert fgnd[1] > fgnd[0]
