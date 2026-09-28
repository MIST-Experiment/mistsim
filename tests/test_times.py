"""Time grid of the mapmaking pipeline.

Earth rotation turns the sky about the equatorial z axis, so a beam's
timestream simulated at ``lmax_sim`` is a Fourier series in the rotation
angle with ``|m| <= lmax_sim``. One sidereal day sampled at
``2 lmax_sim + 1`` evenly spaced rotation angles measures every m-mode.
"""

import logging

import croissant as cro
import numpy as np

from mistsim import pipeline


def _obs(**kw):
    obs = {
        "start_time": "2026-02-25 12:27",
        "n_sidereal_days": 1.0,
        "lmax": 10,
        "lmax_sim": 20,
    }
    obs.update(kw)
    return obs


def _seconds(times):
    """Seconds since the first sample, as croissant's Simulator gets it."""
    return (times.jd - times.jd[0]) * 24 * 3600


def test_default_n_times_is_nyquist_of_forward_model():
    assert pipeline.resolve_n_times(_obs()) == 2 * 20 + 1


def test_default_n_times_falls_back_to_lmax():
    assert pipeline.resolve_n_times(_obs(lmax_sim=None)) == 2 * 10 + 1


def test_default_n_times_scales_with_days():
    obs = _obs(n_sidereal_days=2.0)
    assert pipeline.resolve_n_times(obs) == 2 * (2 * 20 + 1)


def test_explicit_n_times_overrides_default():
    assert pipeline.resolve_n_times(_obs(n_times=500)) == 500


def test_warns_below_nyquist(caplog):
    with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
        pipeline.resolve_n_times(_obs(n_times=2 * 20))
    assert "Nyquist" in caplog.text


def test_times_are_evenly_spaced_rotation_angles():
    n = 2 * 20 + 1
    times = pipeline._make_times(_obs(n_times=n))
    phi = 2 * np.pi * _seconds(times) / cro.constants.sidereal_day["earth"]
    np.testing.assert_allclose(phi, 2 * np.pi * np.arange(n) / n, atol=1e-6)


def test_nyquist_grid_measures_every_m_mode():
    lmax_sim = 20
    times = pipeline._make_times(_obs(n_times=2 * lmax_sim + 1))
    phases = cro.simulator.rot_alm_z(
        lmax_sim, times=_seconds(times), world="earth"
    )
    s = np.linalg.svd(np.asarray(phases), compute_uv=False)
    assert s.min() / s.max() > 0.99
