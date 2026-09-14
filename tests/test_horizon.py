"""Tests for the horizon cut applied when a beam file is loaded."""

import numpy as np
import pytest

from mistsim import pipeline

FREQS = np.array([40.0, 41.0])
NTHETA = 181  # mwss, 1-deg steps including both poles
NPHI = 360

BUILDERS = [pipeline.build_beam, pipeline._build_multi_freq_beam]


def _write_beam(path, theta):
    gain = np.ones((FREQS.size, NTHETA, NPHI))
    phi = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)
    np.savez(path, freqs=FREQS, theta=theta, phi=phi, gain=gain)
    return str(path)


def _build(builder, site_cfg):
    if builder is pipeline.build_beam:
        return builder(site_cfg, FREQS[0])
    return builder(site_cfg, FREQS)


def _above_horizon_rows(beam):
    horizon = np.asarray(beam.horizon)
    assert horizon.shape == (NTHETA, 1)
    return np.flatnonzero(horizon[:, 0])


@pytest.mark.parametrize("builder", BUILDERS)
def test_horizon_max_theta_cuts_radian_beam(tmp_path, builder):
    """horizon_max_theta is in degrees; beam theta is in radians."""
    theta = np.linspace(0, np.pi, NTHETA)
    cfg = {
        "beam_file": _write_beam(tmp_path / "beam.npz", theta),
        "horizon_max_theta": 80,
    }
    beam = _build(builder, cfg)
    # theta = 0..80 deg inclusive stays above the horizon
    np.testing.assert_array_equal(_above_horizon_rows(beam), np.arange(81))


@pytest.mark.parametrize("builder", BUILDERS)
def test_no_horizon_max_theta_uses_default_horizon(tmp_path, builder):
    theta = np.linspace(0, np.pi, NTHETA)
    cfg = {"beam_file": _write_beam(tmp_path / "beam.npz", theta)}
    beam = _build(builder, cfg)
    # croissant default: theta = 0..90 deg inclusive
    np.testing.assert_array_equal(_above_horizon_rows(beam), np.arange(91))


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("horizon_max_theta", [None, 80])
def test_degree_theta_beam_file_rejected(
    tmp_path, builder, horizon_max_theta
):
    """Beam files must store theta in radians."""
    theta = np.arange(NTHETA, dtype=float)  # 0..180 deg
    cfg = {"beam_file": _write_beam(tmp_path / "beam.npz", theta)}
    if horizon_max_theta is not None:
        cfg["horizon_max_theta"] = horizon_max_theta
    with pytest.raises(ValueError, match="radians"):
        _build(builder, cfg)
