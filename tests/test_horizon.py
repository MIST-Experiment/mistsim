"""Tests for the horizon cut applied when a beam file is loaded."""

import croissant as cro
import numpy as np
import pytest

from mistsim import Beam, pipeline

FREQS = np.array([40.0, 41.0])
NTHETA = 181  # mwss, 1-deg steps including both poles
NPHI = 360
THETA = np.linspace(0, np.pi, NTHETA)

BUILDERS = [pipeline.build_beam, pipeline._build_multi_freq_beam]


def _write_beam(path, theta, gain_theta=None):
    if gain_theta is None:
        gain_theta = np.ones(NTHETA)
    gain = np.broadcast_to(
        gain_theta[None, :, None], (FREQS.size, NTHETA, NPHI)
    )
    phi = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)
    np.savez(path, freqs=FREQS, theta=theta, phi=phi, gain=gain)
    return str(path)


def _build(builder, site_cfg):
    if builder is pipeline.build_beam:
        return builder(site_cfg, FREQS[0])
    return builder(site_cfg, FREQS)


def _weights(beam):
    horizon = np.asarray(beam.horizon)
    assert horizon.shape == (NTHETA, 1)
    return horizon[:, 0]


def _flat_horizon_weights(boundary_row, boundary_weight):
    """1 nearer the zenith, *boundary_weight* on the row, 0 below."""
    expected = np.zeros(NTHETA)
    expected[:boundary_row] = 1.0
    expected[boundary_row] = boundary_weight
    return expected


@pytest.mark.parametrize("builder", BUILDERS)
def test_horizon_max_theta_cuts_radian_beam(tmp_path, builder):
    """horizon_max_theta is in degrees; beam theta is in radians.

    A horizon on a grid row gives that row weight 1/2: it stands for
    the cell [79.5, 80.5] deg, half of which is open sky. Updated
    from the boolean mask, which kept rows 0..80 whole (row 80 had
    weight 1, putting the horizon at 80.5 deg).
    """
    cfg = {
        "beam_file": _write_beam(tmp_path / "beam.npz", THETA),
        "horizon_max_theta": 80,
    }
    beam = _build(builder, cfg)
    np.testing.assert_allclose(
        _weights(beam), _flat_horizon_weights(80, 0.5), atol=1e-12
    )


@pytest.mark.parametrize("builder", BUILDERS)
def test_no_horizon_max_theta_uses_default_horizon(tmp_path, builder):
    """Without horizon_max_theta, croissant's default horizon applies.

    croissant's default puts the horizon at 90 deg with the same
    fractional boundary row, so it matches horizon_max_theta = 90.
    Updated from the boolean check that rows 0..90 were nonzero.
    """
    cfg = {"beam_file": _write_beam(tmp_path / "beam.npz", THETA)}
    beam = _build(builder, cfg)
    np.testing.assert_allclose(
        _weights(beam), _flat_horizon_weights(90, 0.5), atol=1e-12
    )
    at_90 = _build(builder, {**cfg, "horizon_max_theta": 90})
    np.testing.assert_allclose(_weights(at_90), _weights(beam), atol=1e-12)


@pytest.mark.parametrize(
    "horizon_max_theta, boundary_row, boundary_weight",
    [
        (80.3, 80, 0.8),  # row 80 is the cell [79.5, 80.5] deg
        (80.7, 81, 0.2),  # row 81 is the cell [80.5, 81.5] deg
        (45.25, 45, 0.75),
    ],
)
def test_horizon_between_rows_is_linear_in_theta(
    horizon_max_theta, boundary_row, boundary_weight
):
    """The boundary row gets the open share of its cell in theta."""
    cfg = {"beam_file": "beam.npz", "horizon_max_theta": horizon_max_theta}
    weights = pipeline._horizon_from_beam_file(cfg, THETA)
    assert weights.shape == (NTHETA, 1)
    np.testing.assert_allclose(
        weights[:, 0],
        _flat_horizon_weights(boundary_row, boundary_weight),
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "horizon_max_theta", [0.0, 10.0, 45.25, 80.0, 80.3, 90.0, 179.6, 180.0]
)
def test_horizon_weights_match_croissant(horizon_max_theta):
    """The weights are croissant.horizon_weights for the same horizon."""
    cfg = {"beam_file": "beam.npz", "horizon_max_theta": horizon_max_theta}
    weights = pipeline._horizon_from_beam_file(cfg, THETA)
    expected = cro.horizon_weights(
        THETA, theta_h=np.deg2rad(horizon_max_theta)
    )
    assert weights.shape == expected.shape
    np.testing.assert_array_equal(weights, np.asarray(expected))
    assert np.all((weights >= 0) & (weights <= 1))


def _open_integral(gain_name, theta_h):
    """Exact integral of the gain times sin(theta) over [0, theta_h]."""
    c = np.cos(theta_h)
    if gain_name == "uniform":
        return 1 - c
    return 2 / 3 - c + c**3 / 3  # sin^2 gain


@pytest.mark.parametrize("gain_name", ["uniform", "sin2"])
@pytest.mark.parametrize("horizon_max_theta", [80.0, 80.3])
def test_ground_fraction_matches_flat_horizon(
    tmp_path, gain_name, horizon_max_theta
):
    """Fractional weights give the analytic ground fraction.

    Ground fraction minus the analytic value, 1-deg MWSS grid:

    =======  =====  ==================  ==================
    gain     theta  boolean mask (old)  fractional weights
    =======  =====  ==================  ==================
    uniform  80.0   -4.30e-3            -2.2e-6
    uniform  80.3   -1.72e-3            -1.1e-6
    sin^2    80.0   -6.26e-3            -9.6e-6
    sin^2    80.3   -2.51e-3            -4.6e-6
    =======  =====  ==================  ==================

    The boolean mask keeps the boundary row whole, opening up to half
    a row of ground to the sky. At 80 deg it misses 5.0 % (uniform)
    and 4.9 % (sin^2) of the ground the cut adds beyond 90 deg.
    """
    gain_theta = np.ones(NTHETA)
    if gain_name == "sin2":
        gain_theta = np.sin(THETA) ** 2
    cfg = {
        "beam_file": _write_beam(tmp_path / "beam.npz", THETA, gain_theta),
        "horizon_max_theta": horizon_max_theta,
    }
    theta_h = np.deg2rad(horizon_max_theta)
    exact = 1 - _open_integral(gain_name, theta_h) / _open_integral(
        gain_name, np.pi
    )

    beam = pipeline.build_beam(cfg, FREQS[0])
    err_new = float(beam.compute_fgnd()[0]) - exact

    # The boolean mask this function returned before the fix.
    boolean = (THETA <= theta_h) | np.isclose(THETA, theta_h)
    old = Beam(beam.data, beam.freqs, horizon=boolean[:, None])
    err_old = float(old.compute_fgnd()[0]) - exact

    assert abs(err_new) < 2e-5
    assert abs(err_old) > 1e-3
    assert abs(err_new) < abs(err_old) / 100


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
