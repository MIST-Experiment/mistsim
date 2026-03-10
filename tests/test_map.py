
import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

import mistsim as ms


@pytest.fixture
def sim():
    nside = 16
    npix = 12 * nside**2
    freqs = jnp.linspace(50, 100, num=50)
    ntimes = 12
    t0 = Time("2024-01-01T00:00:00")
    dt = 1 * u.sday / ntimes
    times = cro.utils.time_array(t_start=t0, N_times=ntimes, delta_t=dt)
    lon = -90
    lat = 80

    tsky = 180 * (freqs / 180)**(-2.5)
    tsky = tsky[:, None] * jnp.ones((1, npix))
    sky = ms.Sky(tsky, freqs)

    beam_map = jnp.ones((freqs.size, 181, 360))
    beam = ms.Beam(beam_map, freqs, sampling="mwss")

    s = ms.Simulator(beam, sky, times.jd, freqs, lon, lat, Tgnd=300)
    return s

@pytest.mark.parametrize("lmax", [16, 32])
def test_pack_unpack_symmetry(lmax):
    Nfreq = 2
    N_total = Nfreq * (lmax + 1)**2

    # 1. Create a random physical (real) state vector
    x_orig = jax.random.normal(jax.random.PRNGKey(0), (N_total,))

    # 2. Unpack to complex s2fft format
    flm_complex = ms.mapmaking.unpack_real_to_s2fft(x_orig, lmax, Nfreq)

    # 3. Test conjugate symmetry: a(l, -m) == (-1)^m * a(l, m)*
    for m in range(1, lmax + 1):
        left_side = flm_complex[:, m:, lmax - m]
        right_side = ((-1)**m) * jnp.conj(flm_complex[:, m:, lmax + m])
        assert jnp.allclose(left_side, right_side), f"Symmetry failed at m={m}"

    # 4. Pack back to 1D real vector and test losslessness
    x_repacked = ms.mapmaking.pack_s2fft_to_real(flm_complex)
    assert jnp.allclose(x_orig, x_repacked)

def test_forward(sim):
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    lmax = cro.utils.lmax_from_shape(sky_alm.shape)
    x_real = ms.mapmaking.pack_s2fft_to_real(sky_alm)

    beam_alm = sim.compute_beam_eq()
    beam_alm = cro.utils.reduce_lmax(beam_alm, lmax)
    phases = sim.phases

    expected_raw = sim.sim()
    # need to ground loss correct since mapmaker assumes it's done
    expected = cro.simulator.correct_ground_loss(
        expected_raw, sim.beam.compute_fgnd(), sim.Tgnd
    )

    # out is a column vector of shape (ntimes*nfreq, 1)
    out = ms.mapmaking._forward(x_real, beam_alm, phases)
    assert out.shape == (sim.freqs.size * sim.times_jd.size,)
    out2d = out.reshape(sim.times_jd.size, sim.freqs.size)
    assert np.allclose(out2d, expected)

def test_A(sim):
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    x_real = ms.mapmaking.pack_s2fft_to_real(sky_alm)

    Amat = ms.mapmaking.make_Amat(sim)
    ntimes = sim.times_jd.size
    nfreq = sim.freqs.size
    assert Amat.shape == (ntimes * nfreq, x_real.size)

    A_wfall_rav = Amat @ x_real
    A_wfall = A_wfall_rav.reshape(ntimes, nfreq)
    # compare with simulator
    sim_wfall = sim.sim()
    # matrix approach does not try to capture ground loss
    fgnd = sim.beam.compute_fgnd()
    sim_wfall = cro.simulator.correct_ground_loss(sim_wfall, fgnd, sim.Tgnd)

    assert A_wfall.shape == sim_wfall.shape
    assert jnp.allclose(A_wfall, sim_wfall)

def test_adjoint(sim):
    """
    Ensures that Amat.rmatvec() correctly computes the conjugate transpose.
    """
    Amat = ms.mapmaking.make_Amat(sim)
    rng = np.random.default_rng(0)

    x = rng.standard_normal(Amat.shape[1])
    y = rng.standard_normal(Amat.shape[0])

    Ax = Amat.matvec(x)
    Ahy = Amat.rmatvec(y)

    inner1 = np.vdot(Ax, y)
    inner2 = np.vdot(x, Ahy)

    # Assert they are equal up to standard floating-point precision
    np.testing.assert_allclose(
        inner1.real, inner2.real,
        rtol=1e-5, atol=1e-8,
        err_msg="Real parts of the inner products do not match!"
    )
    np.testing.assert_allclose(
        inner1.imag, inner2.imag,
        rtol=1e-5, atol=1e-8,
        err_msg="Imaginary parts of the inner products do not match!"
    )
