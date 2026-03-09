from functools import partial

import croissant as cro
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft
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

    beam_map = jnp.ones((freqs.size, npix))
    beam = ms.Beam(beam_map, freqs, sampling="healpix")

    s = ms.Simulator(beam, sky, times.jd, freqs, lon, lat, Tgnd=300)
    return s

@pytest.mark.parametrize("lmax", [16, 32])
def test_hp_to_s2fft(lmax):
    # generate some random alm
    rng = np.random.default_rng(0)
    alm = s2fft.utils.signal_generator.generate_flm(rng, lmax+1, reality=True)

    # this is the inverse transform
    _re = partial(s2fft.sampling.reindex.flm_2d_to_hp_fast, L=lmax+1)
    inv = jax.vmap(_re)

    # one freq
    alm1f = alm[None]
    hp1f = inv(alm1f)
    assert hp1f.shape == (1, hp.Alm.getsize(lmax))
    _alm = ms.mapmaking.hp_to_s2fft(hp1f)
    assert _alm.shape == alm1f.shape
    assert jnp.allclose(_alm, alm1f)

    # multiple freqs
    nfreqs = 10
    alm_nf = jnp.repeat(alm[None], nfreqs, axis=0)
    hp_nf = inv(alm_nf)
    assert hp_nf.shape == (nfreqs, hp.Alm.getsize(lmax))
    _alm = ms.mapmaking.hp_to_s2fft(hp_nf)
    assert _alm.shape == alm_nf.shape
    assert jnp.allclose(_alm, alm_nf)

def test_forward(sim):
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    lmax = cro.utils.lmax_from_shape(sky_alm.shape)
    _re = partial(s2fft.sampling.reindex.flm_2d_to_hp_fast, L=lmax+1)
    assert sky_alm.shape == (sim.freqs.size, sim.lmax + 1, 2 * sim.lmax + 1)
    sky_hp = jax.vmap(_re)(sky_alm)
    sky_hp = sky_hp.reshape(-1, 1)

    beam_alm = sim.compute_beam_eq()
    phases = sim.phases

    expected_raw = sim.sim()
    # need to ground loss correct since mapmaker assumes it's done
    expected = cro.simulator.correct_ground_loss(
        expected_raw, sim.beam.compute_fgnd(), sim.Tgnd
    )

    # out is a column vector of shape (ntimes*nfreq, 1)
    out = ms.mapmaking._forward(sky_hp, beam_alm, phases)
    assert out.shape == (sim.freqs.size * sim.times_jd.size, 1)
    out2d = out.reshape(sim.times_jd.size, sim.freqs.size)
    assert np.allclose(out2d, expected)

def test_A(sim):
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    lmax = cro.utils.lmax_from_shape(sky_alm.shape)
    _re = partial(s2fft.sampling.reindex.flm_2d_to_hp_fast, L=lmax+1)
    assert sky_alm.shape == (sim.freqs.size, sim.lmax + 1, 2 * sim.lmax + 1)
    sky_hp = jax.vmap(_re)(sky_alm)

    Amat = ms.mapmaking.make_Amat(sim)
    ntimes = sim.times_jd.size
    nfreq = sim.freqs.size
    assert Amat.shape == (ntimes * nfreq, sky_hp.size)

    A_wfall_rav = Amat @ sky_hp.reshape(-1, 1)
    A_wfall = A_wfall_rav.reshape(ntimes, nfreq)
    # compare with simulator
    sim_wfall = sim.sim()
    # matrix approach does not try to capture ground loss
    fgnd = sim.beam.compute_fgnd()
    sim_wfall = cro.simulator.correct_ground_loss(sim_wfall, fgnd, sim.Tgnd)

    assert A_wfall.shape == sim_wfall.shape
    assert jnp.allclose(A_wfall, sim_wfall)
