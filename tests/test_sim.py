import croissant as cro
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from astropy.time import Time

import mistsim as ms
from mistsim.sim import correct_ground_loss

freqs = jnp.linspace(50, 100, 50)
nside = 16
npix = 12 * nside**2
times_jd = Time("2024-01-01T00:00:00").jd + jnp.arange(10) * 0.1
lon = 0.0
lat = -30.0

@pytest.fixture
def beam():
    d = jnp.ones((freqs.size, npix)) * (freqs[:, None] / 100) ** 2
    b = ms.Beam(d, freqs, sampling="healpix")
    return b

@pytest.fixture
def sky():
    d = 180 * jnp.arange(npix)[None, :] * (freqs[:, None] / 180) ** -2.5
    s = ms.Sky(d, freqs, sampling="healpix", coord="galactic")
    return s

@pytest.fixture
def vis(beam, sky):
    sim = ms.Simulator(
        beam,
        sky,
        times_jd,
        freqs,
        lon,
        lat,
    )
    v = sim.sim()
    return v

def test_sky_alm(beam, sky, vis):
    """
    Check that Sim can be initialized with a Sky in alm format.
    """
    sim1 = ms.Simulator(
        beam,
        sky,
        times_jd,
        freqs,
        lon,
        lat,
    )
    sky_alm = sky.compute_alm_eq(world="earth")

    with pytest.warns(FutureWarning, match="Providing sky as an alm"):
        sim2 = ms.Simulator(
            beam,
            sky_alm,
            times_jd,
            freqs,
            lon,
            lat,
        )

    # everything should be the same except for the sky
    assert eqx.tree_equal(sim1.beam, sim2.beam)
    assert all(sim1.times_jd == sim2.times_jd)
    assert all(sim1.freqs == sim2.freqs)
    assert sim1.lon == sim2.lon
    assert sim1.lat == sim2.lat
    assert sim1.alt == sim2.alt
    assert sim1.lmax == sim2.lmax
    assert sim1._L == sim2._L
    assert sim1.eul_topo == sim2.eul_topo
    assert jnp.allclose(sim1.dl_topo, sim2.dl_topo)
    assert sim1.Tgnd == sim2.Tgnd
    assert sim1.world == sim2.world
    assert sim1.world == "earth"
    assert jnp.allclose(sim1.phases, sim2.phases)

    # should get the same results
    vis1 = vis
    vis2 = sim2.sim()
    assert jnp.allclose(vis1, vis2)


@pytest.mark.parametrize("disable_jit", [True, False])
@pytest.mark.parametrize("fgnd", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("Tgnd", [0.0, 100.0, 300.0])
def test_correct_ground_loss(vis, fgnd, Tgnd, disable_jit):
    """
    Check that the correct ground loss function raises a Warning and
    matches the croissant function
    """
    with jax.disable_jit(disable_jit):
        with pytest.warns(FutureWarning):
            vc = correct_ground_loss(vis, fgnd, Tgnd)
        vtrue = cro.simulator.correct_ground_loss(vis, fgnd, Tgnd)
        assert jnp.allclose(vc, vtrue)
