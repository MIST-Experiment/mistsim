import jax.numpy as jnp
import pytest

import mistsim.sky as msky

freqs = jnp.linspace(50, 100, 50)
npix = 768  # nside=8
sky_data = jnp.arange(npix)
sky_data = jnp.repeat(sky_data[None, :], len(freqs), axis=0)

def test_coord_invalid():
    with pytest.raises(ValueError):  # works in croissant but not mistsim
        msky.Sky(sky_data, freqs, sampling="healpix", coord="mcmf")

    # these works
    msky.Sky(sky_data, freqs, sampling="healpix", coord="equatorial")
    msky.Sky(sky_data, freqs, sampling="healpix", coord="galactic")

def test_sky_alm():
    """
    Compare initialization of Sky with alm and with data.
    """
    s = msky.Sky(sky_data, freqs, sampling="healpix", coord="equatorial")
    sky_alm_eq = s.compute_alm_eq(world="earth")
    # compute alm works on s.coord so matches alm_eq
    assert jnp.allclose(sky_alm_eq, s.compute_alm())
    # should not match if coord is galactic
    sgal = msky.Sky(sky_data, freqs, sampling="healpix", coord="galactic")
    assert jnp.allclose(sky_alm_eq, sgal.compute_alm())
    assert not jnp.allclose(sky_alm_eq, sgal.compute_alm_eq(world="earth"))

    # ensure initialization with alm works and matches equatorial alm
    salm_class = msky._SkyAlm(sky_alm_eq, freqs)
    assert jnp.allclose(salm_class.compute_alm(), sky_alm_eq)
    assert jnp.allclose(salm_class.compute_alm_eq(), sky_alm_eq)

    # sky alm class does not care about world
    assert jnp.allclose(
        salm_class.compute_alm_eq(world="earth"),
        salm_class.compute_alm_eq(world="moon"),
    )
    assert jnp.allclose(
        salm_class.compute_alm_eq(world="earth"),
        salm_class.compute_alm_eq(world="sun"),  # nonsense but ignored
    )
