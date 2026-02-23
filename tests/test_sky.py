import pytest

import jax.numpy as jnp

import mistsim.sky as msky

freqs = jnp.linspace(50, 100, 50)
npix = 768  # nside=8
sky_data = jnp.ones((len(freqs), npix))

def test_coord_invalid():
    with pytest.raises(ValueError):  # works in croissant but not mistsim
        msky.Sky(data, freqs, sampling="healpix", coord="mcmf")

    # these works
    msky.Sky(data, freqs, sampling="healpix", coord="equatorial")
    msky.Sky(data, freqs, sampling="healpix", coord="galactic")

def test_sky_alm():
    """
    Compare initialization of Sky with alm and with data.
    """
    s = msky.Sky(sky_data, freqs, sampling="healpix", coord="equatorial")
    sky_alm = s.compute_alm_eq()
    # compute alm works on s.coord so matches alm_eq
    assert jnp.allclose(sky_alm, s.compute_alm())

    salm_class = msky._SkyAlm(sky_alm, freqs)
    assert jnp.allclose(salm_class.compute_alm(), sky_alm)
    assert jnp.allclose(salm_class.compute_alm_eq(), sky_alm)
