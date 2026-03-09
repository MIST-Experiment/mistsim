from functools import partial

import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import scipy.sparse.linalg as sla


@jax.jit
def hp_to_s2fft(sky_hp):
    """
    Convert a healpy-style array of alm coefficients to the s2fft
    convention.

    Parameters
    ----------
    sky_hp : jax.Array
        The harmonic coefficients of the sky in equatorial coordinates,
        in the usual healpy ordering. Shape is (nfreq, nalm)

    Returns
    -------
    jax.Array
        The harmonic coefficients of the sky in equatorial coordinates,
        in the s2fft/croissant ordering. Shape is (lmax+1, 2*lmax+1).

    """
    nalm = sky_hp.shape[-1]
    num = -3 + np.sqrt(1 + 8 * nalm)
    lmax = np.floor(num / 2).astype(int)
    r = partial(s2fft.sampling.reindex.flm_hp_to_2d_fast, L=lmax+1)
    return jax.vmap(r)(sky_hp)

@jax.jit
def _forward(sky_hp, beam_alm, phases):
    """
    The forward operator for mapmaking.

    This returns the ground-loss corrected antenna temperature at each
    time and frequency, as a column vector. Mostly a wrapper around
    cro.simulator.convolve, but lets us input sky as a healpy-ordered
    column vector.


    Parameters
    ----------
    sky_hp : jax.Array
        The harmonic coefficients of the sky in equatorial coordinates,
        in healpy ordering and raveled over the frequency axis. Shape
        is (nfreq * nalm, 1) where nalm = (lmax+1) * (lmax+2) // 2.
    beam_alm : jax.Array
        The harmonic coefficients of the beam in equatorial
        coordinates, in the usual s2fft/croissant ordering.
    phases : jax.Array
        The phases of the Earth's rotation at each time, in radians.
        Output of cro.simulator.rot_alm_z.

    Returns
    -------
    y : jax.Array
        The waterfall data, with shape (n_freq * ntimes, 1)

    """
    nfreq = beam_alm.shape[0]
    sky_hp = sky_hp.reshape(nfreq, -1)
    sky_alm = hp_to_s2fft(sky_hp)
    wf = cro.simulator.convolve(beam_alm, sky_alm, phases)
    lmax = cro.utils.lmax_from_shape(beam_alm.shape)
    # beam_alm is already horizon masked so norm is above horizon only
    beam_norm = beam_alm[:, 0, lmax] * jnp.sqrt(4 * jnp.pi)
    wf /= beam_norm[None, :]
    y = wf.ravel()[:, None]
    return y

def _adjoint(v, transpose):
    vcol = jnp.asarray(v).reshape(-1, 1)
    return np.asarray(transpose(vcol)[0])

def make_Amat(sim):
    """
    Make the A matrix for mapmaking. This is a sparse matrix that
    encodes the beam at different times.

    Parameters
    ----------
    sim: Simualtor
        Instance specifying the simulation parameters.

    Returns
    -------
    A : scipy.sparse.linalg.LinearOperator
        The A matrix for mapmaking, as a sparse linear operator.

    """
    beam_alm = sim.compute_beam_eq()
    phases = sim.phases

    matvec = partial(_forward, beam_alm=beam_alm, phases=phases)

    # need a dummy input of same shape as x vector to make transpose
    lmax = cro.utils.lmax_from_shape(beam_alm.shape)
    nalm = (lmax+1) * (lmax+2) // 2
    nfreqs = sim.freqs.size
    x_dummy = jnp.zeros((nfreqs * nalm, 1), dtype=complex)
    _, transpose = jax.vjp(matvec, x_dummy)

    rmatvec = partial(_adjoint, transpose=transpose)

    ntimes = sim.times_jd.size
    shape = (nfreqs * ntimes, nfreqs * nalm)
    A = sla.LinearOperator(
        shape, matvec=matvec, rmatvec=rmatvec, dtype=complex
    )
    return A
