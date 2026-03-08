from functools import partial

import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import s2ftt
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
    lmax = cro.utils.lmax_from_shape(sky_hp.shape)
    r = partial(s2ftt.sampling.reindex.flm_hp_to_2d_fast, L=lmax+1)
    return jax.vmap(r)(sky_hp)

@jax.jit
def _forward(sky_hp, beam_alm, phases):
    """
    Helper function to define the forward operator for mapmaking.

    The output `forward` function is essentially the same as
    `croissant.simulator.conolve`, but wants the beam and phases
    already defined and for the `sky_alm_hp` to be in healpy convention
    and raveled over the frequency axis.

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
    jax.Array
        The timestream data, with shape (ntimes,) or (ntimes, nfreq) if
        the input alms have a frequency axis.

    """
    nfreq = beam_alm.shape[0]
    sky_hp = sky_hp.reshape((nfreq, -1))
    sky_alm = hp_to_s2fft(sky_hp)
    return cro.simulator.convolve(beam_alm, sky_alm, phases)

def _adjoint(v, transpose):
    return np.asarray(transpose(v)[0])

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
    x_dummy = jnp.zeros((nfreqs * nalm,), dtype=complex)
    _, transpose = jax.vjp(matvec, x_dummy)

    rmatvec = partial(_adjoint, transpose=transpose)

    ntimes = sim.times_jd.size
    shape = (ntimes, nfreqs * nalm)
    A = sla.LinearOperator(shape, matvec=matvec, rmatvec=rmatvec)
    return A
