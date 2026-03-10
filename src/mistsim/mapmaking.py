from functools import partial

import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as sla


@partial(jax.jit, static_argnums=(1))
def flm_hp_to_2d_fast(flm_hp: jnp.ndarray, L: int) -> jnp.ndarray:
    """
    Converts from HEALPix (healpy) indexed harmonic coefficients to 2D indexed
    coefficients (JAX).

    Replacing the original s2fft function with a vjp safe one

    """
    flm_2d = jnp.zeros((L, 2 * L - 1), dtype=flm_hp.dtype)
    m_indices, el_indices = (
        np.triu_indices(n=L, k=1, m=L) + np.array([[1], [0]])
    )

    # pass unique_indices=True to allow JAX to linearly transpose
    flm_2d = flm_2d.at[:L, L - 1].set(
        flm_hp[:L],
        unique_indices=True
    )
    flm_2d = flm_2d.at[el_indices, L - 1 + m_indices].set(
        flm_hp[L:],
        unique_indices=True
    )
    flm_2d = flm_2d.at[el_indices, L - 1 - m_indices].set(
        (-1) ** m_indices * flm_hp[L:].conj(),
        unique_indices=True
    )

    return flm_2d

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
    r = partial(flm_hp_to_2d_fast, L=lmax+1)
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

    # convolve with croissant
    wf = cro.simulator.convolve(beam_alm, sky_alm, phases)
    lmax = cro.utils.lmax_from_shape(beam_alm.shape)
    # beam_alm is already horizon masked so norm is above horizon only
    beam_norm = beam_alm[:, 0, lmax] * jnp.sqrt(4 * jnp.pi)
    wf /= beam_norm[None, :]
    y = wf.ravel()[:, None]
    return y

@jax.jit
def _matvec_C(sky_hp, beam_alm, phases):
    xr = sky_hp.real.astype(complex)
    xi = sky_hp.imag.astype(complex)
    outr = _forward(xr, beam_alm, phases)
    outi = _forward(xi, beam_alm, phases)
    return outr + 1j * outi

@jax.jit
def _matvec_R(sky_hp_r, beam_alm, phases):
    sky_hp = sky_hp_r.astype(complex)
    return _forward(sky_hp, beam_alm, phases).real

@jax.jit
def _matvec_I(sky_hp_i, beam_alm, phases):
    sky_hp = sky_hp_i.astype(complex)
    return _forward(sky_hp, beam_alm, phases).imag

def _adjoint(v, transpose_R, transpose_I):
    vcol = v.reshape(-1, 1)
    vr = vcol.real
    vi = vcol.imag

    AR_T_vr = transpose_R(vr)[0]
    AI_T_vi = transpose_I(vi)[0]
    AR_T_vi = transpose_R(vi)[0]
    AI_T_vr = transpose_I(vr)[0]

    out_r = AR_T_vr + AI_T_vi
    out_i = AR_T_vi - AI_T_vr

    outc = out_r + 1j * out_i
    return outc.ravel()

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
    beam_alm = cro.utils.reduce_lmax(beam_alm, sim.lmax)
    phases = sim.phases

    nalm = (sim.lmax+1) * (sim.lmax+2) // 2
    nfreqs = sim.freqs.size
    ntimes = sim.times_jd.size
    shape = (nfreqs * ntimes, nfreqs * nalm)

    fwd_R = partial(_matvec_R, beam_alm=beam_alm, phases=phases)
    fwd_I = partial(_matvec_I, beam_alm=beam_alm, phases=phases)

    x_dummy = jnp.zeros((nfreqs * nalm, 1), dtype=float)
    transpose_R = jax.linear_transpose(fwd_R, x_dummy)
    transpose_I = jax.linear_transpose(fwd_I, x_dummy)

    matvec = partial(_matvec_C, beam_alm=beam_alm, phases=phases)
    rmatvec = partial(
        _adjoint, transpose_R=transpose_R, transpose_I=transpose_I
    )

    A = sla.LinearOperator(
        shape, matvec=matvec, rmatvec=rmatvec, dtype=complex
    )
    return A

def make_SAH(Sdiag, Amat):
    """
    Create the linear operator representing S @ A^H where S is a
    a diagonal prior covariance matrix.

    Parameters
    ----------
    Sdiag : array-like
        The diagonal of the prior covariance matrix, with shape
        (nfreqs * nalm,).
    Amat : scipy.sparse.linalg.LinearOperator
        The A matrix for mapmaking, as a sparse linear operator.
        Expected shape is (nfreqs * ntimes, nfreqs * nalm).

    Returns
    -------
    SAH : scipy.sparse.linalg.LinearOperator
        The linear operator representing S @ A^H, with shape
        (nfreqs * nalm, nfreqs * ntimes).

    """
    SAH = sla.LinearOperator(
        shape=(Amat.shape[1], Amat.shape[0]),
        matvec=lambda v: Sdiag[:, None] * Amat.rmatvec(v),
        rmatvec=lambda v: Amat.matvec(Sdiag[:, None] * v),
        dtype=Amat.dtype
    )
    return SAH

def _Atilde_matvec(v, Ndiag, Amat, Sdiag):
    Nm12 = 1 / np.sqrt(Ndiag)
    S12 = np.sqrt(Sdiag)
    if v.ndim == 2:
        Nm12 = Nm12[:, None]
        S12 = S12[:, None]
    return Nm12 * Amat.matvec(S12 * v)

def _Atilde_rmatvec(v, Ndiag, Amat, Sdiag):
    Nm12 = 1 / np.sqrt(Ndiag)
    S12 = np.sqrt(Sdiag)
    if v.ndim == 2:
        Nm12 = Nm12[:, None]
        S12 = S12[:, None]
    return S12 * Amat.rmatvec(Nm12 * v)

def make_Atilde(Ndiag, Amat, Sdiag):
    """
    Create the linear operator representing the design matrix in the
    whitened least squares problem, Atilde = N^{-1/2} @ A @ S^{1/2}.

    Parameters
    ----------
    Ndiag : array-like
        The diagonal of the noise covariance matrix, with shape
        (nfreqs * ntimes,).
    Amat : scipy.sparse.linalg.LinearOperator
        The A matrix for mapmaking, as a sparse linear operator.
        Expected shape is (nfreqs * ntimes, nfreqs * nalm).
    Sdiag : array-like
        The diagonal of the prior covariance matrix, with shape
        (nfreqs * nalm,).

    Returns
    -------
    Atilde : scipy.sparse.linalg.LinearOperator
        The linear operator representing the design matrix in the
        whitened least squares problem, with shape
        (nfreqs * ntimes, nfreqs * nalm).

    """
    Atilde = sla.LinearOperator(
        shape=Amat.shape,
        matvec=lambda v: _Atilde_matvec(v, Ndiag, Amat, Sdiag),
        rmatvec=lambda v: _Atilde_rmatvec(v, Ndiag, Amat, Sdiag),
        dtype=Amat.dtype,
    )
    return Atilde
