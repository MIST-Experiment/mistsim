from functools import partial

import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as sla


def _unpack_single_freq(x_freq, lmax):
    """
    Map 1d-vector of alm in real/imag to s2fft 2d array of complex alm.
    """
    N_per_freq = (lmax + 1) ** 2
    N_pos_m = (N_per_freq - (lmax + 1)) // 2

    m0_real = x_freq[: lmax + 1]  # m=0 modes
    pos_m_real = x_freq[lmax + 1 : lmax + 1 + N_pos_m]  # m>0 real parts
    pos_m_imag = x_freq[lmax + 1 + N_pos_m :]  # m>0 imag parts

    flm_s2fft = jnp.zeros((lmax + 1, 2 * lmax + 1), dtype=jnp.complex128)
    flm_s2fft = flm_s2fft.at[:, lmax].set(m0_real + 0j)  # set m=0 modes

    # complex m > 0 modes normalized by 1/root(2) to preserve variance
    pos_m_complex = 1 / jnp.sqrt(2) * (pos_m_real + 1j * pos_m_imag)

    current_idx = 0
    for m in range(1, lmax + 1):
        n_ell = lmax + 1 - m
        m_slice = pos_m_complex[current_idx : current_idx + n_ell]
        m_conj_slice = (-1) ** m * jnp.conj(m_slice)

        flm_s2fft = flm_s2fft.at[m:, lmax + m].set(m_slice)
        flm_s2fft = flm_s2fft.at[m:, lmax - m].set(m_conj_slice)

        current_idx += n_ell

    return flm_s2fft


def unpack_real_to_s2fft(x_real, lmax, Nfreq=1):
    """
    Multi-frequency wapper around _unpack_single_freq to reshape the
    input and map over frequencies.
    """
    N_per_freq = (lmax + 1) ** 2
    # Reshape to (Nfreq, N_per_freq) so we can map over frequencies
    x_reshaped = jnp.ravel(x_real).reshape(Nfreq, N_per_freq)

    # vmap over the 0th axis (frequencies)
    unpack_vmap = jax.vmap(_unpack_single_freq, in_axes=(0, None))
    return unpack_vmap(x_reshaped, lmax)


def _pack_single_freq(flm_freq, lmax):
    """
    Map s2fft 2d array of complex alm to 1d-vector of alm in real/imag
    for a single frequency.
    """
    m0_real = jnp.real(flm_freq[:, lmax])  # m=0 modes are purely real

    # complex m > 0 modes
    pos_m_complex = []
    for m in range(1, lmax + 1):
        pos_m_complex.append(flm_freq[m:, lmax + m])

    # multiply by root(2) to preserve variance when going back to real/imag
    pos_m_complex = jnp.sqrt(2) * jnp.concatenate(pos_m_complex)

    pack_alm = jnp.concatenate(
        [m0_real, jnp.real(pos_m_complex), jnp.imag(pos_m_complex)]
    )
    return pack_alm


def pack_s2fft_to_real(flm_s2fft):
    Nfreq, L, _ = flm_s2fft.shape
    lmax = L - 1
    # vmap the packing over the frequency axis
    pack_vmap = jax.vmap(_pack_single_freq, in_axes=(0, None))
    return jnp.ravel(pack_vmap(flm_s2fft, lmax))


@jax.jit
def _forward_jax(x, beam_alm, phases):
    """
    The forward operator for mapmaking.

    This returns the ground-loss corrected antenna temperature at each
    time and frequency, as a column vector. Mostly a wrapper around
    cro.simulator.convolve, but lets us input sky with only the m>=0
    modes, split into real/imag parts.


    Parameters
    ----------
    x : jax.Array
        The input sky ordered like Healpy but split into real and
        imaginary parts, with shape (nfreqs * nalm,).
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
    lmax = cro.utils.lmax_from_shape(beam_alm.shape)
    sky_alm = unpack_real_to_s2fft(x, lmax, Nfreq=nfreq)

    # convolve with croissant
    wf = cro.simulator.convolve(beam_alm, sky_alm, phases)
    # beam_alm is already horizon masked so norm is above horizon only
    beam_norm = beam_alm[:, 0, lmax] * jnp.sqrt(4 * jnp.pi)
    wf /= beam_norm[None, :]
    y = wf.ravel()[:, None]
    return y


def _forward(x, beam_alm, phases):
    xjax = jnp.asarray(x)
    return np.asarray(_forward_jax(xjax, beam_alm, phases)).ravel()


def _adjoint(y, transpose):
    yjax = jnp.asarray(y).reshape(-1, 1).astype(jnp.complex128)
    return np.asarray(transpose(yjax)[0]).ravel()


def alm1d_to_hp(alm):
    """
    Convert packed 1D alm vector (real/imag separated) to healpy
    complex ordering.

    Parameters
    ----------
    alm : array-like
        Packed alm vector with shape ((lmax+1)^2,). Layout is
        [m=0 real, m>0 real, m>0 imag].

    Returns
    -------
    alm_hp : jax.Array
        Complex alm in healpy ordering with shape
        (hp.Alm.getsize(lmax),).

    """
    s = len(alm)
    lmax = int(np.sqrt(s) - 1)
    hp_len = (lmax + 1) * (lmax + 2) // 2
    alm0 = alm[: lmax + 1]
    almre = alm[lmax + 1 : hp_len]
    almim = alm[hp_len:]
    almc = 1 / jnp.sqrt(2) * (almre + 1j * almim)
    return jnp.concatenate((alm0, almc))


def stack_As(*A_list):
    """
    Vertically stack an arbitrary number of LinearOperators.

    Parameters
    ----------
    *A_list : scipy.sparse.linalg.LinearOperator
        Two or more operators with compatible column dimensions.

    Returns
    -------
    Astack : scipy.sparse.linalg.LinearOperator
        Vertically stacked operator with shape (sum(M_i), N).

    Raises
    ------
    ValueError
        If column dimensions are incompatible.

    """
    if len(A_list) < 2:
        raise ValueError("Need at least two operators to stack")
    n = A_list[0].shape[1]
    for A in A_list[1:]:
        if A.shape[1] != n:
            raise ValueError("Incompatible shapes")
    m_total = sum(A.shape[0] for A in A_list)

    def _matvec(v):
        v = np.asarray(v).ravel()
        return np.concatenate(
            [A.matvec(v) for A in A_list],
            axis=0,
        )

    def _rmatvec(v):
        v = np.asarray(v).ravel()
        result = np.zeros(n, dtype=v.dtype)
        offset = 0
        for A in A_list:
            m_i = A.shape[0]
            result += A.rmatvec(v[offset : offset + m_i])
            offset += m_i
        return result

    return sla.LinearOperator(
        (m_total, n),
        matvec=_matvec,
        rmatvec=_rmatvec,
        dtype=np.float64,
    )


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

    nalm = (sim.lmax + 1) ** 2
    nfreqs = sim.freqs.size
    ntimes = sim.times_jd.size
    shape = (nfreqs * ntimes, nfreqs * nalm)

    _fwd_jax = partial(_forward_jax, beam_alm=beam_alm, phases=phases)
    xdummy = jnp.zeros(nfreqs * nalm, dtype=float)
    transpose_fn = jax.linear_transpose(_fwd_jax, xdummy)

    matvec = partial(_forward, beam_alm=beam_alm, phases=phases)
    rmatvec = partial(_adjoint, transpose=transpose_fn)

    A = sla.LinearOperator(
        shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64
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
        dtype=Amat.dtype,
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
