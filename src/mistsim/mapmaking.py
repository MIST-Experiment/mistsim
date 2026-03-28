from functools import partial

import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as sla

# Re-export alm utilities so that ``mapmaking.alm1d_to_hp`` etc.
# continue to work for existing callers.
from .alm import (  # noqa: F401
    _pack_single_freq,
    _unpack_single_freq,
    alm1d_to_hp,
    hp_to_alm1d,
    pack_s2fft_to_real,
    packed_lm_indices,
    unpack_real_to_s2fft,
)


def _forward_single_freq(x, beam_alm_f, phases):
    """Forward for one frequency, one site.

    Parameters
    ----------
    x : jax.Array
        Packed real alm, shape ``(nalm,)``.
    beam_alm_f : jax.Array
        Beam alm for one frequency, shape ``(L, 2L-1)``.
    phases : jax.Array
        Rotation phases, shape ``(ntimes, 2L-1)``.

    Returns
    -------
    y : jax.Array
        Waterfall for this site/freq, shape ``(ntimes,)``.

    """
    lmax = beam_alm_f.shape[0] - 1
    sky_alm = _unpack_single_freq(x, lmax)
    wf = cro.simulator.convolve(beam_alm_f[None], sky_alm[None], phases)
    norm = beam_alm_f[0, lmax] * jnp.sqrt(4 * jnp.pi)
    return (wf[:, 0] / norm).real


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
        shape,
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
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


# ------------------------------------------------------------------
# Pure JAX operators (no scipy / numpy conversion)
# ------------------------------------------------------------------


def make_operators_jax(sim):
    """Build pure JAX forward/adjoint operators.

    Unlike :func:`make_Amat` (which wraps JAX inside a scipy
    ``LinearOperator``), this returns JAX callables that avoid
    JAX-to-NumPy conversion overhead.

    Parameters
    ----------
    sim : Simulator
        Simulation parameters.

    Returns
    -------
    dict
        ``forward_fn`` : ``(nfreqs*nalm,) -> (nfreqs*ntimes,)``
        ``adjoint_fn`` : ``(nfreqs*ntimes,) -> (nfreqs*nalm,)``
        ``shape`` : ``(nfreqs*ntimes, nfreqs*nalm)``
        ``beam_alm``, ``phases`` : arrays used internally.

    """
    beam_alm = sim.compute_beam_eq()
    beam_alm = cro.utils.reduce_lmax(beam_alm, sim.lmax)
    phases = sim.phases

    nalm = (sim.lmax + 1) ** 2
    nfreqs = sim.freqs.size
    ntimes = sim.times_jd.size
    shape = (nfreqs * ntimes, nfreqs * nalm)

    def _fwd(x):
        y = _forward_jax(x, beam_alm, phases)
        return y.ravel().real

    xdummy = jnp.zeros(nfreqs * nalm, dtype=jnp.float64)
    _transpose = jax.linear_transpose(_fwd, xdummy)

    def _adj(y):
        return _transpose(y)[0]

    return {
        "forward_fn": jax.jit(_fwd),
        "adjoint_fn": jax.jit(_adj),
        "shape": shape,
        "beam_alm": beam_alm,
        "phases": phases,
    }


def stack_operators_jax(*op_list):
    """Vertically stack pure JAX operators.

    Parameters
    ----------
    *op_list : dict
        Operator dicts from :func:`make_operators_jax`.

    Returns
    -------
    dict
        Stacked operator with combined forward/adjoint.

    """
    if len(op_list) < 2:
        raise ValueError("Need at least two operators to stack")
    n = op_list[0]["shape"][1]
    for op in op_list[1:]:
        if op["shape"][1] != n:
            raise ValueError("Incompatible shapes")
    m_total = sum(op["shape"][0] for op in op_list)
    m_sizes = tuple(op["shape"][0] for op in op_list)
    fwd_fns = [op["forward_fn"] for op in op_list]
    adj_fns = [op["adjoint_fn"] for op in op_list]

    @jax.jit
    def forward_fn(x):
        return jnp.concatenate([f(x) for f in fwd_fns])

    @jax.jit
    def adjoint_fn(y):
        result = jnp.zeros(n, dtype=jnp.float64)
        offset = 0
        for adj, m in zip(adj_fns, m_sizes):
            result = result + adj(y[offset : offset + m])
            offset += m
        return result

    return {
        "forward_fn": forward_fn,
        "adjoint_fn": adjoint_fn,
        "shape": (m_total, n),
    }


def make_atilde_fns(Ndiag, fwd_fn, adj_fn, Sdiag):
    """Build whitened forward/adjoint as JAX callables.

    Returns ``(atilde_fwd, atilde_adj)`` where::

        atilde_fwd(x) = N^{-1/2} * fwd(S^{1/2} * x)
        atilde_adj(y) = S^{1/2} * adj(N^{-1/2} * y)

    Parameters
    ----------
    Ndiag : array-like
        Noise variance diagonal, shape ``(ndata,)``.
    fwd_fn : callable
        Forward operator ``(nalm,) -> (ndata,)``.
    adj_fn : callable
        Adjoint operator ``(ndata,) -> (nalm,)``.
    Sdiag : array-like
        Prior covariance diagonal, shape ``(nalm,)``.

    Returns
    -------
    atilde_fwd, atilde_adj : callable

    """
    Nm12 = 1.0 / jnp.sqrt(jnp.asarray(Ndiag))
    S12 = jnp.sqrt(jnp.asarray(Sdiag))

    @jax.jit
    def atilde_fwd(x):
        return Nm12 * fwd_fn(S12 * x)

    @jax.jit
    def atilde_adj(y):
        return S12 * adj_fn(Nm12 * y)

    return atilde_fwd, atilde_adj


def randomized_svd_jax(
    atilde_fwd,
    atilde_adj,
    nalm,
    ndata,
    k,
    p=10,
    seed=0,
):
    """Randomized SVD via Halko-Martinsson-Tropp.

    Computes an approximate rank-*k* SVD using only forward
    and adjoint evaluations.  All computation stays in JAX.

    Parameters
    ----------
    atilde_fwd : callable
        Forward ``(nalm,) -> (ndata,)``.
    atilde_adj : callable
        Adjoint ``(ndata,) -> (nalm,)``.
    nalm, ndata : int
        Operator dimensions.
    k : int
        Target rank.
    p : int
        Oversampling parameter.
    seed : int
        PRNG seed.

    Returns
    -------
    U : jax.Array, ``(ndata, k)``
    Sigma : jax.Array, ``(k,)``
    Vh : jax.Array, ``(k, nalm)``

    """
    key = jax.random.PRNGKey(seed)
    Omega = jax.random.normal(key, (k + p, nalm))

    # Y = Atilde @ Omega^T: apply forward to each row
    Y = jax.vmap(atilde_fwd)(Omega).T  # (ndata, k+p)

    Q, _ = jnp.linalg.qr(Y)

    # B = Atilde^H @ Q: apply adjoint to each column
    B = jax.vmap(atilde_adj)(Q.T).T  # (nalm, k+p)

    # SVD of B^T: (k+p, nalm)
    U_B, Sigma_full, Vh_B = jnp.linalg.svd(B.T, full_matrices=False)

    U = Q @ U_B[:, :k]
    Sigma = Sigma_full[:k]
    Vh = Vh_B[:k]
    return U, Sigma, Vh
