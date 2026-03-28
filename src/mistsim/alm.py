"""Utilities for converting between packed real alm vectors and
s2fft / healpy complex alm representations."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=8)
def _make_scatter_indices(lmax):
    """Precompute index arrays for pack/unpack.

    Returns numpy arrays that map between the packed real vector
    and the s2fft 2D complex array.  Pure numpy, executed once
    at trace time and cached.
    """
    rows, cols_pos, cols_neg, signs = [], [], [], []
    for m in range(1, lmax + 1):
        for ell in range(m, lmax + 1):
            rows.append(ell)
            cols_pos.append(lmax + m)
            cols_neg.append(lmax - m)
            signs.append((-1) ** m)
    return (
        np.array(rows),
        np.array(cols_pos),
        np.array(cols_neg),
        np.array(signs, dtype=np.float64),
    )


def _unpack_single_freq(x_freq, lmax):
    """
    Map 1d-vector of alm in real/imag to s2fft 2d array of complex alm.
    """
    N_per_freq = (lmax + 1) ** 2
    N_pos_m = (N_per_freq - (lmax + 1)) // 2

    m0_real = x_freq[: lmax + 1]  # m=0 modes
    pos_m_real = x_freq[lmax + 1 : lmax + 1 + N_pos_m]
    pos_m_imag = x_freq[lmax + 1 + N_pos_m :]

    # complex m > 0 modes normalized by 1/root(2) to preserve variance
    pos_m_complex = 1 / jnp.sqrt(2) * (pos_m_real + 1j * pos_m_imag)

    rows, cols_pos, cols_neg, signs = _make_scatter_indices(lmax)

    flm = jnp.zeros((lmax + 1, 2 * lmax + 1), dtype=jnp.complex128)
    flm = flm.at[:, lmax].set(m0_real + 0j)
    flm = flm.at[rows, cols_pos].set(pos_m_complex, unique_indices=True)
    flm = flm.at[rows, cols_neg].set(
        signs * jnp.conj(pos_m_complex), unique_indices=True
    )
    return flm


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

    rows, cols_pos, _, _ = _make_scatter_indices(lmax)
    # multiply by root(2) to preserve variance going back to real/imag
    pos_m_complex = jnp.sqrt(2) * flm_freq[rows, cols_pos]

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


def hp_to_alm1d(alm_hp):
    """
    Convert healpy complex alm to packed 1D vector (real/imag
    separated).

    Inverse of :func:`alm1d_to_hp`.

    Parameters
    ----------
    alm_hp : array-like
        Complex alm in healpy ordering with shape
        ``(hp.Alm.getsize(lmax),)``.

    Returns
    -------
    alm : np.ndarray
        Packed alm vector with shape ``((lmax+1)^2,)``. Layout is
        ``[m=0 real, m>0 real, m>0 imag]``.

    """
    hp_len = len(alm_hp)
    # hp_len = (lmax+1)*(lmax+2)/2
    # => lmax = (-3 + sqrt(9+8*(hp_len-1)))/2
    lmax = int((-3 + np.sqrt(9 + 8 * (hp_len - 1))) / 2)
    alm0 = np.real(np.asarray(alm_hp[: lmax + 1]))
    almc = np.asarray(alm_hp[lmax + 1 :])
    almre = np.sqrt(2) * np.real(almc)
    almim = np.sqrt(2) * np.imag(almc)
    return np.concatenate((alm0, almre, almim))


def packed_lm_indices(lmax):
    """
    Return ``(ell, m)`` arrays mapping each index in the packed alm
    vector to its spherical harmonic degree and order.

    The layout follows :func:`_pack_single_freq`:

    - ``[0 : lmax+1]`` -- ``m = 0``, ``ell = 0 .. lmax``
    - real block for ``m = 1 .. lmax`` (``lmax+1-m`` entries each)
    - imag block for ``m = 1 .. lmax`` (same layout)

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    ell_arr : np.ndarray
        Spherical harmonic degree for each index, shape
        ``((lmax+1)^2,)``.
    m_arr : np.ndarray
        Spherical harmonic order for each index, shape
        ``((lmax+1)^2,)``.

    """
    nalm = (lmax + 1) ** 2
    ell_arr = np.zeros(nalm, dtype=int)
    m_arr = np.zeros(nalm, dtype=int)

    # m=0 block
    ell_arr[: lmax + 1] = np.arange(lmax + 1)
    # m_arr already 0

    # m>0 real block, then imag block (same l,m mapping)
    offset = lmax + 1
    for _pass in range(2):  # real then imag
        for m in range(1, lmax + 1):
            n_ell = lmax + 1 - m
            ell_arr[offset : offset + n_ell] = np.arange(m, lmax + 1)
            m_arr[offset : offset + n_ell] = m
            offset += n_ell

    return ell_arr, m_arr
