"""
Convert haslam_galactic.npz (HEALPix) to haslam_galactic_mwss.npz (MWSS).

The conversion goes through harmonic space:
    HEALPix pixels → alm (healpy, niter=3) → MWSS pixels (s2fft, exact)

Only the reference frequency is converted; the downstream pipeline applies
power-law scaling in pixel space anyway.
"""

import os
import time
from pathlib import Path

import healpy as hp
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import s2fft

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load healpix data
print("Loading haslam_galactic.npz ...")
d = np.load(DATA_DIR / "haslam_galactic.npz")
m_hp = d["m"]  # (86, 196608)
freqs = d["freqs"]  # (86,)
nside = int(d["nside"])  # 128
lmax = 2 * nside  # 256

print(f"  nside={nside}, lmax={lmax}")

# Use the reference frequency (last one) — same as pipeline
ref_map = m_hp[-1]
f0 = freqs[-1]
print(f"  Reference freq: {f0} MHz")

# HEALPix -> alm via healpy (fast, C implementation)
print("HEALPix -> alm (healpy, iter=3) ...")
t0 = time.time()
alm_hp = hp.map2alm(ref_map, lmax=lmax, iter=3)
print(f"  Done in {time.time() - t0:.1f}s")

# Convert healpy alm (1D) to s2fft alm (2D: lmax+1, 2*lmax+1)
print("Converting alm healpy -> s2fft format ...")
alm_s2fft = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
for ell in range(lmax + 1):
    for m in range(ell + 1):
        idx = hp.Alm.getidx(lmax, ell, m)
        val = alm_hp[idx]
        # s2fft stores m from -lmax to lmax along axis 1
        # index for m in s2fft: lmax + m
        alm_s2fft[ell, lmax + m] = val
        if m > 0:
            alm_s2fft[ell, lmax - m] = (-1) ** m * np.conj(val)

# alm -> MWSS (exact inverse)
print("alm -> MWSS (s2fft, exact) ...")
t0 = time.time()
mwss = s2fft.inverse(
    jnp.array(alm_s2fft),
    L=lmax + 1,
    spin=0,
    sampling="mwss",
    method="jax",
    reality=True,
)
mwss = np.asarray(mwss)
print(f"  Done in {time.time() - t0:.1f}s")
print(f"  MWSS shape: {mwss.shape}")

# Save: single reference map + freq, pipeline scales from here
out_path = DATA_DIR / "haslam_galactic_mwss.npz"
np.savez(out_path, m=mwss, f0=f0, lmax=lmax)
hp_size = os.path.getsize(DATA_DIR / "haslam_galactic.npz") / 1e6
mwss_size = os.path.getsize(out_path) / 1e6
print(f"Saved to {out_path}")
print(f"  HEALPix file: {hp_size:.1f} MB")
print(f"  MWSS file:    {mwss_size:.1f} MB")
