"""Helpers for comparing mistsim with Raul Monsalve's simulations.

Used by ``raul_comparison_normalization.ipynb``. The pixel-domain
convolution reimplements the algorithm of ``convolution`` in
``raul_global21cm/astro.py`` (MIST-Experiment/raul_global21cm @ 4c2d125,
line 960): sky pixels are moved to (AZ, EL) with astropy, the beam is
interpolated onto them with a bicubic ``RectBivariateSpline`` in
(EL, AZ) without wrapping at AZ = 0/360, and the antenna temperature is
sum(B * T * mask) / sum(B * mask) over pixels with EL >= 0.

The time grid follows ``galactic_to_local_coordinates_24h_LST`` (:625)
and ``utc2lst`` (:22) of the same file: UTC 2014-01-01 09:29:45 plus
multiples of 359 s, apparent sidereal time (IAU2006A) with
DUT1 = 0, sorted by LST. Raul's 2026 files reproduce this grid to
1e-9 h (checked in the notebook).
"""

import datetime as dt
from pathlib import Path

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from scipy import sparse
from scipy.interpolate import BSpline, RectBivariateSpline

# Raul's reference epoch: LST = 0 h at the EDGES site (astro.py:671).
RAUL_T0 = dt.datetime(2014, 1, 1, 9, 29, 45)
RAUL_STEP_S = 359
RAUL_NSTEP = int(np.ceil(86164 / RAUL_STEP_S))  # 241

TCMB = 2.725


def read_raul(path):
    """Return (lst [h], freq [MHz], ant_temp [K]) from Raul's hdf5."""
    with h5py.File(path, "r") as hf:
        return (
            np.array(hf["lst"]),
            np.array(hf["freq"]),
            np.array(hf["ant_temp"]),
        )


def raul_times(lon_deg):
    """Raul's UTC sample times and apparent LST, in UTC order.

    Returns
    -------
    times : astropy.time.Time
        RAUL_NSTEP UTC times, in the order Raul generated them.
    lst : np.ndarray
        Apparent sidereal time in hours (IAU2006A, DUT1 = 0), as in
        ``utc2lst``.
    order : np.ndarray
        ``argsort(lst)``: indexing a UTC-ordered waterfall with it gives
        Raul's LST-sorted order.
    """
    t0 = Time(RAUL_T0, scale="utc")
    times = t0 + np.arange(RAUL_NSTEP) * RAUL_STEP_S * u.s
    t_lst = Time(times, copy=True)
    t_lst.delta_ut1_utc = 0
    lst = t_lst.sidereal_time(
        "apparent", longitude=lon_deg * u.deg, model="IAU2006A"
    ).hour
    return times, np.asarray(lst), np.argsort(lst)


def load_haslam(path, nside, freqs, beta=-2.55, f0=408.0):
    """Haslam (Remazeilles 2014) degraded to nside and power-law scaled.

    Tmodel = Tcmb + (h - Tcmb) * (f / 408) ** beta, as in Raul's readme.
    Returns an array of shape (Nfreq, Npix) in RING order, galactic.
    """
    with fits.open(path) as hdul:
        h = hdul[1].data["TEMPERATURE"].ravel()
    h = hp.ud_grade(h, nside, order_in="RING")
    scale = (np.asarray(freqs) / f0) ** beta
    return (h - TCMB)[None, :] * scale[:, None] + TCMB


def pixel_altaz(nside, location, times):
    """AZ, EL [deg] of every galactic HEALPix pixel centre at each time.

    Uses astropy's galactic -> AltAz transform (no refraction), as
    Raul's code does. Returns two arrays of shape (Ntimes, Npix),
    float32 to keep memory modest.
    """
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    gal = SkyCoord(lon, lat, frame="galactic", unit="deg")
    az = np.empty((len(times), npix), dtype=np.float32)
    el = np.empty((len(times), npix), dtype=np.float32)
    for i, t in enumerate(times):
        aa = gal.transform_to(AltAz(location=location, obstime=t))
        az[i] = aa.az.deg
        el[i] = aa.alt.deg
    return az, el


def beam_splines(gain, theta_deg, phi_deg, mirror=False, wrap=False):
    """Bicubic splines of the beam in (EL, AZ), one per frequency.

    gain has shape (Nfreq, Ntheta, Nphi) on theta = 0..90 deg and
    phi = 0..359 deg. Raul maps the beam's phi directly onto AZ
    (readme: "AZ and EL of antenna is the same as AZ and EL of local
    coordinates"). ``mirror=True`` uses AZ = -phi instead (the
    orientation mistsim gives with beam_az_rot = 0; see notebook).
    ``wrap=True`` appends phi = 360 so AZ in (359, 360) is
    interpolated rather than clamped to the AZ = 359 column.
    """
    el = 90.0 - np.asarray(theta_deg)[::-1]
    g = np.asarray(gain)[:, ::-1, :]
    az = np.asarray(phi_deg, dtype=float)
    if mirror:
        # AZ = -phi mod 360: column j holds phi = (-az_j) mod 360
        idx = (-np.round(az).astype(int)) % 360
        g = g[:, :, idx]
    if wrap:
        az = np.concatenate([az, [360.0]])
        g = np.concatenate([g, g[:, :, :1]], axis=2)
    return [RectBivariateSpline(el, az, gf, kx=3, ky=3, s=0) for gf in g]


def _design(x, t, k, lo, hi):
    """B-spline design matrix (Npt, 4) indices and values.

    x is clamped to [lo, hi] first, as FITPACK's bispev does (so
    there is no extrapolation and, for AZ, no wrap).
    """
    x = np.clip(x, lo, hi)
    dm = BSpline.design_matrix(x, t, k).tocsr()
    dm.sort_indices()
    nnz = np.diff(dm.indptr)
    if not np.all(nnz == k + 1):
        # rows with fewer than k+1 nonzeros: pad explicitly
        idx = np.zeros((x.size, k + 1), dtype=int)
        val = np.zeros((x.size, k + 1))
        for r in range(x.size):
            s = slice(dm.indptr[r], dm.indptr[r + 1])
            n = dm.indptr[r + 1] - dm.indptr[r]
            idx[r, :n] = dm.indices[s]
            val[r, :n] = dm.data[s]
        return idx, val
    return dm.indices.reshape(-1, k + 1), dm.data.reshape(-1, k + 1)


def pixel_convolution(splines, sky, az, el, el_min=0.0):
    """Raul's above-horizon convolution for every time and frequency.

    T_ant = sum(B * T * mask) / sum(B * mask) over pixels with EL >= 0,
    with mask = EL >= el_min (the mountain blockage). Returns an array
    of shape (Ntimes, Nfreq).

    All splines share their knots (same grid, s = 0), so the spline
    value is B_f(p) = sum_ij C_f[i, j] N_i(EL_p) M_j(AZ_p). The sums
    over pixels are formed on the coefficient grid with one sparse
    product per time: S = W @ (m * T).T, num_f = <C_f, S_f>. This
    equals evaluating every spline with ``ev`` (checked by
    ``check_pixel_convolution``) and is ~100x faster.
    """
    kx, ky = splines[0].degrees
    tx_full, ty_full = splines[0].tck[:2]
    # data range = FITPACK's evaluation box
    tx, ty = tx_full[kx : len(tx_full) - kx], ty_full[ky : len(ty_full) - ky]
    nx = len(tx_full) - kx - 1
    ny = len(ty_full) - ky - 1
    coef = np.stack([s.tck[2] for s in splines])  # (F, nx*ny)
    ntime = az.shape[0]
    out = np.empty((ntime, len(splines)))
    for i in range(ntime):
        above = el[i] >= 0
        e = el[i][above].astype(float)
        a = az[i][above].astype(float)
        m = (e >= el_min).astype(float)
        ie, ve = _design(e, tx_full, kx, tx[0], tx[-1])
        ia, va = _design(a, ty_full, ky, ty[0], ty[-1])
        cols = (ie[:, :, None] * ny + ia[:, None, :]).reshape(e.size, -1)
        vals = (ve[:, :, None] * va[:, None, :]).reshape(e.size, -1)
        npt = e.size
        w = sparse.csr_array(
            (
                (vals * m[:, None]).ravel(),
                (np.repeat(np.arange(npt), cols.shape[1]), cols.ravel()),
            ),
            shape=(npt, nx * ny),
        )
        s_num = (w.T @ sky[:, above].T).T  # (F, nx*ny)
        s_den = w.T @ np.ones(npt)  # (nx*ny,)
        out[i] = np.sum(coef * s_num, axis=1) / (coef @ s_den)
    return out


def check_pixel_convolution(splines, sky, az, el, el_min=0.0, n=2):
    """Max |fast - direct| over the first n times (direct uses ev)."""
    fast = pixel_convolution(splines, sky, az[:n], el[:n], el_min)
    direct = np.empty_like(fast)
    for i in range(n):
        above = el[i] >= 0
        e = el[i][above].astype(float)
        a = az[i][above].astype(float)
        m = (e >= el_min).astype(float)
        for f, spl in enumerate(splines):
            b = spl.ev(e, a) * m
            direct[i, f] = np.dot(b, sky[f, above]) / b.sum()
    return float(np.max(np.abs(fast - direct)))


def residual_stats(diff):
    """Summary of a (Ntime, Nfreq) residual waterfall, in K."""
    d = np.asarray(diff)
    return {
        "mean_K": float(d.mean()),
        "rms_K": float(np.sqrt(np.mean(d**2))),
        "mean_abs_K": float(np.mean(np.abs(d))),
        "max_abs_K": float(np.max(np.abs(d))),
        "min_K": float(d.min()),
        "max_K": float(d.max()),
    }


def location(test):
    """EarthLocation of each of Raul's tests (readme.txt)."""
    if test in (0, 1):
        return EarthLocation.from_geodetic(
            -90.74750 * u.deg, 79.41833 * u.deg, height=150 * u.m
        )
    if test == 2:
        return EarthLocation.from_geodetic(
            0 * u.deg, 90 * u.deg, height=0 * u.m
        )
    raise ValueError(test)


def find_data_dir(repo):
    """Raul's data: the worktree's data/ or the shared checkout's."""
    for cand in (
        Path(repo) / "data",
        Path(repo).parent / "mistsim" / "data",
    ):
        if (cand / "20260215_for_christian").is_dir():
            return cand
    raise FileNotFoundError("Raul's 2026 test data not found")


def feko_beam_full(path, freqs):
    """Raul's FEKO beam from feko_beam.npz, extended to theta = 180.

    feko_beam.npz holds the total gain of
    beam_best_fit_meas_39_joint_wide.out (linear, theta 0..90 deg,
    phi 0..359 deg, radians in the file). Gain below the horizon is
    zero, as in raul_comparison.ipynb. Returns (gain_full,
    theta_full [rad], gain_above, theta [rad], phi [rad]).
    """
    d = np.load(path)
    theta = d["theta"]
    if theta.max() > np.pi * (1 + 1e-6):
        raise ValueError("feko_beam.npz must store theta in radians")
    fix = np.isin(d["freqs"] / 1e6, freqs)
    if not np.allclose(d["freqs"][fix] / 1e6, freqs):
        raise ValueError("beam frequencies do not cover freqs")
    gain_above = d["gain"][fix]
    gain = np.concatenate(
        (gain_above, np.zeros_like(gain_above[:, :-1, :])), axis=1
    )
    theta_full = np.concatenate((theta, np.deg2rad(np.arange(91, 181))))
    return gain, theta_full, gain_above, theta, d["phi"]


def run_mistsim(
    gain,
    freqs,
    sky_model,
    times_jd,
    loc,
    horizon=None,
    beam_az_rot=0.0,
    lmax=100,
    normalization="above_horizon",
    tgnd=300.0,
    sky_alm=None,
):
    """Simulate with mistsim and return (T_ant, fgnd).

    normalization:
      "above_horizon": sim with Tgnd = 0, then
          croissant.simulator.correct_ground_loss(T, fgnd, 0), i.e.
          divide by the beam integral over the unmasked sky. This is
          what raul_comparison.ipynb does and is Raul's normalisation.
      "full_sphere": sim with Tgnd = 0, no correction: divide by the
          integral of the whole beam; blocked beam sees 0 K.
      "default": sim with Tgnd = tgnd (mistsim's default 300 K):
          blocked beam sees a uniform ground at tgnd.
    sky_alm: optional precomputed sky alm in the simulation frame
      (``Simulator.precompute_sky_alm``); valid only for the same
      times_jd[0].
    """
    import croissant as cro

    import mistsim as ms

    beam = ms.Beam(
        gain,
        freqs,
        sampling="mwss",
        horizon=horizon,
        beam_az_rot=beam_az_rot,
    )
    tg = tgnd if normalization == "default" else 0.0
    sim = ms.Simulator(
        beam,
        sky_model,
        times_jd,
        freqs,
        loc.lon.deg,
        loc.lat.deg,
        alt=loc.height.to_value(u.m),
        lmax=lmax,
        Tgnd=tg,
    )
    tant = sim.sim(sky_alm=sky_alm)
    fgnd = sim.beam.compute_fgnd()
    if normalization == "above_horizon":
        tant = cro.simulator.correct_ground_loss(tant, fgnd, 0.0)
    elif normalization not in ("full_sphere", "default"):
        raise ValueError(normalization)
    return np.asarray(tant), np.asarray(fgnd)


def precompute_sky_alm(sky_model, freqs, times_jd, gain):
    """Sky alm in the CIRS frame of times_jd[0] (location-free)."""
    import mistsim as ms

    beam = ms.Beam(gain, freqs, sampling="mwss")
    sim = ms.Simulator(
        beam, sky_model, times_jd, freqs, 0.0, 0.0, lmax=100, Tgnd=0.0
    )
    return sim.precompute_sky_alm()


# ------------------------------------------------------------------
# Synthetic convention check (beam azimuth direction, mask rotation)
# ------------------------------------------------------------------

ROT_TEST_PHI0 = 30.0  # deg, beam-frame azimuth of the asymmetric lobe


def synthetic_beam(theta_rad, phi_rad):
    """Asymmetric test beam in the beam frame (theta, phi).

    G = cos^2(theta) * (1 + 0.8 sin(theta) cos(phi - 30 deg)) above
    theta = 90 deg, 0 below: one lobe toward beam-frame phi = 30 deg,
    so a mirror (phi -> -phi) or a rotation is visible.
    """
    th = np.asarray(theta_rad)
    ph = np.asarray(phi_rad)
    g = np.cos(th) ** 2 * (
        1 + 0.8 * np.sin(th) * np.cos(ph - np.deg2rad(ROT_TEST_PHI0))
    )
    return np.where(th <= np.pi / 2, g, 0.0)


def synthetic_mask(theta_rad, phi_rad):
    """One-sided horizon: blocked for theta > 60 deg, 0 <= phi < 90."""
    th = np.asarray(theta_rad)
    ph = np.mod(np.rad2deg(np.asarray(phi_rad)), 360.0)
    blocked = (th > np.deg2rad(60.0)) & (ph < 90.0)
    return ~blocked


def analytic_pixel_convolution(sky_f, az, el, rot_deg, mirror, mask_rotates):
    """Pixel-domain T_ant for the synthetic beam at one frequency.

    The beam-frame azimuth is phi_b = AZ - rot (mirror=False, Raul's
    phi = AZ convention) or phi_b = -(AZ - rot) (mirror=True). The
    mask is evaluated at phi_b if mask_rotates, else at the
    unrotated phi_b (rot = 0), i.e. fixed on the ground.
    """
    out = np.empty(az.shape[0])
    sgn = -1.0 if mirror else 1.0
    for i in range(az.shape[0]):
        above = el[i] >= 0
        th = np.deg2rad(90.0 - el[i][above].astype(float))
        a = az[i][above].astype(float)
        phb = np.deg2rad(sgn * (a - rot_deg))
        phm = phb if mask_rotates else np.deg2rad(sgn * a)
        b = synthetic_beam(th, phb) * synthetic_mask(th, phm)
        out[i] = np.dot(b, sky_f[above]) / b.sum()
    return out
