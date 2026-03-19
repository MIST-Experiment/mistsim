"""Pipeline orchestration for MIST mapmaking runs."""

import logging
from pathlib import Path

import astropy.units as u
import croissant as cro
import healpy as hp
import jax
import jax.numpy as jnp
import nbformat
import numpy as np
import scipy.sparse.linalg as sla
import yaml
from astropy.time import Time
from nbconvert.preprocessors import ExecutePreprocessor

from . import mapmaking
from .beam import Beam
from .sim import Simulator
from .sky import Sky

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


def load_config(path, run_name):
    """Load a run from a YAML config file.

    Parameters
    ----------
    path : str or Path
        Path to the ``runs.yaml`` file containing sites, beams,
        defaults, and named runs.
    run_name : str
        Key under the ``runs:`` section to load.

    Returns
    -------
    dict
        Fully expanded config dict ready for :func:`run_pipeline`.
    """
    path = Path(path)
    config_dir = path.resolve().parent
    with open(path) as f:
        registry = yaml.safe_load(f)

    runs = registry.get("runs", {})
    if run_name not in runs:
        available = list(runs)
        raise ValueError(f"Unknown run {run_name!r}. Available: {available}")

    cfg = dict(runs[run_name])
    # Use the run key as default run_name
    cfg.setdefault("run_name", run_name)

    cfg = _expand_beam_config(cfg, registry)
    _resolve_config_paths(cfg, config_dir)
    return cfg


def _expand_beam_config(cfg, registry):
    """Expand compact beam-list config into full sites format."""
    sites_defs = registry["sites"]
    beams_defs = registry["beams"]
    defaults = registry.get("defaults", {})

    # Start from defaults, then overlay run-specific settings
    merged = {}
    for key, val in defaults.items():
        merged[key] = dict(val)
    for key, val in cfg.items():
        if key == "beams":
            continue
        if isinstance(val, dict) and key in merged:
            merged[key].update(val)
        else:
            merged[key] = val

    # Build sites list from beam names
    beam_names = cfg["beams"]
    sites_list = []
    for beam_name in beam_names:
        if beam_name not in beams_defs:
            raise ValueError(
                f"Unknown beam {beam_name!r}. Available: {list(beams_defs)}"
            )
        beam_def = dict(beams_defs[beam_name])
        site_key = beam_def.pop("site")
        if site_key not in sites_defs:
            raise ValueError(
                f"Unknown site {site_key!r} for beam {beam_name!r}"
            )
        site_def = dict(sites_defs[site_key])
        # Merge: site coords + beam-specific settings
        entry = {"name": beam_name, **site_def, **beam_def}
        sites_list.append(entry)

    merged["sites"] = sites_list
    return merged


def _resolve_config_paths(cfg, config_dir):
    """Resolve relative file paths in *cfg* against *config_dir*."""
    # sky.haslam_file
    sky = cfg.get("sky", {})
    if "haslam_file" in sky:
        p = Path(sky["haslam_file"])
        if not p.is_absolute():
            sky["haslam_file"] = str(config_dir / p)

    # sites[*].beam_file
    for site in cfg.get("sites", []):
        if "beam_file" in site:
            p = Path(site["beam_file"])
            if not p.is_absolute():
                site["beam_file"] = str(config_dir / p)

    # output directories
    out = cfg.get("output", {})
    for key in ("results_dir", "notebooks_dir"):
        if key in out:
            p = Path(out[key])
            if not p.is_absolute():
                out[key] = str(config_dir / p)


def _resolve_freq_indices(config):
    """Return frequency indices to simulate.

    - ``freq_indices`` in sky config → use that list
    - Only ``freq_index`` (singular) → ``[freq_index]``
    - Neither → all frequencies in ``freq_range``

    Returns
    -------
    list[int]

    """
    sky_cfg = config["sky"]
    if "freq_indices" in sky_cfg:
        val = sky_cfg["freq_indices"]
        if val == "all":
            fr = sky_cfg["freq_range"]
            return list(range(fr[1] - fr[0]))
        return list(val)
    if "freq_index" in sky_cfg:
        return [sky_cfg["freq_index"]]
    fr = sky_cfg["freq_range"]
    return list(range(fr[1] - fr[0]))


def _pad_and_stack(arrays):
    """Pad 1-D arrays to equal length and stack."""
    max_len = max(len(a) for a in arrays)
    padded = [np.pad(a, (0, max_len - len(a))) for a in arrays]
    return np.stack(padded)


def run_name_from_config(config):
    """
    Derive a run name by joining site names with hyphens.

    If ``run_name`` is set in the config it is returned as-is.
    Otherwise the name is built from the site ``name`` fields,
    e.g. ``"mars-csa2022-dip-nv-dip"``.
    """
    explicit = config.get("run_name")
    if explicit:
        return explicit
    return "-".join(s["name"] for s in config["sites"])


# ------------------------------------------------------------------
# Sky
# ------------------------------------------------------------------


def _scale_map(m, freqs, beta=-2.55, f0=408, tcmb=2.725):
    scale = (freqs / f0) ** beta
    return (m - tcmb)[None, :] * scale[:, None] + tcmb


def setup_sky(config, freq_ix=None):
    """
    Load Haslam map, scale frequencies, build Sky object.

    Parameters
    ----------
    config : dict
    freq_ix : int or None
        Frequency index override.  When *None*, reads from
        ``config["sky"]["freq_index"]``.

    Returns
    -------
    sky : mistsim.Sky
    x_packed : np.ndarray
        Packed real alm vector.
    x_hp : np.ndarray
        Complex alm in healpy ordering.
    freqs : np.ndarray
    sim_freq : float

    """
    sky_cfg = config["sky"]
    d = np.load(sky_cfg["haslam_file"])
    haslam_onefreq = d["m"][-1]
    f0_haslam = d["freqs"][-1]

    fr = sky_cfg["freq_range"]
    freqs = np.arange(fr[0], fr[1])
    beta = sky_cfg.get("spectral_index", -2.55)
    haslam = _scale_map(haslam_onefreq, freqs, beta=beta, f0=f0_haslam)

    if freq_ix is None:
        freq_ix = sky_cfg.get("freq_index", 0)
    sim_freq = freqs[freq_ix]
    sky_map = haslam[freq_ix]

    sky = Sky(
        sky_map[None],
        sim_freq,
        sampling="healpix",
        coord="galactic",
    )

    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    sky_alm = sky.compute_alm_eq(world="earth")
    sky_alm = cro.utils.reduce_lmax(sky_alm, lmax)
    x_packed = np.asarray(mapmaking.pack_s2fft_to_real(sky_alm))
    x_hp = np.asarray(mapmaking.alm1d_to_hp(x_packed))

    return sky, x_packed, x_hp, freqs, sim_freq


# ------------------------------------------------------------------
# Beam
# ------------------------------------------------------------------


def build_beam(site_cfg, sim_freq, freqs, freq_ix):
    """
    Build a Beam object from a site config dict.

    Returns
    -------
    beam : mistsim.Beam

    """
    beam_type = site_cfg.get("beam_type", "file")
    if beam_type == "sin2":
        # Synthetic sin^2 monopole beam on mwss grid
        # mwss: L+1 thetas, 2L phi
        L = 180  # default for mwss with 1-deg sampling
        theta = np.linspace(0, np.pi, L + 1)
        g = np.sin(theta) ** 2
        g = np.repeat(g[:, None], 2 * L, axis=-1)
        return Beam(
            g[None],
            sim_freq,
            sampling="mwss",
            horizon=None,
        )

    d = np.load(site_cfg["beam_file"])
    gain = d["gain"]
    g = gain[freq_ix]

    horizon = None
    if "horizon_max_theta" in site_cfg:
        theta = d["theta"]
        mask = theta <= site_cfg["horizon_max_theta"]
        horizon = mask[:, None]

    az_rot = site_cfg.get("beam_az_rot", 0.0)
    tilt = site_cfg.get("beam_tilt", 0.0)
    return Beam(
        g[None],
        sim_freq,
        sampling="mwss",
        horizon=horizon,
        beam_az_rot=az_rot,
        beam_tilt=tilt,
    )


# ------------------------------------------------------------------
# A-matrix
# ------------------------------------------------------------------


def build_stacked_A(config, sky, times_jd, sim_freq, freq_ix=None):
    """
    Build and stack A-matrices for all sites.

    Returns
    -------
    A : scipy.sparse.linalg.LinearOperator

    """
    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    if freq_ix is None:
        freq_ix = config["sky"].get("freq_index", 0)
    freqs_arr = np.arange(*config["sky"]["freq_range"])

    A_list = []
    for site in config["sites"]:
        beam = build_beam(site, sim_freq, freqs_arr, freq_ix)
        sim = Simulator(
            beam,
            sky,
            times_jd,
            sim_freq,
            site["lon"],
            site["lat"],
            alt=site.get("alt", 0),
            lmax=lmax,
        )
        logger.info("Building A-matrix for %s", site["name"])
        A_list.append(mapmaking.make_Amat(sim))

    if len(A_list) == 1:
        return A_list[0]
    return mapmaking.stack_As(*A_list)


def build_stacked_A_jax(config, sky, times_jd, sim_freq, freq_ix=None):
    """Build and stack pure JAX operators for all sites.

    Returns
    -------
    dict
        Stacked JAX operator dict.

    """
    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    if freq_ix is None:
        freq_ix = config["sky"].get("freq_index", 0)
    freqs_arr = np.arange(*config["sky"]["freq_range"])

    op_list = []
    for site in config["sites"]:
        beam = build_beam(site, sim_freq, freqs_arr, freq_ix)
        sim = Simulator(
            beam,
            sky,
            times_jd,
            sim_freq,
            site["lon"],
            site["lat"],
            alt=site.get("alt", 0),
            lmax=lmax,
        )
        logger.info(
            "Building JAX operator for %s",
            site["name"],
        )
        op_list.append(mapmaking.make_operators_jax(sim))

    if len(op_list) == 1:
        return op_list[0]
    return mapmaking.stack_operators_jax(*op_list)


# ------------------------------------------------------------------
# Noise
# ------------------------------------------------------------------


def compute_noise(y, df, dt, seed=1420):
    """
    Compute radiometer noise diagonal and a noise realization.

    Returns
    -------
    Ndiag : np.ndarray
    noise : np.ndarray

    """
    sigma = np.abs(y) / np.sqrt(dt * df)
    Ndiag = sigma**2
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0, scale=sigma)
    return Ndiag, noise


# ------------------------------------------------------------------
# Prior
# ------------------------------------------------------------------


def compute_prior(x_packed, lmax):
    """
    Compute the diagonal prior covariance from the true sky power
    spectrum.

    Returns
    -------
    Sdiag : np.ndarray

    """
    xhp = np.asarray(mapmaking.alm1d_to_hp(x_packed))
    cl = hp.alm2cl(xhp)
    ells_hp, emms_hp = hp.Alm.getlm(lmax)
    ells_pos = ells_hp[emms_hp != 0]
    ells_full = np.concatenate((ells_hp, ells_pos))
    return cl[ells_full]


# ------------------------------------------------------------------
# SVD + Wiener filter
# ------------------------------------------------------------------


def run_svd(Atilde, k):
    """
    Compute truncated SVD of Atilde, returned in descending order.

    Returns
    -------
    U : np.ndarray
    Sigma : np.ndarray
    Vh : np.ndarray

    """
    U, Sigma, Vh = sla.svds(Atilde, k=k)
    ix = np.argsort(Sigma)[::-1]
    return U[:, ix], Sigma[ix], Vh[ix]


def select_nvec(Sigma, threshold=1e-10):
    """
    Select number of modes to keep above a threshold.

    Returns
    -------
    nvec : int

    """
    return int(np.sum(Sigma > threshold))


def wiener_filter(
    U,
    Sigma,
    Vh,
    nvec,
    Ndiag,
    Sdiag,
    y,
    noise,
):
    """
    Apply Wiener filter via SVD.

    Returns
    -------
    x_rec_hp : np.ndarray
        Recovered alm in healpy ordering.
    D : np.ndarray
        Filter factors.

    """
    Dnum = Sigma[:nvec]
    D = Dnum / (1 + Dnum**2)
    Ut = U[:, :nvec]
    Vht = Vh[:nvec]

    W_tilde = Vht.T @ np.diag(D) @ Ut.T
    y_tilde = Ndiag ** (-0.5) * (y + noise)
    x_tilde = W_tilde @ y_tilde
    x_rec = Sdiag**0.5 * x_tilde
    x_rec_hp = np.asarray(mapmaking.alm1d_to_hp(x_rec))
    return x_rec_hp, D


def wiener_filter_cg(
    atilde_fwd,
    atilde_adj,
    Ndiag,
    Sdiag,
    y,
    noise,
    tol=1e-10,
    maxiter=2000,
):
    """Wiener filter via conjugate gradient.

    Solves ``(Atilde^H Atilde + I) x_tilde = Atilde^H y_tilde``
    using :func:`jax.scipy.sparse.linalg.cg`.

    Parameters
    ----------
    atilde_fwd, atilde_adj : callable
        Whitened forward/adjoint operators.
    Ndiag : array-like
        Noise variance diagonal.
    Sdiag : array-like
        Prior covariance diagonal.
    y : array-like
        Data vector.
    noise : array-like
        Noise realization.
    tol : float
        CG tolerance.
    maxiter : int
        Maximum CG iterations.

    Returns
    -------
    x_rec_hp : jax.Array
        Recovered alm in healpy ordering.
    info : int
        CG convergence info (0 = success).

    """
    Nm12 = 1.0 / jnp.sqrt(jnp.asarray(Ndiag))
    S12 = jnp.sqrt(jnp.asarray(Sdiag))

    y_tilde = Nm12 * (jnp.asarray(y) + jnp.asarray(noise))
    rhs = atilde_adj(y_tilde)

    def normal_op(x):
        return atilde_adj(atilde_fwd(x)) + x

    x_tilde, info = jax.scipy.sparse.linalg.cg(
        normal_op, rhs, tol=tol, maxiter=maxiter
    )
    x_rec = S12 * x_tilde
    x_rec_hp = mapmaking.alm1d_to_hp(x_rec)
    return jnp.asarray(x_rec_hp), info


# ------------------------------------------------------------------
# Posterior uncertainty
# ------------------------------------------------------------------


def posterior_uncertainty(
    Vh,
    Sigma,
    Sdiag,
    nvec,
    lmax,
    nside=128,
    n_realizations=1000,
    seed=1420,
):
    """
    Monte Carlo posterior uncertainty estimation.

    Returns
    -------
    result : dict
        Keys: std_alm, std_map, cl_prior, sigma2_prior, best_map_nside

    """
    Dnum = Sigma[:nvec]
    Vht = Vh[:nvec]
    n_alm = Vh.shape[1]

    rng = np.random.default_rng(seed=seed)
    wfull = rng.normal(size=(n_alm, n_realizations))

    post_std_svd = 1.0 / np.sqrt(Dnum**2 + 1.0)
    std_reduce = 1 - post_std_svd

    corr = Vht.T @ (std_reduce[:, None] * (Vht @ wfull))
    x_tilde_sim = wfull - corr

    x_sim = np.sqrt(Sdiag)[:, None] * x_tilde_sim
    x_sim_hp = np.array(
        [
            np.asarray(mapmaking.alm1d_to_hp(x_sim[:, i]))
            for i in range(n_realizations)
        ]
    )
    x_sim_map = np.array([hp.alm2map(xs, nside) for xs in x_sim_hp])

    std_alm_re = np.std(x_sim_hp.real, axis=0)
    std_alm_im = np.std(x_sim_hp.imag, axis=0)
    std_alm = std_alm_re + 1j * std_alm_im
    std_map = np.std(x_sim_map, axis=0)

    xhp_dummy = np.asarray(
        mapmaking.alm1d_to_hp(np.sqrt(Sdiag) * np.ones(n_alm))
    )
    cl_prior = hp.alm2cl(xhp_dummy)
    ell = np.arange(len(cl_prior))
    sigma2_prior = np.sum((2 * ell + 1) / (4 * np.pi) * cl_prior)

    return {
        "std_alm": std_alm,
        "std_map": std_map,
        "cl_prior": cl_prior,
        "sigma2_prior": sigma2_prior,
    }


# ------------------------------------------------------------------
# Multi-frequency data preparation
# ------------------------------------------------------------------


def _prepare_freq_data(config, times, freqs):
    """Build per-frequency arrays for the ``lax.map`` solve.

    Returns numpy/jax arrays stacked over frequencies,
    plus per-site phases (shared across freqs).
    """
    freq_indices = _resolve_freq_indices(config)
    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    freqs_arr = np.arange(*config["sky"]["freq_range"])
    post_cfg = config.get("posterior", {})
    noise_seed = post_cfg.get("seed", 1420)
    df = (freqs[1] - freqs[0]) * 1e6
    dt = (times[1] - times[0]).to_value(u.s)

    beam_alms_list = []
    y_list, Ndiag_list, noise_list = [], [], []
    Sdiag_list, x_packed_list, x_hp_list = [], [], []
    phases_per_site = []
    sim_freqs = []

    for i, fi in enumerate(freq_indices):
        sky_f, xp_f, xhp_f, _, sf = setup_sky(config, freq_ix=fi)
        x_packed_list.append(xp_f)
        x_hp_list.append(xhp_f)
        sim_freqs.append(sf)

        # Per-site beam + phases
        beams_f = []
        y_parts = []
        for s, site in enumerate(config["sites"]):
            beam = build_beam(site, sf, freqs_arr, fi)
            sim = Simulator(
                beam,
                sky_f,
                times.jd,
                sf,
                site["lon"],
                site["lat"],
                alt=site.get("alt", 0),
                lmax=lmax,
            )
            ba = sim.compute_beam_eq()
            ba = cro.utils.reduce_lmax(ba, lmax)
            beams_f.append(ba[0])  # squeeze freq dim
            if i == 0:
                phases_per_site.append(sim.phases)
            y_s = mapmaking._forward_single_freq(
                jnp.asarray(xp_f),
                ba[0],
                phases_per_site[s],
            )
            y_parts.append(np.asarray(y_s))

        beam_alms_list.append(jnp.stack(beams_f))
        y_f = np.concatenate(y_parts)
        y_list.append(y_f)
        Nd_f, n_f = compute_noise(y_f, df, dt, seed=noise_seed + i)
        Ndiag_list.append(Nd_f)
        noise_list.append(n_f)
        Sdiag_list.append(compute_prior(xp_f, lmax))

        logger.info(
            "Prepared freq %d/%d (%.0f MHz)",
            i + 1,
            len(freq_indices),
            sf,
        )

    return {
        "beam_alms": jnp.stack(beam_alms_list),
        "phases": jnp.stack(phases_per_site),
        "y": jnp.asarray(np.stack(y_list)),
        "Ndiag": jnp.asarray(np.stack(Ndiag_list)),
        "Sdiag": jnp.asarray(np.stack(Sdiag_list)),
        "noise": jnp.asarray(np.stack(noise_list)),
        "x_packed": np.stack(x_packed_list),
        "x_hp": np.stack(x_hp_list),
        "sim_freqs": np.array(sim_freqs),
        "freq_indices": freq_indices,
    }


def _solve_all_freqs(fdata, config):
    """CG + randomized SVD for all frequencies via lax.map.

    Parameters
    ----------
    fdata : dict
        Output of :func:`_prepare_freq_data`.
    config : dict

    Returns
    -------
    x_rec_hp : jax.Array, ``(nfreq, nalm_hp)``

    """
    phases = fdata["phases"]
    nsites = phases.shape[0]
    nalm = fdata["Sdiag"].shape[1]

    cg_cfg = config.get("cg", {})
    tol = cg_cfg.get("tol", 1e-10)
    maxiter = cg_cfg.get("maxiter", 2000)

    def _solve_one(args):
        ba_f, y_f, Nd_f, Sd_f, n_f = args

        def fwd(x):
            outs = [
                mapmaking._forward_single_freq(x, ba_f[s], phases[s])
                for s in range(nsites)
            ]
            return jnp.concatenate(outs)

        xd = jnp.zeros(nalm, dtype=jnp.float64)
        _tr = jax.linear_transpose(fwd, xd)

        def adj(y):
            return _tr(y)[0]

        Nm12 = 1.0 / jnp.sqrt(Nd_f)
        S12 = jnp.sqrt(Sd_f)

        def at_fwd(x):
            return Nm12 * fwd(S12 * x)

        def at_adj(y):
            return S12 * adj(Nm12 * y)

        y_tilde = Nm12 * (y_f + n_f)
        rhs = at_adj(y_tilde)

        def normal_op(x):
            return at_adj(at_fwd(x)) + x

        x_tilde, _ = jax.scipy.sparse.linalg.cg(
            normal_op, rhs, tol=tol, maxiter=maxiter
        )
        x_rec = S12 * x_tilde
        return mapmaking.alm1d_to_hp(x_rec)

    batch = (
        fdata["beam_alms"],
        fdata["y"],
        fdata["Ndiag"],
        fdata["Sdiag"],
        fdata["noise"],
    )
    logger.info(
        "Solving %d frequencies via lax.map (first call compiles)",
        fdata["beam_alms"].shape[0],
    )
    return jax.lax.map(_solve_one, batch)


def _build_atilde_for_freq(fdata, i):
    """Build Atilde operators for frequency *i*."""
    phases = fdata["phases"]
    nsites = phases.shape[0]
    nalm = fdata["Sdiag"].shape[1]
    ba_f = fdata["beam_alms"][i]

    def fwd(x):
        outs = [
            mapmaking._forward_single_freq(x, ba_f[s], phases[s])
            for s in range(nsites)
        ]
        return jnp.concatenate(outs)

    xd = jnp.zeros(nalm, dtype=jnp.float64)
    _tr = jax.linear_transpose(fwd, xd)

    def adj(y):
        return _tr(y)[0]

    return mapmaking.make_atilde_fns(
        fdata["Ndiag"][i], fwd, adj, fdata["Sdiag"][i]
    )


# ------------------------------------------------------------------
# Top-level orchestrator
# ------------------------------------------------------------------


def _run_single_freq(config):
    """Original single-frequency pipeline (backward compat)."""
    logger.info("Setting up sky")
    sky, x_packed, x_hp, freqs, sim_freq = setup_sky(config)

    obs = config["observation"]
    tstart = Time(obs["start_time"])
    tend = tstart + obs["n_sidereal_days"] * u.sday
    times = cro.utils.time_array(
        t_start=tstart,
        t_end=tend,
        N_times=obs["n_times"],
    )
    lmax = obs["lmax"]
    solver = config.get("solver", "cg")
    svd_cfg = config.get("svd", {})
    post_cfg = config.get("posterior", {})
    noise_seed = post_cfg.get("seed", 1420)

    if solver == "cg":
        logger.info("Building JAX operators")
        ops = build_stacked_A_jax(config, sky, times.jd, sim_freq)
        logger.info("Running forward model")
        y = np.asarray(ops["forward_fn"](jnp.asarray(x_packed)))
    else:
        logger.info("Building A-matrices")
        A = build_stacked_A(config, sky, times.jd, sim_freq)
        logger.info("Running forward model")
        y = A @ x_packed

    df = (freqs[1] - freqs[0]) * 1e6
    dt = (times[1] - times[0]).to_value(u.s)
    Ndiag, noise = compute_noise(y, df, dt, seed=noise_seed)
    Sdiag = compute_prior(x_packed, lmax)

    if solver == "cg":
        atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
            Ndiag,
            ops["forward_fn"],
            ops["adjoint_fn"],
            Sdiag,
        )
        logger.info("Solving Wiener filter via CG")
        cg_cfg = config.get("cg", {})
        x_rec_hp, cg_info = wiener_filter_cg(
            atilde_fwd,
            atilde_adj,
            Ndiag,
            Sdiag,
            y,
            noise,
            tol=cg_cfg.get("tol", 1e-10),
            maxiter=cg_cfg.get("maxiter", 2000),
        )
        x_rec_hp = np.asarray(x_rec_hp)
        logger.info("CG info=%s", cg_info)

        logger.info("Randomized SVD for posterior")
        k = svd_cfg.get("n_singular_values", 800)
        U, Sigma, Vh = mapmaking.randomized_svd_jax(
            atilde_fwd,
            atilde_adj,
            ops["shape"][1],
            ops["shape"][0],
            k,
            p=svd_cfg.get("oversampling", 10),
        )
        Sigma = np.asarray(Sigma)
        Vh = np.asarray(Vh)
    else:
        logger.info("Building Atilde and running SVD")
        Atilde = mapmaking.make_Atilde(Ndiag, A, Sdiag)
        k = svd_cfg.get("n_singular_values", 1200)
        U, Sigma, Vh = run_svd(Atilde, k=k)

    threshold = svd_cfg.get("threshold", 1e-10)
    nvec = select_nvec(Sigma, threshold=threshold)
    logger.info(
        "Selected nvec = %d (threshold = %e)",
        nvec,
        threshold,
    )

    if solver != "cg":
        logger.info("Applying Wiener filter")
        x_rec_hp, _ = wiener_filter(
            U,
            Sigma,
            Vh,
            nvec,
            Ndiag,
            Sdiag,
            y,
            noise,
        )

    Dnum = Sigma[:nvec]
    D = Dnum / (1 + Dnum**2)

    logger.info("Computing posterior uncertainty")
    nside = 128
    n_real = post_cfg.get("n_realizations", 1000)
    post = posterior_uncertainty(
        Vh,
        Sigma,
        Sdiag,
        nvec,
        lmax,
        nside=nside,
        n_realizations=n_real,
        seed=noise_seed,
    )

    fl = np.ones(lmax + 1)
    plot_lmax = min(10, lmax)
    fl[plot_lmax + 1 :] = 0.0
    best_map = hp.alm2map(hp.almxfl(x_rec_hp, fl), nside=nside)

    return {
        "run_name": run_name_from_config(config),
        "freqs": freqs,
        "lmax": lmax,
        "x_true": x_hp,
        "x_rec": x_rec_hp,
        "std_alm": post["std_alm"],
        "std_map": post["std_map"],
        "cl_prior": post["cl_prior"],
        "sigma2_prior": post["sigma2_prior"],
        "Sigma": Sigma,
        "D": D,
        "nvec": nvec,
        "best_map": best_map,
        "config": config,
    }


def _run_multi_freq(config):
    """Multi-frequency pipeline with lax.map solve."""
    obs = config["observation"]
    tstart = Time(obs["start_time"])
    tend = tstart + obs["n_sidereal_days"] * u.sday
    times = cro.utils.time_array(
        t_start=tstart,
        t_end=tend,
        N_times=obs["n_times"],
    )
    lmax = obs["lmax"]
    fr = config["sky"]["freq_range"]
    freqs = np.arange(fr[0], fr[1])

    # Data prep (Python loop)
    fdata = _prepare_freq_data(config, times, freqs)
    nfreq = len(fdata["freq_indices"])

    # CG solve (lax.map — lightweight output)
    x_rec_all = np.asarray(_solve_all_freqs(fdata, config))

    # rSVD + posterior (Python loop — one freq at a time
    # to avoid OOM from stacking Vh across frequencies)
    svd_cfg = config.get("svd", {})
    post_cfg = config.get("posterior", {})
    noise_seed = post_cfg.get("seed", 1420)
    threshold = svd_cfg.get("threshold", 1e-10)
    k = svd_cfg.get("n_singular_values", 800)
    p_over = svd_cfg.get("oversampling", 10)
    nside = 128
    n_real = post_cfg.get("n_realizations", 1000)
    nalm = fdata["Sdiag"].shape[1]
    ndata = fdata["y"].shape[1]

    Sigma_list, nvec_list = [], []
    std_alm_list, std_map_list = [], []
    cl_prior_list, sigma2_prior_list = [], []
    best_map_list = []

    for i in range(nfreq):
        logger.info(
            "rSVD + posterior freq %d/%d",
            i + 1,
            nfreq,
        )
        at_fwd, at_adj = _build_atilde_for_freq(fdata, i)
        _, Sig_i, Vh_i = mapmaking.randomized_svd_jax(
            at_fwd, at_adj, nalm, ndata, k, p=p_over
        )
        Sig_i = np.asarray(Sig_i)
        Vh_i = np.asarray(Vh_i)
        Sigma_list.append(Sig_i)

        nv = select_nvec(Sig_i, threshold=threshold)
        nvec_list.append(nv)

        post = posterior_uncertainty(
            Vh_i,
            Sig_i,
            np.asarray(fdata["Sdiag"][i]),
            nv,
            lmax,
            nside=nside,
            n_realizations=n_real,
            seed=noise_seed,
        )
        std_alm_list.append(post["std_alm"])
        std_map_list.append(post["std_map"])
        cl_prior_list.append(post["cl_prior"])
        sigma2_prior_list.append(post["sigma2_prior"])

        fl = np.ones(lmax + 1)
        fl[min(10, lmax) + 1 :] = 0.0
        bm = hp.alm2map(hp.almxfl(x_rec_all[i], fl), nside=nside)
        best_map_list.append(bm)
        del Vh_i  # free ~200 MB per freq

    return {
        "run_name": run_name_from_config(config),
        "freqs": freqs,
        "sim_freqs": fdata["sim_freqs"],
        "lmax": lmax,
        "x_true": fdata["x_hp"],
        "x_rec": x_rec_all,
        "std_alm": np.stack(std_alm_list),
        "std_map": np.stack(std_map_list),
        "cl_prior": np.stack(cl_prior_list),
        "sigma2_prior": np.array(sigma2_prior_list),
        "Sigma": np.stack(Sigma_list),
        "nvec": np.array(nvec_list),
        "best_map": np.stack(best_map_list),
        "config": config,
        "multi_freq": True,
    }


def run_pipeline(config):
    """Run the full mapmaking pipeline from a config dict.

    When ``freq_index`` is set in the sky config, runs the
    single-frequency path (backward compatible).  Otherwise
    runs all frequencies via ``lax.map``.

    Set ``config["solver"]`` to ``"cg"`` (default) or
    ``"svd"`` for the original ARPACK path.

    Returns
    -------
    results : dict

    """
    sky_cfg = config.get("sky", {})
    # freq_indices (plural) activates multi-freq even if
    # freq_index is also present from defaults.
    if "freq_index" in sky_cfg and "freq_indices" not in sky_cfg:
        return _run_single_freq(config)
    return _run_multi_freq(config)


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


def save_results(results, path):
    """Save results dict to npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "freqs": results["freqs"],
        "lmax": results["lmax"],
        "x_true": results["x_true"],
        "x_rec": results["x_rec"],
        "cl_prior": results["cl_prior"],
        "sigma2_prior": results["sigma2_prior"],
        "std_alm": results["std_alm"],
        "std_map": results["std_map"],
        "Sigma": results["Sigma"],
        "nvec": results["nvec"],
        "config_yaml": yaml.dump(results["config"]),
    }
    if results.get("multi_freq"):
        d["sim_freqs"] = results["sim_freqs"]
        d["multi_freq"] = True
    np.savez(path, **d)
    logger.info("Saved results to %s", path)


# ------------------------------------------------------------------
# Notebook generation
# ------------------------------------------------------------------


def _make_code_cell(source):
    return nbformat.v4.new_code_cell(source=source)


def _make_md_cell(source):
    return nbformat.v4.new_markdown_cell(source=source)


def generate_notebook(results, npz_path, output_path):
    """Generate and execute a diagnostic notebook."""
    npz_path = Path(npz_path).resolve()
    output_path = Path(output_path).resolve()
    nb = nbformat.v4.new_notebook()
    cells = []
    is_multi = results.get("multi_freq", False)
    cfg = results["config"]
    site_names = ", ".join(s["name"] for s in cfg["sites"])

    if is_multi:
        cells += _nb_cells_multi(results, npz_path, cfg, site_names)
    else:
        cells += _nb_cells_single(results, npz_path, cfg, site_names)

    nb.cells = cells
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(
            nb,
            {"metadata": {"path": str(output_path.parent)}},
        )
    except Exception:
        logger.warning("Notebook execution failed; saving unexecuted notebook")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        nbformat.write(nb, f)
    logger.info("Saved notebook to %s", output_path)


def _nb_cells_single(results, npz_path, cfg, sites):
    """Notebook cells for single-frequency results."""
    cells = []
    cells.append(
        _make_code_cell(
            "import numpy as np\n"
            "import healpy as hp\n"
            "import matplotlib.pyplot as plt\n"
            "import mistsim.plotting as msplt\n"
            "%matplotlib inline\n"
            "\n"
            f'd = np.load("{npz_path}")\n'
            'freqs = d["freqs"]\n'
            'lmax = int(d["lmax"])\n'
            'x_true = d["x_true"]\n'
            'x_rec = d["x_rec"]\n'
            'std_alm = d["std_alm"]\n'
            'cl_prior = d["cl_prior"]\n'
            'sigma2_prior = float(d["sigma2_prior"])\n'
            'Sigma = d["Sigma"]\n'
            'nvec = int(d["nvec"])\n'
            'std_map = d["std_map"]\n'
        )
    )
    lmax_cfg = cfg["observation"]["lmax"]
    k_cfg = cfg.get("svd", {}).get("n_singular_values", "N/A")
    cells.append(
        _make_md_cell(
            f"# Diagnostics: {results['run_name']}\n\n"
            f"**Sites:** {sites}  \n"
            f"**lmax:** {lmax_cfg}  \n"
            f"**nvec:** {results['nvec']}  \n"
            f"**SVD k:** {k_cfg}"
        )
    )
    cells.append(_make_md_cell("## Singular Value Spectrum"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_singular_values(Sigma, nvec=nvec)\nplt.show()"
        )
    )
    cells.append(_make_md_cell("## Filter Factors"))
    cells.append(
        _make_code_cell(
            "Dnum = Sigma[:nvec]\n"
            "D = Dnum / (1 + Dnum**2)\n"
            "fig = msplt.plot_filter_factors(D)\n"
            "plt.show()"
        )
    )
    cells.append(_make_md_cell("## Alm Comparison"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_alm_comparison(\n"
            "    x_true, x_rec, std_alm, "
            "lmax, lmax_plot=8\n"
            ")\nplt.show()"
        )
    )
    cells.append(_make_md_cell("## Power Spectra"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_power_spectra("
            "x_true, x_rec, std_alm)\n"
            "plt.show()"
        )
    )
    cells.append(_make_md_cell("## Transfer Function"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_transfer_function(x_true, x_rec)\nplt.show()"
        )
    )
    cells.append(_make_md_cell("## Maps & Residuals (Equatorial)"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_maps_and_residuals(\n"
            "    x_true, x_rec, lmax, plot_lmax=10,\n"
            "    nside=128, plot_galactic=False\n"
            ")\nplt.show()"
        )
    )
    cells.append(_make_md_cell("## Maps & Residuals (Galactic)"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_maps_and_residuals(\n"
            "    x_true, x_rec, lmax, plot_lmax=10,\n"
            "    nside=128, plot_galactic=True\n"
            ")\nplt.show()"
        )
    )
    cells.append(_make_md_cell("## Posterior Uncertainty"))
    cells.append(
        _make_code_cell(
            "fl = np.ones(lmax + 1)\n"
            "fl[11:] = 0.0\n"
            "best_map = hp.alm2map(\n"
            "    hp.almxfl(x_rec, fl), nside=128\n"
            ")\n"
            "fig = msplt.plot_posterior_maps(\n"
            "    std_map, sigma2_prior, "
            "best_map, nside=128\n"
            ")\nplt.show()"
        )
    )
    return cells


def _nb_cells_multi(results, npz_path, cfg, sites):
    """Notebook cells for multi-frequency results."""
    cells = []
    cells.append(
        _make_code_cell(
            "import numpy as np\n"
            "import healpy as hp\n"
            "import matplotlib.pyplot as plt\n"
            "import mistsim.plotting as msplt\n"
            "%matplotlib inline\n\n"
            f'd = np.load("{npz_path}")\n'
            'freqs = d["freqs"]\n'
            'sim_freqs = d["sim_freqs"]\n'
            'lmax = int(d["lmax"])\n'
            "nf = len(sim_freqs)\n"
            'x_true = d["x_true"]\n'
            'x_rec = d["x_rec"]\n'
            'std_alm = d["std_alm"]\n'
            'Sigma = d["Sigma"]\n'
            'nvec = d["nvec"]\n'
            'std_map = d["std_map"]\n'
            'cl_prior = d["cl_prior"]\n'
            'sigma2_prior = d["sigma2_prior"]\n'
        )
    )
    lmax_cfg = cfg["observation"]["lmax"]
    k_cfg = cfg.get("svd", {}).get("n_singular_values", "N/A")
    cells.append(
        _make_md_cell(
            f"# Diagnostics: {results['run_name']}\n\n"
            f"**Sites:** {sites}  \n"
            f"**lmax:** {lmax_cfg}  \n"
            f"**Frequencies:** {len(results['sim_freqs'])}"
            f"  \n**SVD k:** {k_cfg}"
        )
    )
    # Per-frequency diagnostics
    cells.append(_make_md_cell("## Per-Frequency Diagnostics"))
    cells.append(
        _make_code_cell(
            "for fi in range(nf):\n"
            "    freq_mhz = sim_freqs[fi]\n"
            '    print(f"\\n=== {freq_mhz:.0f} MHz ==='
            '")\n'
            "    nv = int(nvec[fi])\n"
            "    fig = msplt.plot_singular_values(\n"
            "        Sigma[fi], nvec=nv)\n"
            "    plt.show()\n"
            "    fig = msplt.plot_alm_comparison(\n"
            "        x_true[fi], x_rec[fi],\n"
            "        std_alm[fi], lmax, lmax_plot=8)\n"
            "    plt.show()\n"
            "    fig = msplt.plot_power_spectra(\n"
            "        x_true[fi], x_rec[fi],\n"
            "        std_alm[fi])\n"
            "    plt.show()\n"
            "    fig = msplt.plot_transfer_function(\n"
            "        x_true[fi], x_rec[fi])\n"
            "    plt.show()\n"
        )
    )
    # Cross-frequency comparison
    cells.append(_make_md_cell("## Cross-Frequency Comparison"))
    cells.append(
        _make_code_cell(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for fi in range(nf):\n"
            "    ax.semilogy(Sigma[fi],\n"
            "        label=f'{sim_freqs[fi]:.0f} MHz')\n"
            "ax.set_xlabel('Index')\n"
            "ax.set_ylabel('Singular Value')\n"
            "ax.set_title("
            "'Singular Values Across Frequencies')\n"
            "ax.legend()\n"
            "ax.grid(alpha=0.3)\n"
            "plt.show()\n\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for fi in range(nf):\n"
            "    cl_t = hp.alm2cl(x_true[fi])\n"
            "    cl_x = hp.alm2cl(\n"
            "        x_true[fi], x_rec[fi])\n"
            "    v = cl_t > 0\n"
            "    tf = np.zeros_like(cl_t)\n"
            "    tf[v] = cl_x[v] / cl_t[v]\n"
            "    ax.plot(tf,\n"
            "        label=f'{sim_freqs[fi]:.0f} MHz')\n"
            "ax.axhline(1, color='k', ls='--', "
            "alpha=0.5)\n"
            "ax.set_xlabel(r'$\\ell$')\n"
            "ax.set_ylabel('Transfer Function')\n"
            "ax.set_title("
            "'Transfer Functions Across Frequencies')\n"
            "ax.legend()\n"
            "ax.grid(alpha=0.3)\n"
            "plt.show()\n"
        )
    )
    return cells
