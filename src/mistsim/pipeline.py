"""Pipeline orchestration for MIST mapmaking runs."""

import copy
import logging
from pathlib import Path

import astropy.units as u
import croissant as cro
import healpy as hp
import jax
import jax.numpy as jnp
import nbformat
import numpy as np
import s2fft
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
        Fully expanded config dict ready for :func:`run_mapmaking`.
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
    shape = (-1,) + (1,) * m.ndim
    return (m - tcmb)[None] * scale.reshape(shape) + tcmb


def setup_sky(config, freq_ix=None, haslam_scaled=None):
    """
    Load Haslam map, scale frequencies, build Sky object.

    Parameters
    ----------
    config : dict
    freq_ix : int or None
        Frequency index override.  When *None*, reads from
        ``config["sky"]["freq_index"]``.
    haslam_scaled : np.ndarray or None
        Pre-loaded and pre-scaled Haslam map with shape
        ``(n_freqs, n_pix)``.  When provided the map is not
        reloaded from disk, which avoids redundant I/O when
        calling this function inside a frequency loop.

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
    fr = sky_cfg["freq_range"]
    freqs = np.arange(fr[0], fr[1])

    if haslam_scaled is None:
        d = np.load(sky_cfg["haslam_file"])
        beta = sky_cfg.get("spectral_index", -2.55)
        if "f0" in d:
            # MWSS format: single ref map + reference freq
            haslam_onefreq = d["m"]
            f0_haslam = float(d["f0"])
        else:
            # Legacy HEALPix format: all freqs stacked
            haslam_onefreq = d["m"][-1]
            f0_haslam = d["freqs"][-1]
        haslam_scaled = _scale_map(
            haslam_onefreq, freqs, beta=beta, f0=f0_haslam
        )

    if freq_ix is None:
        freq_ix = sky_cfg.get("freq_index", 0)
    sim_freq = freqs[freq_ix]
    sky_map = haslam_scaled[freq_ix]

    sampling = sky_cfg.get("sampling", "mwss")
    sky = Sky(
        sky_map[None],
        sim_freq,
        sampling=sampling,
        coord="galactic",
    )

    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    sky_alm = sky.compute_alm_eq(world="earth")
    sky_alm = cro.utils.reduce_lmax(sky_alm, lmax)
    x_packed = np.asarray(mapmaking.pack_s2fft_to_real(sky_alm))
    x_hp = np.asarray(mapmaking.alm1d_to_hp(x_packed))

    return sky, x_packed, x_hp, freqs, sim_freq


def setup_sky_multi_freq(config):
    """Load Haslam map at all frequencies. Beam-independent.

    Computes sky SHT once for reuse across multiple beams.
    Returns truth at mapmaking ``lmax`` and forward-model sky
    alm at ``lmax_sim``.

    Parameters
    ----------
    config : dict
        Pipeline config with ``sky`` and ``observation`` sections.

    Returns
    -------
    dict
        Keys: sky, sky_alm_sim, x_packed, x_hp, freqs,
        sim_freqs, freq_indices, lmax, lmax_sim, times.

    """
    obs = config["observation"]
    lmax = obs["lmax"]
    lmax_sim = obs.get("lmax_sim") or lmax
    times = _make_times(obs)
    sky_cfg = config["sky"]
    freqs_arr = np.arange(*sky_cfg["freq_range"])

    freq_indices = _resolve_freq_indices(config)
    nfreq = len(freq_indices)
    sim_freqs_arr = freqs_arr[np.array(freq_indices)]

    if lmax_sim > lmax:
        logger.info(
            "Forward model at lmax_sim=%d, mapmaking at lmax=%d",
            lmax_sim,
            lmax,
        )

    # Load and scale Haslam map once for all frequencies.
    d = np.load(sky_cfg["haslam_file"])
    beta = sky_cfg.get("spectral_index", -2.55)
    if "f0" in d:
        haslam_onefreq = d["m"]
        f0_haslam = float(d["f0"])
    else:
        haslam_onefreq = d["m"][-1]
        f0_haslam = d["freqs"][-1]
    haslam_scaled = _scale_map(
        haslam_onefreq, freqs_arr, beta=beta, f0=f0_haslam
    )

    # Sky SHT
    sampling = sky_cfg.get("sampling", "mwss")
    sky_maps = haslam_scaled[np.array(freq_indices)]
    sky = Sky(
        sky_maps,
        sim_freqs_arr,
        sampling=sampling,
        coord="galactic",
    )
    sky_alm_full = sky.compute_alm_eq(world="earth")

    # Truth at mapmaking lmax
    sky_alm = cro.utils.reduce_lmax(sky_alm_full, lmax)
    nalm = (lmax + 1) ** 2
    x_packed_flat = np.asarray(
        mapmaking.pack_s2fft_to_real(sky_alm)
    )
    x_packed_all = x_packed_flat.reshape(nfreq, nalm)
    x_hp_list = [
        np.asarray(mapmaking.alm1d_to_hp(x_packed_all[i]))
        for i in range(nfreq)
    ]

    # Sky at forward-model resolution
    if lmax_sim > lmax:
        sky_alm_sim = cro.utils.reduce_lmax(
            sky_alm_full, lmax_sim
        )
    else:
        sky_alm_sim = sky_alm

    logger.info("Sky SHT done for %d frequencies", nfreq)

    return {
        "sky": sky,
        "sky_alm_sim": sky_alm_sim,
        "x_packed": x_packed_all,
        "x_hp": np.stack(x_hp_list),
        "freqs": freqs_arr,
        "sim_freqs": sim_freqs_arr,
        "freq_indices": freq_indices,
        "lmax": lmax,
        "lmax_sim": lmax_sim,
        "times": times,
    }


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
    beam_freqs = d["freqs"]
    # Map global freq to beam-local index
    idx = np.where(np.isclose(beam_freqs, sim_freq))[0]
    if len(idx) == 0:
        raise ValueError(
            f"Frequency {sim_freq} MHz not found in beam file "
            f"{site_cfg['beam_file']} "
            f"(range {beam_freqs[0]}-{beam_freqs[-1]} MHz)"
        )
    g = gain[idx[0]]

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


def _build_multi_freq_beam(site_cfg, sim_freqs, freqs, freq_indices):
    """Build a multi-frequency Beam from a site config.

    Like :func:`build_beam` but creates a single Beam object
    spanning all requested frequencies so the SHT and rotation
    are vmapped in one call.

    Parameters
    ----------
    site_cfg : dict
    sim_freqs : array-like
        Frequencies (MHz) for the selected channels.
    freqs : array-like
        Full frequency array.
    freq_indices : list[int]

    Returns
    -------
    beam : mistsim.Beam

    """
    beam_type = site_cfg.get("beam_type", "file")
    nfreq = len(freq_indices)
    if beam_type == "sin2":
        L = 180
        theta = np.linspace(0, np.pi, L + 1)
        g = np.sin(theta) ** 2
        g = np.repeat(g[:, None], 2 * L, axis=-1)
        g_multi = np.broadcast_to(g[None], (nfreq,) + g.shape).copy()
        return Beam(
            g_multi,
            sim_freqs,
            sampling="mwss",
            horizon=None,
        )

    d = np.load(site_cfg["beam_file"])
    gain = d["gain"]
    beam_freqs = d["freqs"]
    # Map global freq_indices to beam-local indices
    local_indices = []
    for f in sim_freqs:
        idx = np.where(np.isclose(beam_freqs, f))[0]
        if len(idx) == 0:
            raise ValueError(
                f"Frequency {f} MHz not found in beam file "
                f"{site_cfg['beam_file']} "
                f"(range {beam_freqs[0]}-{beam_freqs[-1]} MHz)"
            )
        local_indices.append(idx[0])
    g = gain[np.array(local_indices)]

    horizon = None
    if "horizon_max_theta" in site_cfg:
        theta = d["theta"]
        mask = theta <= site_cfg["horizon_max_theta"]
        horizon = mask[:, None]

    az_rot = site_cfg.get("beam_az_rot", 0.0)
    tilt = site_cfg.get("beam_tilt", 0.0)
    return Beam(
        g,
        sim_freqs,
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
            np.array([sim_freq]),
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
            np.array([sim_freq]),
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


def select_nvec(Sigma, method="threshold", **kwargs):
    """Select the number of SVD modes to keep.

    Three selection methods are available:

    ``"threshold"``
        Keep all modes with singular value above *threshold*
        (default ``1e-10``).  This is the safest default — it
        retains every mode that carries measurable signal.

    ``"manual"``
        Use a fixed *nvec* provided by the caller.  A warning is
        emitted if the mode at the cut still has a Wiener filter
        factor D > 0.01 (i.e. it contributes > 1 % of its
        amplitude), which indicates significant information is
        being discarded.

    ``"auto"``
        Detect the elbow (knee) of the singular-value curve in
        log space.  The elbow is the point of maximum distance
        from the line connecting the first and last computed
        singular values.  No tuning parameters needed.

    In all cases a warning is logged when the selected *nvec*
    equals the total number of computed singular values *k*,
    since this means the spectrum may not have been fully
    captured and *n_singular_values* should be increased.

    Parameters
    ----------
    Sigma : array-like
        Singular values in descending order, shape ``(k,)``.
    method : ``{"threshold", "manual", "auto"}``
        Selection strategy.
    **kwargs
        Extra arguments forwarded to the chosen method:

        - *threshold* : ``float`` — cutoff for ``"threshold"``
          (default ``1e-10``).
        - *nvec* : ``int`` — fixed count for ``"manual"``.

    Returns
    -------
    nvec : int

    """
    Sigma = np.asarray(Sigma)
    k = len(Sigma)

    if method == "threshold":
        threshold = kwargs.get("threshold", 1e-10)
        nvec = int(np.sum(Sigma > threshold))

    elif method == "manual":
        nvec = int(kwargs["nvec"])
        if nvec > k:
            logger.warning(
                "Requested nvec=%d but only k=%d singular "
                "values were computed; clamping to k.",
                nvec,
                k,
            )
            nvec = k
        # Warn if the cut discards well-measured modes
        if nvec < k:
            sigma_cut = Sigma[nvec]
            D_cut = sigma_cut / (1 + sigma_cut**2)
            if D_cut > 0.01:
                logger.warning(
                    "Manual nvec=%d drops mode with "
                    "sigma=%.2e (filter factor D=%.3f). "
                    "Consider increasing nvec.",
                    nvec,
                    sigma_cut,
                    D_cut,
                )

    elif method == "auto":
        nvec = _find_elbow(Sigma)

    else:
        raise ValueError(
            f"Unknown nvec method {method!r}. "
            "Use 'threshold', 'manual', or 'auto'."
        )

    # Global guard: all computed SVs exceeded the cut
    if nvec == k and k > 0:
        logger.warning(
            "nvec == k (%d): all computed singular values "
            "are retained. The smallest is %.2e. "
            "Increase n_singular_values to ensure the full "
            "spectrum is captured.",
            k,
            Sigma[-1],
        )

    return nvec


def _find_elbow(Sigma):
    """Detect the elbow of a singular-value curve.

    Uses the maximum-distance-from-chord method in log space:
    the elbow is the index farthest from the straight line
    connecting the first and last points of log(Sigma).

    Returns 0 if *Sigma* has fewer than 3 elements.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = len(Sigma)
    if n < 3:
        return n

    # Work in log space (clip to avoid log(0))
    log_s = np.log(np.maximum(Sigma, 1e-300))

    # Chord from first to last point
    x = np.arange(n, dtype=np.float64)
    dx = float(n - 1)
    dy = log_s[-1] - log_s[0]
    chord_len = np.sqrt(dx**2 + dy**2)

    # Perpendicular distance of each point to the chord
    # Using cross-product formula: |n x (p - p0)| / |n|
    dist = np.abs(dy * x - dx * (log_s - log_s[0])) / chord_len

    # Elbow is the point of maximum distance
    elbow = int(np.argmax(dist))

    # Return elbow + 1 since we want to *keep* that mode
    return elbow + 1


def _select_nvec_from_config(Sigma, svd_cfg):
    """Read nvec selection settings from config and dispatch."""
    method = svd_cfg.get("nvec_method", "threshold")
    kwargs = {}
    if method == "threshold":
        kwargs["threshold"] = svd_cfg.get("threshold", 1e-10)
    elif method == "manual":
        kwargs["nvec"] = svd_cfg["nvec"]
    return select_nvec(Sigma, method=method, **kwargs)


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


# ------------------------------------------------------------------
# Prior mismatch study helpers
# ------------------------------------------------------------------


def prior_mismatch_products(config, freq_index):
    """Compute SVD products for a prior mismatch study.

    Re-runs data preparation for a single frequency, then
    computes the full rSVD (including *U*) and the projected
    data vector ``c = U^T y_tilde``.

    Parameters
    ----------
    config : dict
        Full pipeline config (e.g. loaded from npz config_yaml).
    freq_index : int
        Index into the frequency array to analyse.

    Returns
    -------
    dict
        Keys: U, Sigma, Vh, c, y_tilde, Sdiag, Ndiag,
        x_true_hp, lmax, nvec, fdata.

    """
    cfg = copy.deepcopy(config)
    cfg["sky"]["freq_indices"] = [freq_index]

    obs = cfg["observation"]
    tstart = Time(obs["start_time"])
    tend = tstart + obs["n_sidereal_days"] * u.sday
    times = cro.utils.time_array(
        t_start=tstart,
        t_end=tend,
        N_times=obs["n_times"],
    )
    fr = cfg["sky"]["freq_range"]
    freqs = np.arange(fr[0], fr[1])

    fdata = _prepare_freq_data(cfg, times, freqs)

    atilde_fwd, atilde_adj = _build_atilde_for_freq(fdata, 0)

    nalm = fdata["Sdiag"].shape[1]
    ndata = fdata["y"].shape[1]
    svd_cfg = cfg.get("svd", {})
    k = svd_cfg.get("n_singular_values", 800)
    p_over = svd_cfg.get("oversampling", 10)

    logger.info(
        "Running rSVD (k=%d, p=%d) for freq index %d",
        k,
        p_over,
        freq_index,
    )
    U, Sigma, Vh = mapmaking.randomized_svd_jax(
        atilde_fwd,
        atilde_adj,
        nalm,
        ndata,
        k,
        p=p_over,
    )
    U = np.asarray(U)
    Sigma = np.asarray(Sigma)
    Vh = np.asarray(Vh)

    Ndiag = np.asarray(fdata["Ndiag"][0])
    Sdiag = np.asarray(fdata["Sdiag"][0])
    y = np.asarray(fdata["y"][0])
    noise = np.asarray(fdata["noise"][0])

    y_tilde = Ndiag ** (-0.5) * (y + noise)
    c = U[:, :k].T @ y_tilde

    nvec = _select_nvec_from_config(Sigma, svd_cfg)
    lmax = obs["lmax"]

    return {
        "U": U,
        "Sigma": Sigma,
        "Vh": Vh,
        "c": c,
        "y_tilde": y_tilde,
        "Sdiag": Sdiag,
        "Ndiag": Ndiag,
        "x_true_hp": fdata["x_hp"][0],
        "lmax": lmax,
        "nvec": nvec,
    }


def wiener_filter_alpha(
    Sigma,
    Vh,
    c,
    Sdiag,
    alpha,
    nvec,
    lmax,
    nside=128,
    n_realizations=1000,
    seed=1420,
):
    """Reapply Wiener filter with prior scaled by *alpha*.

    Uses pre-computed SVD products and projected data to avoid
    re-running the expensive SVD.

    Parameters
    ----------
    Sigma : array, shape (k,)
        Singular values from rSVD with true prior.
    Vh : array, shape (k, nalm)
        Right singular vectors.
    c : array, shape (k,)
        Projected data ``U^T y_tilde``.
    Sdiag : array, shape (nalm,)
        True prior diagonal (will be scaled by *alpha*).
    alpha : float
        Prior scaling factor.
    nvec : int
        Number of modes to retain.
    lmax : int
        Maximum harmonic degree.
    nside : int
        HEALPix nside for posterior std map.
    n_realizations : int
        Monte Carlo realizations for posterior uncertainty.
    seed : int
        Random seed for MC realizations.

    Returns
    -------
    dict
        Keys: x_rec_hp, D_alpha, std_alm, std_map.

    """
    Sig_a = np.sqrt(alpha) * Sigma[:nvec]
    D_alpha = Sig_a / (1 + Sig_a**2)

    x_tilde = Vh[:nvec].T @ (D_alpha * c[:nvec])
    x_rec = np.sqrt(alpha * Sdiag) * x_tilde
    x_rec_hp = np.asarray(mapmaking.alm1d_to_hp(x_rec))

    # Posterior uncertainty (adapted from posterior_uncertainty)
    Vht = Vh[:nvec]
    n_alm = Vh.shape[1]
    rng = np.random.default_rng(seed=seed)
    wfull = rng.normal(size=(n_alm, n_realizations))

    post_std_svd = 1.0 / np.sqrt(Sig_a**2 + 1.0)
    std_reduce = 1 - post_std_svd

    corr = Vht.T @ (std_reduce[:, None] * (Vht @ wfull))
    x_tilde_sim = wfull - corr

    Sdiag_a = alpha * Sdiag
    x_sim = np.sqrt(Sdiag_a)[:, None] * x_tilde_sim
    x_sim_hp = np.array(
        [
            np.asarray(mapmaking.alm1d_to_hp(x_sim[:, i]))
            for i in range(n_realizations)
        ]
    )

    std_alm_re = np.std(x_sim_hp.real, axis=0)
    std_alm_im = np.std(x_sim_hp.imag, axis=0)
    std_alm = std_alm_re + 1j * std_alm_im

    x_sim_map = np.array([hp.alm2map(xs, nside) for xs in x_sim_hp])
    std_map = np.std(x_sim_map, axis=0)

    return {
        "x_rec_hp": x_rec_hp,
        "D_alpha": D_alpha,
        "std_alm": std_alm,
        "std_map": std_map,
    }


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


def _prepare_freq_data(config, times, freqs, y=None, x_packed=None, x_hp=None):
    """Build per-frequency arrays for the ``lax.map`` solve.

    When *y*, *x_packed*, and *x_hp* are provided the forward
    model is skipped and the solver is prepared directly from the
    supplied data.  This is the path used by ``run_mapmaking``.

    Returns numpy/jax arrays stacked over frequencies,
    plus per-site phases (shared across freqs).
    """
    freq_indices = _resolve_freq_indices(config)
    nfreq = len(freq_indices)
    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    sky_cfg = config["sky"]
    freqs_arr = np.arange(*sky_cfg["freq_range"])
    sim_freqs_arr = freqs_arr[np.array(freq_indices)]
    post_cfg = config.get("posterior", {})
    noise_seed = post_cfg.get("seed", 1420)
    df = (freqs[1] - freqs[0]) * 1e6
    dt = (times[1] - times[0]).to_value(u.s)

    have_data = y is not None

    # --- Sky model (always needed for beam rotation) ---
    d = np.load(sky_cfg["haslam_file"])
    beta = sky_cfg.get("spectral_index", -2.55)
    if "f0" in d:
        haslam_onefreq = d["m"]
        f0_haslam = float(d["f0"])
    else:
        haslam_onefreq = d["m"][-1]
        f0_haslam = d["freqs"][-1]
    haslam_scaled = _scale_map(
        haslam_onefreq, freqs_arr, beta=beta, f0=f0_haslam
    )

    sampling = sky_cfg.get("sampling", "mwss")
    sky_maps = haslam_scaled[np.array(freq_indices)]
    sky = Sky(
        sky_maps,
        sim_freqs_arr,
        sampling=sampling,
        coord="galactic",
    )

    # --- Truth / prior sky ---
    if x_packed is None:
        sky_alm_full = sky.compute_alm_eq(world="earth")
        sky_alm = cro.utils.reduce_lmax(sky_alm_full, lmax)
        nalm = (lmax + 1) ** 2
        x_packed_flat = np.asarray(mapmaking.pack_s2fft_to_real(sky_alm))
        x_packed = x_packed_flat.reshape(nfreq, nalm)
    if x_hp is None:
        x_hp_list = [
            np.asarray(mapmaking.alm1d_to_hp(x_packed[i]))
            for i in range(nfreq)
        ]
        x_hp = np.stack(x_hp_list)

    logger.info("Sky model ready for %d frequencies", nfreq)

    # --- Beams at mapmaking lmax ---
    beam_alms_per_site = []
    phases_per_site = []
    for s, site in enumerate(config["sites"]):
        beam = _build_multi_freq_beam(
            site, sim_freqs_arr, freqs_arr, freq_indices
        )
        sim = Simulator(
            beam,
            sky,
            times.jd,
            sim_freqs_arr,
            site["lon"],
            site["lat"],
            alt=site.get("alt", 0),
            lmax=lmax,
        )
        ba = sim.compute_beam_eq()
        ba = cro.utils.reduce_lmax(ba, lmax)
        beam_alms_per_site.append(ba)
        phases_per_site.append(sim.phases)
        logger.info(
            "Beam SHT done for site %s (%d freqs)",
            site["name"],
            nfreq,
        )

    # --- Forward model (if no data provided) + noise + prior ---
    y_list, Ndiag_list, noise_list, Sdiag_list = (
        [],
        [],
        [],
        [],
    )
    beam_alms_list = []
    for i in range(nfreq):
        beams_f = [ba[i] for ba in beam_alms_per_site]
        beam_alms_list.append(jnp.stack(beams_f))

        if have_data:
            y_f = y[i]
        else:
            y_parts = []
            for s in range(len(config["sites"])):
                y_s = mapmaking._forward_single_freq(
                    jnp.asarray(x_packed[i]),
                    beam_alms_per_site[s][i],
                    phases_per_site[s],
                )
                y_parts.append(np.asarray(y_s))
            y_f = np.concatenate(y_parts)

        y_list.append(y_f)
        Nd_f, n_f = compute_noise(y_f, df, dt, seed=noise_seed + i)
        Ndiag_list.append(Nd_f)
        noise_list.append(n_f)
        Sdiag_list.append(compute_prior(x_packed[i], lmax))
        logger.info(
            "Prepared freq %d/%d (%.0f MHz)",
            i + 1,
            nfreq,
            sim_freqs_arr[i],
        )

    return {
        "beam_alms": jnp.stack(beam_alms_list),
        "phases": jnp.stack(phases_per_site),
        "y": jnp.asarray(np.stack(y_list)),
        "Ndiag": jnp.asarray(np.stack(Ndiag_list)),
        "Sdiag": jnp.asarray(np.stack(Sdiag_list)),
        "noise": jnp.asarray(np.stack(noise_list)),
        "x_packed": x_packed,
        "x_hp": x_hp,
        "sim_freqs": sim_freqs_arr,
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


def _rsvd_all_freqs(fdata, config):
    """Randomized SVD for all frequencies via lax.map.

    Compiles the rSVD graph once and applies it to each frequency,
    avoiding per-frequency recompilation.

    Parameters
    ----------
    fdata : dict
        Output of :func:`_prepare_freq_data`.
    config : dict

    Returns
    -------
    Sigma_all : jax.Array, ``(nfreq, k)``
    Vh_all : jax.Array, ``(nfreq, k, nalm)``

    """
    phases = fdata["phases"]
    nsites = phases.shape[0]
    nalm = fdata["Sdiag"].shape[1]
    ndata = fdata["y"].shape[1]

    svd_cfg = config.get("svd", {})
    k = svd_cfg.get("n_singular_values", 800)
    p_over = svd_cfg.get("oversampling", 10)

    def _rsvd_one(args):
        ba_f, Nd_f, Sd_f = args

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

        at_fwd, at_adj = mapmaking.make_atilde_fns(Nd_f, fwd, adj, Sd_f)
        _, Sigma, Vh = mapmaking.randomized_svd_jax(
            at_fwd, at_adj, nalm, ndata, k, p=p_over
        )
        return Sigma, Vh

    batch = (
        fdata["beam_alms"],
        fdata["Ndiag"],
        fdata["Sdiag"],
    )
    logger.info(
        "rSVD for %d frequencies via lax.map (first call compiles)",
        fdata["beam_alms"].shape[0],
    )
    return jax.lax.map(_rsvd_one, batch)


# ------------------------------------------------------------------
# Time helper
# ------------------------------------------------------------------


def _make_times(obs):
    """Build time array from observation config."""
    tstart = Time(obs["start_time"])
    tend = tstart + obs["n_sidereal_days"] * u.sday
    return cro.utils.time_array(
        t_start=tstart,
        t_end=tend,
        N_times=obs["n_times"],
    )


def _time_spacing(obs, freqs):
    """Return df (Hz) and dt (seconds) from config and freqs."""
    times = _make_times(obs)
    df = (freqs[1] - freqs[0]) * 1e6
    dt = (times[1] - times[0]).to_value(u.s)
    return df, dt


# ------------------------------------------------------------------
# Stage 1: Forward simulation (data generation)
# ------------------------------------------------------------------


def _generate_single_freq(config):
    """Generate forward-model timestream for a single frequency.

    Returns
    -------
    dict
        Keys: y, x_packed, x_hp, freqs, sim_freq, lmax, lmax_sim,
        config.

    """
    logger.info("Setting up sky")
    sky, x_packed, x_hp, freqs, sim_freq = setup_sky(config)

    obs = config["observation"]
    times = _make_times(obs)
    lmax = obs["lmax"]
    lmax_sim = obs.get("lmax_sim") or lmax

    # Forward model at lmax_sim
    if lmax_sim > lmax:
        logger.info(
            "Forward model at lmax_sim=%d, mapmaking at lmax=%d",
            lmax_sim,
            lmax,
        )
        config_sim = copy.deepcopy(config)
        config_sim["observation"]["lmax"] = lmax_sim
        sky_sim, x_sim, _, _, _ = setup_sky(config_sim)
        ops_sim = build_stacked_A_jax(config_sim, sky_sim, times.jd, sim_freq)
        logger.info("Running forward model")
        y = np.asarray(ops_sim["forward_fn"](jnp.asarray(x_sim)))
    else:
        ops = build_stacked_A_jax(config, sky, times.jd, sim_freq)
        logger.info("Running forward model")
        y = np.asarray(ops["forward_fn"](jnp.asarray(x_packed)))

    return {
        "y": y,
        "x_packed": x_packed,
        "x_hp": x_hp,
        "freqs": freqs,
        "sim_freq": sim_freq,
        "lmax": lmax,
        "lmax_sim": lmax_sim,
        "config": config,
    }


def _generate_multi_freq(config):
    """Generate forward-model timestreams for multiple frequencies.

    Returns
    -------
    dict
        Keys: y, x_packed, x_hp, freqs, sim_freqs, freq_indices,
        lmax, lmax_sim, config.

    """
    sky_data = setup_sky_multi_freq(config)

    y_parts = []
    for site in config["sites"]:
        data = simulate_waterfall(site, sky_data)
        y_parts.append(data["y"])

    # Concatenate across beams/sites per frequency
    y_all = np.concatenate(y_parts, axis=1)

    return {
        "y": y_all,
        "x_packed": sky_data["x_packed"],
        "x_hp": sky_data["x_hp"],
        "freqs": sky_data["freqs"],
        "sim_freqs": sky_data["sim_freqs"],
        "freq_indices": sky_data["freq_indices"],
        "lmax": sky_data["lmax"],
        "lmax_sim": sky_data["lmax_sim"],
        "config": config,
    }


def simulate_waterfall(beam_cfg, sky_data):
    """Forward-simulate one beam across all frequencies.

    Uses ``cro.simulator.convolve`` for wideband simulation in a
    single call.

    Parameters
    ----------
    beam_cfg : dict
        Expanded beam/site entry with keys ``name``, ``lat``,
        ``lon``, ``alt``, and beam parameters (``beam_file``,
        ``beam_az_rot``, etc.).
    sky_data : dict
        Output of :func:`setup_sky_multi_freq`.

    Returns
    -------
    dict
        Keys: y, beam_name.

    """
    sky = sky_data["sky"]
    sky_alm_sim = sky_data["sky_alm_sim"]
    sim_freqs = sky_data["sim_freqs"]
    freqs = sky_data["freqs"]
    freq_indices = sky_data["freq_indices"]
    lmax_sim = sky_data["lmax_sim"]
    times = sky_data["times"]

    beam = _build_multi_freq_beam(
        beam_cfg, sim_freqs, freqs, freq_indices
    )
    sim = Simulator(
        beam,
        sky,
        times.jd,
        sim_freqs,
        beam_cfg["lon"],
        beam_cfg["lat"],
        alt=beam_cfg.get("alt", 0),
        lmax=lmax_sim,
    )
    beam_alm = cro.utils.reduce_lmax(
        sim.compute_beam_eq(), lmax_sim
    )

    wf = cro.simulator.convolve(beam_alm, sky_alm_sim, sim.phases)
    wf /= beam.compute_norm()[None, :]
    y = np.asarray(wf.real.T)  # (nfreq, ntimes)

    logger.info(
        "Forward model done for %s (%d freqs)",
        beam_cfg["name"],
        len(sim_freqs),
    )

    return {
        "y": y,
        "beam_name": beam_cfg["name"],
    }


# ------------------------------------------------------------------
# Top-level orchestrator
# ------------------------------------------------------------------


def _solve_single_freq(config, y, x_packed, x_hp, freqs):
    """Mapmaking solver for single frequency.

    Parameters
    ----------
    config : dict
        Full pipeline config.
    y : np.ndarray
        Timestream data (from simulation or real observation).
    x_packed : np.ndarray or None
        True sky packed alm (for prior).  If *None*, computed
        from the Haslam sky model via ``setup_sky``.
    x_hp : np.ndarray or None
        True sky in healpy ordering (for diagnostics).
    freqs : np.ndarray
        Full frequency array.

    """
    obs = config["observation"]
    times = _make_times(obs)
    lmax = obs["lmax"]
    lmax_sim = obs.get("lmax_sim") or lmax
    solver = config.get("solver", "cg")
    svd_cfg = config.get("svd", {})
    post_cfg = config.get("posterior", {})
    noise_seed = post_cfg.get("seed", 1420)

    # Sky for prior and A-matrix
    if x_packed is None:
        logger.info("Computing sky model for prior")
        sky, x_packed, x_hp, _, sim_freq = setup_sky(config)
    else:
        sky, _, _, _, sim_freq = setup_sky(config)

    # Mapmaking operators (at lmax)
    if solver == "cg":
        logger.info("Building JAX operators at lmax=%d", lmax)
        ops = build_stacked_A_jax(config, sky, times.jd, sim_freq)
    else:
        logger.info("Building A-matrices at lmax=%d", lmax)
        A = build_stacked_A(config, sky, times.jd, sim_freq)

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

    nvec = _select_nvec_from_config(Sigma, svd_cfg)
    logger.info("Selected nvec = %d", nvec)

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
        "lmax_sim": lmax_sim,
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


def _solve_multi_freq(config, y=None, x_packed=None, x_hp=None):
    """Multi-frequency mapmaking solver.

    When *y* is provided, the forward model is skipped.
    """
    obs = config["observation"]
    times = _make_times(obs)
    lmax = obs["lmax"]
    lmax_sim = obs.get("lmax_sim") or lmax
    fr = config["sky"]["freq_range"]
    freqs = np.arange(fr[0], fr[1])

    fdata = _prepare_freq_data(
        config,
        times,
        freqs,
        y=y,
        x_packed=x_packed,
        x_hp=x_hp,
    )
    nfreq = len(fdata["freq_indices"])

    # CG solve (lax.map)
    x_rec_all = np.asarray(_solve_all_freqs(fdata, config))

    # rSVD (lax.map)
    Sigma_all, Vh_all = _rsvd_all_freqs(fdata, config)
    Sigma_all = np.asarray(Sigma_all)
    Vh_all = np.asarray(Vh_all)

    # Posterior (Python loop — healpy cannot be JIT'd)
    svd_cfg = config.get("svd", {})
    post_cfg = config.get("posterior", {})
    noise_seed = post_cfg.get("seed", 1420)
    nside = 128
    n_real = post_cfg.get("n_realizations", 1000)

    nvec_list = []
    std_alm_list, std_map_list = [], []
    cl_prior_list, sigma2_prior_list = [], []
    best_map_list = []

    for i in range(nfreq):
        logger.info(
            "Posterior freq %d/%d",
            i + 1,
            nfreq,
        )
        Sig_i = Sigma_all[i]
        Vh_i = Vh_all[i]

        nv = _select_nvec_from_config(Sig_i, svd_cfg)
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

    return {
        "run_name": run_name_from_config(config),
        "freqs": freqs,
        "sim_freqs": fdata["sim_freqs"],
        "lmax": lmax,
        "lmax_sim": lmax_sim,
        "x_true": fdata["x_hp"],
        "x_rec": x_rec_all,
        "std_alm": np.stack(std_alm_list),
        "std_map": np.stack(std_map_list),
        "cl_prior": np.stack(cl_prior_list),
        "sigma2_prior": np.array(sigma2_prior_list),
        "Sigma": Sigma_all,
        "nvec": np.array(nvec_list),
        "best_map": np.stack(best_map_list),
        "config": config,
        "multi_freq": True,
    }


def _is_single_freq(config):
    """Check whether config selects single- or multi-freq path."""
    sky_cfg = config.get("sky", {})
    return "freq_index" in sky_cfg and "freq_indices" not in sky_cfg


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def generate_data(config):
    """Stage 1: Forward simulation only.

    Generates simulated timestream data from the sky model and
    beam configuration.  The result can be saved with
    :func:`save_data` and later loaded for mapmaking with
    :func:`run_mapmaking`.

    Returns
    -------
    dict
        Keys: y, x_packed, x_hp, freqs, lmax, lmax_sim, config.
        Multi-freq runs also include sim_freqs, freq_indices.

    """
    if _is_single_freq(config):
        return _generate_single_freq(config)
    return _generate_multi_freq(config)


def save_data(data, path):
    """Save forward-simulation data to npz.

    Parameters
    ----------
    data : dict
        Output of :func:`generate_data`.
    path : str or Path
        Output file path.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "y": data["y"],
        "x_packed": data["x_packed"],
        "x_hp": data["x_hp"],
        "freqs": data["freqs"],
        "lmax": data["lmax"],
        "lmax_sim": data["lmax_sim"],
        "config_yaml": yaml.dump(data["config"]),
    }
    if "sim_freqs" in data:
        d["sim_freqs"] = data["sim_freqs"]
        d["multi_freq"] = True
    np.savez(path, **d)
    logger.info("Saved data to %s", path)


def load_data(path):
    """Load forward-simulation data (or real data) from npz.

    Parameters
    ----------
    path : str or Path
        Path to npz file.

    Returns
    -------
    dict
        Keys: y, x_packed (may be None), x_hp (may be None),
        freqs, lmax, lmax_sim, config.

    """
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    config = yaml.safe_load(str(d["config_yaml"]))
    result = {
        "y": d["y"],
        "x_packed": d.get("x_packed"),
        "x_hp": d.get("x_hp"),
        "freqs": d["freqs"],
        "lmax": int(d["lmax"]),
        "lmax_sim": int(d.get("lmax_sim", d["lmax"])),
        "config": config,
    }
    if "sim_freqs" in d:
        result["sim_freqs"] = d["sim_freqs"]
        result["freq_indices"] = d.get("freq_indices")
    logger.info("Loaded data from %s", path)
    return result


def save_sim_data(data, sky_data, path):
    """Save per-beam forward-simulation data to npz.

    Parameters
    ----------
    data : dict
        Output of :func:`simulate_waterfall`.
    sky_data : dict
        Output of :func:`setup_sky_multi_freq`.
    path : str or Path
        Output file path.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        y=data["y"],
        beam_name=data["beam_name"],
        x_packed=sky_data["x_packed"],
        x_hp=sky_data["x_hp"],
        freqs=sky_data["sim_freqs"],
        lmax=sky_data["lmax"],
        lmax_sim=sky_data["lmax_sim"],
    )
    logger.info("Saved %s to %s", data["beam_name"], path)


def load_sim_data(path):
    """Load per-beam forward-simulation data from npz.

    Parameters
    ----------
    path : str or Path
        Path to per-beam npz file.

    Returns
    -------
    dict
        Keys: y, beam_name, x_packed, x_hp, freqs, lmax,
        lmax_sim.

    """
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    result = {
        "y": d["y"],
        "beam_name": str(d["beam_name"]),
        "x_packed": d["x_packed"],
        "x_hp": d["x_hp"],
        "freqs": d["freqs"],
        "lmax": int(d["lmax"]),
        "lmax_sim": int(d["lmax_sim"]),
    }
    logger.info("Loaded %s from %s", result["beam_name"], path)
    return result


def load_and_concat_sim_data(paths, freq_range=None):
    """Load per-beam NPZs and concatenate for mapmaking.

    Parameters
    ----------
    paths : list of str or Path
        Paths to per-beam npz files.
    freq_range : list of int or None
        ``[f_min, f_max)`` in MHz.  If given, only frequencies
        within this range are kept.

    Returns
    -------
    dict
        Keys: y (nfreq, n_beams * ntimes), x_packed, x_hp,
        freqs, sim_freqs, lmax, lmax_sim.

    """
    beams = [load_sim_data(p) for p in paths]
    ref = beams[0]

    # Determine target frequencies
    target_freqs = ref["freqs"]
    if freq_range is not None:
        mask = (target_freqs >= freq_range[0]) & (
            target_freqs < freq_range[1]
        )
        target_freqs = target_freqs[mask]

    # Select and concatenate y for each beam
    y_parts = []
    for b in beams:
        # Find indices of target_freqs in this beam's freqs
        idx = np.array([
            int(np.where(b["freqs"] == f)[0][0])
            for f in target_freqs
        ])
        y_parts.append(b["y"][idx])
    # y_parts[i] shape: (nfreq, ntimes)
    # Concatenate along time axis
    y = np.concatenate(y_parts, axis=1)

    # Sky truth from first beam (same for all)
    ref_idx = np.array([
        int(np.where(ref["freqs"] == f)[0][0])
        for f in target_freqs
    ])
    x_packed = ref["x_packed"][ref_idx]
    x_hp = ref["x_hp"][ref_idx]

    names = [b["beam_name"] for b in beams]
    logger.info(
        "Concatenated %d beams (%s), %d freqs, y shape %s",
        len(beams),
        ", ".join(names),
        len(target_freqs),
        y.shape,
    )

    return {
        "y": y,
        "x_packed": x_packed,
        "x_hp": x_hp,
        "freqs": target_freqs,
        "sim_freqs": target_freqs,
        "lmax": ref["lmax"],
        "lmax_sim": ref["lmax_sim"],
    }


def run_mapmaking(config, y, x_true=None, x_packed=None):
    """Stage 2: Mapmaking from data.

    Builds the A-matrix from the beam/observer configuration,
    computes noise and prior, and solves for the sky.

    Parameters
    ----------
    config : dict
        Full pipeline config (beams, sites, times, lmax, etc.).
    y : np.ndarray
        Timestream data — from simulation or real observation.
    x_true : np.ndarray or None
        True sky alm in healpy ordering (for diagnostics).
        Pass *None* for real data.
    x_packed : np.ndarray or None
        True sky in packed-real format (for prior).
        If *None*, the prior is computed from the Haslam sky
        model specified in the config.

    Returns
    -------
    results : dict

    """
    if _is_single_freq(config):
        fr = config["sky"]["freq_range"]
        freqs = np.arange(fr[0], fr[1])
        return _solve_single_freq(config, y, x_packed, x_true, freqs)
    return _solve_multi_freq(config, y=y, x_packed=x_packed, x_hp=x_true)


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
        "lmax_sim": results.get("lmax_sim", results["lmax"]),
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


def add_beam_maps(npz_path, freq, nside=None):
    """Compute equatorial beam maps and add them to an npz file.

    Loads the run config from the saved ``config_yaml``, builds
    each beam at the requested frequency, rotates to equatorial
    coordinates, and saves HEALPix maps back into the npz.

    Parameters
    ----------
    npz_path : str or Path
        Path to the results npz file.
    freq : float
        Frequency in MHz at which to evaluate the beams.
    nside : int or None
        HEALPix nside for the output beam maps.  Must satisfy
        ``nside <= (lmax + 1) // 2``.  Default is the largest
        power-of-2 nside that fits.

    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files}

    config = yaml.safe_load(str(data["config_yaml"]))

    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    L = lmax + 1
    if nside is None:
        nside = 2 ** int(np.floor(np.log2(L // 2)))
    if L < 2 * nside:
        raise ValueError(
            f"nside={nside} requires L >= {2 * nside} "
            f"but lmax={lmax} gives L={L}"
        )
    fr = config["sky"]["freq_range"]
    freqs_arr = np.arange(fr[0], fr[1])
    freq_ix = int(np.where(np.isclose(freqs_arr, freq))[0][0])

    # Minimal time array (only rotation matrices needed)
    start_time = Time(obs_cfg["start_time"])
    end_time = start_time + 1.0 * u.sday
    times = cro.utils.time_array(
        t_start=start_time,
        t_end=end_time,
        N_times=2,
    )

    # Dummy sky — compute_beam_eq only uses observer
    # location, not the sky itself.
    nside_dummy = 64
    dummy_sky = Sky(
        np.ones((1, hp.nside2npix(nside_dummy))),
        freq,
        sampling="healpix",
        coord="equatorial",
    )

    beam_maps = []
    beam_names = []
    for site in config["sites"]:
        logger.info(
            "Computing beam map for %s at %.0f MHz",
            site["name"],
            freq,
        )
        beam = build_beam(site, freq, freqs_arr, freq_ix)
        sim = Simulator(
            beam,
            dummy_sky,
            times.jd,
            np.array([freq]),
            site["lon"],
            site["lat"],
            alt=site.get("alt", 0),
            lmax=lmax,
        )
        beam_alm = sim.compute_beam_eq()
        beam_alm = cro.utils.reduce_lmax(beam_alm, lmax)
        bmap = s2fft.inverse_numpy(
            np.array(beam_alm[0]),
            L=lmax + 1,
            nside=nside,
            sampling="healpix",
        )
        beam_maps.append(np.real(bmap))
        beam_names.append(site["name"])

    data["beam_maps"] = np.stack(beam_maps)
    data["beam_names"] = np.array(beam_names)
    data["beam_freq"] = np.float64(freq)
    np.savez(npz_path, **data)
    logger.info(
        "Added %d beam maps at %.0f MHz to %s",
        len(beam_maps),
        freq,
        npz_path,
    )


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
    cells.append(_make_md_cell("## Cumulative Residual RMS vs ell"))
    cells.append(
        _make_code_cell(
            "cl_res = hp.alm2cl(x_rec - x_true)\n"
            "ell = np.arange(len(cl_res))\n"
            "weights = (2 * ell + 1) / (4 * np.pi)\n"
            "rms_cum = np.sqrt(\n"
            "    np.cumsum(weights * cl_res)\n"
            ")\n"
            "fig, ax = plt.subplots(\n"
            "    figsize=(10, 5),\n"
            "    constrained_layout=True,\n"
            ")\n"
            "ax.semilogy(ell[1:], rms_cum[1:])\n"
            "ax.set_xlabel(r'$\\ell_{\\rm max}$',\n"
            "    fontsize=14)\n"
            "ax.set_ylabel(\n"
            "    'Cumulative residual RMS [K]',\n"
            "    fontsize=14,\n"
            ")\n"
            "ax.set_title(\n"
            "    'RMS of (recovered - true) vs '\n"
            "    'included modes',\n"
            ")\n"
            "ax.grid(alpha=0.3)\n"
            "plt.show()"
        )
    )
    cells.append(_make_md_cell("## Fractional Residual Power vs ell"))
    cells.append(
        _make_code_cell(
            "cl_true = hp.alm2cl(x_true)\n"
            "cl_res = hp.alm2cl(x_rec - x_true)\n"
            "frac = np.zeros_like(cl_true)\n"
            "valid = cl_true > 0\n"
            "frac[valid] = (\n"
            "    cl_res[valid] / cl_true[valid]\n"
            ")\n"
            "ell = np.arange(len(frac))\n"
            "fig, ax = plt.subplots(\n"
            "    figsize=(10, 5),\n"
            "    constrained_layout=True,\n"
            ")\n"
            "ax.semilogy(ell[1:], frac[1:])\n"
            "ax.axhline(1, color='k', ls='--',\n"
            "    alpha=0.5)\n"
            "ax.set_xlabel(r'$\\ell$',\n"
            "    fontsize=14)\n"
            "ax.set_ylabel(\n"
            "    r'$C_\\ell^{\\rm res}'\n"
            "    r' / C_\\ell^{\\rm true}$',\n"
            "    fontsize=14,\n"
            ")\n"
            "ax.set_title(\n"
            "    'Fractional residual power '\n"
            "    'per multipole',\n"
            ")\n"
            "ax.grid(alpha=0.3)\n"
            "plt.show()"
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
