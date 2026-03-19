"""Pipeline orchestration for MIST mapmaking runs."""

import logging
from pathlib import Path

import astropy.units as u
import croissant as cro
import healpy as hp
import nbformat
import numpy as np
import scipy.sparse.linalg as sla
import yaml
from astropy.time import Time
from nbconvert.preprocessors import ExecutePreprocessor

import mistsim as ms
from mistsim import mapmaking

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


def load_config(path):
    """Load and return a YAML config dict.

    The config lists beam names defined in ``beams.yaml`` (must live
    in the same directory)::

        beams: [mars-csa2022-dip, nv-dip]
        svd:
          n_singular_values: 800

    Defaults for sky, observation, posterior, and output come from
    ``beams.yaml`` and can be overridden per-run.  Relative paths
    are resolved against the config file's parent directory.
    """
    path = Path(path)
    config_dir = path.resolve().parent
    with open(path) as f:
        cfg = yaml.safe_load(f)

    cfg = _expand_beam_config(cfg, config_dir)
    _resolve_config_paths(cfg, config_dir)
    return cfg


def _expand_beam_config(cfg, config_dir):
    """Expand compact beam-list config into full sites format."""
    registry_path = config_dir / "beams.yaml"
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

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

    # output.results_dir
    out = cfg.get("output", {})
    if "results_dir" in out:
        p = Path(out["results_dir"])
        if not p.is_absolute():
            out["results_dir"] = str(config_dir / p)


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


def setup_sky(config):
    """
    Load Haslam map, scale frequencies, build Sky object.

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

    freq_ix = sky_cfg.get("freq_index", 0)
    sim_freq = freqs[freq_ix]
    sky_map = haslam[freq_ix]

    sky = ms.Sky(
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
        return ms.Beam(
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
    return ms.Beam(
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


def build_stacked_A(config, sky, times_jd, sim_freq):
    """
    Build and stack A-matrices for all sites.

    Returns
    -------
    A : scipy.sparse.linalg.LinearOperator

    """
    obs_cfg = config["observation"]
    lmax = obs_cfg["lmax"]
    freq_ix = config["sky"].get("freq_index", 0)
    freqs_arr = np.arange(*config["sky"]["freq_range"])

    A_list = []
    for site in config["sites"]:
        beam = build_beam(site, sim_freq, freqs_arr, freq_ix)
        sim = ms.Simulator(
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
# Top-level orchestrator
# ------------------------------------------------------------------


def run_pipeline(config):
    """
    Run the full mapmaking pipeline from a config dict.

    Returns
    -------
    results : dict

    """
    # Sky
    logger.info("Setting up sky")
    sky, x_packed, x_hp, freqs, sim_freq = setup_sky(config)

    # Times
    obs = config["observation"]
    tstart = Time(obs["start_time"])
    tend = tstart + obs["n_sidereal_days"] * u.sday
    times = cro.utils.time_array(
        t_start=tstart,
        t_end=tend,
        N_times=obs["n_times"],
    )
    lmax = obs["lmax"]

    # Build stacked A
    logger.info("Building A-matrices")
    A = build_stacked_A(config, sky, times.jd, sim_freq)

    # Forward model
    logger.info("Running forward model")
    y = A @ x_packed

    # Noise
    df = (freqs[1] - freqs[0]) * 1e6  # Hz
    dt = (times[1] - times[0]).to_value(u.s)
    post_cfg = config.get("posterior", {})
    noise_seed = post_cfg.get("seed", 1420)
    Ndiag, noise = compute_noise(y, df, dt, seed=noise_seed)

    # Prior
    Sdiag = compute_prior(x_packed, lmax)

    # Whitened operator + SVD
    logger.info("Building Atilde and running SVD")
    Atilde = mapmaking.make_Atilde(Ndiag, A, Sdiag)
    svd_cfg = config.get("svd", {})
    k = svd_cfg.get("n_singular_values", 1200)
    U, Sigma, Vh = run_svd(Atilde, k=k)

    # Select modes
    threshold = svd_cfg.get("threshold", 1e-10)
    nvec = select_nvec(Sigma, threshold=threshold)
    logger.info("Selected nvec = %d (threshold = %e)", nvec, threshold)

    # Wiener filter
    logger.info("Applying Wiener filter")
    x_rec_hp, D = wiener_filter(
        U,
        Sigma,
        Vh,
        nvec,
        Ndiag,
        Sdiag,
        y,
        noise,
    )

    # Posterior uncertainty
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

    # Compute best map for diagnostics
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


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


def save_results(results, path):
    """Save results dict to npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        freqs=results["freqs"],
        lmax=results["lmax"],
        x_true=results["x_true"],
        x_rec=results["x_rec"],
        cl_prior=results["cl_prior"],
        sigma2_prior=results["sigma2_prior"],
        std_alm=results["std_alm"],
        std_map=results["std_map"],
        Sigma=results["Sigma"],
        nvec=results["nvec"],
        config_yaml=yaml.dump(results["config"]),
    )
    logger.info("Saved results to %s", path)


# ------------------------------------------------------------------
# Notebook generation
# ------------------------------------------------------------------


def _make_code_cell(source):
    return nbformat.v4.new_code_cell(source=source)


def _make_md_cell(source):
    return nbformat.v4.new_markdown_cell(source=source)


def generate_notebook(results, npz_path, output_path):
    """
    Generate and execute a diagnostic notebook.

    Parameters
    ----------
    results : dict
        Pipeline results.
    npz_path : str or Path
        Path to the saved npz file (used in notebook cells).
    output_path : str or Path
        Where to save the notebook.

    """
    npz_path = Path(npz_path).resolve()
    output_path = Path(output_path).resolve()
    nb = nbformat.v4.new_notebook()
    cells = []

    # Setup
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

    # Config summary
    cfg = results["config"]
    site_names = ", ".join(s["name"] for s in cfg["sites"])
    cells.append(
        _make_md_cell(
            f"# Diagnostics: {results['run_name']}\n\n"
            f"**Sites:** {site_names}  \n"
            f"**lmax:** {cfg['observation']['lmax']}  \n"
            f"**nvec:** {results['nvec']}  \n"
            f"**SVD k:** {cfg.get('svd', {}).get('n_singular_values', 'N/A')}"
        )
    )

    # Singular values
    cells.append(_make_md_cell("## Singular Value Spectrum"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_singular_values(Sigma, nvec=nvec)\nplt.show()"
        )
    )

    # Filter factors
    cells.append(_make_md_cell("## Filter Factors"))
    cells.append(
        _make_code_cell(
            "Dnum = Sigma[:nvec]\n"
            "D = Dnum / (1 + Dnum**2)\n"
            "fig = msplt.plot_filter_factors(D)\n"
            "plt.show()"
        )
    )

    # Alm comparison
    cells.append(_make_md_cell("## Alm Comparison"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_alm_comparison(\n"
            "    x_true, x_rec, std_alm, lmax, lmax_plot=8\n"
            ")\n"
            "plt.show()"
        )
    )

    # Power spectra
    cells.append(_make_md_cell("## Power Spectra"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_power_spectra(x_true, x_rec, std_alm)\n"
            "plt.show()"
        )
    )

    # Transfer function
    cells.append(_make_md_cell("## Transfer Function"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_transfer_function(x_true, x_rec)\nplt.show()"
        )
    )

    # Maps equatorial
    cells.append(_make_md_cell("## Maps & Residuals (Equatorial)"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_maps_and_residuals(\n"
            "    x_true, x_rec, lmax, plot_lmax=10,\n"
            "    nside=128, plot_galactic=False\n"
            ")\n"
            "plt.show()"
        )
    )

    # Maps galactic
    cells.append(_make_md_cell("## Maps & Residuals (Galactic)"))
    cells.append(
        _make_code_cell(
            "fig = msplt.plot_maps_and_residuals(\n"
            "    x_true, x_rec, lmax, plot_lmax=10,\n"
            "    nside=128, plot_galactic=True\n"
            ")\n"
            "plt.show()"
        )
    )

    # Posterior maps
    cells.append(_make_md_cell("## Posterior Uncertainty"))
    cells.append(
        _make_code_cell(
            "fl = np.ones(lmax + 1)\n"
            "fl[11:] = 0.0\n"
            "best_map = hp.alm2map(\n"
            "    hp.almxfl(x_rec, fl), nside=128\n"
            ")\n"
            "fig = msplt.plot_posterior_maps(\n"
            "    std_map, sigma2_prior, best_map, nside=128\n"
            ")\n"
            "plt.show()"
        )
    )

    nb.cells = cells

    # Execute
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(output_path.parent)}})
    except Exception:
        logger.warning("Notebook execution failed; saving unexecuted notebook")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        nbformat.write(nb, f)
    logger.info("Saved notebook to %s", output_path)
