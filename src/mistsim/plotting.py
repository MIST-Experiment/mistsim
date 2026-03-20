import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def plot_alm_comparison(x_true, x_rec, std_alm, lmax, lmax_plot=5):
    """
    Plot true vs recovered alm coefficients with error bars.

    Parameters
    ----------
    x_true : array-like
        True alm in healpy ordering.
    x_rec : array-like
        Recovered alm in healpy ordering.
    std_alm : array-like
        Complex error bars (real part = error on Re, imag = on Im).
    lmax : int
        The lmax of the full alm arrays.
    lmax_plot : int
        Maximum ell to include in the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    labels = []
    true_real, rec_real, err_real = [], [], []
    true_imag, rec_imag, err_imag = [], [], []

    for ell in range(lmax_plot + 1):
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            labels.append(f"({ell},{m})")
            true_real.append(x_true[idx].real)
            rec_real.append(x_rec[idx].real)
            err_real.append(std_alm[idx].real)
            if m > 0:
                true_imag.append(x_true[idx].imag)
                rec_imag.append(x_rec[idx].imag)
                err_imag.append(std_alm[idx].imag)
            else:
                true_imag.append(0.0)
                rec_imag.append(0.0)
                err_imag.append(0.0)

    x_indices = np.arange(len(labels))
    true_real = np.array(true_real)
    rec_real = np.array(rec_real)
    err_real = np.array(err_real)
    true_imag = np.array(true_imag)
    rec_imag = np.array(rec_imag)
    err_imag = np.array(err_imag)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.plot(
        x_indices,
        true_real,
        "ks",
        markersize=6,
        label="True (Real)",
    )
    ax1.errorbar(
        x_indices,
        rec_real,
        yerr=err_real,
        fmt="ro",
        markersize=5,
        capsize=3,
        label="Recovered (Real)",
    )
    ax1.set_ylabel(r"$\mathrm{Re}(a_{\ell m})$", fontsize=14)
    ax1.set_title(
        r"Spherical Harmonic Coefficients True vs Recovered "
        r"($\ell_{\mathrm{max}} = %d$)" % lmax_plot,
        fontsize=16,
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.plot(
        x_indices,
        true_imag,
        "ks",
        markersize=6,
        label="True (Imag)",
    )
    ax2.errorbar(
        x_indices,
        rec_imag,
        yerr=err_imag,
        fmt="bo",
        markersize=5,
        capsize=3,
        label="Recovered (Imag)",
    )
    ax2.set_ylabel(r"$\mathrm{Im}(a_{\ell m})$", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=10)
    ax2.set_xlabel(r"$(\ell, m)$ indices", fontsize=14)

    l_transitions = [i for i, lab in enumerate(labels) if lab.endswith(",0)")]
    for t in l_transitions:
        ax1.axvline(t - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax2.axvline(t - 0.5, color="gray", linestyle=":", alpha=0.5)

    return fig


def plot_power_spectra(x_true, x_rec, std_alm):
    """
    Plot angular power spectra for true, recovered, cross, and noise.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    cl_true = hp.alm2cl(x_true)
    cl_rec = hp.alm2cl(x_rec)
    nl_err = hp.alm2cl(std_alm)
    cl_cross = hp.alm2cl(x_true, x_rec)
    l_arr = np.arange(len(cl_true))

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )
    ax.plot(
        l_arr,
        cl_true,
        "k-",
        linewidth=2.5,
        label=r"True Signal ($C_\ell^{\mathrm{true}}$)",
    )
    ax.plot(
        l_arr,
        cl_rec,
        "b-",
        linewidth=2,
        label=r"Recovered Signal ($C_\ell^{\mathrm{rec}}$)",
    )
    ax.plot(
        l_arr,
        cl_cross,
        "g-.",
        linewidth=2,
        label=r"Cross-Spectrum (True $\times$ Rec)",
    )
    ax.plot(
        l_arr,
        nl_err,
        "r--",
        linewidth=2,
        label=r"Error Variance ($N_\ell$)",
    )
    ax.set_yscale("log")
    if len(l_arr) > 100:
        ax.set_xscale("log")
    ax.set_xlabel(r"Multipole $\ell$", fontsize=14)
    ax.set_ylabel(r"Power Spectrum $C_\ell$", fontsize=14)
    ax.set_title(
        "Angular Power Spectra: True vs. Wiener Filtered Recovery",
        fontsize=16,
    )
    ax.legend(fontsize=12, loc="lower left")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.set_xlim(left=1 if l_arr[-1] > 10 else 0)
    return fig


def plot_transfer_function(x_true, x_rec):
    """
    Plot empirical Wiener filter transfer function.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    cl_true = hp.alm2cl(x_true)
    cl_cross = hp.alm2cl(x_true, x_rec)
    valid = cl_true > 0
    transfer = np.zeros_like(cl_true)
    transfer[valid] = cl_cross[valid] / cl_true[valid]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(transfer)), transfer, "m-", lw=2)
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$C_\ell^{\mathrm{cross}} / C_\ell^{\mathrm{true}}$")
    ax.set_title("Wiener Filter Empirical Transfer Function")
    ax.grid(alpha=0.3)
    return fig


def plot_maps_and_residuals(
    x_true,
    x_rec,
    lmax,
    plot_lmax,
    nside=128,
    plot_galactic=False,
    ratio=False,
):
    """
    Mollweide projections of true, recovered, and residual maps.

    Parameters
    ----------
    ratio : bool
        If True, plot fractional residual (true - rec) / true instead
        of the absolute residual.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    fl = np.ones(lmax + 1)
    fl[plot_lmax + 1 :] = 0.0
    x_true_low = hp.almxfl(x_true, fl)
    x_rec_low = hp.almxfl(x_rec, fl)
    map_true = hp.alm2map(x_true_low, nside=nside)
    map_rec = hp.alm2map(x_rec_low, nside=nside)

    if ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            map_res = np.where(
                map_true != 0,
                (map_true - map_rec) / map_true,
                0.0,
            )
        res_title = "Fractional Residual (True - Rec) / True"
        res_label = "Fractional Residual"
    else:
        map_res = map_true - map_rec
        res_title = "Residual (True - Rec)"
        res_label = "Residual Error"

    if plot_galactic:
        coord = ["C", "G"]
    else:
        coord = ["C", "C"]

    fig = plt.figure(figsize=(8, 14))
    vmin = min(map_true.min(), map_rec.min())
    vmax = max(map_true.max(), map_rec.max())
    res_max = np.max(np.abs(map_res))

    def _add_cbar(label=""):
        ax = plt.gca()
        image = ax.get_images()[0]
        fig.colorbar(
            image,
            ax=ax,
            orientation="vertical",
            shrink=0.6,
            pad=0.05,
        ).set_label(label, fontsize=12)

    hp.mollview(
        map_true,
        sub=(3, 1, 1),
        cbar=False,
        title=f"True Map (l <= {plot_lmax})",
        cmap="viridis",
        min=vmin,
        max=vmax,
        coord=coord,
    )
    _add_cbar("Signal Amplitude")
    hp.mollview(
        map_rec,
        sub=(3, 1, 2),
        cbar=False,
        title=f"Recovered Map (l <= {plot_lmax})",
        cmap="viridis",
        min=vmin,
        max=vmax,
        coord=coord,
    )
    _add_cbar("Signal Amplitude")
    hp.mollview(
        map_res,
        sub=(3, 1, 3),
        cbar=False,
        title=res_title,
        cmap="coolwarm",
        min=-res_max,
        max=res_max,
        coord=coord,
    )
    _add_cbar(res_label)
    return fig


def plot_comparison_grid(
    x_true,
    x_recs,
    labels,
    lmax,
    plot_lmax,
    nside=128,
    plot_galactic=False,
    frac_range=1.0,
    ratio=True,
):
    """
    Grid comparing multiple recovered maps against the true map.

    Top row: true map then each recovered map (shared colorscale).
    Bottom row: empty then residual for each (shared colorscale).

    Parameters
    ----------
    x_true : array-like
        True alm in healpy ordering.
    x_recs : list of array-like
        Recovered alms, one per run.
    labels : list of str
        Short label for each run.
    lmax : int
        Maximum ell of the alm arrays.
    plot_lmax : int
        Maximum ell to keep in the low-pass filter.
    nside : int
        HEALPix nside for map synthesis.
    plot_galactic : bool
        If True, rotate from equatorial to galactic.
    frac_range : float
        Symmetric range for residual colorbar. For ratio=True
        this is the fractional range; for ratio=False it is
        auto-scaled and this parameter is ignored.
    ratio : bool
        If True, plot fractional residual (true - rec) / true.
        If False, plot absolute residual (true - rec).

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    n = len(x_recs)
    coord = ["C", "G"] if plot_galactic else ["C", "C"]

    fl = np.ones(lmax + 1)
    fl[plot_lmax + 1 :] = 0.0
    map_true = hp.alm2map(hp.almxfl(x_true, fl), nside=nside)

    maps_rec = []
    maps_res = []
    for xr in x_recs:
        mr = hp.alm2map(hp.almxfl(xr, fl), nside=nside)
        maps_rec.append(mr)
        if ratio:
            with np.errstate(divide="ignore", invalid="ignore"):
                res = np.where(
                    map_true != 0,
                    (map_true - mr) / map_true,
                    0.0,
                )
        else:
            res = map_true - mr
        maps_res.append(res)

    vmin = min(map_true.min(), *(m.min() for m in maps_rec))
    vmax = max(map_true.max(), *(m.max() for m in maps_rec))

    if ratio:
        res_max = frac_range
        res_prefix = "Frac. Resid."
    else:
        res_max = max(np.max(np.abs(m)) for m in maps_res)
        res_prefix = "Residual"

    ncols = n + 1
    fig = plt.figure(figsize=(5 * ncols, 7))

    # Row 1: true + recovered maps
    hp.mollview(
        map_true,
        sub=(2, ncols, 1),
        title=f"True (l <= {plot_lmax})",
        cmap="viridis",
        min=vmin,
        max=vmax,
        coord=coord,
        cbar=True,
    )
    for i, (mr, lab) in enumerate(zip(maps_rec, labels)):
        hp.mollview(
            mr,
            sub=(2, ncols, i + 2),
            title=lab,
            cmap="viridis",
            min=vmin,
            max=vmax,
            coord=coord,
            cbar=True,
        )

    # Row 2: blank under true, then residuals
    for i, (mres, lab) in enumerate(zip(maps_res, labels)):
        hp.mollview(
            mres,
            sub=(2, ncols, ncols + i + 2),
            title=f"{res_prefix}: {lab}",
            cmap="coolwarm",
            min=-res_max,
            max=res_max,
            coord=coord,
            cbar=True,
        )

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig


def plot_singular_values(Sigma, nvec=None):
    """
    Plot singular value spectrum on log scale.

    Parameters
    ----------
    Sigma : array-like
        Singular values in descending order.
    nvec : int or None
        If given, mark the truncation point.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(Sigma, "k-", lw=1.5)
    if nvec is not None:
        ax.axvline(
            nvec,
            color="r",
            ls="--",
            lw=1.5,
            label=f"nvec = {nvec}",
        )
        ax.legend()
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("Singular Value Spectrum")
    ax.grid(alpha=0.3)
    return fig


def plot_filter_factors(D):
    """
    Plot the Wiener filter factors D_i = sigma_i / (1 + sigma_i^2).

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(D, "b-", lw=1.5)
    ax.set_xlabel("Index")
    ax.set_ylabel("Filter Factor")
    ax.set_title("Wiener Filter Factors")
    ax.grid(alpha=0.3)
    return fig


def plot_posterior_maps(std_map, sigma2_prior, best_map, nside):
    """
    Plot posterior standard deviation, variance ratio, and SNR maps.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    fig = plt.figure(figsize=(8, 14))
    hp.mollview(
        std_map,
        sub=(3, 1, 1),
        cbar=True,
        title="Posterior Std Dev",
        cmap="inferno",
    )
    var_ratio = std_map**2 / sigma2_prior
    hp.mollview(
        var_ratio,
        sub=(3, 1, 2),
        cbar=True,
        title=r"$\sigma^2_{\rm post} / \sigma^2_{\rm prior}$",
        cmap="viridis",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(std_map > 0, np.abs(best_map) / std_map, 0)
    hp.mollview(
        snr,
        sub=(3, 1, 3),
        cbar=True,
        title="SNR = |map| / std",
        cmap="magma",
    )
    return fig
