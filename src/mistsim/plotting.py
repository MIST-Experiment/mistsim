import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator


def _nice_clim(vmin, vmax):
    """Round vmin down and vmax up to 1 significant figure.

    Uses the data span to choose the rounding precision, so colorbar
    ticks land on clean numbers and the endpoints are evenly spaced
    with the interior ticks.
    """
    span = vmax - vmin
    if span == 0:
        return vmin, vmax
    decade = 10 ** np.floor(np.log10(span))
    return (
        np.floor(vmin / decade) * decade,
        np.ceil(vmax / decade) * decade,
    )


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


def plot_pull_distribution(x_true, x_rec, std_alm, lmax):
    """
    Histogram of normalized residuals (pulls) for alm coefficients.

    Pull = (rec - true) / std for each real and imaginary component.
    Well-calibrated error bars produce pulls distributed as N(0, 1).

    Parameters
    ----------
    x_true : array-like
        True alm in healpy ordering.
    x_rec : array-like
        Recovered alm in healpy ordering.
    std_alm : array-like
        Complex error bars (real part = error on Re, imag = on Im).
    lmax : int
        Maximum ell of the alm arrays.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    pulls_real = []
    pulls_imag = []

    for ell in range(lmax + 1):
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            err_re = std_alm[idx].real
            if err_re > 0:
                pulls_real.append(
                    (x_rec[idx].real - x_true[idx].real) / err_re
                )
            if m > 0:
                err_im = std_alm[idx].imag
                if err_im > 0:
                    pulls_imag.append(
                        (x_rec[idx].imag - x_true[idx].imag) / err_im
                    )

    pulls_real = np.array(pulls_real)
    pulls_imag = np.array(pulls_imag)
    pulls_all = np.concatenate([pulls_real, pulls_imag])

    z = np.linspace(-5, 5, 200)
    gauss = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    for ax, data, title in zip(
        axes,
        [pulls_real, pulls_imag, pulls_all],
        ["Real part", "Imaginary part", "All"],
    ):
        ax.hist(
            data,
            bins="auto",
            density=True,
            alpha=0.7,
            color="C0",
            edgecolor="white",
        )
        ax.plot(z, gauss, "k-", lw=2, label=r"$\mathcal{N}(0,1)$")
        ax.set_xlabel("Pull", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        mu, sig = np.mean(data), np.std(data)
        ax.set_title(
            f"{title}\n"
            rf"$\mu={mu:.2f}$, $\sigma={sig:.2f}$, "
            f"$N={len(data)}$",
            fontsize=12,
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(-5, 5)

    return fig


def plot_alm_error_heatmap(x_true, std_alm, lmax, lmax_plot=None):
    """
    2D heatmap of posterior error and SNR in (ell, m) space.

    Left: posterior uncertainty |sigma_{ell m}| (log scale).
    Right: per-mode SNR |a^true_{ell m}| / |sigma_{ell m}|.
    The power spectrum C_ell averages over m — this plot
    reveals any m-dependent structure in constraining power.

    Parameters
    ----------
    x_true : array-like
        True alm in healpy ordering.
    std_alm : array-like
        Complex error bars (real = error on Re, imag on Im).
    lmax : int
        Maximum ell of the alm arrays.
    lmax_plot : int or None
        Maximum ell to show (defaults to lmax).

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    if lmax_plot is None:
        lmax_plot = lmax

    err = np.full((lmax_plot + 1, lmax_plot + 1), np.nan)
    snr = np.full((lmax_plot + 1, lmax_plot + 1), np.nan)

    for ell in range(lmax_plot + 1):
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            e = np.abs(std_alm[idx])
            err[m, ell] = e
            if e > 0:
                snr[m, ell] = np.abs(x_true[idx]) / e

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    # --- posterior error ---
    valid = err[np.isfinite(err) & (err > 0)]
    im1 = ax1.imshow(
        err,
        origin="lower",
        aspect="equal",
        norm=LogNorm(vmin=valid.min(), vmax=valid.max()),
        cmap="inferno",
        extent=(
            -0.5,
            lmax_plot + 0.5,
            -0.5,
            lmax_plot + 0.5,
        ),
    )
    fig.colorbar(
        im1,
        ax=ax1,
        shrink=0.8,
        label=r"$|\sigma_{\ell m}|$",
    )
    ax1.set_xlabel(r"$\ell$", fontsize=14)
    ax1.set_ylabel(r"$m$", fontsize=14)
    ax1.set_title("Posterior Error", fontsize=14)

    # --- SNR ---
    valid_s = snr[np.isfinite(snr) & (snr > 0)]
    vmin_s = max(valid_s.min(), 1e-2)
    im2 = ax2.imshow(
        snr,
        origin="lower",
        aspect="equal",
        norm=LogNorm(vmin=vmin_s, vmax=valid_s.max()),
        cmap="viridis",
        extent=(
            -0.5,
            lmax_plot + 0.5,
            -0.5,
            lmax_plot + 0.5,
        ),
    )
    fig.colorbar(
        im2,
        ax=ax2,
        shrink=0.8,
        label="SNR",
    )
    # contour at SNR = 1
    ell_arr = np.arange(lmax_plot + 1)
    snr_filled = np.where(np.isfinite(snr), snr, 0)
    ax2.contour(
        ell_arr,
        ell_arr,
        snr_filled,
        levels=[1.0],
        colors="red",
        linewidths=1.5,
        linestyles="--",
    )
    ax2.set_xlabel(r"$\ell$", fontsize=14)
    ax2.set_ylabel(r"$m$", fontsize=14)
    ax2.set_title(
        r"SNR $= |a_{\ell m}^{\rm true}|"
        r" / |\sigma_{\ell m}|$",
        fontsize=14,
    )

    return fig


def plot_ell_freq_residual(
    x_true, x_rec, lmax, freqs, lmax_plot=None, vmin=None, vmax=None,
    ):
    """
    Heatmap of per-ell fractional residual power vs frequency.

    Color encodes C_ell^res / C_ell^true at each (ell, freq).
    Values below 1 indicate constrained modes; the black
    contour at 1.0 marks the constraining frontier.

    Parameters
    ----------
    x_true : array-like, shape (nf, nalm)
        True alm at each frequency.
    x_rec : array-like, shape (nf, nalm)
        Recovered alm at each frequency.
    lmax : int
        Maximum ell.
    freqs : array-like, shape (nf,)
        Frequencies in MHz.
    lmax_plot : int or None
        Maximum ell to show (defaults to lmax).

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    nf = len(freqs)
    if lmax_plot is None:
        lmax_plot = lmax

    frac = np.full((nf, lmax_plot + 1), np.nan)
    for fi in range(nf):
        cl_t = hp.alm2cl(x_true[fi])[: lmax_plot + 1]
        cl_r = hp.alm2cl(x_rec[fi] - x_true[fi])[: lmax_plot + 1]
        ok = cl_t > 0
        frac[fi, ok] = cl_r[ok] / cl_t[ok]

    ell_arr = np.arange(lmax_plot + 1)

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )
    im = ax.pcolormesh(
        ell_arr,
        freqs,
        frac,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="RdYlGn_r",
        shading="nearest",
    )
    fig.colorbar(
        im,
        ax=ax,
        label=(
            r"$C_\ell^{\rm res}"
            r" / C_\ell^{\rm true}$"
        ),
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel("Frequency [MHz]", fontsize=14)
    ax.set_title(
        r"Per-$\ell$ Fractional Residual Power",
        fontsize=14,
    )

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
    res_range=None,
):
    """
    Mollweide projections of true, recovered, and residual maps.

    Parameters
    ----------
    ratio : bool
        If True, plot fractional residual (true - rec) / true instead
        of the absolute residual.
    res_range : float or None
        Symmetric colorbar range for the residual map. If None,
        the range is determined automatically from the data.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    fl = np.ones(lmax + 1)
    if plot_lmax is not None:
        fl[plot_lmax + 1 :] = 0.0
    else:
        plot_lmax = lmax
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
    vmin, vmax = _nice_clim(
        min(map_true.min(), map_rec.min()),
        max(map_true.max(), map_rec.max()),
    )
    if res_range is not None:
        res_max = res_range
    else:
        res_raw = np.max(np.abs(map_res))
        ticks = MaxNLocator(symmetric=True).tick_values(-res_raw, res_raw)
        res_max = ticks[-1]

    def _add_cbar(label="", loc=None):
        ax = plt.gca()
        image = ax.get_images()[0]
        cb = fig.colorbar(
            image,
            ax=ax,
            orientation="vertical",
            shrink=0.6,
            pad=0.05,
        )
        if loc is not None:
            cb.locator = loc
            cb.update_ticks()
        cb.set_label(label, fontsize=12)

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
    _add_cbar(res_label, loc=MaxNLocator(symmetric=True))
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
    orientation="horizontal",
):
    """
    Grid comparing multiple recovered maps against the true map.

    Two layouts controlled by *orientation*:

    ``"horizontal"`` (default)
        Top row: true map then each recovered map (shared colorscale).
        Bottom row: empty then residual for each (shared colorscale).
        Each recovered column gets its label as a title.

    ``"vertical"``
        Top row: true map centred across both columns.
        Subsequent rows: recovered map (left) and residual (right).
        Each row is labelled with a ylabel on the recovered map.

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
    orientation : str
        ``"horizontal"`` or ``"vertical"``.

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

    vmin, vmax = _nice_clim(
        min(map_true.min(), *(m.min() for m in maps_rec)),
        max(map_true.max(), *(m.max() for m in maps_rec)),
    )

    if ratio:
        res_raw = frac_range
        res_prefix = "Fractional Residual"
    else:
        res_raw = max(np.max(np.abs(m)) for m in maps_res)
        res_prefix = "Residual"
    ticks = MaxNLocator(symmetric=True).tick_values(-res_raw, res_raw)
    res_max = ticks[-1]

    if orientation == "vertical":
        fig = _comparison_grid_vertical(
            map_true,
            maps_rec,
            maps_res,
            labels,
            plot_lmax,
            coord,
            vmin,
            vmax,
            res_max,
            res_prefix,
            n,
        )
    else:
        fig = _comparison_grid_horizontal(
            map_true,
            maps_rec,
            maps_res,
            labels,
            plot_lmax,
            coord,
            vmin,
            vmax,
            res_max,
            res_prefix,
            n,
        )
    return fig


def _comparison_grid_horizontal(
    map_true,
    maps_rec,
    maps_res,
    labels,
    plot_lmax,
    coord,
    vmin,
    vmax,
    res_max,
    res_prefix,
    n,
):
    ncols = n + 1
    fig = plt.figure(figsize=(5 * ncols, 7))

    def _add_cbar(ax, loc=None):
        im = ax.get_images()[0]
        cb = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            shrink=0.6,
        )
        if loc is not None:
            cb.locator = loc
            cb.update_ticks()

    # Row 1: true + recovered maps
    hp.mollview(
        map_true,
        sub=(2, ncols, 1),
        title=f"True (l <= {plot_lmax})",
        cmap="viridis",
        min=vmin,
        max=vmax,
        coord=coord,
        cbar=False,
    )
    _add_cbar(plt.gca())
    for i, (mr, lab) in enumerate(zip(maps_rec, labels)):
        hp.mollview(
            mr,
            sub=(2, ncols, i + 2),
            title=lab,
            cmap="viridis",
            min=vmin,
            max=vmax,
            coord=coord,
            cbar=False,
        )
        _add_cbar(plt.gca())

    # Row 2: blank under true, then residuals
    sym = MaxNLocator(symmetric=True)
    for i, (mres, lab) in enumerate(zip(maps_res, labels)):
        hp.mollview(
            mres,
            sub=(2, ncols, ncols + i + 2),
            unit=res_prefix,
            cmap="coolwarm",
            min=-res_max,
            max=res_max,
            coord=coord,
            cbar=False,
        )
        _add_cbar(plt.gca(), loc=sym)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig


def _comparison_grid_vertical(
    map_true,
    maps_rec,
    maps_res,
    labels,
    plot_lmax,
    coord,
    vmin,
    vmax,
    res_max,
    res_prefix,
    n,
):
    nrows = n + 1
    fig = plt.figure(figsize=(10, 3.5 * nrows))

    def _add_cbar(ax, loc=None):
        im = ax.get_images()[0]
        cb = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            shrink=0.6,
        )
        if loc is not None:
            cb.locator = loc
            cb.update_ticks()

    # Row 0: true map centred (use left column, leave right blank)
    hp.mollview(
        map_true,
        sub=(nrows, 2, 1),
        title=f"True (l <= {plot_lmax})",
        cmap="viridis",
        min=vmin,
        max=vmax,
        coord=coord,
        cbar=False,
    )
    _add_cbar(plt.gca())

    # Rows 1..n: recovered (left) + residual (right)
    sym = MaxNLocator(symmetric=True)
    for i, (mr, mres, lab) in enumerate(zip(maps_rec, maps_res, labels)):
        row = i + 1
        # Recovered map (left column)
        hp.mollview(
            mr,
            sub=(nrows, 2, 2 * row + 1),
            title=lab,
            unit="Recovered",
            cmap="viridis",
            min=vmin,
            max=vmax,
            coord=coord,
            cbar=False,
        )
        _add_cbar(plt.gca())

        # Residual map (right column)
        hp.mollview(
            mres,
            sub=(nrows, 2, 2 * row + 2),
            title=lab,
            unit=res_prefix,
            cmap="coolwarm",
            min=-res_max,
            max=res_max,
            coord=coord,
            cbar=False,
        )
        _add_cbar(plt.gca(), loc=sym)

    fig.subplots_adjust(hspace=0.3, wspace=0.05)
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


def plot_beam_patterns(
    beam_maps,
    beam_names,
    nrows,
    ncols,
    freq=None,
    coord=None,
):
    """Plot beam patterns on a grid of Mollweide projections.

    Parameters
    ----------
    beam_maps : array-like, shape (n_beams, npix)
        HEALPix beam maps in equatorial coordinates.
    beam_names : list of str
        Label for each beam.
    nrows, ncols : int
        Grid layout.
    freq : float or None
        Frequency in MHz (used in the suptitle).
    coord : list of str or None
        Coordinate rotation for ``hp.projview``,
        e.g. ``["C", "G"]`` for equatorial to galactic.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    n = len(beam_maps)
    fig = plt.figure(
        figsize=(7 * ncols, 4.5 * nrows),
    )
    for i in range(n):
        kw = {}
        if coord is not None:
            kw["coord"] = coord
        hp.projview(
            beam_maps[i],
            title=beam_names[i],
            projection_type="mollweide",
            sub=(nrows, ncols, i + 1),
            **kw,
        )
    if freq is not None:
        fig.suptitle(
            f"Beam Patterns at {freq:.0f} MHz",
            fontsize=16,
            y=1.01,
        )
    plt.tight_layout()
    return fig
