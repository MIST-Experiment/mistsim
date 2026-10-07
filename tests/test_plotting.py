import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mistsim.plotting import plot_comparison_grid

plt.switch_backend("agg")


def _titles(fig):
    return [ax.get_title() for ax in fig.axes]


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_comparison_grid_true_label(orientation):
    lmax = 3
    rng = np.random.default_rng(0)
    alm = hp.map2alm(10 + rng.normal(size=hp.nside2npix(4)), lmax=lmax)
    kw = dict(
        lmax=lmax, plot_lmax=2, nside=4, ratio=False, orientation=orientation
    )

    fig = plot_comparison_grid(alm, [0.9 * alm], ["rec"], **kw)
    assert "True (l <= 2)" in _titles(fig)

    fig = plot_comparison_grid(
        alm, [0.9 * alm], ["rec"], true_label="True, row space", **kw
    )
    assert "True, row space" in _titles(fig)
    assert "True (l <= 2)" not in _titles(fig)
    plt.close("all")


def _spiky_alms(lmax=8, nside=8):
    """A smooth true sky, and an estimate off by one bright pixel."""
    npix = hp.nside2npix(nside)
    m_true = 10 + np.cos(hp.pix2ang(nside, np.arange(npix))[0])
    m_rec = m_true.copy()
    m_rec[0] += 1e3
    return hp.map2alm(m_true, lmax=lmax), hp.map2alm(m_rec, lmax=lmax)


def _residual_limit(fig):
    """Upper colour limit of the first residual panel (the untitled maps)."""
    for ax in fig.axes:
        if ax.get_title() == "" and ax.get_images():
            return ax.get_images()[0].get_clim()[1]
    raise AssertionError("no residual panel")


def test_comparison_grid_res_percentile():
    x_true, x_rec = _spiky_alms()
    kw = dict(lmax=8, plot_lmax=8, nside=8, ratio=False)
    full = _residual_limit(plot_comparison_grid(x_true, [x_rec], ["r"], **kw))
    cut = _residual_limit(
        plot_comparison_grid(x_true, [x_rec], ["r"], res_percentile=50, **kw)
    )
    assert cut < 0.5 * full
    plt.close("all")


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_comparison_grid_removed_panel(orientation):
    x_true, x_rec = _spiky_alms()
    kw = dict(lmax=8, plot_lmax=8, nside=8, ratio=False)
    kw["orientation"] = orientation
    fig = plot_comparison_grid(x_true, [x_rec], ["r"], **kw)
    n_axes = len(fig.axes)
    fig = plot_comparison_grid(
        x_true,
        [x_rec],
        ["r"],
        x_removed=x_rec - x_true,
        removed_label="True - shown",
        **kw,
    )
    assert "True - shown" in _titles(fig)
    assert len(fig.axes) == n_axes + 2  # the map and its colour bar
    # every colour bar's tick locator is bound to its own axis: a shared
    # locator draws one bar's ticks on another when their ranges differ,
    # as the removed panel's and the residuals' do
    for ax in fig.axes:
        for im in ax.get_images():
            cb = im.colorbar
            if cb is not None:
                assert cb.ax.xaxis.get_major_locator().axis is cb.ax.xaxis
    plt.close("all")
