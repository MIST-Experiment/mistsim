"""Slide figures for the mistsim vs Raul recap (from cached runs).

The story, in slide order:

1. Last time: the February 3x3 figure, exactly as it was shown
   (extracted from git, commit 25d1967).
2. Today: the same layout with today's best like-for-like run, on
   February's colour range, then zoomed.
3. What it took: one cumulative chain of runs, rms per test.
4. The floor: what is left is the size of an independent
   re-implementation of Raul's code against Raul's own output.

Backups: each step of the chain on its own, and the February
dipole-orientation bug.

mistsim keeps its own beam azimuth throughout. Raul's spec maps the
beam's phi directly onto compass azimuth ("AZ and EL of antenna is the
same as AZ and EL of local coordinates", N -> E); mistsim keeps FEKO's
right-handed frame with z at zenith (N -> W). Matching his convention
(the normalisation study's "mirror" runs) moves the total rms by only
~0.02 K, which 40 MHz dominates, so the chain leaves it out. Above
~60 MHz it is the largest residual left (~0.15 K); slide 4 shows it.

Inputs (all cached, nothing is simulated here):

- ``results_versions/``: the version ladder on Raul's time grid
  (``versions/run_all.py``), plus Raul's waterfalls in ``inputs.npz``.
- ``results_normalization/cache/``: the dev4 variants
  (``normalization/run.py``) and the pixel replica.
- ``MARCH``: the March 2026 run (croissant 5.1.4, LSTs on 2022-07-17),
  from the main mistsim checkout.
"""

import base64
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "slides"
VERS = HERE / "results_versions"
CACHE = HERE / "results_normalization/cache"
MARCH = Path(
    "/home/christian/Documents/research/MIST/mistsim/notebooks/"
    "sim_comparisons/results/raul_cmp_5.1.4_az0.npz"
)
FEB_COMMIT = "25d1967"
FEB_NB = "notebooks/sim_comparisons/sim_comparison.ipynb"

C = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
MARK = ["o", "s", "^"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIV = "RdBu_r"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "image.interpolation": "none",
    }
)

TESTS = (0, 1, 2)
TITLE = {
    0: "Test 0: MARS, 10° blockage",
    1: "Test 1: MARS",
    2: "Test 2: North Pole",
}

z = np.load(VERS / "inputs.npz")
freqs = z["freqs"]
raul = {k: z[f"raul{k}"] for k in TESTS}
lsts = {k: z[f"lst{k}"] for k in TESTS}


def version(key):
    f = np.load(VERS / f"{key}.npz")
    return {k: f[f"test{k}"] for k in TESTS}


def variant(name):
    out = {}
    for k in TESTS:
        p = CACHE / f"ms_{name}_test{k}.npz"
        if p.exists():
            out[k] = np.load(p)["tant"]
    return out


march = np.load(MARCH, allow_pickle=True)
march = {k: march[f"test{k}"] for k in TESTS}
# The pixel replica of Raul's algorithm: "pix" with his phi = AZ,
# "pix_mirror" with mistsim's beam azimuth (AZ = -phi).
replica = {k: np.load(CACHE / f"pix_test{k}.npz") for k in TESTS}
pix = {k: replica[k]["pix"] for k in TESTS}
pix_ms = {k: replica[k]["pix_mirror"] for k in TESTS}


def rms(d, axis=None):
    return np.sqrt(np.mean(np.square(d), axis=axis))


def plain_log(ax):
    """Log y axis with 1-2-5 ticks as plain numbers."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def extent(k):
    return [freqs[0], freqs[-1], lsts[k][-1], lsts[k][0]]


def waterfalls(panels, fname, width=3.0):
    """Row of residual waterfalls on one symmetric colour scale."""
    lim = max(np.max(np.abs(d)) for _, d, _ in panels)
    fig, axs = plt.subplots(
        1,
        len(panels),
        figsize=(width * len(panels) + 1, 3.4),
        sharey=True,
        constrained_layout=True,
    )
    for ax, (title, d, k) in zip(axs, panels):
        im = ax.imshow(
            d, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto", extent=extent(k)
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Frequency [MHz]")
    axs[0].set_ylabel("LST [h]")
    fig.colorbar(im, ax=axs, label="mistsim − Raul [K]", shrink=0.9)
    fig.savefig(OUT / fname)
    plt.close(fig)


# The cumulative chain: each step adds one change to the one before.
# All but the first run on Raul's time grid. "kind" says what sort of
# change it is, for the slide.
dev4, edge = version("dev4"), variant("edge_unmirrored")
CHAIN = [
    ("March code,\nLSTs on 2022-07-17", "≈ Feb plot", march),
    ("+ Raul's\ntime grid", "setup", version("cro514")),
    ("+ croissant\nrotation fixes", "bugs: #147, #152", dev4),
    ("+ half-weight\nhorizon edge", "definition", {0: edge[0]}),
]
# Today's best like-for-like run: the end of the chain, per test.
today = {
    k: next(run[k] for *_, run in reversed(CHAIN) if k in run) for k in TESTS
}

# 1. Last time: the February figure, as shown -------------------------
nb = json.loads(
    subprocess.run(
        ["git", "show", f"{FEB_COMMIT}:{FEB_NB}"],
        cwd=HERE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
)
cell = next(
    c
    for c in nb["cells"]
    if c["cell_type"] == "code"
    and "".join(c["source"]).startswith("diff_kwargs")
)
png = next(
    o["data"]["image/png"]
    for o in cell["outputs"]
    if "image/png" in o.get("data", {})
)
(OUT / "slide1_last_time.png").write_bytes(base64.b64decode(png))

# 2. Today, in February's layout --------------------------------------
FEB_RANGE = 30.0  # the February residual colour bar ran -10 to +30 K


def three_by_three(resid_lim, fname, note):
    fig, axs = plt.subplots(
        3,
        3,
        figsize=(10, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for j, k in enumerate(TESTS):
        a = axs[0, j].imshow(
            raul[k], norm=LogNorm(400, 12000), aspect="auto", extent=extent(k)
        )
        axs[1, j].imshow(
            today[k], norm=LogNorm(400, 12000), aspect="auto", extent=extent(k)
        )
        r = axs[2, j].imshow(
            today[k] - raul[k],
            cmap=DIV,
            vmin=-resid_lim,
            vmax=resid_lim,
            aspect="auto",
            extent=extent(k),
        )
        axs[0, j].set_title(TITLE[k], fontsize=11)
        axs[2, j].set_xlabel("Frequency [MHz]")
        axs[2, j].text(
            0.97,
            0.03,
            f"rms {rms(today[k] - raul[k]):.2f} K",
            transform=axs[2, j].transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color=INK,
            bbox={"fc": "white", "ec": "none", "alpha": 0.8},
        )
    for i in range(3):
        axs[i, 0].set_ylabel("LST [h]")
    fig.colorbar(a, ax=axs[0, :], label="Raul [K]", shrink=0.9)
    fig.colorbar(a, ax=axs[1, :], label="mistsim today [K]", shrink=0.9)
    fig.colorbar(
        r, ax=axs[2, :], label=f"mistsim − Raul [K]\n({note})", shrink=0.9
    )
    fig.savefig(OUT / fname)
    plt.close(fig)


three_by_three(FEB_RANGE, "slide2_today.png", "February's range")
zoom = max(np.max(np.abs(today[k] - raul[k])) for k in TESTS)
three_by_three(zoom, "slide2b_today_zoom.png", "own range")

# 3. What it took: rms along the chain -------------------------------
fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
x = np.arange(len(CHAIN))
print(f"{'step':34s}" + "".join(f"  test{k} rms" for k in TESTS))
for c, m, k in zip(C, MARK, TESTS):
    xs = [i for i, (*_, run) in enumerate(CHAIN) if k in run]
    ys = [rms(CHAIN[i][2][k] - raul[k]) for i in xs]
    ax.plot(xs, ys, c=c, marker=m, ms=8, lw=1.5, label=TITLE[k])
    ax.annotate(
        f"{ys[-1]:.2f} K",
        (xs[-1], ys[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )
    floor = rms(pix[k] - raul[k])
    ax.axhline(floor, c=c, lw=1, ls=":")
for i, (lab, kind, run) in enumerate(CHAIN):
    print(
        f"{lab.replace(chr(10), ' '):34s}"
        + "".join(
            f"  {rms(run[k] - raul[k]):10.3f}" if k in run else "           -"
            for k in TESTS
        )
    )
print(
    f"{'pixel replica (floor)':34s}"
    + "".join(f"  {rms(pix[k] - raul[k]):10.3f}" for k in TESTS)
)
ax.set_yscale("log")
plain_log(ax)
ax.set_xticks(x, [f"{lab}\n({kind})" for lab, kind, _ in CHAIN], fontsize=9.5)
ax.set_xlim(-0.3, len(CHAIN) - 0.4)
ax.set_ylabel("rms(mistsim − Raul) [K]")
ax.grid(True, axis="y", which="both", c=GRID, lw=0.6)
h, labels = ax.get_legend_handles_labels()
h.append(plt.Line2D([], [], c=MUTED, lw=1, ls=":"))
labels.append("independent replica of Raul's code (floor)")
ax.legend(h, labels, fontsize=9, loc="upper right")
fig.savefig(OUT / "slide3_what_it_took.png")
plt.close(fig)

# 4. The floor: rms over LST vs frequency ----------------------------
fig, axs = plt.subplots(
    1, 3, figsize=(13, 4.3), sharey=True, layout="constrained"
)
# Above ~60 MHz mistsim tracks the replica laid out in mistsim's beam
# azimuth; the gap down to the replica in Raul's (phi = AZ) is the
# mirror between the two conventions, which grows with frequency as the
# beam's left-right asymmetry does.
for ax, k in zip(axs, TESTS):
    for c, ls, (lab, d) in zip(
        C,
        ["-", "--", "-"],
        [
            ("mistsim today − Raul", today[k] - raul[k]),
            ("replica, mistsim's φ layout − Raul", pix_ms[k] - raul[k]),
            ("replica, Raul's φ layout − Raul (floor)", pix[k] - raul[k]),
        ],
    ):
        ax.plot(freqs, rms(d, axis=0), c=c, ls=ls, label=lab)
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_title(TITLE[k], fontsize=11)
    ax.set_xlabel("Frequency [MHz]")
    ax.grid(True, which="major", c=GRID, lw=0.6)
axs[0].set_ylabel("rms over LST [K]")
fig.legend(
    *axs[0].get_legend_handles_labels(),
    loc="outside lower center",
    ncol=3,
    fontsize=10,
)
fig.savefig(OUT / "slide4_floor.png")
plt.close(fig)

# Backups: one step of the chain each ---------------------------------
waterfalls(
    [
        ("Test 1, LSTs on 2022-07-17", variant("base")[1] - raul[1], 1),
        ("Test 1, Raul's 2014 grid", dev4[1] - raul[1], 1),
        ("Test 2, LSTs on 2022-07-17", variant("base")[2] - raul[2], 2),
        ("Test 2, Raul's 2014 grid", dev4[2] - raul[2], 2),
    ],
    "backup_time_grid.png",
)

dev2 = version("dev2")
cro514 = version("cro514")
waterfalls(
    [
        ("Test 1, March code", cro514[1] - raul[1], 1),
        ("Test 1, + pole of date (#147)", dev2[1] - raul[1], 1),
        ("Test 1, + nearest rotation (#152)", dev4[1] - raul[1], 1),
    ],
    "backup_rotation_fixes_test1.png",
    width=3.4,
)
waterfalls(
    [
        ("Test 2, March code", cro514[2] - raul[2], 2),
        ("Test 2, + pole of date (#147)", dev2[2] - raul[2], 2),
        ("Test 2, + nearest rotation (#152)", dev4[2] - raul[2], 2),
    ],
    "backup_rotation_fixes_test2.png",
    width=3.4,
)

waterfalls(
    [
        ("θ ≤ 80° (edge ≈ 80.5°)", dev4[0] - raul[0], 0),
        ("80° row at half weight", edge[0] - raul[0], 0),
    ],
    "backup_horizon_edge.png",
    width=3.4,
)

# The mask on mistsim's 1° beam grid, in the style of eigsep_analysis
# M004 (016_mask): each cell is one (theta, phi) sample standing for
# ±0.5°, coloured by its weight; the line is the true 10° horizon. The
# mask does not depend on azimuth, so any 12° of it looks the same.
az_edges = np.arange(0.0, 13.0)
el_rows = np.arange(6.0, 15.0)  # theta = 84 ... 76 deg
el_edges = np.append(el_rows - 0.5, el_rows[-1] + 0.5)
weights = {
    "Boolean mask: θ ≤ 80°": np.where(el_rows >= 10, 1.0, 0.0),
    "Fractional: θ = 80° row at ½": np.where(
        el_rows > 10, 1.0, np.where(el_rows == 10, 0.5, 0.0)
    ),
}
WCMAP = LinearSegmentedColormap.from_list("w", ["#eeede8", "#1f4e9a"])
fig, axs = plt.subplots(
    1, 2, figsize=(8, 3.6), sharey=True, constrained_layout=True
)
for ax, (title, w) in zip(axs, weights.items()):
    im = ax.pcolormesh(
        az_edges,
        el_edges,
        np.repeat(w[:, None], len(az_edges) - 1, axis=1),
        cmap=WCMAP,
        vmin=0,
        vmax=1,
        edgecolors="white",
        lw=1.5,
    )
    ax.axhline(10, c=INK, lw=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Azimuth [°]")
    ax.set_xlim(az_edges[0], az_edges[-1])
    ax.set_ylim(el_edges[0], el_edges[-1])
axs[0].set_ylabel("Elevation [°]")
axs[1].annotate(
    "line: horizon at 10° elevation",
    (12, 10),
    xytext=(-6, -16),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=9,
    color=INK,
)
fig.colorbar(im, ax=axs, location="bottom", shrink=0.5, label="weight W")
fig.savefig(OUT / "backup_horizon_mask.png")
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
