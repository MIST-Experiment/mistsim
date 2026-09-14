# Consolidated Raul Comparison Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One notebook that puts Raul's reference simulator, mistsim under `main`, and mistsim under PR 12 on the same axes, with plots and a durable paper trail.

**Architecture:** A single notebook is run twice under two Jupyter kernels — one backed by a git worktree of `main` (croissant 5.1.4), one by the current branch (croissant v5.3.0.dev2). It detects its own croissant revision, caches its three simulated waterfalls to a tagged npz, and its comparison section globs every tagged npz present and renders whatever it finds. The second execution is therefore the complete artifact.

**Tech Stack:** Python 3.12, JAX (x64), croissant, mistsim, h5py, healpy, astropy, matplotlib, Jupyter/ipykernel, uv.

**Spec:** `docs/superpowers/specs/2026-09-14-raul-comparison-notebook-design.md`

## Global Constraints

- Ruff line length 79; rules E, F, W, I. Notebook cells are not linted, but keep lines within 79 for readability.
- Never use `noqa` to silence lint; fix the underlying issue.
- The notebook must not write to `data/`. `WRITE_SHARED_INPUTS = False` gates every such write; `data/beam.npz` is the `beam_file` for `mars-lake-dip` in `notebooks/mapmaking/configs/runs.yaml`.
- `BEAM_AZ_ROT = 0`. Since PR 8, mistsim's `beam_az_rot` is the astronomical azimuth of the beam X-axis (0 = North), converted internally via `beam_rot = beam_az_rot - 90` (`src/mistsim/beam.py:81`). The FEKO beam has phi=0 = North. Passing 90 produces a ~130 K disagreement and is the single most likely source of a spurious "regression".
- Curves are labelled by **environment** (`main` / `PR 12`), never by feature. The two environments differ in croissant, s2fft and jax together.
- `jax.config.update("jax_enable_x64", True)` must run before any croissant import work.
- Notebook path: `notebooks/sim_comparisons/raul_comparison.ipynb`. Cache path: `notebooks/sim_comparisons/results/raul_cmp_<TAG>.npz`.

---

### Task 1: The `main` worktree and its Jupyter kernel

**Files:**
- Create: `../mistsim-main/` (git worktree, outside the repo tree — not committed)
- Create: kernelspec `mistsim-cro514` in `~/.local/share/jupyter/kernels/`

**Interfaces:**
- Consumes: nothing.
- Produces: a Jupyter kernel named `mistsim-cro514`, display name `mistsim (croissant 5.1.4)`, whose interpreter imports croissant 5.1.4 and a mistsim identical to `main`.

- [ ] **Step 1: Create the worktree and sync main's locked environment**

```bash
cd /home/christian/Documents/research/MIST/mistsim
git worktree add ../mistsim-main main
cd ../mistsim-main && uv sync --all-extras --dev
```

- [ ] **Step 2: Verify the worktree resolved croissant 5.1.4, not the git pin**

```bash
cd /home/christian/Documents/research/MIST/mistsim/../mistsim-main
uv run python -c "
import croissant as cro, jax, importlib.metadata as md
print('croissant', md.version('croissant-sim'))
print('jax', jax.__version__)
print('has eq2cirs:', hasattr(cro.rotations, 'eq2cirs'))
"
```

Expected: `croissant 5.1.4`, a jax below 0.6, and `has eq2cirs: False`. If `eq2cirs` is True the worktree picked up the git pin and the A/B is meaningless — stop and investigate before continuing.

- [ ] **Step 3: Register the kernel**

```bash
cd /home/christian/Documents/research/MIST/mistsim/../mistsim-main
uv run python -m ipykernel install --user --name mistsim-cro514 \
    --display-name "mistsim (croissant 5.1.4)"
```

- [ ] **Step 4: Verify both kernels are visible**

```bash
cd /home/christian/Documents/research/MIST/mistsim
uv run jupyter kernelspec list
```

Expected: both `mistsim` (or `python3`) and `mistsim-cro514` are listed.

- [ ] **Step 5: No commit**

The worktree and kernelspec live outside the repository. Nothing to commit for this task.

---

### Task 2: Notebook setup, tagging, and Raul's reference

**Files:**
- Create: `notebooks/sim_comparisons/raul_comparison.ipynb`
- Create: `notebooks/sim_comparisons/results/.gitignore` (contents: `*.npz`)

**Interfaces:**
- Consumes: Task 1's kernels.
- Produces: in-notebook names `REPO`, `DATA_DIR`, `OUT_DIR`, `TAG`, `LABEL`, `BEAM_AZ_ROT`, `WRITE_SHARED_INPUTS`, `NSIDE` (int, 128), `SIM_LMAX` (int, 100), `lsts` (dict `int -> np.ndarray`), `freqs` (`np.ndarray`, shape `(86,)`), `raul` (dict `int -> np.ndarray` of shape `(241, 86)`), `TESTS` (dict `int -> str`).

- [ ] **Step 1: Create the results directory and ignore its contents**

```bash
cd /home/christian/Documents/research/MIST/mistsim
mkdir -p notebooks/sim_comparisons/results
printf '*.npz\n' > notebooks/sim_comparisons/results/.gitignore
```

Cached waterfalls are regenerable and large; the notebook is the artifact, not the cache.

- [ ] **Step 2: Write the title markdown cell**

```markdown
# Raul vs mistsim: `main` versus PR 12

Compares Raul's reference simulator against mistsim run under two
environments: `main` (croissant 5.1.4) and this branch (croissant
v5.3.0.dev2, PR 12).

**Run this notebook twice** — once under the `mistsim (croissant
5.1.4)` kernel, once under the project kernel. It tags its own output
by the croissant revision it detects, and the comparison section
renders whichever tagged results exist. The second run is the complete
artifact.

The two environments differ in croissant, s2fft **and** jax together;
they cannot be separated cheaply because croissant 5.1.4 and PyPI
s2fft both require `jax<0.6`. This notebook therefore measures *`main`
versus PR 12*, not the pole-of-date fix in isolation.
```

- [ ] **Step 3: Write the setup code cell**

```python
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import importlib.metadata as md

import astropy.units as u
import croissant as cro
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time

import mistsim as ms

%matplotlib inline

REPO = Path("/home/christian/Documents/research/MIST/mistsim")
DATA_DIR = REPO / "data"
OUT_DIR = REPO / "notebooks/sim_comparisons/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Astronomical azimuth of the beam X-axis. The FEKO beam has
# phi=0=North, so 0 is correct; see beam.py (beam_rot = az_rot - 90).
BEAM_AZ_ROT = 0

# This notebook never writes to data/. beam.npz there is a live
# mapmaking input (mars-lake-dip in runs.yaml).
WRITE_SHARED_INPUTS = False

NSIDE = 128
SIM_LMAX = 100

TESTS = {
    0: "MARS / with mountains",
    1: "MARS / no mountains",
    2: "North Pole / no mountains",
}
```

- [ ] **Step 4: Write the environment-tagging cell**

```python
def croissant_tag():
    """Machine tag and human label for the installed croissant.

    The v5.3.0.dev2 tag still reports __version__ == "5.2.1", so the
    version string alone cannot distinguish the environments. Prefer
    the git revision the installer recorded; fall back to probing for
    eq2cirs, which exists only after the bump.
    """
    version = md.version("croissant-sim")
    rev = None
    try:
        raw = md.distribution("croissant-sim").read_text(
            "direct_url.json"
        )
        if raw:
            rev = json.loads(raw).get("vcs_info", {}).get("commit_id")
    except Exception:
        rev = None
    tag = f"{version}+git{rev[:7]}" if rev else version
    is_new = hasattr(cro.rotations, "eq2cirs")
    label = (
        "PR 12 (croissant v5.3.0.dev2)"
        if is_new
        else "main (croissant 5.1.4)"
    )
    return tag, label


TAG, LABEL = croissant_tag()
print(f"TAG   = {TAG}")
print(f"LABEL = {LABEL}")
print(f"jax   = {jax.__version__}")
```

- [ ] **Step 5: Write the cell that loads Raul's reference**

```python
def read_raul(path):
    with h5py.File(path, "r") as hf:
        return (
            np.array(hf["lst"]),
            np.array(hf["freq"]),
            np.array(hf["ant_temp"]),
        )


paths = {
    0: "20260215_for_christian/antenna_temperature_20260215_test1.hdf5",
    1: "20260220_for_christian/antenna_temperature_20260220_test1.hdf5",
    2: "20260220_for_christian/antenna_temperature_20260220_test2.hdf5",
}

raul, lsts = {}, {}
freqs = None
for k, rel in paths.items():
    lst_k, freq_k, temp_k = read_raul(DATA_DIR / rel)
    if freqs is None:
        freqs = freq_k
    assert np.allclose(freq_k, freqs), f"freq mismatch in test {k}"
    assert temp_k.shape == (lst_k.size, freqs.size), (
        f"test {k}: got {temp_k.shape}"
    )
    raul[k], lsts[k] = temp_k, lst_k

# Tests 0 and 1 share a MARS LST grid; test 2 is the North Pole grid.
assert np.allclose(lsts[0], lsts[1])
print(f"freqs {freqs.shape}: {freqs[0]:.0f}-{freqs[-1]:.0f} MHz")
for k in raul:
    print(f"test {k}: {raul[k].shape}, LST "
          f"{lsts[k][0]:.2f}-{lsts[k][-1]:.2f} hr")
```

- [ ] **Step 6: Execute the notebook headless to verify setup and loading**

```bash
cd /home/christian/Documents/research/MIST/mistsim
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/sim_comparisons/raul_comparison.ipynb
```

Expected: exit 0. The printed shapes are `(241, 86)` for all three tests, `freqs (86,): 40-125 MHz`, and `LABEL = PR 12 (croissant v5.3.0.dev2)` under the project kernel. The in-cell `assert`s are the test — a frequency or shape mismatch fails the execution.

- [ ] **Step 7: Commit**

```bash
git add notebooks/sim_comparisons/raul_comparison.ipynb \
        notebooks/sim_comparisons/results/.gitignore
git commit -m "feat: add Raul comparison notebook setup and reference loading"
```

---

### Task 3: Sky, beams, and the cached simulations

**Files:**
- Modify: `notebooks/sim_comparisons/raul_comparison.ipynb`

**Interfaces:**
- Consumes: `DATA_DIR`, `OUT_DIR`, `TAG`, `LABEL`, `BEAM_AZ_ROT`, `WRITE_SHARED_INPUTS`, `NSIDE`, `SIM_LMAX`, `freqs`, `lsts`, `TESTS` from Task 2.
- Produces: `sky_model` (`ms.Sky`), `beams` (dict `int -> ms.Beam`), `gain` (`np.ndarray`, shape `(86, 181, 360)`), `theta_full` (`np.ndarray`, shape `(181,)`, radians), `locs` (dict `int -> EarthLocation`), `times` (dict `int -> astropy.time.Time`), `mine` (dict `int -> np.ndarray` of shape `(241, 86)`), and the file `OUT_DIR / f"raul_cmp_{TAG}.npz"` holding arrays `test0`, `test1`, `test2`, `lst0`, `lst2`, `freqs`, plus scalars `tag`, `label`, `beam_az_rot`.

- [ ] **Step 1: Write the sky cell**

```python
def load_haslam():
    path = (
        DATA_DIR
        / "20260215_for_christian/haslam408_ds_Remazeilles2014.fits"
    )
    with fits.open(path) as hdul:
        return hdul[1].data["TEMPERATURE"].ravel()


def scale_map(m, freqs, beta=-2.55, f0=408, tcmb=2.725):
    scale = (freqs / f0) ** beta
    return (m - tcmb)[None, :] * scale[:, None] + tcmb


haslam = hp.ud_grade(load_haslam(), NSIDE, order_in="RING")
haslam = scale_map(haslam, freqs)
sky_model = ms.Sky(haslam, freqs, sampling="healpix", coord="galactic")
print(f"sky {sky_model.data.shape}")
```

- [ ] **Step 2: Write the beam cell**

```python
d = np.load(DATA_DIR / "feko_beam.npz")
theta = d["theta"]
assert theta.max() <= np.pi * (1 + 1e-6), (
    "feko_beam.npz must store theta in radians"
)

fix = np.isin(d["freqs"] / 1e6, freqs)
gain_above = d["gain"][fix]
assert np.allclose(d["freqs"][fix] / 1e6, freqs)

# Extend theta to 180 deg with zero gain below the horizon.
gain = np.concatenate(
    (gain_above, np.zeros_like(gain_above[:, :-1, :])), axis=1
)
theta_full = np.concatenate(
    (theta, np.deg2rad(np.arange(91, 181)))
)

horizon = (theta_full <= np.deg2rad(80))[:, None]
beam_mtn = ms.Beam(
    gain, freqs, sampling="mwss", horizon=horizon,
    beam_az_rot=BEAM_AZ_ROT,
)
beam_flat = ms.Beam(
    gain, freqs, sampling="mwss", horizon=None,
    beam_az_rot=BEAM_AZ_ROT,
)
beams = {0: beam_mtn, 1: beam_flat, 2: beam_flat}

if WRITE_SHARED_INPUTS:
    np.savez(
        DATA_DIR / "beam.npz",
        freqs=d["freqs"][fix] / 1e6, phi=d["phi"],
        theta=theta_full, gain=gain,
    )
else:
    print("data/beam.npz left untouched (WRITE_SHARED_INPUTS=False)")

plt.figure(figsize=(5, 3), constrained_layout=True)
plt.imshow(gain[0], aspect="auto", interpolation="none",
           extent=[0, 359, 180, 0])
plt.colorbar(label="gain")
plt.title(f"Beam at {freqs[0]:.0f} MHz, az_rot={BEAM_AZ_ROT}")
plt.xlabel(r"$\phi$ [deg]")
plt.ylabel(r"$\theta$ [deg]")
plt.show()
```

- [ ] **Step 3: Write the observation-geometry cell**

```python
mars = EarthLocation.from_geodetic(-90.74750, 79.41833, height=150)
npole = EarthLocation.from_geodetic(0, 90, height=0)
locs = {0: mars, 1: mars, 2: npole}


def lst_to_time(lst_arr, t0):
    """Convert an array of LST in hours to astropy Times."""
    lst_ref = t0.sidereal_time("mean").hour
    return t0 + (lst_arr - lst_ref) / 24 * u.sday


times = {}
for k, loc in locs.items():
    t0 = Time("2022-07-17 00:00", location=loc)
    times[k] = lst_to_time(lsts[k], t0)
    got = times[k].sidereal_time("mean").hour
    assert np.allclose(got, lsts[k]), f"LST mismatch in test {k}"
print("LST round-trip OK for all three tests")
```

- [ ] **Step 4: Write the simulation cell with caching**

```python
cache_path = OUT_DIR / f"raul_cmp_{TAG}.npz"

if cache_path.exists():
    print(f"Loading cached {cache_path.name}")
    c = np.load(cache_path, allow_pickle=True)
    mine = {k: c[f"test{k}"] for k in TESTS}
else:
    mine = {}
    for k in TESTS:
        print(f"Simulating test {k}: {TESTS[k]}")
        sim = ms.Simulator(
            beams[k], sky_model, times[k].jd, freqs,
            locs[k].lon.deg, locs[k].lat.deg,
            alt=locs[k].height.value, lmax=SIM_LMAX, Tgnd=0,
        )
        tant = sim.sim()
        fgnd = sim.beam.compute_fgnd()
        mine[k] = np.asarray(
            cro.simulator.correct_ground_loss(tant, fgnd, 0)
        )
    np.savez(
        cache_path,
        test0=mine[0], test1=mine[1], test2=mine[2],
        lst0=lsts[0], lst2=lsts[2], freqs=freqs,
        tag=TAG, label=LABEL, beam_az_rot=BEAM_AZ_ROT,
    )
    print(f"Wrote {cache_path}")

for k in TESTS:
    assert mine[k].shape == raul[k].shape, (
        f"test {k}: {mine[k].shape} vs Raul {raul[k].shape}"
    )
print("All three waterfalls match Raul's shape")
```

- [ ] **Step 5: Execute headless and verify the cache is written**

```bash
cd /home/christian/Documents/research/MIST/mistsim
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/sim_comparisons/raul_comparison.ipynb
ls -l notebooks/sim_comparisons/results/
```

Expected: exit 0, and one `raul_cmp_5.2.1+git*.npz` present. Compilation makes the first run several minutes.

- [ ] **Step 6: Re-execute to verify the cache short-circuits**

```bash
cd /home/christian/Documents/research/MIST/mistsim
time uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/sim_comparisons/raul_comparison.ipynb
```

Expected: exit 0 and "Loading cached ..." printed, with a runtime substantially below the first run's. Re-execution still rebuilds the HEALPix sky and both beam transforms, so do not expect it to be instant — what matters is that the three `ms.Simulator` runs are skipped. "Simulating test 0" appearing again means the cache key is wrong.

- [ ] **Step 7: Commit**

```bash
git add notebooks/sim_comparisons/raul_comparison.ipynb
git commit -m "feat: simulate the three Raul comparison cases with caching"
```

---

### Task 4: The three-way comparison section

**Files:**
- Modify: `notebooks/sim_comparisons/raul_comparison.ipynb`

**Interfaces:**
- Consumes: `OUT_DIR`, `TESTS`, `raul`, `lsts`, `freqs`.
- Produces: `runs` (list of dicts with keys `tag`, `label`, `az_rot`, `data`); `residual_profile(mine, ref, lst)` returning a dict with keys `mean_abs_K`, `max_abs_K`, `mean_rel_pct`, `edge_mean_K`, `middle_mean_K`, `arch_ratio`, `per_lst`; `profiles`, a dict keyed `(test_index, tag)` holding those dicts — Task 5 reads `profiles[(0, TAG)]`; and the rendered figures.

- [ ] **Step 1: Write the loader cell**

```python
runs = []
for path in sorted(OUT_DIR.glob("raul_cmp_*.npz")):
    c = np.load(path, allow_pickle=True)
    runs.append({
        "tag": str(c["tag"]),
        "label": str(c["label"]),
        "az_rot": float(c["beam_az_rot"]),
        "data": {k: c[f"test{k}"] for k in TESTS},
    })
    print(f"{c['label']}  (az_rot={float(c['beam_az_rot']):.0f})")

if len(runs) < 2:
    print(
        "\nOnly one environment present. Re-run this notebook under "
        "the other kernel to populate the comparison."
    )
```

- [ ] **Step 2: Write the metric cell**

```python
def residual_profile(mine, ref, lst):
    """Agreement with the reference, and how it varies across the day.

    croissant's pre-bump pointing error was zero at times_jd[0], grew
    to ~17' by mid-day and closed again after a full sidereal day. If
    that error dominates, per-LST residual is an arch and arch_ratio
    is well above 1. A flat profile (arch_ratio ~ 1) means the
    disagreement is something else.
    """
    resid = np.asarray(mine) - np.asarray(ref)
    per_lst = np.mean(np.abs(resid), axis=1)
    n = per_lst.size
    third = max(n // 3, 1)
    edges = np.concatenate(
        [per_lst[: third // 2], per_lst[-(third // 2):]]
    )
    middle = per_lst[third: 2 * third]
    return {
        "mean_abs_K": float(np.mean(np.abs(resid))),
        "max_abs_K": float(np.max(np.abs(resid))),
        "mean_rel_pct": float(
            100 * np.mean(np.abs(resid)) / np.mean(np.abs(ref))
        ),
        "edge_mean_K": float(np.mean(edges)),
        "middle_mean_K": float(np.mean(middle)),
        "arch_ratio": float(np.mean(middle) / (np.mean(edges) or 1.0)),
        "per_lst": per_lst,
    }
```

- [ ] **Step 3: Write the summary-table cell**

```python
hdr = (
    f"{'test':<28} {'environment':<30} {'mean K':>8} "
    f"{'rel %':>7} {'max K':>9} {'arch':>6}"
)
print(hdr)
print("-" * len(hdr))
profiles = {}
for k in TESTS:
    for run in runs:
        p = residual_profile(run["data"][k], raul[k], lsts[k])
        profiles[(k, run["tag"])] = p
        print(
            f"{TESTS[k]:<28} {run['label']:<30} "
            f"{p['mean_abs_K']:8.3f} {p['mean_rel_pct']:7.2f} "
            f"{p['max_abs_K']:9.3f} {p['arch_ratio']:6.2f}"
        )
    print()
```

- [ ] **Step 4: Write the LST-residual figure cell — the key plot**

```python
fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(12, 3.5), sharex=False,
    constrained_layout=True,
)
for ax, k in zip(axs, TESTS):
    for run in runs:
        p = profiles[(k, run["tag"])]
        ax.plot(
            lsts[k], p["per_lst"],
            label=f"{run['label']} (arch {p['arch_ratio']:.2f})",
        )
    ax.set_title("\n".join(TESTS[k].split(" / ")))
    ax.set_xlabel("LST [hr]")
    ax.set_yscale("log")
axs[0].set_ylabel(r"mean $|T_{\rm mistsim} - T_{\rm Raul}|$ [K]")
for ax in axs:
    ax.legend(fontsize=7)
fig.suptitle(
    "Disagreement with Raul across the day. An arch peaking mid-day "
    "is the pre-bump pointing drift; flat is something else."
)
plt.show()
```

- [ ] **Step 5: Write the waterfall and residual-image cell**

```python
for k in TESTS:
    ncols = 1 + len(runs)
    fig, axs = plt.subplots(
        nrows=1, ncols=ncols, figsize=(3.2 * ncols, 3.6),
        sharey=True, constrained_layout=True,
    )
    ext = [freqs[0], freqs[-1], lsts[k][-1], lsts[k][0]]
    kw = dict(extent=ext, interpolation="none", aspect="auto")
    im = axs[0].imshow(raul[k], vmin=500, vmax=12000, **kw)
    axs[0].set_title("Raul")
    axs[0].set_ylabel("LST [hr]")
    for ax, run in zip(axs[1:], runs):
        d_im = run["data"][k] - raul[k]
        lim = np.max(np.abs(d_im))
        im2 = ax.imshow(d_im, cmap="bwr", vmin=-lim, vmax=lim, **kw)
        ax.set_title(f"{run['label']}\nminus Raul")
        fig.colorbar(im2, ax=ax, label="[K]")
    for ax in axs:
        ax.set_xlabel("Frequency [MHz]")
    fig.colorbar(im, ax=axs[0], label="[K]", location="bottom")
    fig.suptitle(TESTS[k])
    plt.show()
```

- [ ] **Step 6: Write the horizon-effect cell**

```python
fig, axs = plt.subplots(
    nrows=1, ncols=1 + len(runs), figsize=(3.2 * (1 + len(runs)), 3.6),
    sharey=True, constrained_layout=True,
)
ext = [freqs[0], freqs[-1], lsts[0][-1], lsts[0][0]]
kw = dict(extent=ext, interpolation="none", aspect="auto",
          cmap="bwr", vmin=-90, vmax=90)
im = axs[0].imshow(raul[1] - raul[0], **kw)
axs[0].set_title("Raul")
axs[0].set_ylabel("LST [hr]")
for ax, run in zip(axs[1:], runs):
    ax.imshow(run["data"][1] - run["data"][0], **kw)
    ax.set_title(run["label"])
for ax in axs:
    ax.set_xlabel("Frequency [MHz]")
fig.colorbar(im, ax=axs, label="(no mountains) - (with mountains) [K]")
fig.suptitle("Horizon blockage effect")
plt.show()
```

- [ ] **Step 7: Execute headless and verify the section renders with one run present**

```bash
cd /home/christian/Documents/research/MIST/mistsim
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/sim_comparisons/raul_comparison.ipynb
```

Expected: exit 0. With only the PR 12 npz present it prints "Only one environment present…" and every figure still renders with a single curve. A crash on `zip(axs[1:], runs)` here means the single-run case was not handled.

- [ ] **Step 8: Commit**

```bash
git add notebooks/sim_comparisons/raul_comparison.ipynb
git commit -m "feat: add three-way comparison figures and residual metrics"
```

---

### Task 5: The beam-convention sanity check

**Files:**
- Modify: `notebooks/sim_comparisons/raul_comparison.ipynb`

**Interfaces:**
- Consumes: `gain`, `freqs`, `theta_full`, `sky_model`, `times`, `locs`, `raul`, `lsts`, `SIM_LMAX`.
- Produces: nothing downstream; a documented, disabled check.

- [ ] **Step 1: Write the explanatory markdown cell**

```markdown
## Beam convention sanity check

`beam_az_rot` is the astronomical azimuth of the beam X-axis
(0 = North), converted internally to croissant's convention via
`beam_rot = beam_az_rot - 90`. The FEKO beam has phi=0 = North, so 0
is correct.

Passing 90 instead puts the beam 90 degrees out, which for this
dipole's two-fold symmetry is the *maximally* wrong rotation: it
produces a ~130 K disagreement with Raul rather than the ~1 K the
correct convention gives. Set `RUN_CONVENTION_CHECK = True` to
reproduce that, so the convention stays documented rather than
tribal knowledge.
```

- [ ] **Step 2: Write the check cell**

```python
RUN_CONVENTION_CHECK = False

if RUN_CONVENTION_CHECK:
    bad = ms.Beam(
        gain, freqs, sampling="mwss",
        horizon=(theta_full <= np.deg2rad(80))[:, None],
        beam_az_rot=90,
    )
    sim = ms.Simulator(
        bad, sky_model, times[0].jd, freqs,
        locs[0].lon.deg, locs[0].lat.deg,
        alt=locs[0].height.value, lmax=SIM_LMAX, Tgnd=0,
    )
    wrong = np.asarray(
        cro.simulator.correct_ground_loss(
            sim.sim(), sim.beam.compute_fgnd(), 0
        )
    )
    p_bad = residual_profile(wrong, raul[0], lsts[0])
    p_good = profiles[(0, TAG)]
    print(f"az_rot=0  : {p_good['mean_abs_K']:8.3f} K")
    print(f"az_rot=90 : {p_bad['mean_abs_K']:8.3f} K")
else:
    print("Convention check disabled (RUN_CONVENTION_CHECK=False)")
```

- [ ] **Step 3: Execute headless to verify the disabled path is inert**

```bash
cd /home/christian/Documents/research/MIST/mistsim
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/sim_comparisons/raul_comparison.ipynb
```

Expected: exit 0, prints "Convention check disabled", and adds no runtime.

- [ ] **Step 4: Commit**

```bash
git add notebooks/sim_comparisons/raul_comparison.ipynb
git commit -m "docs: add disabled beam_az_rot convention check to the notebook"
```

---

### Task 6: Populate both environments and record the result

**Files:**
- Modify: `notebooks/sim_comparisons/raul_comparison.ipynb` (executed outputs)

**Interfaces:**
- Consumes: Task 1's `mistsim-cro514` kernel; Tasks 2-5's notebook.
- Produces: two files in `notebooks/sim_comparisons/results/`, and a notebook whose comparison section shows both environments.

- [ ] **Step 1: Execute under the `main` kernel**

```bash
cd /home/christian/Documents/research/MIST/mistsim
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=mistsim-cro514 \
    --output-dir /tmp --output raul_comparison_main.ipynb \
    notebooks/sim_comparisons/raul_comparison.ipynb
```

Expected: exit 0, and a second npz appears tagged `5.1.4`. Output goes to `/tmp` so this run does not overwrite the committed notebook; its only lasting product is the cached npz.

- [ ] **Step 2: Verify both environments are now cached**

```bash
ls -l /home/christian/Documents/research/MIST/mistsim/notebooks/sim_comparisons/results/
```

Expected: exactly two npz files, one tagged `5.1.4` and one `5.2.1+git*`. If only one exists, the `main` run silently reused the PR cache — check `TAG` differs between kernels.

- [ ] **Step 3: Execute under the project kernel to produce the final artifact**

```bash
cd /home/christian/Documents/research/MIST/mistsim
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/sim_comparisons/raul_comparison.ipynb
```

Expected: exit 0. The loader prints both labels, the summary table has two rows per test, and every figure carries two curves.

- [ ] **Step 4: Read off the result**

Record, from the summary table, whether the arch ratio for the two MARS tests falls between `main` and PR 12, and what the North Pole test does. Both outcomes are publishable: the notebook's job is to answer, not to confirm.

- [ ] **Step 5: Commit the executed notebook**

```bash
git add notebooks/sim_comparisons/raul_comparison.ipynb
git commit -m "results: run the Raul comparison under main and PR 12"
```

---

## Cleanup (after the comparison is settled)

The worktree is disposable once both npz files exist:

```bash
cd /home/christian/Documents/research/MIST/mistsim
git worktree remove ../mistsim-main
uv run jupyter kernelspec uninstall mistsim-cro514
```

Keep them until the North Pole question is resolved — re-running under `main` is the first diagnostic if that result is surprising.
