# Consolidated Raul comparison notebook

**Date:** 2026-09-14
**Status:** design, awaiting review
**Branch:** `deps/croissant-v5.3.0.dev2` (PR 12)

## Problem

PR 12 moves mistsim onto croissant `v5.3.0.dev2`, whose headline change
is that Earth simulations rotate the sky about the pole of date rather
than the J2000 pole. The mapmaking results are insensitive to this —
data and forward model are built from the same `Simulator`, so the
error cancels in the linear inverse, and a re-run of `mars-lmax40`
moved the relative reconstruction error by 0.006%.

The comparison against Raul's reference simulator is a different
matter. Raul's implementation is independent, so nothing cancels and
any croissant-side pointing error lands directly in the residual.
That comparison therefore has to be redone.

It cannot be redone with the notebooks as they stood. There were three
of them (`sim_comparison.ipynb`, `sim_comparison_azrot0.ipynb`,
`sim_comparison_azrot90.ipynb`), none referenced each other, and none
had any notion of comparing two versions of mistsim. The stored
`comparison_022526.npz` was written at 11:09 on 2026-02-25, thirty-six
minutes before that day's croissant-v5 upgrade commits, so it predates
five subsequent changes and cannot serve as a controlled baseline.

## Goals

- One notebook that puts Raul's reference, mistsim under `main`, and
  mistsim under PR 12 on the same axes.
- Plots and a durable paper trail, not summary statistics in a
  terminal.
- Enough visibility into the simulation setup that a convention bug is
  apparent from reading the notebook.

## Non-goals

- Isolating the pole-of-date fix from the rest of the dependency bump
  (see "What this can and cannot claim").
- The beam-interpolation study in `interp_beam.ipynb`, which
  investigates sampling rather than frame correctness and stays where
  it is.
- Any change to the mapmaking pipeline or its stored results.

## Decisions taken

**"Old" means a re-run under `main`, not the stored February arrays.**
The alternative — reading `chb-*` out of `comparison_022526.npz` —
costs no compute but mixes five changes into one difference.

**The comparison absorbs the three `sim_comparison` variants only.**
`beam_az_rot` becomes a parameter defaulting to 0, which subsumes what
the two `azrot` notebooks tested.

**Both versions run byte-identical notebook code under two kernels.**
The notebook is run twice, once per environment. The alternatives —
shelling out to a second venv, or reducing the notebook to analysis
over precomputed files — both move the simulation code out of the
notebook's visible cells, which is the transparency the notebook
format is being chosen for.

## Environment

The "old" environment is a git worktree of `main`, so it reproduces
main's lockfile exactly rather than being hand-assembled, and "old"
means all of main rather than main's croissant grafted onto the PR
branch's mistsim:

```bash
git worktree add ../mistsim-main main
cd ../mistsim-main && uv sync --all-extras --dev
uv run python -m ipykernel install --user --name mistsim-cro514 \
    --display-name "mistsim (croissant 5.1.4)"
```

For the record, the sim-comparison code path is identical between the
two branches: `beam.py` and `sim.py` are byte-identical to main, and
`sky.py`'s only change is inside `_SkyAlm`, the raw-alm compatibility
class that `ms.Sky` does not use. PR 12's `pipeline.py` changes are
mapmaking-only. The variable is the dependency set, not mistsim.

## Notebook

`notebooks/sim_comparisons/raul_comparison.ipynb`

1. **Setup.** Detects the installed croissant revision and derives
   `TAG` from it, so results are self-labelling and do not depend on
   remembering which kernel was selected. Parameters: `BEAM_AZ_ROT = 0`,
   `WRITE_SHARED_INPUTS = False`.
2. **Raul's reference.** Loads the three hdf5 waterfalls from
   `data/20260215_for_christian/` and `data/20260220_for_christian/`.
3. **Sky and beams.** Haslam to HEALPix `nside=128`; `feko_beam.npz`
   extended to `theta = 180 deg` with zero gain below the horizon;
   the mountains / no-mountains beam pair.
4. **Simulations.** Three tests (MARS with mountains, MARS without,
   North Pole without), cached to
   `notebooks/sim_comparisons/results/raul_cmp_<TAG>.npz`
   (a new directory) and
   skipped when that file already exists, so iterating on a figure
   costs seconds rather than the full run.
5. **Three-way comparison.** Globs every `results/raul_cmp_*.npz` present and
   renders what it finds: waterfalls side by side; residual images
   against Raul; mean absolute residual against LST for each version
   on shared axes; a summary table; and the horizon-effect panel
   (test 0 minus test 1).
6. **Convention sanity check.** Off by default. Re-runs test 0 at
   `beam_az_rot = 90` to reproduce the ~130 K disagreement that a
   90-degree azimuthal offset produces for this two-fold-symmetric
   dipole, so the convention is documented rather than tribal.

Run under the 5.1.4 kernel first, then the PR kernel; the second
execution is the complete artifact.

### The figure that matters

Mean absolute residual against LST, per version. croissant's changelog
puts the old pointing error at zero at `times_jd[0]`, about 9 arcmin
after 4 h and 17 arcmin after 12 h, closing again after a full
sidereal day. Against a 24 h observation that predicts an arch in the
residual — small at both ends, largest in the middle — which the fix
should flatten. A scalar improvement in mean agreement could come from
anything; an arch collapsing to flat is specific to this fix.

## What this can and cannot claim

`main` and PR 12 differ in three dependencies:

| | main | PR 12 |
|---|---|---|
| croissant | 5.1.4 (PyPI) | v5.3.0.dev2 (git `d972c5f`) |
| s2fft | 1.4.0 (PyPI) | slosar fork, 20 commits past v1.4.0 |
| jax | `<0.6` | 0.11.1 |

These cannot be separated cheaply: croissant 5.1.4 and PyPI s2fft
require `jax<0.6`. The notebook therefore measures **main versus
PR 12**, which is the decision actually on the table, but it cannot on
its own attribute an observed change to the pole fix. Curves are
labelled by environment, not by feature, and a markdown cell states
this limitation. Attributing the change to the pole fix specifically
would require a third environment holding s2fft and jax fixed, which
is out of scope here.

## Open question: the North Pole test

An exploratory run at `beam_az_rot = 0` under PR 12 gave, against
Raul: MARS with mountains 1.303 K (arch 1.42, from 3.99), MARS without
0.921 K (arch 1.15, from 7.19) — both consistent with the prediction —
but North Pole 117.237 K with arch 0.99, against 2.964 K stored.

A large, flat residual is neither the pole-of-date signature (arched)
nor the 90-degree beam error (that run was at 0, and this test was
*better* at 90). The working hypothesis is that at latitude 90 the
local azimuth reference is degenerate, so a beam azimuth rotation is
indistinguishable from a shift in LST, and a convention change that is
harmless at MARS's 79.4 degrees becomes a constant LST offset at
exactly 90. This is inference from the shape of the numbers and is not
demonstrated.

The notebook gives this plots instead of three summary numbers, which
is what diagnosing it needs. If it survives the re-run, the next step
is to cross-correlate the North Pole waterfall against Raul's over LST
lags: a residual that collapses at some nonzero lag confirms an
offset.

`runs.yaml` defines `southpole` at latitude -90.0, so if the effect is
real it would reach `southpole-csa2022-dip` and
`southpole-mono-analytic`. Whether any executed mapmaking run used
them should be checked before anyone relies on those results.

## Cleanup

`sim_comparison.ipynb`, `sim_comparison_azrot0.ipynb` and
`sim_comparison_azrot90.ipynb` are deleted, superseded by this
notebook; all three remain in git history on `main`. The throwaway
`scripts/spot_check_croissant_bump.py` is deleted.
`interp_beam.ipynb` and the two stored npz files are kept.

## Success criteria

- The notebook runs end to end under both kernels without writing to
  `data/`.
- The three-way comparison section renders from cached npz alone, so
  figures can be revised without re-simulating.
- The LST-residual figure either shows the arch flattening between
  main and PR 12, or shows that it does not — both are publishable
  answers, and the notebook records which.
- The North Pole result is either reproduced or shown to be an
  artefact of the exploratory run.
