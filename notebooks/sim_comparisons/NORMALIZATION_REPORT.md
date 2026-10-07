# mistsim vs Raul: normalisation, epoch and conventions

MISTIC analysis queue item S9 (2026-10-06). Notebook:
`raul_comparison_normalization.ipynb` (executed). Helpers:
`raul_tools.py`. Numbers: `results_normalization/summary.json`; figures:
`results_normalization/fig*.png`. Environment: mistsim `4d12bfd` +
this branch, croissant v5.3.0.dev3 (`58bf56e`), jax 0.11.1, lmax 100,
nside 128. Raul's inputs are read in place from the shared checkout's
untracked `data/2026021{5,20}_for_christian`.

## Result

Residual mistsim - Raul in K, mean / rms / max |d| over 241 LST x 86
frequencies (40-125 MHz):

| Run | Test 0 (10 deg blockage) | Test 1 (MARS) | Test 2 (North Pole) |
|---|---|---|---|
| `base` = `raul_comparison.ipynb` | 1.07 / 2.01 / 14.9 | 0.75 / 1.43 / 7.3 | 0.72 / 1.30 / 5.1 |
| `epoch`: Raul's UTC grid | 0.22 / 1.36 / 11.7 | -0.09 / 0.28 / 2.3 | -0.20 / 0.40 / 1.9 |
| `mirror`: + beam gain(-phi) | 0.22 / 1.35 / 11.7 | -0.09 / 0.26 / 2.3 | -0.20 / 0.37 / 1.9 |
| `edge`: + half-weight edge row | -0.09 / 0.26 / 2.4 | (no edge) | (no edge) |
| pixel replica of Raul's 2022 `convolution` | -0.00 / 0.19 / 1.9 | -0.01 / 0.19 / 1.9 | -0.12 / 0.21 / 1.0 |

Test 0 with Raul's grid, by normalisation:

| Normalisation | mean / rms / max |d| |
|---|---|
| above-horizon (Raul's) | 0.22 / 1.36 / 11.7 |
| full sphere, blocked beam at 0 K | -29.8 / 48.8 / 214 |
| full sphere + `Tgnd = 300 K` (mistsim default) | -25.9 / 44.2 / 209 |

1. **Raul's normalisation does not explain the residual: mistsim
   already used it.** Raul divides by the above-horizon, unmasked beam
   (`astro.py:960`). `raul_comparison.ipynb` simulated with `Tgnd = 0`
   and applied `correct_ground_loss(T, fgnd, 0)`, which is the same
   estimator. The alternatives are wrong by up to 214 K (about fgnd x T
   at 40 MHz, fgnd = 1.7 %).
2. **The epoch was the main cause.** Raul's 2026 LSTs are his 2022 grid
   exactly (rebuilt to 5e-14 h): UTC 2014-01-01 09:29:45 + i x 359 s,
   apparent sidereal time (IAU2006A, DUT1 = 0), sorted by LST. The old
   notebook placed the same LSTs on 2022-07-17 with mean sidereal time.
   The 8.5-year precession and nutation of the frame is not a pure LST
   shift: the best-fit lag is -21 s, but removing it lowers the Test 1
   rms only from 1.43 to 1.34 K. Using Raul's UTC grid takes it to
   0.28 K.
3. **Test 0's excess is the mask edge on the 1-deg beam grid.**
   `theta <= 80` keeps the theta = 80 row, which stands for 79.5-80.5
   deg, so the effective horizon sits near 80.5 deg (+0.22 K mean,
   11.7 K max). `theta < 80` moves it to about 79.5 deg (-0.41 K mean,
   13.2 K max). Half weight on that row gives 0.26 K rms, Test 1's
   level. fgnd at 40 MHz is 1.74 %, 1.97 % and 2.21 % for the three
   choices. The Test 1 - Test 0 blockage effect (rms 11.7 K) is matched
   to 1.36 K rms with `theta <= 80` and to 0.066 K with half weight.
4. **Small or nil:** the beam mirror (-0.02 K rms), lmax 179
   (< 0.003 K) and a 3-iteration HEALPix SHT of the sky (+0.003 to
   +0.006 K).
5. **What is left** (0.26 / 0.26 / 0.37 K rms, 2-4e-5 of T_ant, max
   2.4 K at 40 MHz) splits into mistsim - replica (rms 0.16-0.17 K,
   mean -0.08 K on every test) and replica - Raul (rms 0.19-0.21 K,
   frequency-structured). The second comes from Raul's 2026 code and is
   not in his 2022 code. One candidate is the FEKO file's printed
   precision.

## What mistsim normalises by

croissant v5.3.0.dev3, `site-packages/croissant/`:

- `beam.py:184`: `compute_alm` transforms `data * horizon`.
- `simulator.py:429-433`: `sim` divides by `compute_norm()`, the
  **full-sphere** beam integral (`beam.py:136-148` →
  `_compute_norm(use_horizon=False)`, `beam.py:100-133`). It then adds
  `fgnd * Tgnd` (`simulator.py:305-320`), with
  `fgnd = 1 - int(B * horizon) / int(B)` (`beam.py:151-165`).
- `simulator.py:97-125`: `correct_ground_loss` gives
  `(vis - fgnd Tgnd) / (1 - fgnd)`, i.e. division by the
  **above-horizon** integral, which is Raul's estimator.

mistsim:

- `sim.py:20`: `Tgnd = 300.0` is the default.
- `pipeline.py:1705-1757`: `simulate_waterfall` uses
  `normalize_beam_alm(..., ground_loss=True)` (`mapmaking.py:22-56`).
  That is the full-sphere norm with no ground term, so the blocked beam
  sees 0 K. The map-making operators use the same norm unless
  `observation.ground_loss: false`.

## Conventions (synthetic asymmetric beam and one-sided mask, section 6)

- **Beam azimuth.** With `beam_az_rot = 0`, mistsim puts beam phi = 0 at
  North and phi = 90 deg at **West**: a right-handed frame with z up
  (2.2 K mean |d| against 236 K for phi = AZ). Raul maps FEKO phi onto
  astronomical AZ (N through E), the mirror image. The replica confirms
  his 2026 code still does (0.108 K against 0.166 K mirrored). For this
  nearly symmetric dipole it is 0.02 K; for an asymmetric beam it is
  not.
- **Horizon mask.** The mask lives in the beam frame. It is mirrored the
  same way and **rotates with `beam_az_rot`**: at 40 deg, 3.1 K mean
  |d| with the mask rotating against 37 K with it fixed on the ground.
  A terrain horizon in astronomical azimuth must be passed as
  mask(theta, phi_b) with phi_b = -(AZ - beam_az_rot) mod 360.
- **Units.** The mask is boolean, so units do not enter. The pipeline
  converts `horizon_max_theta` from degrees (`pipeline.py:380`) and
  requires beam theta in radians (`pipeline.py:373`).

## Implications and next steps

1. **Horizon edge bias (mistsim, all users).** `_horizon_from_beam_file`
   (`pipeline.py:381`) keeps the edge row, so a horizon at theta_h acts
   at about theta_h + 0.5 deg on a 1-deg grid. Here that is a 12 %
   relative error in fgnd and up to 12 K at 40 MHz. Proposed fix (a
   `src/` change with tests, for CHB to decide): fractional edge
   weights, or a supersampled mask.
2. **MISTIC terrain horizon and HFSS beam.** Any code that builds a
   `horizon` from an azimuth profile, or loads an HFSS beam, must follow
   the convention above: mirrored, and counter-rotated by `beam_az_rot`.
3. **Comparisons with ground-loss-corrected references** (Raul's sims,
   or calibrated data corrected for ground loss) need the above-horizon
   norm: `correct_ground_loss`, or `ground_loss: false` in the pipeline.
4. **Raul:** ask whether his 2026 code still maps phi = AZ, and how it
   reads the FEKO gain (dB column or fields). That would settle the last
   0.2 K.
5. Raul's sims are on 2014 dates. Simulate them on his UTC grid
   (`raul_tools.raul_times`), not by LST alone.
