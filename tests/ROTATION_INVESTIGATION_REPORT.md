# Rotation Convention Investigation: NEU-to-ENU Fix

**Date:** 2026-03-19
**Context:** Croissant 5.1.1 -> 5.1.2 upgrade; only source change was a
NEU-to-ENU axes swap fix in `get_rot_mat` for topocentric frames.

## Problem

After the fix, disagreement with Raul's reference simulator increased from
~10-30 K to ~130 K mean (7% relative). The fix is mathematically correct
(AltAz uses left-handed NEU; the swap gives right-handed ENU with det=+1),
but the regression in agreement needed explanation.

## Summary of Findings

**The ENU fix is mathematically correct but exposed a pre-existing beam
azimuth convention mismatch.** The old NEU code was wrong (det=-1, improper
rotation), but it happened to partially compensate for the fact that the
sim_comparison notebook uses beam data in AltAz convention (phi=0=North)
without setting `beam_az_rot`.

The agreement with Raul was **not a coincidence** -- it was a direct
consequence of NEU's x-axis being North, which matched the beam data's
azimuthal convention. The fix changed the assumed convention to ENU
(x=East), introducing a real azimuthal mismatch that beam_az_rot is
designed to correct.

## The core issue: two intertwined conventions

There are two separate conventions at play:

1. **Rotation matrix handedness**: NEU (det=-1) vs ENU (det=+1)
2. **Beam azimuth reference**: phi=0=North (AltAz/FEKO) vs phi=0=East (ENU)

The old code got convention 1 wrong (improper rotation) but convention 2
accidentally right (NEU x=North matched beam's phi=0=North). The fix
corrected convention 1 but broke convention 2. The solution is to keep the
fix and also set `beam_az_rot` correctly.

### Why NEU "worked" for AltAz beams

In the HEALPix/SHT convention, phi=0 corresponds to the x-axis. The
physical meaning of phi=0 depends on which frame's x-axis we use:

| Frame | x-axis | phi=0 means | Matches AltAz az=0? |
|-------|--------|-------------|---------------------|
| NEU   | North  | North       | YES                 |
| ENU   | East   | East        | NO (90 deg off)     |

FEKO beams (after AZ_antenna_axis correction) have az=0=North. When loaded
into HEALPix, phi=0=North. In the old NEU code, the SHT's phi=0 mapped to
x=North, matching the beam data. In the new ENU code, phi=0 maps to x=East,
creating a 90-degree azimuthal offset.

The `beam_az_rot` parameter exists precisely for this: it rotates the beam
from its native frame to ENU. For a beam with phi=0=North, `beam_az_rot=90`
is correct (North is 90 deg CCW from East).

### What gamma does

From the Euler angle comparison:

| Matrix | gamma (deg) | Effective beam azimuth offset |
|--------|-------------|------------------------------|
| R_new  | +89.8       | 0 (correct for ENU beam)     |
| R_old  | -179.8      | ~270 from ENU = ~180 from NEU|

The gamma difference between old and new is ~270 deg. For the old code with
an AltAz beam (phi=0=North in NEU frame), the gamma=-180 deg offset means
the beam was rotated ~180 deg in azimuth. For a beam with approximate
2-fold symmetry (dipole), a 180 deg rotation is nearly the identity. This
is why the old code agreed well with Raul for the MIST beam -- the residual
error from the 180 deg offset was small because of the beam's symmetry.

With the new code and no beam_az_rot, the beam has a 90 deg azimuthal
offset (ENU vs AltAz). For the MIST beam's 2-fold symmetry, a 90 deg
rotation produces maximum deviation -- explaining the jump from ~10-30 K
to ~130 K.

## Test Results

### Test 1: `rotate_flms` convention (s2fft)

s2fft uses the **active rotation** convention:

    D^l_mn(alpha, beta, gamma) = exp(-i*m*alpha) * d^l_mn(beta) * exp(-i*n*gamma)

Applied to alm, this gives f'(r) = f(R^{-1} r). Confirmed by setting
f_{1,1} = 1 and checking that rotate_flms with alpha=pi/4 gives
exp(-i*pi/4), not exp(+i*pi/4).

### Test 2: Old vs new rotation matrices

At lat=45, lon=0:

| Matrix | det  | alpha   | beta  | gamma  |
|--------|------|---------|-------|--------|
| R_new  | +1   | -1.588  | 0.785 | 1.568  |
| R_old  | -1   | -1.588  | 0.785 | -3.139 |
| R_new^T| +1   | 1.574   | 0.785 | -1.553 |

Key observations:
- **alpha and beta are identical** between R_new and R_old.
- **Only gamma differs**, by approximately 270 deg.
- gamma is the rotation about the z-axis in the *source* frame (zenith in
  topocentric coords). This is the azimuthal orientation of the beam.
- For any axially symmetric beam, gamma has no effect.

### Test 3-4: Beam peak placement (symmetric Gaussian)

Rotating a Gaussian beam peaked at zenith:

| Rotation | Mid-lat peak (RA, Dec) | Sep from expected |
|----------|------------------------|-------------------|
| R_new    | (268.5, 45.0)          | 0.4 deg           |
| R_old    | (268.5, 45.0)          | 0.4 deg           |
| R_new^T  | (91.5, 45.0)           | 90.0 deg          |

R_new and R_old give identical results for a symmetric beam because the only
difference (gamma) doesn't affect an axially symmetric pattern. R_new^T
(inverse) is unambiguously wrong.

### Test 5: Asymmetric beam (ENU east-direction test)

Beam pattern: 1 + 0.5*sin(theta)*cos(phi) in ENU (phi=0 = East).
Expected visibility ratio (East source / North source) = 1.354.

| Rotation | Vis ratio | Relative error |
|----------|-----------|----------------|
| R_new    | 1.230     | 9.1%           |
| R_old    | 0.676     | **50%**        |

R_new is correct for this beam (which IS defined in ENU). The 9.1% error is
from SHT truncation at nside=32. R_old gives catastrophically wrong results:
the ratio is below 1.0, meaning the beam response toward East is *weaker*
than toward North -- physically impossible for this beam definition.

**Important caveat:** This test uses a beam defined in ENU (phi=0=East). For
actual MIST beam data (phi=0=North), the relevant comparison would be
different. R_old would perform better for a phi=0=North beam because NEU's
x-axis is North.

### Test 6: healpy cross-check

Rotated a beam using both s2fft (via croissant) and healpy, comparing
against the known expected zenith position (Dec=45, RA=269):

| Method        | Peak (RA, Dec)   | Sep from expected |
|---------------|------------------|-------------------|
| s2fft(R)      | (274.5, 45.0)    | 3.9 deg           |
| healpy ZYX(R) | (274.5, 45.0)    | 3.9 deg           |
| healpy ZYX(R^T)| (85.5, 45.0)   | 89.9 deg          |

healpy independently confirms that the forward rotation R (with the ENU fix)
is correct for the ENU frame.

## Raul's code: no rotation matrix bug

Raul uses `SkyCoord.transform_to(AltAz)` to get (az, alt) angles per pixel,
then evaluates the beam via `RectBivariateSpline(EL_beam, AZ_beam, beam)`.
No rotation matrices, no Cartesian representation, no NEU/ENU ambiguity.
His code is correct by construction.

His convolution is:
```python
beam_above = spline.ev(EL_above_horizon, AZ_above_horizon)
antenna_temp = sum(beam * sky * mask) / sum(beam * mask)
```

Everything is in standard AltAz (az=0=North), self-consistent.

## Why the old code agreed better (not a coincidence)

The agreement was NOT a coincidence. It was a direct consequence of the
NEU frame's x-axis (North) being close to the MIST antenna orientation.

The MIST beam NPZ files are processed from raw FEKO output. phi=0
corresponds to the FEKO antenna x-axis, which runs along the dipole arms.
The beam data confirms this: phi=0/180 is the weak direction (along arms),
phi=90/270 is the strong direction (perpendicular to arms).

For MARS, the dipole is oriented at **10 deg from North toward East**
(AltAz az=10 deg). So the beam's phi=0 = AltAz az=10 deg.

### Old code (NEU, det=-1)

- SHT phi=0 maps to x=North (AltAz az=0)
- The beam's phi=0 is at az=10 deg, so there's only a ~10 deg mismatch
- The gamma error (~180 deg) is invisible for a dipole (2-fold symmetry)
- **Net azimuthal error: ~10 deg** -> small disagreement with Raul

### New code (ENU, det=+1) with beam_az_rot=0

- SHT phi=0 maps to x=East (AltAz az=90)
- The beam's phi=0 is at az=10 deg -> **80 deg mismatch**
- For a dipole, 80 deg ≈ 90 deg is the worst-case orientation
- **Net azimuthal error: ~80 deg** -> large disagreement with Raul

### New code (ENU, det=+1) with beam_az_rot=80

- beam_az_rot=80 rotates beam from antenna frame to ENU: phi=0 (az=10)
  is mapped 80 deg CCW to land on East (az=90)
- This is the correct ENU alignment
- **Net azimuthal error: 0 deg** -> should agree well with Raul

The mapmaking configs already use `beam_az_rot=80` for MARS and
`beam_az_rot=26` for Nevada. The sim_comparison notebook was the only
place missing this correction.

## Mapping between Raul's AZ_antenna_axis and beam_az_rot

Raul's `AZ_antenna_axis` shifts the beam in azimuth so that AZ=0 aligns
with North. With `AZ_antenna_axis=0`, the beam's phi=0 (FEKO x-axis =
dipole arms) is implicitly placed at AZ=0 = North.

In croissant (ENU code), `beam_az_rot` is the angle from East to the beam's
phi=0 direction, measured CCW. To match Raul's `AZ_antenna_axis=0` (dipole
arms at North), use `beam_az_rot=90` (North is 90 deg CCW from East).

In general: **beam_az_rot = 90 - AZ_antenna_axis**.

## Verified configurations

The mapmaking configs specify the antenna orientation as degrees from
North toward East. The conversion is: `beam_az_rot = 90 - az_from_north`.

| Site   | Orientation       | AltAz az | beam_az_rot | Config | Status |
|--------|-------------------|----------|-------------|--------|--------|
| MARS   | 10 deg N→E        | 10       | 80          | 80     | OK     |
| Nevada | 64 deg N→E        | 64       | 26          | 26     | OK     |

Both mapmaking configs are correct for the ENU code.

## Confidence assessment

**High confidence** that the ENU fix is mathematically correct:
- det=+1, proper rotation, right-handed ENU
- Tests 3-6 confirm correct rotation direction
- healpy independently validates

**High confidence** that the disagreement with Raul is caused by a missing
beam_az_rot in the sim_comparison notebook:
- Old code: NEU phi=0=North, with beam_az_rot=0 the dipole arms are at
  North. Only ~10 deg from the MARS antenna orientation. The ~180 deg gamma
  error is invisible for a 2-fold symmetric dipole. Net error ~10 deg.
- New code: ENU phi=0=East, with beam_az_rot=0 the dipole arms are at
  East. That's 80 deg from the MARS orientation. Net error ~80 deg.
- With beam_az_rot=90, the dipole arms are at North, matching Raul's
  likely AZ_antenna_axis=0. Net error ~0 deg.

## Recommended next steps

1. **Keep the ENU fix in croissant.** It is correct.

2. **Rerun sim_comparison with `beam_az_rot=90`** (assuming Raul used
   `AZ_antenna_axis=0`). Agreement should improve significantly.

3. **Confirm with Raul** what `AZ_antenna_axis` he used. If non-zero,
   adjust: `beam_az_rot = 90 - AZ_antenna_axis`.

## Test script

All tests are in `tests/test_rotation_investigation.py`. Run with:

    JAX_ENABLE_X64=1 uv run python tests/test_rotation_investigation.py
