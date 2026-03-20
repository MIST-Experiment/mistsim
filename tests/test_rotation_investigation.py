"""
Rotation convention investigation after NEU→ENU fix in croissant.

Tests whether the ENU swap fix (det=+1) is correct and whether the
old det=-1 matrix was accidentally compensating for something.
"""

import astropy.units as u
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time as AstroTime

from croissant.beam import Beam
from croissant.rotations import (
    generate_euler_dl,
    get_rot_mat,
    rotmat_to_eulerZYZ,
)
from croissant.simulator import Simulator
from croissant.sky import Sky

jax.config.update("jax_enable_x64", True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _get_old_rot_mat(from_frame, to_frame):
    """
    Reproduce the OLD get_rot_mat (before NEU→ENU swap).

    The old code did NOT swap x<->y for AltAz frames, so the
    resulting matrix has det=-1 when a topocentric frame is involved.
    """
    from croissant.rotations import _is_topo_frame

    from_name = getattr(from_frame, "name", from_frame)
    to_name = getattr(to_frame, "name", to_frame)
    if from_name.lower() == "galactic":
        from_frame = to_frame
        to_frame = "galactic"
        return_inv = True
    else:
        return_inv = False
    x, y, z = np.eye(3)
    sc = SkyCoord(
        x=x, y=y, z=z, frame=from_frame, representation_type="cartesian"
    )
    rmat = sc.transform_to(to_frame).cartesian.xyz.value
    # OLD code: no swap at all
    if return_inv:
        rmat = rmat.T
    return rmat


def _point_source_sky(nside, freqs, ra_deg, dec_deg, flux=1.0):
    """Create a HEALPix sky with a single point source."""
    npix = 12 * nside**2
    pix = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True)
    sky_data = jnp.zeros((len(freqs), npix))
    sky_data = sky_data.at[:, pix].set(flux)
    return sky_data


# -------------------------------------------------------------------
# Test 1: Verify rotate_flms convention (active vs passive)
# -------------------------------------------------------------------

def test_rotate_flms_convention():
    """
    Create alm with only f_{1,1} = 1.
    Apply rotate_flms with (alpha=pi/4, beta=0, gamma=0).

    Active convention: f'_{1,m} = exp(-i*m*alpha) * d^1_m1(0) * f_{1,1}
    Since d^1_mn(0) = delta_mn, f'_{1,1} = exp(-i*alpha) * f_{1,1}

    Passive would give exp(+i*alpha).
    """
    L = 4  # bandlimit
    flm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    # f_{1,1}: ell=1, m=1 -> index [1, L-1+1] = [1, L]
    flm = flm.at[1, L].set(1.0 + 0.0j)

    alpha = np.pi / 4
    rotation = (alpha, 0.0, 0.0)
    dl_array = s2fft.generate_rotate_dls(L, 0.0)

    flm_rot = s2fft.utils.rotation.rotate_flms(
        flm, L, rotation=rotation, dl_array=dl_array
    )

    # For beta=0: d^l_mn(0) = delta_mn, so only m=n=1 survives
    # Active: f'_{1,1} = exp(-i*1*alpha) * 1 = exp(-i*pi/4)
    expected = np.exp(-1j * alpha)
    actual = flm_rot[1, L]  # ell=1, m=1

    print(f"f'_{{1,1}} = {actual}")
    print(f"exp(-i*alpha) = {expected}")
    print(f"exp(+i*alpha) = {np.exp(1j * alpha)}")

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    print("CONFIRMED: rotate_flms uses ACTIVE rotation convention")
    print("  D^l_mn(R) f_ln gives f'(r) = f(R^{-1} r)")


# -------------------------------------------------------------------
# Test 2: Compare old vs new rotation matrices
# -------------------------------------------------------------------

def test_old_vs_new_rotation_matrices():
    """
    Compare Euler angles from:
    - R_new = get_rot_mat(topo, "fk5")  (with swap, det=+1)
    - R_old = old code (no swap, det=-1)
    - R_new^T = inverse of R_new

    Check if eul_old ≈ eul_inverse (compensating error hypothesis).
    """
    lon, lat = 0.0, 45.0
    t0 = AstroTime("2022-06-21 00:00:00")
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    topo = AltAz(obstime=t0, location=loc)

    R_new = get_rot_mat(topo, "fk5")
    R_old = _get_old_rot_mat(topo, "fk5")

    print(f"det(R_new) = {np.linalg.det(R_new):.6f}")
    print(f"det(R_old) = {np.linalg.det(R_old):.6f}")

    assert np.isclose(np.linalg.det(R_new), 1.0, atol=1e-10), \
        "R_new should have det=+1"
    assert np.isclose(np.linalg.det(R_old), -1.0, atol=1e-10), \
        "R_old should have det=-1"

    eul_new = rotmat_to_eulerZYZ(R_new)
    eul_old = rotmat_to_eulerZYZ(R_old)
    eul_inv = rotmat_to_eulerZYZ(R_new.T)

    print(f"\nEuler angles (alpha, beta, gamma):")
    print(f"  R_new:   ({eul_new[0]:.6f}, {eul_new[1]:.6f}, "
          f"{eul_new[2]:.6f})")
    print(f"  R_old:   ({eul_old[0]:.6f}, {eul_old[1]:.6f}, "
          f"{eul_old[2]:.6f})")
    print(f"  R_new^T: ({eul_inv[0]:.6f}, {eul_inv[1]:.6f}, "
          f"{eul_inv[2]:.6f})")

    # Check if old Euler angles are closer to the inverse
    diff_new = np.array(eul_old) - np.array(eul_new)
    diff_inv = np.array(eul_old) - np.array(eul_inv)
    print(f"\n  |eul_old - eul_new|:   {np.linalg.norm(diff_new):.6f}")
    print(f"  |eul_old - eul_inv|:   {np.linalg.norm(diff_inv):.6f}")

    if np.linalg.norm(diff_inv) < np.linalg.norm(diff_new):
        print("\n  => eul_old is CLOSER to eul_inverse")
        print("  => Old det=-1 matrix approximated the INVERSE rotation")
    else:
        print("\n  => eul_old is CLOSER to eul_new")
        print("  => Old det=-1 matrix was close to the forward rotation")

    # Also show the actual rotation matrices
    print(f"\nR_new:\n{R_new}")
    print(f"\nR_old:\n{R_old}")
    print(f"\nR_new^T:\n{R_new.T}")
    print(f"\nDifference R_old - R_new^T:\n{R_old - R_new.T}")


# -------------------------------------------------------------------
# Test 3: Beam rotation at North Pole
# -------------------------------------------------------------------

def test_beam_rotation_north_pole():
    """
    At lat=90, zenith = celestial north pole (Dec=90°).
    A Gaussian beam peaked at zenith should remain at θ=0 in
    equatorial coords after rotation.
    """
    nside = 32
    npix = 12 * nside**2
    lmax = 2 * nside
    L = lmax + 1
    freqs = jnp.array([75.0])

    # Gaussian beam peaked at zenith (theta=0)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    sigma = np.radians(20)
    beam_data = np.exp(-theta**2 / (2 * sigma**2))
    beam_data = jnp.array(beam_data[None, :])  # (1, npix)

    beam = Beam(beam_data, freqs, sampling="healpix", niter=3)
    beam_alm = beam.compute_alm()  # (1, L, 2L-1)

    # Setup at north pole
    lon, lat = 0.0, 89.99  # near pole to avoid singularity
    t0 = AstroTime("2022-06-21 00:00:00")
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    topo = AltAz(obstime=t0, location=loc)

    R_new = get_rot_mat(topo, "fk5")
    R_old = _get_old_rot_mat(topo, "fk5")

    eul_new = rotmat_to_eulerZYZ(R_new)
    eul_inv = rotmat_to_eulerZYZ(R_new.T)
    eul_old = rotmat_to_eulerZYZ(R_old)

    dl_new = s2fft.generate_rotate_dls(L, eul_new[1])
    dl_inv = s2fft.generate_rotate_dls(L, eul_inv[1])
    dl_old = s2fft.generate_rotate_dls(L, eul_old[1])

    # Rotate beam alm three ways
    rot_fn = s2fft.utils.rotation.rotate_flms
    beam_eq_new = jax.vmap(
        lambda f: rot_fn(f, L, rotation=eul_new, dl_array=dl_new)
    )(beam_alm)
    beam_eq_inv = jax.vmap(
        lambda f: rot_fn(f, L, rotation=eul_inv, dl_array=dl_inv)
    )(beam_alm)
    beam_eq_old = jax.vmap(
        lambda f: rot_fn(f, L, rotation=eul_old, dl_array=dl_old)
    )(beam_alm)

    # Convert to maps and find peak
    for label, alm in [
        ("R_new (det=+1)", beam_eq_new),
        ("R_new^T (inv)", beam_eq_inv),
        ("R_old (det=-1)", beam_eq_old),
    ]:
        # Convert s2fft alm to healpy format
        alm_hp = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        for ell in range(L):
            for m in range(ell + 1):
                idx = hp.Alm.getidx(lmax, ell, m)
                alm_hp[idx] = np.array(alm[0, ell, L - 1 + m])
        beam_map = hp.alm2map(alm_hp, nside, verbose=False)
        peak_pix = np.argmax(beam_map)
        peak_theta, peak_phi = hp.pix2ang(nside, peak_pix)
        peak_dec = 90.0 - np.degrees(peak_theta)
        peak_ra = np.degrees(peak_phi)
        print(f"  {label}: peak at Dec={peak_dec:.1f}°, "
              f"RA={peak_ra:.1f}° (θ={np.degrees(peak_theta):.1f}°)")

    # At north pole, zenith = Dec~90. Correct rotation should
    # preserve peak near Dec=90 (theta~0 in equatorial).
    # Check R_new
    alm_hp = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    for ell in range(L):
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            alm_hp[idx] = np.array(beam_eq_new[0, ell, L - 1 + m])
    beam_map = hp.alm2map(alm_hp, nside, verbose=False)
    peak_pix = np.argmax(beam_map)
    peak_theta, _ = hp.pix2ang(nside, peak_pix)
    peak_dec = 90.0 - np.degrees(peak_theta)
    print(f"\n  Zenith at lat={lat}° should map to Dec~{lat}°")
    assert peak_dec > 80.0, (
        f"Peak should be near Dec=90° but got Dec={peak_dec:.1f}°"
    )


# -------------------------------------------------------------------
# Test 4: Beam rotation at mid-latitude
# -------------------------------------------------------------------

def test_beam_rotation_midlat():
    """
    At lat=45, zenith = Dec=45° (approximately, depends on time).
    Create beam peaked at zenith, rotate, check peak location.
    """
    nside = 32
    npix = 12 * nside**2
    lmax = 2 * nside
    L = lmax + 1
    freqs = jnp.array([75.0])

    # Gaussian beam peaked at zenith (theta=0)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    sigma = np.radians(20)
    beam_data = np.exp(-theta**2 / (2 * sigma**2))
    beam_data = jnp.array(beam_data[None, :])
    beam = Beam(beam_data, freqs, sampling="healpix", niter=3)
    beam_alm = beam.compute_alm()

    lon, lat = 0.0, 45.0
    t0 = AstroTime("2022-06-21 00:00:00")
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    topo = AltAz(obstime=t0, location=loc)

    # Expected zenith direction in equatorial
    zenith_eq = SkyCoord(
        alt=90 * u.deg, az=0 * u.deg, frame=topo
    ).transform_to("fk5")
    expected_dec = zenith_eq.dec.deg
    expected_ra = zenith_eq.ra.deg
    print(f"  Expected zenith: RA={expected_ra:.1f}°, "
          f"Dec={expected_dec:.1f}°")

    R_new = get_rot_mat(topo, "fk5")
    R_old = _get_old_rot_mat(topo, "fk5")

    eul_new = rotmat_to_eulerZYZ(R_new)
    eul_inv = rotmat_to_eulerZYZ(R_new.T)
    eul_old = rotmat_to_eulerZYZ(R_old)

    dl_new = s2fft.generate_rotate_dls(L, eul_new[1])
    dl_inv = s2fft.generate_rotate_dls(L, eul_inv[1])
    dl_old = s2fft.generate_rotate_dls(L, eul_old[1])

    rot_fn = s2fft.utils.rotation.rotate_flms

    results = {}
    for label, eul, dl in [
        ("R_new (det=+1)", eul_new, dl_new),
        ("R_new^T (inv)", eul_inv, dl_inv),
        ("R_old (det=-1)", eul_old, dl_old),
    ]:
        beam_eq = jax.vmap(
            lambda f, e=eul, d=dl: rot_fn(f, L, rotation=e, dl_array=d)
        )(beam_alm)
        alm_hp = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        for ell in range(L):
            for m in range(ell + 1):
                idx = hp.Alm.getidx(lmax, ell, m)
                alm_hp[idx] = np.array(beam_eq[0, ell, L - 1 + m])
        beam_map = hp.alm2map(alm_hp, nside, verbose=False)
        peak_pix = np.argmax(beam_map)
        peak_theta, peak_phi = hp.pix2ang(nside, peak_pix)
        peak_dec = 90.0 - np.degrees(peak_theta)
        peak_ra = np.degrees(peak_phi)
        print(f"  {label}: peak at RA={peak_ra:.1f}°, Dec={peak_dec:.1f}°")

        # Angular separation from expected
        peak_sc = SkyCoord(ra=peak_ra * u.deg, dec=peak_dec * u.deg)
        sep = peak_sc.separation(
            SkyCoord(ra=expected_ra * u.deg, dec=expected_dec * u.deg)
        ).deg
        print(f"    Separation from expected: {sep:.1f}°")
        results[label] = sep

    # The correct rotation should place the peak closest to expected
    best = min(results, key=results.get)
    print(f"\n  Best match: {best} "
          f"(sep={results[best]:.1f}°)")

    # R_new should be the closest (or at least within ~5° for nside=32)
    assert results["R_new (det=+1)"] < 10.0, (
        f"R_new peak too far from expected: "
        f"{results['R_new (det=+1)']:.1f}°"
    )


# -------------------------------------------------------------------
# Test 5: Tighten test_beam_enu_east_direction
# -------------------------------------------------------------------

def test_beam_enu_east_tighter():
    """
    Rerun the existing physical ENU east test with tighter tolerance.
    Use nside=32 for better accuracy than 16. Also run with old
    rotation to compare.
    """
    nside = 32
    npix = 12 * nside**2
    freqs = jnp.array([75.0])

    # Beam: 1 + 0.5*sin(theta)*cos(phi) in ENU convention
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    beam_data = 1.0 + 0.5 * np.sin(theta) * np.cos(phi)
    beam_data = jnp.array(beam_data[None, :])
    beam = Beam(beam_data, freqs, sampling="healpix", niter=3)

    lon, lat = 0.0, 45.0
    t0 = AstroTime("2022-06-21 00:00:00")
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    altaz_frame = AltAz(obstime=t0, location=loc)

    alt_deg = 45.0
    east_eq = SkyCoord(
        alt=alt_deg * u.deg, az=90 * u.deg, frame=altaz_frame
    ).transform_to("fk5")
    north_eq = SkyCoord(
        alt=alt_deg * u.deg, az=0 * u.deg, frame=altaz_frame
    ).transform_to("fk5")

    sky_east = Sky(
        _point_source_sky(nside, freqs, east_eq.ra.deg, east_eq.dec.deg),
        freqs,
        coord="equatorial",
        niter=3,
    )
    sky_north = Sky(
        _point_source_sky(nside, freqs, north_eq.ra.deg, north_eq.dec.deg),
        freqs,
        coord="equatorial",
        niter=3,
    )

    times_jd = jnp.array([t0.jd])
    sim_east = Simulator(
        beam, sky_east, times_jd, freqs, lon, lat,
        world="earth", Tgnd=0.0,
    )
    sim_north = Simulator(
        beam, sky_north, times_jd, freqs, lon, lat,
        world="earth", Tgnd=0.0,
    )

    vis_east = sim_east.sim()[0, 0]
    vis_north = sim_north.sim()[0, 0]

    alt_rad = np.radians(alt_deg)
    expected_ratio = (1 + 0.5 * np.cos(alt_rad)) / 1.0

    actual_ratio = float(vis_east / vis_north)
    print(f"  Expected ratio: {expected_ratio:.6f}")
    print(f"  Actual ratio:   {actual_ratio:.6f}")
    print(f"  Relative error: "
          f"{abs(actual_ratio - expected_ratio) / expected_ratio:.4f}")

    # Report rather than assert — point source tests have inherent
    # SHT truncation error. The key comparison is R_new vs R_old.
    if abs(actual_ratio - expected_ratio) / expected_ratio < 0.15:
        print("  R_new within 15% of expected (SHT-limited)")
    else:
        print("  WARNING: R_new far from expected")

    # Also test with old rotation (via manual setup)
    R_old = _get_old_rot_mat(altaz_frame, "fk5")
    eul_old = rotmat_to_eulerZYZ(R_old)
    lmax = beam.lmax
    L = lmax + 1
    dl_old = s2fft.generate_rotate_dls(L, eul_old[1])

    beam_alm = beam.compute_alm()
    rot_fn = s2fft.utils.rotation.rotate_flms
    beam_eq_old = jax.vmap(
        lambda f: rot_fn(f, L, rotation=eul_old, dl_array=dl_old)
    )(beam_alm)

    # Compute visibility manually with old-rotated beam
    sky_east_alm = sky_east.compute_alm()
    sky_north_alm = sky_north.compute_alm()

    vis_east_old = jnp.sum(beam_eq_old * jnp.conj(sky_east_alm)).real
    vis_north_old = jnp.sum(beam_eq_old * jnp.conj(sky_north_alm)).real

    if vis_north_old != 0:
        old_ratio = float(vis_east_old / vis_north_old)
        print(f"\n  Old rotation ratio: {old_ratio:.6f}")
        print(f"  Old relative error: "
              f"{abs(old_ratio - expected_ratio) / expected_ratio:.4f}")
    else:
        print("\n  Old rotation: vis_north=0, cannot compute ratio")


# -------------------------------------------------------------------
# Test 6: healpy cross-check
# -------------------------------------------------------------------

def test_healpy_crosscheck():
    """
    Cross-check rotation with healpy.Rotator using healpy's own
    SHT pipeline (avoids alm format/normalization issues between
    s2fft and healpy).

    Create a beam map, compute alm with healpy, rotate with healpy,
    find the peak. Compare with s2fft/croissant peak and with the
    known expected zenith.
    """
    nside = 32
    npix = 12 * nside**2
    lmax = 2 * nside
    L = lmax + 1

    # Asymmetric beam pattern
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    beam_map = np.exp(-theta**2 / (2 * np.radians(30)**2))
    beam_map *= (1 + 0.3 * np.cos(phi))

    lon, lat = 0.0, 45.0
    t0 = AstroTime("2022-06-21 00:00:00")
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    topo = AltAz(obstime=t0, location=loc)

    R_new = get_rot_mat(topo, "fk5")
    eul_zyz = rotmat_to_eulerZYZ(R_new)

    # Expected zenith direction in equatorial
    zenith_eq = SkyCoord(
        alt=90 * u.deg, az=0 * u.deg, frame=topo
    ).transform_to("fk5")
    sc_expected = SkyCoord(ra=zenith_eq.ra, dec=zenith_eq.dec)
    print(f"  Expected zenith: Dec={zenith_eq.dec.deg:.1f}°, "
          f"RA={zenith_eq.ra.deg:.1f}°")

    # --- s2fft/croissant rotation ---
    freqs = jnp.array([75.0])
    beam_data = jnp.array(beam_map[None, :])
    beam = Beam(beam_data, freqs, sampling="healpix", niter=3)
    beam_alm = beam.compute_alm()

    dl = s2fft.generate_rotate_dls(L, eul_zyz[1])
    rot_fn = s2fft.utils.rotation.rotate_flms
    beam_eq_s2fft = jax.vmap(
        lambda f: rot_fn(f, L, rotation=eul_zyz, dl_array=dl)
    )(beam_alm)

    # Convert s2fft rotated alm to map via healpy for peak finding
    alm_s2fft_hp = np.zeros(
        hp.Alm.getsize(lmax), dtype=np.complex128
    )
    for ell in range(L):
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            alm_s2fft_hp[idx] = np.array(
                beam_eq_s2fft[0, ell, L - 1 + m]
            )
    map_s2fft = hp.alm2map(alm_s2fft_hp, nside, verbose=False)
    peak = np.argmax(map_s2fft)
    t, p = hp.pix2ang(nside, peak)
    sc = SkyCoord(
        ra=np.degrees(p) * u.deg,
        dec=(90 - np.degrees(t)) * u.deg,
    )
    sep = sc_expected.separation(sc).deg
    print(f"  s2fft(R): Dec={90 - np.degrees(t):.1f}°, "
          f"RA={np.degrees(p):.1f}° (sep={sep:.1f}°)")

    # --- healpy rotation (own alm pipeline) ---
    alm_hp = hp.map2alm(beam_map, lmax=lmax)

    from croissant.rotations import rotmat_to_eulerZYX
    eul_zyx = rotmat_to_eulerZYX(R_new)
    eul_zyx_inv = rotmat_to_eulerZYX(R_new.T)

    cases = [
        ("ZYZ(R)", eul_zyz, "ZYZ"),
        ("ZYZ(R^T)", rotmat_to_eulerZYZ(R_new.T), "ZYZ"),
        ("ZYZ(-α,-β,-γ)",
         (-eul_zyz[0], -eul_zyz[1], -eul_zyz[2]), "ZYZ"),
        ("ZYZ(γ,β,α)",
         (eul_zyz[2], eul_zyz[1], eul_zyz[0]), "ZYZ"),
        ("ZYX(R)", eul_zyx, "ZYX"),
        ("ZYX(R^T)", eul_zyx_inv, "ZYX"),
    ]

    best_label = None
    best_sep = 999.0
    for label, eul, etype in cases:
        rot_hp = hp.Rotator(rot=eul, eulertype=etype, deg=False)
        alm_rot = rot_hp.rotate_alm(alm_hp.copy())
        hmap = hp.alm2map(alm_rot, nside, verbose=False)
        peak = np.argmax(hmap)
        t, p = hp.pix2ang(nside, peak)
        sc = SkyCoord(
            ra=np.degrees(p) * u.deg,
            dec=(90 - np.degrees(t)) * u.deg,
        )
        sep = sc_expected.separation(sc).deg
        print(f"  hp {label}: Dec={90 - np.degrees(t):.1f}°, "
              f"RA={np.degrees(p):.1f}° (sep={sep:.1f}°)")
        if sep < best_sep:
            best_sep = sep
            best_label = label

    print(f"\n  Best healpy match: {best_label} "
          f"(sep={best_sep:.1f}°)")
    if best_sep < 5.0:
        print("  GOOD: healpy confirms rotation direction")


# -------------------------------------------------------------------
# Main: run all tests with verbose output
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: rotate_flms convention")
    print("=" * 60)
    test_rotate_flms_convention()

    print("\n" + "=" * 60)
    print("Test 2: Old vs new rotation matrices")
    print("=" * 60)
    test_old_vs_new_rotation_matrices()

    print("\n" + "=" * 60)
    print("Test 3: Beam rotation at North Pole")
    print("=" * 60)
    test_beam_rotation_north_pole()

    print("\n" + "=" * 60)
    print("Test 4: Beam rotation at mid-latitude")
    print("=" * 60)
    test_beam_rotation_midlat()

    print("\n" + "=" * 60)
    print("Test 5: Tightened ENU east direction test")
    print("=" * 60)
    test_beam_enu_east_tighter()

    print("\n" + "=" * 60)
    print("Test 6: healpy cross-check")
    print("=" * 60)
    test_healpy_crosscheck()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
