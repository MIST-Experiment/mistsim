"""Compute everything raul_comparison_normalization.ipynb shows.

Runs in the project environment (current mistsim and croissant):
  uv run python notebooks/sim_comparisons/normalization/run.py [--force]

Writes, under results_normalization/:
  cache/ms_<variant>_test<k>.npz  mistsim runs (tant, fgnd), in Raul's
                                  LST-sorted row order (git-ignored)
  cache/pix_test<k>.npz           pixel replica of Raul's convolution
  cache/synthetic_convention.npz  the synthetic convention test
  cache/raul.npz                  Raul's waterfalls, LSTs, frequencies
  summary.json                    every number the notebook quotes

Cached runs are reused unless --force. A from-scratch run takes about
30 min on 4 cores and peaks near 11 GB of RAM (the niter = 3 sky
transform).
"""

import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="recompute all")
    args = ap.parse_args()

    # Keep CPU use moderate: other analyses share the machine. Set
    # before jax is first imported.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4",
    )
    import jax

    jax.config.update("jax_enable_x64", True)
    import astropy
    import astropy.units as u
    import croissant as cro
    import healpy as hp
    import numpy as np
    import raul_tools as rt
    import variants as V
    from astropy.time import Time
    from astropy.utils import iers

    import mistsim as ms

    # Raul's epoch (2014) is in the bundled IERS-B table.
    iers.conf.auto_download = False
    repo = HERE.parents[2]
    data = rt.find_data_dir(repo)
    V.CACHE.mkdir(parents=True, exist_ok=True)

    def git(*a):
        return subprocess.run(
            ["git", "-C", str(repo), *a], capture_output=True, text=True
        ).stdout.strip()

    cro_url = json.loads(
        md.distribution("croissant-sim").read_text("direct_url.json")
    )
    prov = {
        "mistsim_commit": git("rev-parse", "--short", "HEAD"),
        "mistsim_src_dirty": bool(git("status", "--porcelain", "src")),
        "croissant": cro_url["vcs_info"].get("requested_revision"),
        "croissant_commit": cro_url["vcs_info"]["commit_id"][:7],
        "jax": jax.__version__,
        "astropy": astropy.__version__,
        "healpy": hp.__version__,
        "python": sys.version.split()[0],
        "lmax": V.LMAX,
        "nside": V.NSIDE,
    }
    print(json.dumps(prov, indent=1))

    def cached(path, fn):
        if path.exists() and not args.force:
            z = np.load(path)
            return {n: z[n] for n in z.files}
        t0 = time.time()
        out = fn()
        np.savez(path, **out)
        print(f"  {path.name}: {time.time() - t0:.0f} s", flush=True)
        return out

    # --- Raul's inputs ---------------------------------------------
    raul, lsts, locs, utc, order, grid_err = {}, {}, {}, {}, {}, {}
    freqs = None
    for k, rel in V.RAUL_FILES.items():
        lst_k, f_k, t_k = rt.read_raul(data / rel)
        freqs = f_k if freqs is None else freqs
        assert np.allclose(f_k, freqs)
        raul[k], lsts[k] = t_k, lst_k
        locs[k] = rt.location(k)
        utc[k], lst_mine, order[k] = rt.raul_times(locs[k].lon.deg)
        grid_err[k] = float(np.max(np.abs(lst_mine[order[k]] - lst_k)))
        assert grid_err[k] < 1e-9
    assert all(np.allclose(utc[k].jd, utc[0].jd) for k in V.TESTS)
    # The notebook reads Raul's data from here: no h5py, no data dir.
    np.savez(
        V.CACHE / "raul.npz",
        freqs=freqs,
        **{f"raul{k}": raul[k] for k in V.TESTS},
        **{f"lst{k}": lsts[k] for k in V.TESTS},
    )

    gain, theta_full, gain_above, theta, phi = rt.feko_beam_full(
        data / "feko_beam.npz", freqs
    )
    beam_check = check_feko(
        np, data / V.FEKO_OUT, gain_above, freqs, {40e6, 80e6, 125e6}
    )
    sky = rt.load_haslam(data / V.HASLAM, V.NSIDE, freqs)
    sky_model = ms.Sky(sky, freqs, sampling="healpix", coord="galactic")

    thd = np.rad2deg(theta_full)
    masks = {
        "le80": (thd <= 80 + 1e-9)[:, None],
        "lt80": (thd < 80 - 1e-9)[:, None],
        "half": np.where(
            thd < 80 - 1e-9, 1.0, np.where(np.isclose(thd, 80), 0.5, 0.0)
        )[:, None],
    }
    gains = {False: gain, True: gain[:, :, (-np.arange(360)) % 360]}

    def times_for(v, k):
        if v["times"] == "raul":
            return utc[k].jd, order[k]
        # The old notebook's mapping: mean LST on 2022-07-17.
        t0 = Time("2022-07-17 00:00", location=locs[k])
        lst_ref = t0.sidereal_time("mean").hour
        t = t0 + (lsts[k] - lst_ref) / 24 * u.sday
        return t.jd, None

    sky_alm = {}

    def sky_alm_for(times_jd, niter):
        # Sky alm in CIRS at times_jd[0]; location-independent.
        key = (round(float(times_jd[0]), 9), niter)
        if key not in sky_alm:
            if niter == 0:
                sky_alm[key] = rt.precompute_sky_alm(
                    sky_model, freqs, times_jd, gain
                )
            else:
                csky = cro.Sky(
                    sky,
                    freqs,
                    sampling="healpix",
                    coord="galactic",
                    niter=niter,
                )
                b = ms.Beam(gain, freqs, sampling="mwss")
                s0 = ms.Simulator(
                    b,
                    sky_model,
                    times_jd,
                    freqs,
                    0.0,
                    0.0,
                    lmax=V.LMAX,
                    Tgnd=0.0,
                )
                sky_alm[key] = csky.compute_alm_eq(world="earth", et=s0.et_ref)
        return sky_alm[key]

    # --- mistsim variants ------------------------------------------
    runs, fgnd = {}, {}
    for v in V.VARIANTS:
        for k in V.tests_of(v):

            def fn(v=v, k=k):
                jd, reorder = times_for(v, k)
                tant, fg = rt.run_mistsim(
                    gains[v["mirror"]],
                    freqs,
                    sky_model,
                    jd,
                    locs[k],
                    horizon=masks[v["mask"]] if k == 0 else None,
                    lmax=v.get("lmax", V.LMAX),
                    normalization=v.get("norm", "above_horizon"),
                    sky_alm=sky_alm_for(jd, v.get("niter", 0)),
                )
                if reorder is not None:
                    tant = tant[reorder]
                return {"tant": tant, "fgnd": fg}

            out = cached(V.cache_path(v["key"], k), fn)
            runs[(k, v["key"])] = out["tant"]
            fgnd[(k, v["key"])] = out["fgnd"]

    # --- pixel replica of Raul's convolution -----------------------
    spl = {
        m: rt.beam_splines(
            gain_above, np.rad2deg(theta), np.rad2deg(phi), mirror=m
        )
        for m in (False, True)
    }
    pix_check = {}
    for k in V.TESTS:

        def pfn(k=k):
            az, el = rt.pixel_altaz(V.NSIDE, locs[k], utc[k][order[k]])
            el_min = 10.0 if k == 0 else 0.0
            return {
                "pix": rt.pixel_convolution(spl[False], sky, az, el, el_min),
                "pix_mirror": rt.pixel_convolution(
                    spl[True], sky, az, el, el_min
                ),
                "check": np.array(
                    rt.check_pixel_convolution(
                        spl[False], sky, az, el, el_min, n=1
                    )
                ),
            }

        out = cached(V.CACHE / f"pix_test{k}.npz", pfn)
        runs[(k, "pix")] = out["pix"]
        runs[(k, "pix_mirror")] = out["pix_mirror"]
        pix_check[f"test{k}"] = float(out["check"])

    # --- synthetic convention test ---------------------------------
    syn = cached(
        V.CACHE / "synthetic_convention.npz",
        lambda: synthetic(np, rt, ms, data, V, locs, utc, order),
    )

    # --- numbers ---------------------------------------------------
    def stats(k, name):
        d = runs[(k, name)] - raul[k]
        s = rt.residual_stats(d)
        i, j = np.unravel_index(np.argmax(np.abs(d)), d.shape)
        s["argmax_lst_h"] = float(lsts[k][i])
        s["argmax_freq_MHz"] = float(freqs[j])
        s["rel_mean_abs"] = float(np.mean(np.abs(d)) / np.mean(raul[k]))
        # Effective time lag: least-squares d ~ a * dT/dLST.
        dT = np.gradient(raul[k], lsts[k] * 3600, axis=0)
        a = float(np.sum(d * dT) / np.sum(dT * dT))
        s["lag_fit_s"] = a
        s["rms_after_lag_fit_K"] = float(np.sqrt(np.mean((d - a * dT) ** 2)))
        return s

    residual = {f"test{k}/{name}": stats(k, name) for (k, name) in runs}
    f0 = fgnd[(0, "epoch")][None, :]
    algebra = {
        "full_sphere_eq_above_x_1_minus_fgnd_K": float(
            np.max(
                np.abs(
                    runs[(0, "full_sphere")] - runs[(0, "epoch")] * (1 - f0)
                )
            )
        ),
        "default_eq_above_x_1_minus_fgnd_plus_300fgnd_K": float(
            np.max(
                np.abs(
                    runs[(0, "default")]
                    - (runs[(0, "epoch")] * (1 - f0) + 300 * f0)
                )
            )
        ),
    }
    eff_r = raul[1] - raul[0]
    blockage = {}
    for nm0, nm1 in (("epoch", "epoch"), ("edge", "mirror"), ("pix", "pix")):
        d = (runs[(1, nm1)] - runs[(0, nm0)]) - eff_r
        blockage[nm0] = rt.residual_stats(d)
    blockage["effect_rms_K"] = float(np.sqrt(np.mean(eff_r**2)))
    split = {
        f"test{k}": {
            "best_variant": V.BEST[k],
            "mistsim_minus_replica": rt.residual_stats(
                runs[(k, V.BEST[k])] - runs[(k, "pix")]
            ),
            "replica_minus_raul": rt.residual_stats(
                runs[(k, "pix")] - raul[k]
            ),
        }
        for k in V.TESTS
    }
    ifr = [0, 40, len(freqs) - 1]
    fgnd_by_mask = {
        nm: {f"{freqs[i]:.0f}MHz": float(fgnd[(0, nm)][i]) for i in ifr}
        for nm in ("mirror", "edge", "edge_lt80")
    }
    conv = {}
    for key in syn:
        if not key.startswith("ms_"):
            continue
        _, rot, frame = key.split("_")
        for mir in (0, 1):
            for mrot in (0, 1):
                d = syn[key] - syn[f"pix_{rot}_mir{mir}_mrot{mrot}"]
                conv[f"{frame}/{rot}/mir{mir}/mrot{mrot}"] = {
                    "mean_abs_K": float(np.mean(np.abs(d))),
                    "max_abs_K": float(np.max(np.abs(d))),
                }
    conv["mean_tant_K"] = float(np.mean(syn["ms_rot0_beam"]))

    summary = {
        "provenance": prov,
        "raul_time_grid": {
            "utc_start": utc[0][0].isot,
            "step_s": rt.RAUL_STEP_S,
            "n": int(rt.RAUL_NSTEP),
            "lst": "apparent, IAU2006A, DUT1 = 0, sorted",
            "max_abs_lst_error_h": {
                f"test{k}": e for k, e in grid_err.items()
            },
        },
        "beam_check": beam_check,
        "pixel_fast_vs_direct_K": pix_check,
        "fgnd_test0_by_mask": fgnd_by_mask,
        "residual_mistsim_minus_raul": residual,
        "normalisation_algebra": algebra,
        "blockage_effect_residual": blockage,
        "residual_split": split,
        "convention_test": conv,
    }
    V.SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {V.SUMMARY}")


def check_feko(np, path, gain_above, freqs, want_hz):
    """feko_beam.npz against the FEKO .out file, and beam symmetry.

    Reads the total gain (dB column) on theta 0..90, phi 0..359 at
    want_hz; the parser scans the 470 MB file once.
    """
    out, cur, rows, reading = {}, None, [], False
    with open(path) as f:
        for line in f:
            if "Frequency in Hz" in line and "FREQ =" in line:
                cur = float(line.split("=")[1])
            elif cur in want_hz and "THETA    PHI" in line:
                reading, rows = True, []
            elif reading:
                p = line.split()
                if len(p) < 10:
                    if rows:
                        out[cur], reading = np.array(rows), False
                        if len(out) == len(want_hz):
                            break
                    continue
                rows.append([float(p[0]), float(p[1]), float(p[8])])
    check = {}
    for fr, arr in sorted(out.items()):
        th, ph, gdb = arr.T
        g = np.zeros((91, 360))
        g[th.astype(int), ph.astype(int)] = np.where(
            gdb < -900, 0.0, 10 ** (gdb / 10)
        )
        i = int(np.argmin(np.abs(freqs - fr / 1e6)))
        check[f"{fr / 1e6:.0f}MHz"] = {
            "max_rel_diff_npz_vs_out": float(
                np.max(np.abs(gain_above[i] - g)) / g.max()
            ),
            "asym_phi_plus_180": float(
                np.max(np.abs(g - np.roll(g, 180, 1))) / g.max()
            ),
            "asym_phi_minus_phi": float(
                np.max(np.abs(g - g[:, (-np.arange(360)) % 360])) / g.max()
            ),
        }
    return check


def synthetic(np, rt, ms, data, V, locs, utc, order):
    """Asymmetric beam + one-sided mask at 60 MHz, Test 1's site.

    mistsim at beam_az_rot 0 and 40 deg, with the mask in either
    horizon frame, against the pixel-domain sum under four hypotheses
    (beam azimuth phi_b = AZ - rot or -(AZ - rot); mask rotating with
    the beam or fixed on the ground). Every 8th of Raul's times.
    """
    f1 = np.array([60.0])
    sky1 = rt.load_haslam(data / V.HASLAM, V.NSIDE, f1)
    sel = order[1][::8]
    az, el = rt.pixel_altaz(V.NSIDE, locs[1], utc[1][sel])
    th = np.deg2rad(np.arange(181.0))
    ph = np.deg2rad(np.arange(360.0))
    g = rt.synthetic_beam(th[:, None], ph[None, :])[None]
    h = rt.synthetic_mask(th[:, None], ph[None, :])
    sm = ms.Sky(sky1, f1, sampling="healpix", coord="galactic")
    out = {}
    for rot in (0.0, 40.0):
        for frame in ("beam", "topocentric"):
            tant, _ = rt.run_mistsim(
                g,
                f1,
                sm,
                utc[1].jd,
                locs[1],
                horizon=h,
                beam_az_rot=rot,
                lmax=V.LMAX,
                horizon_frame=frame,
            )
            out[f"ms_rot{rot:.0f}_{frame}"] = tant[sel, 0]
        for mir in (False, True):
            for mrot in (False, True):
                out[f"pix_rot{rot:.0f}_mir{int(mir)}_mrot{int(mrot)}"] = (
                    rt.analytic_pixel_convolution(
                        sky1[0], az, el, rot, mir, mrot
                    )
                )
    return out


if __name__ == "__main__":
    main()
