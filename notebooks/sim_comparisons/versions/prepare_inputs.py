"""Write the inputs every version simulates, once, from the project env.

Old environments then need only numpy, jax, croissant and mistsim:
no astropy time handling, no file formats, and identical inputs.

Writes ``results_versions/inputs.npz`` (git-ignored) with
  freqs           (Nf,) MHz
  sky             (Nf, Npix) K, Haslam (Remazeilles 2014) at nside 128,
                  galactic RING, scaled with beta = -2.55 above T_CMB
  gain            (Nf, 181, 360) FEKO total gain on the 1-deg mwss
                  grid, zero below the horizon
  horizon_le80    (181, 1) bool, theta <= 80 deg (Test 0)
  jd{k}           Raul's UTC sample times as JD, in UTC order
  order{k}        argsort of their apparent LST: tant[order] is in
                  Raul's LST-sorted row order
  lst{k}, raul{k} Raul's LSTs [h] and antenna temperature [K]
  lon{k}, lat{k}, alt{k}  site (deg, deg, m)

Usage: uv run python notebooks/sim_comparisons/versions/prepare_inputs.py
"""

import sys
from pathlib import Path

import numpy as np
from astropy.utils import iers

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

# Raul's epoch (2014) is in the bundled IERS-B table; never download.
iers.conf.auto_download = False

NSIDE = 128
RAUL_FILES = {
    0: "20260215_for_christian/antenna_temperature_20260215_test1.hdf5",
    1: "20260220_for_christian/antenna_temperature_20260220_test1.hdf5",
    2: "20260220_for_christian/antenna_temperature_20260220_test2.hdf5",
}


def main():
    import raul_tools as rt
    import registry

    repo = HERE.parents[2]
    data = rt.find_data_dir(repo)
    out = {}
    freqs = None
    for k, rel in RAUL_FILES.items():
        lst, f, tant = rt.read_raul(data / rel)
        if freqs is None:
            freqs = f
        if not np.allclose(f, freqs):
            raise ValueError(f"test {k}: frequency grid differs")
        loc = rt.location(k)
        utc, lst_mine, order = rt.raul_times(loc.lon.deg)
        err = np.max(np.abs(lst_mine[order] - lst))
        if err > 1e-6:
            raise ValueError(f"test {k}: rebuilt LST off by {err:.1e} h")
        out[f"jd{k}"] = utc.jd
        out[f"order{k}"] = order
        out[f"lst{k}"] = lst
        out[f"raul{k}"] = tant
        out[f"lon{k}"] = loc.lon.deg
        out[f"lat{k}"] = loc.lat.deg
        out[f"alt{k}"] = loc.height.to_value("m")
        print(f"test {k}: max |LST rebuilt - Raul| = {err:.1e} h")

    gain, theta_full, *_ = rt.feko_beam_full(data / "feko_beam.npz", freqs)
    out["freqs"] = freqs
    out["gain"] = gain
    out["horizon_le80"] = (np.rad2deg(theta_full) <= 80 + 1e-9)[:, None]
    out["sky"] = rt.load_haslam(
        data / "20260215_for_christian/haslam408_ds_Remazeilles2014.fits",
        NSIDE,
        freqs,
    )
    registry.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(registry.INPUTS, **out)
    print(f"wrote {registry.INPUTS}")


if __name__ == "__main__":
    main()
