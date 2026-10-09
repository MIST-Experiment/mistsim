"""Simulate Raul's three tests with whichever mistsim is installed.

Runs inside each version's own environment, so it uses only the API
every version since mistsim ff99a9d (croissant 5.0.0) shares:
``Sky(data, freqs, sampling, coord)``, ``Beam(data, freqs, sampling,
horizon, beam_az_rot)``, ``Simulator(beam, sky, times_jd, freqs, lon,
lat, alt, lmax, Tgnd)``, ``sim()``, ``beam.compute_fgnd()`` and
``croissant.simulator.correct_ground_loss``. It imports nothing from
this repo, so an old environment never sees new code.

Normalisation is Raul's: simulate with Tgnd = 0, then divide by the
above-horizon beam integral via ``correct_ground_loss(T, fgnd, 0)``.

Usage:
  <env python> simulate.py INPUTS OUT --beam-az-rot DEG [--lmax 100]
"""

import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def provenance(mistsim_module):
    """What ran: package versions and install sources."""

    def direct_url(dist):
        try:
            raw = md.distribution(dist).read_text("direct_url.json")
        except md.PackageNotFoundError:
            return None
        return json.loads(raw) if raw else None

    src = Path(mistsim_module.__file__).resolve().parent
    git = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(src), "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "mistsim_path": str(src),
        "mistsim_commit": git,
        "mistsim_src_dirty": bool(dirty),
        "croissant_version": md.version("croissant-sim"),
        "croissant_source": direct_url("croissant-sim"),
        "s2fft_version": md.version("s2fft"),
        "s2fft_source": direct_url("s2fft"),
        "jax": md.version("jax"),
        "jaxlib": md.version("jaxlib"),
        "numpy": md.version("numpy"),
        "python": platform.python_version(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--beam-az-rot", type=float, required=True)
    ap.add_argument("--lmax", type=int, default=100)
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
    import croissant as cro
    import numpy as np

    import mistsim as ms

    z = np.load(args.inputs)
    freqs = z["freqs"]
    sky = ms.Sky(z["sky"], freqs, sampling="healpix", coord="galactic")
    horizons = {0: z["horizon_le80"], 1: None, 2: None}

    out = {"freqs": freqs, "beam_az_rot": args.beam_az_rot}
    seconds = {}
    for k, horizon in horizons.items():
        t0 = time.time()
        beam = ms.Beam(
            z["gain"],
            freqs,
            sampling="mwss",
            horizon=horizon,
            beam_az_rot=args.beam_az_rot,
        )
        sim = ms.Simulator(
            beam,
            sky,
            z[f"jd{k}"],
            freqs,
            float(z[f"lon{k}"]),
            float(z[f"lat{k}"]),
            alt=float(z[f"alt{k}"]),
            lmax=args.lmax,
            Tgnd=0.0,
        )
        tant = sim.sim()
        fgnd = sim.beam.compute_fgnd()
        tant = cro.simulator.correct_ground_loss(tant, fgnd, 0.0)
        # UTC order -> Raul's LST-sorted rows.
        out[f"test{k}"] = np.asarray(tant)[z[f"order{k}"]]
        out[f"fgnd{k}"] = np.asarray(fgnd)
        seconds[k] = round(time.time() - t0, 1)
        print(f"test {k}: {seconds[k]:.0f} s", flush=True)

    prov = provenance(ms)
    prov.update(lmax=args.lmax, seconds=seconds)
    out["provenance"] = json.dumps(prov)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
