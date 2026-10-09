"""Run every version in registry.RUNS, each in its own environment.

Prerequisites: ``bash setup_envs.sh`` (the environments) and
``prepare_inputs.py`` (the shared inputs). Runs whose result file
exists are skipped unless named with --force.

Usage:
  python run_all.py            # all missing runs
  python run_all.py dev3 dev4  # only these
  python run_all.py --force dev4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    import registry

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", help="run keys (default: all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    known = [r["key"] for r in registry.RUNS]
    unknown = set(args.runs) - set(known)
    if unknown:
        ap.error(f"unknown runs {sorted(unknown)}; known: {known}")
    if not registry.INPUTS.exists():
        sys.exit(f"{registry.INPUTS} missing: run prepare_inputs.py")

    for run in registry.RUNS:
        if args.runs and run["key"] not in args.runs:
            continue
        out = registry.result_path(run["key"])
        if out.exists() and not args.force:
            print(f"{run['key']}: exists, skipped")
            continue
        py = registry.python(run["env"])
        if not py.exists():
            sys.exit(f"{py} missing: run setup_envs.sh")
        print(f"{run['key']}: {run['label']} ({py})", flush=True)
        t0 = time.time()
        subprocess.run(
            [
                str(py),
                str(HERE / "simulate.py"),
                str(registry.INPUTS),
                str(out),
                "--beam-az-rot",
                str(run["beam_az_rot"]),
            ],
            check=True,
            # Not the repo: a stray mistsim/ there must not shadow the
            # environment's own install.
            cwd=registry.RESULTS,
        )
        print(f"{run['key']}: {time.time() - t0:.0f} s total", flush=True)


if __name__ == "__main__":
    main()
