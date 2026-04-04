#!/usr/bin/env python
"""Generate per-beam synthetic data for the MIST mapmaking pipeline.

Produces one NPZ file per beam definition in runs.yaml, containing
the forward-simulated timestream at all available frequencies.
The sky model is computed once and reused across all beams.

Usage
-----
    uv run python scripts/generate_all_data.py
    uv run python scripts/generate_all_data.py --beams mars-csa2022-dip
    uv run python scripts/generate_all_data.py --output-dir path/to/dir
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import yaml

# Suppress JAX backend discovery noise and enable 64-bit precision
# before JAX is first imported by mistsim.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

from mistsim.pipeline import (
    _resolve_config_paths,
    save_sim_data,
    setup_sky_multi_freq,
    simulate_waterfall,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logging.getLogger("healpy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_RUNS = (
    Path(__file__).resolve().parent
    / "../notebooks/mapmaking/configs/runs.yaml"
)


def _beam_freq_range(beam_def, config_dir, default_range):
    """Determine the frequency range a beam supports.

    For analytic beams (sin2), returns the full default range.
    For file-based beams, reads the beam NPZ to find the
    intersection with the default range.

    Returns [f_min, f_max) as a list.
    """
    beam_type = beam_def.get("beam_type", "file")
    if beam_type != "file":
        return default_range

    beam_file = beam_def.get("beam_file")
    if beam_file is None:
        return default_range

    p = Path(beam_file)
    if not p.is_absolute():
        p = config_dir / p
    d = np.load(str(p))
    beam_freqs = d["freqs"]
    f_min = max(int(beam_freqs.min()), default_range[0])
    f_max = min(int(beam_freqs.max()) + 1, default_range[1])
    return [f_min, f_max]


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-beam synthetic data for MIST",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_RUNS),
        help="Path to runs.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/synthetic_data)",
    )
    parser.add_argument(
        "--beams",
        nargs="+",
        default=None,
        help="Generate only these beams (default: all)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip beams whose output file already exists",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    with open(config_path) as f:
        registry = yaml.safe_load(f)

    beams_defs = registry["beams"]
    defaults = registry.get("defaults", {})
    # Default freqs from config (e.g. [40] for single-freq)
    # For data generation we need the full range the beams cover.
    default_freq_range = [25, 126]

    # Which beams to generate
    beam_names = args.beams or list(beams_defs.keys())
    for name in beam_names:
        if name not in beams_defs:
            raise ValueError(
                f"Unknown beam {name!r}. "
                f"Available: {list(beams_defs)}"
            )

    # Output directory
    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(
        args.output_dir
        or str(project_root / "data" / "synthetic_data")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group beams by frequency range so we only compute the sky
    # SHT once per unique range.
    range_to_beams = {}
    for name in beam_names:
        beam_def = dict(beams_defs[name])
        fr = tuple(
            _beam_freq_range(
                beam_def, config_dir, default_freq_range
            )
        )
        range_to_beams.setdefault(fr, []).append(name)

    for freq_range, group_beams in range_to_beams.items():
        logger.info(
            "Frequency range %d-%d MHz: %d beams (%s)",
            freq_range[0],
            freq_range[1] - 1,
            len(group_beams),
            ", ".join(group_beams),
        )

        # Build config from defaults for this freq range
        cfg = {}
        for key, val in defaults.items():
            cfg[key] = dict(val)
        cfg["sky"]["freqs"] = f"{freq_range[0]}:{freq_range[1]}"
        cfg.setdefault("sites", [])
        _resolve_config_paths(cfg, config_dir)

        # Sky SHT (once per freq range)
        sky_data = setup_sky_multi_freq(cfg)

        # Generate each beam
        for name in group_beams:
            out_path = out_dir / f"{name}.npz"
            if args.no_overwrite and out_path.exists():
                logger.info("Skipping %s (exists)", name)
                continue

            beam_def = dict(beams_defs[name])
            site_key = beam_def.pop("site")
            site_def = dict(registry["sites"][site_key])
            beam_cfg = {"name": name, **site_def, **beam_def}

            # Resolve beam_file path
            if "beam_file" in beam_cfg:
                p = Path(beam_cfg["beam_file"])
                if not p.is_absolute():
                    beam_cfg["beam_file"] = str(config_dir / p)

            data = simulate_waterfall(beam_cfg, sky_data)
            save_sim_data(data, sky_data, out_path)

    logger.info("Done. Output in %s", out_dir)


if __name__ == "__main__":
    main()
