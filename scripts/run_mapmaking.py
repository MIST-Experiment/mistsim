#!/usr/bin/env python
"""CLI entry point for the MIST mapmaking pipeline."""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import logging
from pathlib import Path

from mistsim.pipeline import (
    generate_notebook,
    load_config,
    run_name_from_config,
    run_pipeline,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run MIST SVD mapmaking pipeline",
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--nvec",
        type=int,
        default=None,
        help="Override automatic nvec selection",
    )
    parser.add_argument(
        "--n-singular-values",
        type=int,
        default=None,
        help="Override k for svds",
    )
    parser.add_argument(
        "--no-notebook",
        action="store_true",
        help="Skip notebook generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if args.n_singular_values is not None:
        config.setdefault("svd", {})["n_singular_values"] = (
            args.n_singular_values
        )

    # Run pipeline
    results = run_pipeline(config)

    # Override nvec if requested
    if args.nvec is not None:
        logger.info("Overriding nvec: %d -> %d", results["nvec"], args.nvec)
        results["nvec"] = args.nvec

    # Output directory
    out_cfg = config.get("output", {})
    output_dir = Path(args.output_dir or out_cfg.get("results_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_name_from_config(config)
    npz_path = output_dir / f"{run_name}.npz"
    save_results(results, npz_path)

    # Notebook
    gen_nb = out_cfg.get("generate_notebook", True)
    if not args.no_notebook and gen_nb:
        nb_path = output_dir / f"{run_name}.ipynb"
        generate_notebook(results, npz_path, nb_path)

    logger.info("Done: %s", run_name)


if __name__ == "__main__":
    main()
