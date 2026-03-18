# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mistsim is a differentiable simulator of MIST (radio astronomy) observations, built on the `croissant` package. It uses JAX for automatic differentiation and GPU acceleration, with spherical harmonic transforms via `s2fft`.

## Commands

```bash
# Install dependencies (uses uv)
uv sync --all-extras --dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_beam.py -v

# Run a specific test
uv run pytest tests/test_beam.py::test_name -v

# Lint
uv run ruff check src/
uv run ruff format --check src/
```

## Architecture

Source lives in `src/mistsim/` with four modules:

- **`beam.py`** — Beam pattern model (extends `croissant.Beam`). Handles horizon masking, azimuth rotation, tilt, and multiple sampling schemes (mw, mwss, dh, gl, healpix).
- **`sky.py`** — Sky model (extends `croissant.Sky`). Supports galactic/equatorial coordinates and various pixelization schemes. Internal `_SkyAlm` provides backward compat for direct alm input.
- **`sim.py`** — Simulator (extends `croissant.Simulator`). Combines beam + sky + observer location + time to produce simulated visibilities. Includes ground loss correction.
- **`mapmaking.py`** — Map reconstruction from simulated data. Builds sparse linear operators (A-matrix) relating sky alms to timestreams, supports SVD-based mapmaking. Uses JAX JIT and scipy sparse linear algebra.

The `croissant` package (croissant-sim) is the core engine that performs beam-sky convolution in spherical harmonic space.

## Key Constraints

- Python 3.10–3.12 (no 3.13)
- JAX pinned to <0.6.0
- Tests enable JAX x64 mode via `conftest.py`
- Ruff line length: 79 characters
- Ruff rules: E, F, W, I (pycodestyle, pyflakes, isort)
