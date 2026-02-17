# mistsim

`mistsim` is a differentiable simulator of MIST observations, built on the `croissant` package. It is written in Python and uses JAX for automatic differentiation and GPU acceleration. The goal of `mistsim` is to enable end-to-end differentiable modeling of MIST observations, allowing for efficient optimization and inference.

## Installation
Clone the reporistory and install the package using pip:
```bash
   git clone git@github.com:MIST-Experiment/mistsim.git
   cd mistsim
   pip install .
```
This will install `mistsim` and its dependencies, including `croissant` and `jax`. If you want to develop `mistsim`, you can also install the development dependencies:
```bash
   pip install -e .[dev]
```

## Getting Started
The directory `examples` contains a Jupyter notebook that demonstrates how to use `mistsim` to simulate MIST observations.

## CROISSANT
`croissant` is the engine of the `mistsim` package and performs the convolution of the sky with the MIST beam. The convolution is implemented in spherical harmonic space. See the [https://github.com/christianhbye/croissant](CROISSANT repository) for more details on the implementation and usage of `croissant`.
