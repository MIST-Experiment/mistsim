# Posterior uncertainty in the whitened SVD basis

How `mistsim` turns an SVD of the whitened design matrix into a
posterior standard-deviation map, and what the
$\sigma^2_{\rm post}/\sigma^2_{\rm prior}$ plots actually show.

**Code:**

| Piece | Location |
| --- | --- |
| Whitened design matrix $\tilde{A}$ | `src/mistsim/mapmaking.py:281` (`make_Atilde`) |
| Prior diagonal $S$ | `src/mistsim/pipeline.py:565` (`compute_prior`) |
| Truncated SVD | `src/mistsim/pipeline.py:588` (`run_svd`) |
| Mode selection | `src/mistsim/pipeline.py:604` (`select_nvec`) |
| Wiener mean | `src/mistsim/pipeline.py:749` (`wiener_filter`) |
| Posterior sampling | `src/mistsim/pipeline.py:1026` (`posterior_uncertainty`) |
| Plotting | `src/mistsim/plotting.py:949` (`plot_posterior_maps`) |

Rendered examples live in `notebooks/mapmaking/notebooks_out/`
(cell 21 of each executed notebook).

---

## 1. The model

The forward problem is

$$ y = A x + n $$

where $x$ is the packed real spherical-harmonic vector, $A$ the
beam-convolution / drift-scan operator (`make_Amat`,
`mapmaking.py:188`), $y$ the simulated timestream, and $n$ the
radiometer noise.

Two Gaussian ingredients close the problem:

- **Noise:** $n \sim \mathcal{N}(0, N)$ with $N = \mathrm{diag}(\texttt{Ndiag})$,
  one variance per (time, frequency) sample.
- **Prior:** $x \sim \mathcal{N}(0, S)$ with $S = \mathrm{diag}(\texttt{Sdiag})$,
  where `Sdiag = cl[ells_full]` — the sky power spectrum evaluated
  at each coefficient's $\ell$.

### Why the prior is exactly diagonal

`compute_prior` assigns $C_\ell$ to *every* real degree of freedom,
including the real and imaginary parts of $m > 0$ separately:

```python
ells_pos  = ells_hp[emms_hp != 0]
ells_full = np.concatenate((ells_hp, ells_pos))
return cl[ells_full]
```

This is consistent because of the $1/\sqrt{2}$ in the packing
convention (`alm.py:122`):

$$ a_{\ell m} = \tfrac{1}{\sqrt{2}}\left(a^{\rm re}_{\ell m} + i\,a^{\rm im}_{\ell m}\right) $$

A complex coefficient with $\langle |a_{\ell m}|^2 \rangle = C_\ell$
splits into two real d.o.f. each with variance $C_\ell$. So in the
packed basis the prior covariance really is diagonal with entries
$C_\ell$ — no $m$-dependent bookkeeping, and the whitened prior below
is exactly the identity.

---

## 2. Whitening

Define

$$ \tilde{x} = S^{-1/2} x, \qquad \tilde{y} = N^{-1/2} y, \qquad \tilde{A} = N^{-1/2} A\, S^{1/2} $$

so that

$$ \tilde{y} = \tilde{A}\,\tilde{x} + \tilde{n}, \qquad \tilde{x} \sim \mathcal{N}(0, I), \quad \tilde{n} \sim \mathcal{N}(0, I). $$

This is `make_Atilde` (`mapmaking.py:281`), applied matrix-free:

```python
def _Atilde_matvec(v, Ndiag, Amat, Sdiag):
    Nm12 = 1 / np.sqrt(Ndiag)
    S12 = np.sqrt(Sdiag)
    return Nm12 * Amat.matvec(S12 * v)
```

and on the data side, `y_tilde = Ndiag**(-0.5) * (y + noise)`
(`pipeline.py:776`).

**Why bother.** After whitening, the prior and the noise are the
same object — the identity. A single SVD of $\tilde{A}$ therefore
diagonalizes the entire posterior, and the singular values come out
as pure dimensionless signal-to-noise numbers: $\sigma_k$ is how
well the data constrains mode $k$ *in units of its own prior width*.
Nothing carries units, and no mode is privileged by an arbitrary
choice of amplitude scale.

---

## 3. The posterior

With both covariances the identity,

$$ -2 \ln p(\tilde{x} \mid \tilde{y}) = \lVert \tilde{y} - \tilde{A}\tilde{x} \rVert^2 + \lVert \tilde{x} \rVert^2 + \text{const} $$

Both terms are quadratic, so the posterior is Gaussian:

$$ C_{\rm post} = \left(\tilde{A}^{\mathsf H}\tilde{A} + I\right)^{-1}, \qquad \mu = C_{\rm post}\,\tilde{A}^{\mathsf H} \tilde{y} $$

The $+I$ *is* the prior. Drop it and this is ordinary least
squares, which is singular: the drift scan has exact null
directions (modes the beam never sees), and those blow up.

---

## 4. The SVD diagonalizes it

Write $\tilde{A} = U \Sigma V^{\mathsf H}$ (`run_svd`,
`pipeline.py:588`, via `scipy.sparse.linalg.svds`). Then

$$ \tilde{A}^{\mathsf H}\tilde{A} + I = V\left(\Sigma^2 + I\right)V^{\mathsf H} \quad\Longrightarrow\quad C_{\rm post} = V\left(\Sigma^2 + I\right)^{-1}V^{\mathsf H} $$

The right singular vectors $v_k$ are the principal axes of the
posterior. Along $v_k$:

| | value |
| --- | --- |
| prior std (whitened) | $1$ |
| posterior std | $1/\sqrt{\sigma_k^2 + 1}$ |
| variance ratio | $1/(\sigma_k^2 + 1)$ |

In code (`pipeline.py:1052`):

```python
post_std_svd = 1.0 / np.sqrt(Dnum**2 + 1.0)
std_reduce = 1 - post_std_svd
```

The two limits are the whole story:

- $\sigma_k \gg 1$ — well-measured mode. Posterior std $\to 1/\sigma_k$,
  far tighter than the prior. The data dominates.
- $\sigma_k \ll 1$ — unconstrained mode. Posterior std $\to 1$: the
  posterior hands back the prior unchanged, which is the correct
  and honest answer for something the instrument never saw.

`std_reduce` is the *shrinkage* — the fraction of the prior width
that the data removes.

---

## 5. Sampling: a low-rank square root

`posterior_uncertainty` draws from the posterior without ever
forming $C_{\rm post}$:

```python
wfull = rng.normal(size=(n_alm, n_realizations))
corr = Vht.T @ (std_reduce[:, None] * (Vht @ wfull))
x_tilde_sim = wfull - corr
```

Written out, with $w \sim \mathcal{N}(0, I)$:

$$ \tilde{x}_{\rm sim} = \Big[\,I - VV^{\mathsf H} + V\left(\Sigma^2+I\right)^{-1/2}V^{\mathsf H}\Big]\,w $$

**This bracket is a symmetric square root of $C_{\rm post}$.** The
projector $P = VV^{\mathsf H}$ and its complement $I - P$ are
orthogonal, so the two blocks square independently:

$$ \Big[\,(I - P) + V(\Sigma^2+I)^{-1/2}V^{\mathsf H}\Big]^2 = (I-P) + V\left(\Sigma^2+I\right)^{-1}V^{\mathsf H} = C_{\rm post} $$

so $\tilde{x}_{\rm sim}$ is an exact posterior draw.

Two things make this the right formulation:

1. **It never builds an $n_{\rm alm} \times n_{\rm alm}$ matrix.** Only
   `nvec` projections are computed, so cost is
   $O(n_{\rm alm}\cdot n_{\rm vec}\cdot n_{\rm real})$.
2. **It handles the unretained subspace for free.** Modes outside
   $V$ keep coefficient 1 — full prior width — which is exactly
   right for genuine null modes.

`V` is real here (real packed basis), so `Vht.T` is legitimately
$V^{\mathsf H}$.

---

## 6. Truncation

`svds` is truncated at $k$ modes and `select_nvec`
(`pipeline.py:604`) then keeps `nvec` of them. Three strategies:

- `"threshold"` (default) — keep $\sigma_k > 10^{-10}$. Safest:
  retains everything carrying measurable signal.
- `"auto"` — elbow of the singular-value curve in log space.
- `"manual"` — fixed count, with a warning if the mode at the cut
  still has Wiener factor $D > 0.01$.

Since dropped modes are implicitly assigned the prior width,
truncation is exact for $\sigma \to 0$ and a mild approximation for
any mode cut while $\sigma$ is not yet $\ll 1$ — it slightly
*over*-states the posterior variance there. That is the conservative
direction, but it is the reason the `"manual"` path warns.

---

## 7. Back to the sky

```python
x_sim     = np.sqrt(Sdiag)[:, None] * x_tilde_sim   # un-whiten
x_sim_hp  = [alm1d_to_hp(col) for col in x_sim.T]   # packed -> healpy
x_sim_map = [hp.alm2map(xs, nside) for xs in x_sim_hp]
std_map   = np.std(x_sim_map, axis=0)               # posterior sigma per pixel
```

Un-whitening restores the $C_\ell$ scaling, and `alm2map` rotates
into pixel space — where the covariance is emphatically *not*
diagonal. This is the real payoff of the Monte Carlo: the
pixel-space diagonal falls out of the sample scatter, instead of
requiring a dense covariance to be transformed by $Y_{\ell m}$.

The denominator is the per-pixel variance of a Gaussian random field
with spectrum $C_\ell$ (`pipeline.py:1077`):

$$ \sigma^2_{\rm prior} = \sum_\ell \frac{2\ell+1}{4\pi} C_\ell $$

A single scalar, because the prior is statistically isotropic. So
the plotted map

$$ \frac{\sigma^2_{\rm post}(\hat{n})}{\sigma^2_{\rm prior}} = \frac{\texttt{std\_map}^2}{\texttt{sigma2\_prior}} $$

is the **fraction of prior variance surviving in each pixel**: 0 means
fully determined by the data, 1 means the data said nothing there.

Its structure is purely instrumental. In the 40 MHz `all-nominal`
run the ratio spans $\approx 0.13$ in the Galactic plane to
$\approx 0.32$ near the poles — pixels the drift scan weights
heavily project onto high-$\sigma_k$ modes and shrink the most.

---

## 8. Relation to the Wiener filter factors

`wiener_filter` (`pipeline.py:771`) uses a *different* factor for
the posterior mean:

```python
Dnum = Sigma[:nvec]
D = Dnum / (1 + Dnum**2)
```

which comes from the same core with one extra $\Sigma$, contributed
by the $\tilde{A}^{\mathsf H}$ acting on the data:

$$ C_{\rm post}\tilde{A}^{\mathsf H} = V\left(\Sigma^2+I\right)^{-1}\Sigma\, U^{\mathsf H} $$

| Quantity | Factor | $\sigma \gg 1$ | $\sigma \ll 1$ |
| --- | --- | --- | --- |
| Posterior std | $1/\sqrt{\sigma^2+1}$ | $\to 1/\sigma \to 0$ | $\to 1$ (prior width) |
| Wiener mean | $\sigma/(1+\sigma^2)$ | $\to 1/\sigma \to 0$ | $\to \sigma \to 0$ |

Both vanish for unmeasured modes, but for opposite reasons: the
**mean** goes to zero because that is the prior mean; the **std** goes
to one because that is the prior width. Confusing the two is the
classic way to convince yourself a null mode was measured.

---

## 9. Monte Carlo convergence

`std_map` is a sample standard deviation over
`n_realizations = 1000` draws (seeded, `seed=1420`, so runs are
reproducible). A sample std from $N$ draws carries fractional error

$$ \frac{\Delta\sigma}{\sigma} \approx \frac{1}{\sqrt{2(N-1)}} \approx 2.2\,\% \quad (N = 1000) $$

and the variance ratio, being a square, carries about $4.5\,\%$.

**This is visible in the plots.** The fine-grained speckle in the
"Posterior Std Dev" and ratio panels is Monte Carlo noise, not sky
structure. It shrinks as $1/\sqrt{N}$ — raise `n_realizations` if
you want those panels publication-smooth. The large-scale
plane-versus-pole gradient is real signal and is unaffected.

---

## 10. Reproducing the figure

`plot_posterior_maps` (`plotting.py:949`) draws three Mollweide
panels: posterior std, the variance ratio, and
$\mathrm{SNR} = |{\rm map}|/\sigma$.

```python
fl = np.ones(lmax + 1)
fl[11:] = 0.0                                  # keep ell <= 10
best_map = hp.alm2map(hp.almxfl(x_rec, fl), nside=128)
fig = msplt.plot_posterior_maps(std_map, sigma2_prior, best_map, nside=128)
```

Note the `fl` low-pass: the *mean* map is truncated to $\ell \le 10$
for display, while `std_map` is computed over the full $\ell$ range.
The SNR panel is therefore a smoothed numerator over an unsmoothed
denominator — fine for a visual check, but not a quantity to quote.

Executed examples, all at cell 21:

- `notebooks/mapmaking/notebooks_out/40mhz/all-nominal.ipynb`
- `notebooks/mapmaking/notebooks_out/25mhz/mars.ipynb`
- `notebooks/mapmaking/notebooks_out/mars-lmax{40,60,90,179}.ipynb`
  — the $\ell_{\rm max}$ series
- plus the per-site combinations under `25mhz/` and `40mhz/`
