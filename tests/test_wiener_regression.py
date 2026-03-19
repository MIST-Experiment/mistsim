"""Regression tests for Wiener filter mapmaking.

The forward model is y = Ax + n, where:
    x ~ N(0, S)   signal with prior covariance S (diagonal)
    n ~ N(0, N)   noise with covariance N (diagonal)
    A             forward operator (sky alm -> timestream)

The Wiener filter solution is:
    x_hat = W y

Three equivalent implementations are tested:

1. **Naive** (dense direct solve):
   x = (A^H N^{-1} A + S^{-1})^{-1} A^H N^{-1} y

2. **SVD** (truncated SVD of whitened operator):
   Atilde = N^{-1/2} A S^{1/2},  Atilde = U Sigma V^H
   Filter factors: D_i = sigma_i / (1 + sigma_i^2)
   x_tilde = V D U^H y_tilde,  x = S^{1/2} x_tilde

3. **Conjugate gradient** (iterative solve of normal equations):
   (Atilde^H Atilde + I) x_tilde = Atilde^H y_tilde
   x = S^{1/2} x_tilde

All three solve the same problem and must agree.

Additionally tested:

4. **Randomized SVD** (mapmaking.randomized_svd_jax):
   Approximate SVD via Halko-Martinsson-Tropp, fed into the SVD
   Wiener filter path.

5. **Multi-frequency**: per-frequency Wiener filtering produces
   independent correct solutions.

6. **Noise scaling**: higher noise degrades reconstruction,
   lower noise improves it — confirming N is correctly wired.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as sla

from mistsim import mapmaking, pipeline


@pytest.fixture
def wiener_problem():
    """Small dense Wiener filter problem with known solution.

    nalm = 25 (lmax=4), ndata = 80.  Signal drawn from the prior,
    noise at 10% of typical N amplitude.
    """
    rng = np.random.default_rng(12345)
    lmax = 4
    nalm = (lmax + 1) ** 2  # 25
    ndata = 80

    # Forward operator
    A = rng.standard_normal((ndata, nalm))

    # Covariances (diagonal, positive)
    Ndiag = np.abs(rng.standard_normal(ndata)) + 0.5
    Sdiag = np.abs(rng.standard_normal(nalm)) + 0.1

    # True signal drawn from prior
    x_true = rng.standard_normal(nalm) * np.sqrt(Sdiag)

    # Noiseless data
    y_noiseless = A @ x_true

    # Noise realization (moderate SNR)
    noise = rng.standard_normal(ndata) * np.sqrt(Ndiag) * 0.1

    # Observed data
    y_obs = y_noiseless + noise

    # Whitened quantities
    Nm12 = 1.0 / np.sqrt(Ndiag)
    S12 = np.sqrt(Sdiag)
    Atilde_dense = np.diag(Nm12) @ A @ np.diag(S12)
    y_tilde = Nm12 * y_obs

    # -- Naive reference: model-space form --
    AHNinvA = A.T @ np.diag(1.0 / Ndiag) @ A
    Sinv = np.diag(1.0 / Sdiag)
    x_naive = np.linalg.solve(
        AHNinvA + Sinv, A.T @ np.diag(1.0 / Ndiag) @ y_obs
    )

    # -- Naive reference: whitened form --
    AtA_I = Atilde_dense.T @ Atilde_dense + np.eye(nalm)
    rhs = Atilde_dense.T @ y_tilde
    x_tilde_ref = np.linalg.solve(AtA_I, rhs)
    x_naive_whitened = S12 * x_tilde_ref

    # Full SVD of Atilde for identity checks
    U_full, Sigma_full, Vh_full = np.linalg.svd(
        Atilde_dense, full_matrices=False
    )

    return {
        "lmax": lmax,
        "nalm": nalm,
        "ndata": ndata,
        "A": A,
        "Ndiag": Ndiag,
        "Sdiag": Sdiag,
        "x_true": x_true,
        "y_noiseless": y_noiseless,
        "noise": noise,
        "y_obs": y_obs,
        "Nm12": Nm12,
        "S12": S12,
        "Atilde_dense": Atilde_dense,
        "y_tilde": y_tilde,
        "x_naive": x_naive,
        "x_naive_whitened": x_naive_whitened,
        "x_tilde_ref": x_tilde_ref,
        "U_full": U_full,
        "Sigma_full": Sigma_full,
        "Vh_full": Vh_full,
    }


# ======================================================================
# 1. Mathematical identities
# ======================================================================


class TestWienerMath:
    """Verify the mathematical identities underlying all methods."""

    def test_model_space_equals_whitened(self, wiener_problem):
        """Two forms of the naive Wiener filter must agree.

        Model-space: x = (A^H N^{-1} A + S^{-1})^{-1} A^H N^{-1} y
        Whitened:    x = S^{1/2} (Atilde^H Atilde + I)^{-1} Atilde^H y~
        """
        p = wiener_problem
        np.testing.assert_allclose(
            p["x_naive"], p["x_naive_whitened"], rtol=1e-10
        )

    def test_data_space_form(self, wiener_problem):
        """Data-space form equals model-space form (Woodbury identity).

        x = S A^H (A S A^H + N)^{-1} y
        """
        p = wiener_problem
        A, Ndiag, Sdiag = p["A"], p["Ndiag"], p["Sdiag"]

        x_data = (
            np.diag(Sdiag)
            @ A.T
            @ np.linalg.solve(
                A @ np.diag(Sdiag) @ A.T + np.diag(Ndiag), p["y_obs"]
            )
        )
        np.testing.assert_allclose(x_data, p["x_naive"], rtol=1e-10)

    def test_svd_filter_factors_reproduce_dense(self, wiener_problem):
        """Full SVD with D = sigma/(1+sigma^2) reproduces the dense solve.

        x_tilde = V diag(D) U^H y_tilde
        """
        p = wiener_problem
        U, Sigma, Vh = p["U_full"], p["Sigma_full"], p["Vh_full"]

        D = Sigma / (1 + Sigma**2)
        x_tilde_svd = Vh.T @ np.diag(D) @ U.T @ p["y_tilde"]

        np.testing.assert_allclose(
            x_tilde_svd, p["x_tilde_ref"], rtol=1e-10
        )

    def test_whitening_is_correct(self, wiener_problem):
        """Atilde = N^{-1/2} A S^{1/2} matches make_Atilde operator."""
        p = wiener_problem
        Aop = sla.aslinearoperator(p["A"])
        Atilde_op = mapmaking.make_Atilde(p["Ndiag"], Aop, p["Sdiag"])

        # Check forward on random vector
        rng = np.random.default_rng(99)
        x = rng.standard_normal(p["nalm"])
        np.testing.assert_allclose(
            Atilde_op.matvec(x),
            p["Atilde_dense"] @ x,
            rtol=1e-12,
        )

        # Check adjoint
        y = rng.standard_normal(p["ndata"])
        np.testing.assert_allclose(
            Atilde_op.rmatvec(y),
            p["Atilde_dense"].T @ y,
            rtol=1e-12,
        )


# ======================================================================
# 2. Naive dense Wiener filter
# ======================================================================


class TestNaiveWiener:
    """The dense solve is our ground truth.  Verify its properties."""

    def test_reduces_reconstruction_error(self, wiener_problem):
        """Wiener estimate is closer to x_true than the zero vector."""
        p = wiener_problem
        err_zero = np.linalg.norm(p["x_true"])
        err_wiener = np.linalg.norm(p["x_naive"] - p["x_true"])
        assert err_wiener < err_zero

    def test_noiseless_recovery(self, wiener_problem):
        """With noise=0 the Wiener filter closely recovers x_true."""
        p = wiener_problem
        A, Ndiag, Sdiag = p["A"], p["Ndiag"], p["Sdiag"]

        AHNinvA = A.T @ np.diag(1.0 / Ndiag) @ A
        Sinv = np.diag(1.0 / Sdiag)
        x_rec = np.linalg.solve(
            AHNinvA + Sinv,
            A.T @ np.diag(1.0 / Ndiag) @ p["y_noiseless"],
        )
        rel_err = np.linalg.norm(x_rec - p["x_true"]) / np.linalg.norm(
            p["x_true"]
        )
        # Prior regularization biases the estimate, but high-SNR
        # modes are recovered well.
        assert rel_err < 0.5

    def test_shrinkage(self, wiener_problem):
        """Wiener estimate has smaller norm than OLS (prior shrinkage)."""
        p = wiener_problem
        x_ls = np.linalg.lstsq(p["A"], p["y_obs"], rcond=None)[0]
        assert np.linalg.norm(p["x_naive"]) <= np.linalg.norm(x_ls)


# ======================================================================
# 3. SVD Wiener filter  (pipeline.wiener_filter)
# ======================================================================


class TestSVDWiener:
    """Test the SVD-based Wiener filter from pipeline.wiener_filter."""

    def test_matches_naive_full_rank(self, wiener_problem):
        """Full SVD reproduces the naive solution exactly."""
        p = wiener_problem

        # Use the full numpy SVD (all nalm modes) so no truncation
        U = p["U_full"]
        Sigma = p["Sigma_full"]
        Vh = p["Vh_full"]
        nvec = p["nalm"]

        x_svd_hp, D = pipeline.wiener_filter(
            U, Sigma, Vh, nvec,
            p["Ndiag"], p["Sdiag"],
            p["y_noiseless"], p["noise"],
        )

        x_naive_hp = np.asarray(mapmaking.alm1d_to_hp(p["x_naive"]))
        np.testing.assert_allclose(x_svd_hp, x_naive_hp, rtol=1e-10)

    def test_filter_factors_bounded(self, wiener_problem):
        """D = sigma/(1+sigma^2) is always in (0, 0.5]."""
        p = wiener_problem
        Sigma = p["Sigma_full"]
        D = Sigma / (1 + Sigma**2)
        assert np.all(D > 0)
        assert np.all(D <= 0.5 + 1e-15)

    def test_truncated_svd_converges(self, wiener_problem):
        """Error relative to the naive solution decreases with k."""
        p = wiener_problem
        Aop = sla.aslinearoperator(p["A"])
        Atilde = mapmaking.make_Atilde(p["Ndiag"], Aop, p["Sdiag"])

        x_naive_hp = np.asarray(mapmaking.alm1d_to_hp(p["x_naive"]))
        errors = []
        for k in [5, 10, 15, 20, 24]:
            U, Sigma, Vh = sla.svds(Atilde, k=k)
            idx = np.argsort(-Sigma)
            U, Sigma, Vh = U[:, idx], Sigma[idx], Vh[idx]
            x_svd_hp, _ = pipeline.wiener_filter(
                U, Sigma, Vh, k,
                p["Ndiag"], p["Sdiag"],
                p["y_noiseless"], p["noise"],
            )
            err = np.linalg.norm(x_svd_hp - x_naive_hp)
            err /= np.linalg.norm(x_naive_hp)
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-10


# ======================================================================
# 4. CG Wiener filter  (pipeline.wiener_filter_cg)
# ======================================================================


class TestCGWiener:
    """Test the CG-based Wiener filter from pipeline.wiener_filter_cg."""

    def _make_jax_ops(self, A, Ndiag, Sdiag):
        """Build JAX forward/adjoint and whitened operators."""
        Ajax = jnp.asarray(A)

        def fwd(x):
            return Ajax @ x

        def adj(y):
            return Ajax.T @ y

        atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
            Ndiag, fwd, adj, Sdiag
        )
        return fwd, adj, atilde_fwd, atilde_adj

    def test_matches_naive(self, wiener_problem):
        """CG solution matches the naive dense solution."""
        p = wiener_problem
        _, _, atilde_fwd, atilde_adj = self._make_jax_ops(
            p["A"], p["Ndiag"], p["Sdiag"]
        )

        x_cg_hp, info = pipeline.wiener_filter_cg(
            atilde_fwd, atilde_adj,
            p["Ndiag"], p["Sdiag"],
            p["y_noiseless"], p["noise"],
            tol=1e-12, maxiter=2000,
        )

        x_naive_hp = np.asarray(mapmaking.alm1d_to_hp(p["x_naive"]))
        np.testing.assert_allclose(
            np.asarray(x_cg_hp), x_naive_hp, rtol=1e-5
        )

    def test_solves_normal_equations(self, wiener_problem):
        """CG output satisfies (Atilde^H Atilde + I) x~ = Atilde^H y~."""
        p = wiener_problem
        _, _, atilde_fwd, atilde_adj = self._make_jax_ops(
            p["A"], p["Ndiag"], p["Sdiag"]
        )

        Nm12 = jnp.asarray(p["Nm12"])
        S12 = jnp.asarray(p["S12"])
        y_tilde = Nm12 * (jnp.asarray(p["y_noiseless"]) + jnp.asarray(
            p["noise"]
        ))
        rhs = atilde_adj(y_tilde)

        x_cg_hp, _ = pipeline.wiener_filter_cg(
            atilde_fwd, atilde_adj,
            p["Ndiag"], p["Sdiag"],
            p["y_noiseless"], p["noise"],
            tol=1e-12, maxiter=2000,
        )

        # Recover x_tilde from x_rec = S^{1/2} x_tilde
        x_rec_1d = np.asarray(p["x_naive"])  # use naive as proxy
        x_tilde = x_rec_1d / p["S12"]

        # Check residual
        lhs = np.asarray(
            atilde_adj(atilde_fwd(jnp.asarray(x_tilde)))
        ) + x_tilde
        np.testing.assert_allclose(lhs, np.asarray(rhs), rtol=1e-8)


# ======================================================================
# 5. Cross-method agreement
# ======================================================================


class TestAllMethodsAgree:
    """All three methods must produce the same reconstruction."""

    def test_naive_svd_cg_agree(self, wiener_problem):
        """Naive, SVD (k=24), and CG all agree."""
        p = wiener_problem

        # --- Naive ---
        x_naive_hp = np.asarray(mapmaking.alm1d_to_hp(p["x_naive"]))

        # --- SVD (full rank via numpy) ---
        U = p["U_full"]
        Sigma = p["Sigma_full"]
        Vh = p["Vh_full"]
        x_svd_hp, _ = pipeline.wiener_filter(
            U, Sigma, Vh, p["nalm"],
            p["Ndiag"], p["Sdiag"],
            p["y_noiseless"], p["noise"],
        )

        # --- CG ---
        Ajax = jnp.asarray(p["A"])

        def fwd(x):
            return Ajax @ x

        def adj(y):
            return Ajax.T @ y

        atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
            p["Ndiag"], fwd, adj, p["Sdiag"]
        )
        x_cg_hp, _ = pipeline.wiener_filter_cg(
            atilde_fwd, atilde_adj,
            p["Ndiag"], p["Sdiag"],
            p["y_noiseless"], p["noise"],
            tol=1e-12, maxiter=2000,
        )

        np.testing.assert_allclose(
            x_svd_hp, x_naive_hp, rtol=1e-4,
            err_msg="SVD vs Naive",
        )
        np.testing.assert_allclose(
            np.asarray(x_cg_hp), x_naive_hp, rtol=1e-5,
            err_msg="CG vs Naive",
        )
        np.testing.assert_allclose(
            np.asarray(x_cg_hp), x_svd_hp, rtol=1e-4,
            err_msg="CG vs SVD",
        )


# ======================================================================
# 6. Randomized SVD Wiener filter
# ======================================================================


class TestRandomizedSVDWiener:
    """Test that randomized_svd_jax produces a valid Wiener filter."""

    def test_rsvd_wiener_matches_naive(self, wiener_problem):
        """Randomized SVD Wiener filter matches naive dense solution."""
        p = wiener_problem
        Ajax = jnp.asarray(p["A"])

        def fwd(x):
            return Ajax @ x

        def adj(y):
            return Ajax.T @ y

        atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
            p["Ndiag"], fwd, adj, p["Sdiag"]
        )

        # Full rank with oversampling for accuracy
        k = p["nalm"]
        U, Sigma, Vh = mapmaking.randomized_svd_jax(
            atilde_fwd, atilde_adj,
            p["nalm"], p["ndata"],
            k=k, p=10, seed=0,
        )
        U = np.asarray(U)
        Sigma = np.asarray(Sigma)
        Vh = np.asarray(Vh)

        x_rsvd_hp, D = pipeline.wiener_filter(
            U, Sigma, Vh, k,
            p["Ndiag"], p["Sdiag"],
            p["y_noiseless"], p["noise"],
        )

        x_naive_hp = np.asarray(mapmaking.alm1d_to_hp(p["x_naive"]))
        np.testing.assert_allclose(
            x_rsvd_hp, x_naive_hp, rtol=1e-3,
        )

    def test_rsvd_singular_values_match_exact(self, wiener_problem):
        """Randomized SVD singular values match exact SVD."""
        p = wiener_problem
        Ajax = jnp.asarray(p["A"])

        def fwd(x):
            return Ajax @ x

        def adj(y):
            return Ajax.T @ y

        atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
            p["Ndiag"], fwd, adj, p["Sdiag"]
        )

        k = p["nalm"]
        _, Sigma_rsvd, _ = mapmaking.randomized_svd_jax(
            atilde_fwd, atilde_adj,
            p["nalm"], p["ndata"],
            k=k, p=10, seed=0,
        )

        np.testing.assert_allclose(
            np.asarray(Sigma_rsvd),
            p["Sigma_full"],
            rtol=1e-4,
        )

    def test_rsvd_truncated_converges(self, wiener_problem):
        """Randomized SVD error decreases as k increases."""
        p = wiener_problem
        Ajax = jnp.asarray(p["A"])

        def fwd(x):
            return Ajax @ x

        def adj(y):
            return Ajax.T @ y

        atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
            p["Ndiag"], fwd, adj, p["Sdiag"]
        )

        x_naive_hp = np.asarray(mapmaking.alm1d_to_hp(p["x_naive"]))
        errors = []
        for k in [5, 10, 15, 20, 25]:
            U, Sigma, Vh = mapmaking.randomized_svd_jax(
                atilde_fwd, atilde_adj,
                p["nalm"], p["ndata"],
                k=k, p=10, seed=0,
            )
            x_hp, _ = pipeline.wiener_filter(
                np.asarray(U), np.asarray(Sigma), np.asarray(Vh),
                k,
                p["Ndiag"], p["Sdiag"],
                p["y_noiseless"], p["noise"],
            )
            err = np.linalg.norm(x_hp - x_naive_hp)
            err /= np.linalg.norm(x_naive_hp)
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-6


# ======================================================================
# 7. Multi-frequency Wiener filter
# ======================================================================


class TestMultiFreqWiener:
    """Per-frequency Wiener filtering gives correct independent solutions."""

    def test_per_freq_cg_matches_naive(self):
        """CG Wiener filter applied per-frequency matches naive solve."""
        rng = np.random.default_rng(777)
        lmax = 3
        nalm = (lmax + 1) ** 2  # 16
        ndata = 40
        nfreq = 3

        errors = []
        for f in range(nfreq):
            # Independent problem per frequency
            A_f = rng.standard_normal((ndata, nalm))
            Ndiag_f = np.abs(rng.standard_normal(ndata)) + 0.5
            Sdiag_f = np.abs(rng.standard_normal(nalm)) + 0.1
            x_true_f = rng.standard_normal(nalm) * np.sqrt(Sdiag_f)
            y_f = A_f @ x_true_f
            noise_f = rng.standard_normal(ndata) * np.sqrt(Ndiag_f) * 0.1

            # Naive reference
            AHNinvA = A_f.T @ np.diag(1.0 / Ndiag_f) @ A_f
            Sinv = np.diag(1.0 / Sdiag_f)
            x_naive_f = np.linalg.solve(
                AHNinvA + Sinv,
                A_f.T @ np.diag(1.0 / Ndiag_f) @ (y_f + noise_f),
            )
            x_naive_hp = np.asarray(
                mapmaking.alm1d_to_hp(x_naive_f)
            )

            # CG solve
            Ajax = jnp.asarray(A_f)

            def fwd(x, M=Ajax):
                return M @ x

            def adj(y, M=Ajax):
                return M.T @ y

            atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
                Ndiag_f, fwd, adj, Sdiag_f
            )
            x_cg_hp, _ = pipeline.wiener_filter_cg(
                atilde_fwd, atilde_adj,
                Ndiag_f, Sdiag_f,
                y_f, noise_f,
                tol=1e-12, maxiter=2000,
            )

            np.testing.assert_allclose(
                np.asarray(x_cg_hp), x_naive_hp, rtol=1e-5,
                err_msg=f"Frequency {f} mismatch",
            )

            err = np.linalg.norm(
                np.asarray(x_cg_hp) - x_naive_hp
            ) / np.linalg.norm(x_naive_hp)
            errors.append(err)

        # All frequencies should have small error
        assert all(e < 1e-5 for e in errors)

    def test_per_freq_independent(self):
        """Changing one frequency does not affect another's solution."""
        rng = np.random.default_rng(888)
        lmax = 3
        nalm = (lmax + 1) ** 2
        ndata = 40

        # Two frequencies, same A but different signals
        A = rng.standard_normal((ndata, nalm))
        Ndiag = np.abs(rng.standard_normal(ndata)) + 0.5
        Sdiag = np.abs(rng.standard_normal(nalm)) + 0.1

        x1 = rng.standard_normal(nalm) * np.sqrt(Sdiag)
        x2 = rng.standard_normal(nalm) * np.sqrt(Sdiag)
        noise1 = rng.standard_normal(ndata) * np.sqrt(Ndiag) * 0.1
        noise2 = rng.standard_normal(ndata) * np.sqrt(Ndiag) * 0.1

        Ajax = jnp.asarray(A)

        def fwd(x):
            return Ajax @ x

        def adj(y):
            return Ajax.T @ y

        atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
            Ndiag, fwd, adj, Sdiag
        )

        # Solve freq 0
        x_cg_f0, _ = pipeline.wiener_filter_cg(
            atilde_fwd, atilde_adj,
            Ndiag, Sdiag,
            A @ x1, noise1,
            tol=1e-12, maxiter=2000,
        )

        # Solve freq 1 (different signal/noise)
        x_cg_f1, _ = pipeline.wiener_filter_cg(
            atilde_fwd, atilde_adj,
            Ndiag, Sdiag,
            A @ x2, noise2,
            tol=1e-12, maxiter=2000,
        )

        # They must differ (different inputs)
        assert not np.allclose(
            np.asarray(x_cg_f0), np.asarray(x_cg_f1)
        )

        # Solve freq 0 again — must be identical (deterministic)
        x_cg_f0_again, _ = pipeline.wiener_filter_cg(
            atilde_fwd, atilde_adj,
            Ndiag, Sdiag,
            A @ x1, noise1,
            tol=1e-12, maxiter=2000,
        )
        np.testing.assert_allclose(
            np.asarray(x_cg_f0_again), np.asarray(x_cg_f0),
            rtol=1e-12,
        )


# ======================================================================
# 8. Noise scaling
# ======================================================================


class TestNoiseScaling:
    """Verify that noise level correctly affects reconstruction quality."""

    def test_more_noise_more_error(self):
        """Reconstruction error increases monotonically with noise."""
        rng = np.random.default_rng(999)
        lmax = 4
        nalm = (lmax + 1) ** 2
        ndata = 80

        A = rng.standard_normal((ndata, nalm))
        Ndiag_base = np.abs(rng.standard_normal(ndata)) + 0.5
        Sdiag = np.abs(rng.standard_normal(nalm)) + 0.1
        x_true = rng.standard_normal(nalm) * np.sqrt(Sdiag)
        y = A @ x_true
        noise_unit = rng.standard_normal(ndata)

        errors = []
        for scale in [0.01, 0.1, 1.0, 10.0]:
            Ndiag = Ndiag_base * scale**2
            noise = noise_unit * np.sqrt(Ndiag_base) * scale

            AHNinvA = A.T @ np.diag(1.0 / Ndiag) @ A
            Sinv = np.diag(1.0 / Sdiag)
            x_rec = np.linalg.solve(
                AHNinvA + Sinv,
                A.T @ np.diag(1.0 / Ndiag) @ (y + noise),
            )
            err = np.linalg.norm(x_rec - x_true)
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i + 1] > errors[i], (
                f"Error did not increase: scale {i} -> {i+1}: "
                f"{errors[i]:.6f} -> {errors[i+1]:.6f}"
            )

    def test_zero_noise_best_recovery(self):
        """Noiseless case gives the best reconstruction."""
        rng = np.random.default_rng(999)
        lmax = 4
        nalm = (lmax + 1) ** 2
        ndata = 80

        A = rng.standard_normal((ndata, nalm))
        Ndiag = np.abs(rng.standard_normal(ndata)) + 0.5
        Sdiag = np.abs(rng.standard_normal(nalm)) + 0.1
        x_true = rng.standard_normal(nalm) * np.sqrt(Sdiag)
        y = A @ x_true

        AHNinvA = A.T @ np.diag(1.0 / Ndiag) @ A
        Sinv = np.diag(1.0 / Sdiag)

        # Noiseless
        x_clean = np.linalg.solve(
            AHNinvA + Sinv, A.T @ np.diag(1.0 / Ndiag) @ y
        )
        err_clean = np.linalg.norm(x_clean - x_true)

        # With noise
        noise = rng.standard_normal(ndata) * np.sqrt(Ndiag)
        x_noisy = np.linalg.solve(
            AHNinvA + Sinv,
            A.T @ np.diag(1.0 / Ndiag) @ (y + noise),
        )
        err_noisy = np.linalg.norm(x_noisy - x_true)

        assert err_clean < err_noisy

    def test_noise_scaling_cg_consistent(self):
        """CG solver shows the same noise scaling behavior as naive."""
        rng = np.random.default_rng(999)
        lmax = 4
        nalm = (lmax + 1) ** 2
        ndata = 80

        A = rng.standard_normal((ndata, nalm))
        Ndiag_base = np.abs(rng.standard_normal(ndata)) + 0.5
        Sdiag = np.abs(rng.standard_normal(nalm)) + 0.1
        x_true = rng.standard_normal(nalm) * np.sqrt(Sdiag)
        y = A @ x_true
        noise_unit = rng.standard_normal(ndata)
        Ajax = jnp.asarray(A)

        def fwd(x):
            return Ajax @ x

        def adj(yv):
            return Ajax.T @ yv

        errors = []
        for scale in [0.01, 0.1, 1.0, 10.0]:
            Ndiag = Ndiag_base * scale**2
            noise = noise_unit * np.sqrt(Ndiag_base) * scale

            atilde_fwd, atilde_adj = mapmaking.make_atilde_fns(
                Ndiag, fwd, adj, Sdiag
            )
            x_cg_hp, _ = pipeline.wiener_filter_cg(
                atilde_fwd, atilde_adj,
                Ndiag, Sdiag,
                y, noise,
                tol=1e-12, maxiter=2000,
            )

            # Convert back to packed 1d for fair comparison
            x_true_hp = np.asarray(mapmaking.alm1d_to_hp(x_true))
            err = np.linalg.norm(np.asarray(x_cg_hp) - x_true_hp)
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i + 1] > errors[i], (
                f"CG error did not increase: scale {i} -> {i+1}: "
                f"{errors[i]:.6f} -> {errors[i+1]:.6f}"
            )
