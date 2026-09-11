"""Tests for posterior uncertainty and the posterior eigenmodes.

The whitened posterior is

    Ctilde_post = (Atilde^H Atilde + I)^{-1}
                = V (Sigma^2 + I)^{-1} V^H  +  (I - V V^H)

where the second term covers modes the truncated SVD never saw, which
keep their full prior width.  `posterior_uncertainty` draws samples
from this and reports per-pixel and per-alm standard deviations; these
tests pin those against dense linear algebra on a small problem.

`posterior_eigenmodes` exposes the eigendecomposition itself: the rows
of Vh are the eigenvectors and 1/(sigma^2 + 1) are the eigenvalues.
"""

import healpy as hp
import numpy as np
import pytest

from mistsim import pipeline
from mistsim.alm import alm1d_to_hp


@pytest.fixture
def posterior_problem():
    """Small posterior problem with an isotropic prior.

    lmax = 4 (nalm = 25), ndata = 80, nside = 8 (npix = 768).  The SVD
    is deliberately truncated at nvec = 12 so the untouched subspace is
    exercised.

    A is scaled so the retained singular values straddle unity
    (0.55..2.10).  This matters: the ``+1`` in 1/sqrt(sigma^2 + 1) *is*
    the prior, and for sigma >> 1 it is numerically invisible — a test
    built on a high-SNR problem passes even if that term is wrong.
    """
    rng = np.random.default_rng(20240917)
    lmax, ndata, nside, nvec = 4, 80, 8, 12
    nalm = (lmax + 1) ** 2
    a_scale = 0.25

    # Isotropic prior: Sdiag = cl[ell] for every real degree of freedom
    cl = (np.arange(lmax + 1) + 1.0) ** -2.0
    ells_hp, emms_hp = hp.Alm.getlm(lmax)
    ells_full = np.concatenate((ells_hp, ells_hp[emms_hp != 0]))
    Sdiag = cl[ells_full]

    A = rng.standard_normal((ndata, nalm)) * a_scale
    Ndiag = np.abs(rng.standard_normal(ndata)) + 0.5

    Atilde = (1 / np.sqrt(Ndiag))[:, None] * A * np.sqrt(Sdiag)[None, :]
    _, Sigma, Vh = np.linalg.svd(Atilde, full_matrices=False)

    # Dense reference for the whitened and alm-space posteriors
    Vt = Vh[:nvec]
    M = Sigma[:nvec] ** 2 / (Sigma[:nvec] ** 2 + 1)
    Ct = np.eye(nalm) - Vt.T @ np.diag(M) @ Vt
    C_alm = np.sqrt(Sdiag)[:, None] * Ct * np.sqrt(Sdiag)[None, :]

    # Synthesis matrix Y: column j is alm2map of packed basis vector j
    Y = np.empty((hp.nside2npix(nside), nalm))
    for j in range(nalm):
        e = np.zeros(nalm)
        e[j] = 1.0
        Y[:, j] = hp.alm2map(np.asarray(alm1d_to_hp(e)).astype(complex), nside)

    return {
        "lmax": lmax,
        "nalm": nalm,
        "nside": nside,
        "nvec": nvec,
        "cl": cl,
        "Sdiag": Sdiag,
        "Sigma": Sigma,
        "Vh": Vh,
        "Ct": Ct,
        "C_alm": C_alm,
        "Y": Y,
        "var_pix_dense": np.einsum("ij,jk,ik->i", Y, C_alm, Y),
    }


# ======================================================================
# posterior_uncertainty
# ======================================================================


class TestPosteriorUncertainty:
    """Monte Carlo posterior moments against dense linear algebra."""

    def test_std_map_matches_dense_pixel_variance(self, posterior_problem):
        p = posterior_problem
        post = pipeline.posterior_uncertainty(
            p["Vh"],
            p["Sigma"],
            p["Sdiag"],
            p["nvec"],
            p["lmax"],
            nside=p["nside"],
            n_realizations=4000,
            seed=7,
        )
        rel = (
            np.abs(post["std_map"] ** 2 - p["var_pix_dense"])
            / p["var_pix_dense"]
        )
        # 4000 draws => ~1.6% expected scatter on the variance
        assert rel.mean() < 0.03
        assert rel.max() < 0.10

    def test_std_alm_matches_dense_diagonal(self, posterior_problem):
        """Pins the 1/sqrt(2) real/imag packing convention."""
        p = posterior_problem
        post = pipeline.posterior_uncertainty(
            p["Vh"],
            p["Sigma"],
            p["Sdiag"],
            p["nvec"],
            p["lmax"],
            nside=p["nside"],
            n_realizations=4000,
            seed=7,
        )
        lmax = p["lmax"]
        var_packed = np.diag(p["C_alm"])
        n_m0 = lmax + 1
        hp_len = (lmax + 1) * (lmax + 2) // 2

        # m = 0 coefficients are real and carry the packed variance
        got_m0 = post["std_alm"][:n_m0].real
        np.testing.assert_allclose(
            got_m0, np.sqrt(var_packed[:n_m0]), rtol=0.06
        )

        # m > 0: a_lm = (re + i im)/sqrt(2), so each part halves
        got_re = post["std_alm"][n_m0:].real
        got_im = post["std_alm"][n_m0:].imag
        np.testing.assert_allclose(
            got_re, np.sqrt(var_packed[n_m0:hp_len] / 2), rtol=0.06
        )
        np.testing.assert_allclose(
            got_im, np.sqrt(var_packed[hp_len:] / 2), rtol=0.06
        )

    def test_sigma2_prior_matches_isotropic_formula(self, posterior_problem):
        p = posterior_problem
        post = pipeline.posterior_uncertainty(
            p["Vh"],
            p["Sigma"],
            p["Sdiag"],
            p["nvec"],
            p["lmax"],
            nside=p["nside"],
            n_realizations=50,
            seed=7,
        )
        np.testing.assert_allclose(post["cl_prior"], p["cl"], rtol=1e-10)
        ell = np.arange(p["lmax"] + 1)
        expected = np.sum((2 * ell + 1) / (4 * np.pi) * p["cl"])
        assert post["sigma2_prior"] == pytest.approx(expected, rel=1e-10)

    def test_posterior_never_exceeds_prior(self, posterior_problem):
        """The data can only remove variance, never add it."""
        p = posterior_problem
        post = pipeline.posterior_uncertainty(
            p["Vh"],
            p["Sigma"],
            p["Sdiag"],
            p["nvec"],
            p["lmax"],
            nside=p["nside"],
            n_realizations=4000,
            seed=7,
        )
        ratio = post["std_map"] ** 2 / post["sigma2_prior"]
        assert ratio.max() < 1.0


# ======================================================================
# posterior_eigenmodes
# ======================================================================


class TestPosteriorEigenmodes:
    """The SVD already is the eigendecomposition of the posterior."""

    def test_eigenvalues_are_one_over_sigma_squared_plus_one(
        self, posterior_problem
    ):
        p = posterior_problem
        out = pipeline.posterior_eigenmodes(
            p["Vh"][:5],
            p["Sigma"],
            p["cl"],
            p["lmax"],
            nside=p["nside"],
        )
        expected = 1.0 / (p["Sigma"][:5] ** 2 + 1.0)
        np.testing.assert_allclose(out["eigenvalues"], expected, rtol=1e-12)

    def test_eigenvalues_match_dense_eigh_of_whitened_posterior(
        self, posterior_problem
    ):
        """The claimed eigenvalues really are eigenvalues of Ctilde."""
        p = posterior_problem
        out = pipeline.posterior_eigenmodes(
            p["Vh"][: p["nvec"]],
            p["Sigma"],
            p["cl"],
            p["lmax"],
            nside=p["nside"],
        )
        dense = np.linalg.eigvalsh(p["Ct"])
        full = np.concatenate(
            [out["eigenvalues"], np.ones(p["nalm"] - p["nvec"])]
        )
        np.testing.assert_allclose(np.sort(full), dense, atol=1e-12)

    def test_maps_are_unwhitened_by_sqrt_prior(self, posterior_problem):
        """S^{1/2} must be applied, or the modes are whitened patterns."""
        p = posterior_problem
        out = pipeline.posterior_eigenmodes(
            p["Vh"][:3],
            p["Sigma"],
            p["cl"],
            p["lmax"],
            nside=p["nside"],
        )
        for k in range(3):
            mode = np.sqrt(p["Sdiag"]) * p["Vh"][k]
            expected = hp.alm2map(
                np.asarray(alm1d_to_hp(mode)).astype(complex), p["nside"]
            )
            np.testing.assert_allclose(
                out["maps"][k], expected, rtol=1e-10, atol=1e-12
            )

    def test_maps_have_one_row_per_requested_mode(self, posterior_problem):
        p = posterior_problem
        out = pipeline.posterior_eigenmodes(
            p["Vh"][:7],
            p["Sigma"],
            p["cl"],
            p["lmax"],
            nside=p["nside"],
        )
        assert out["maps"].shape == (7, hp.nside2npix(p["nside"]))
        assert out["eigenvalues"].shape == (7,)


# ======================================================================
# n_modes_saved
# ======================================================================


class TestModesToSave:
    """Selecting how many right singular vectors to persist."""

    def test_returns_none_when_not_configured(self, posterior_problem):
        assert pipeline._modes_to_save(posterior_problem["Vh"], {}) is None

    def test_returns_none_when_zero(self, posterior_problem):
        got = pipeline._modes_to_save(
            posterior_problem["Vh"], {"n_modes_saved": 0}
        )
        assert got is None

    def test_returns_requested_number_of_modes(self, posterior_problem):
        p = posterior_problem
        got = pipeline._modes_to_save(p["Vh"], {"n_modes_saved": 6})
        assert got.shape == (6, p["nalm"])
        np.testing.assert_array_equal(got, p["Vh"][:6])

    def test_clamps_to_available_rows(self, posterior_problem):
        p = posterior_problem
        n_avail = p["Vh"].shape[0]
        got = pipeline._modes_to_save(p["Vh"], {"n_modes_saved": n_avail + 50})
        assert got.shape == (n_avail, p["nalm"])


class TestSaveResultsModes:
    """Vh_modes reaches the npz only when it was requested."""

    def _results(self, extra=None):
        d = {
            "freqs": np.array([40]),
            "lmax": 4,
            "x_true": np.zeros(15, dtype=complex),
            "x_rec": np.zeros(15, dtype=complex),
            "cl_prior": np.ones(5),
            "sigma2_prior": 1.0,
            "std_alm": np.zeros(15, dtype=complex),
            "std_map": np.zeros(768),
            "Sigma": np.ones(10),
            "nvec": 10,
            "config": {},
        }
        if extra:
            d.update(extra)
        return d

    def test_absent_when_not_requested(self, tmp_path):
        path = tmp_path / "r.npz"
        pipeline.save_results(self._results(), path)
        with np.load(path, allow_pickle=False) as d:
            assert "Vh_modes" not in d.files

    def test_present_when_supplied(self, tmp_path):
        path = tmp_path / "r.npz"
        modes = np.arange(3 * 25, dtype=float).reshape(3, 25)
        pipeline.save_results(self._results({"Vh_modes": modes}), path)
        with np.load(path, allow_pickle=False) as d:
            np.testing.assert_array_equal(d["Vh_modes"], modes)
