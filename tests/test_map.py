import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as sla
from astropy import units as u
from astropy.time import Time

import mistsim as ms
from mistsim import pipeline


@pytest.fixture
def sim():
    nside = 16
    npix = 12 * nside**2
    freqs = jnp.linspace(50, 100, num=50)
    ntimes = 12
    t0 = Time("2024-01-01T00:00:00")
    dt = 1 * u.sday / ntimes
    times = cro.utils.time_array(t_start=t0, N_times=ntimes, delta_t=dt)
    lon = -90
    lat = 80

    tsky = 180 * (freqs / 180) ** (-2.5)
    tsky = tsky[:, None] * jnp.ones((1, npix))
    sky = ms.Sky(tsky, freqs)

    beam_map = jnp.ones((freqs.size, 181, 360))
    beam = ms.Beam(beam_map, freqs, sampling="mwss")

    s = ms.Simulator(beam, sky, times.jd, freqs, lon, lat, Tgnd=300)
    return s


@pytest.mark.parametrize("lmax", [16, 32])
def test_pack_unpack_symmetry(lmax):
    Nfreq = 2
    N_total = Nfreq * (lmax + 1) ** 2

    # 1. Create a random physical (real) state vector
    x_orig = jax.random.normal(jax.random.PRNGKey(0), (N_total,))

    # 2. Unpack to complex s2fft format
    flm_complex = ms.mapmaking.unpack_real_to_s2fft(x_orig, lmax, Nfreq)

    # 3. Test conjugate symmetry: a(l, -m) == (-1)^m * a(l, m)*
    for m in range(1, lmax + 1):
        left_side = flm_complex[:, m:, lmax - m]
        right_side = ((-1) ** m) * jnp.conj(flm_complex[:, m:, lmax + m])
        assert jnp.allclose(left_side, right_side), f"Symmetry failed at m={m}"

    # 4. Pack back to 1D real vector and test losslessness
    x_repacked = ms.mapmaking.pack_s2fft_to_real(flm_complex)
    assert jnp.allclose(x_orig, x_repacked)


def test_forward(sim):
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    lmax = cro.utils.lmax_from_shape(sky_alm.shape)
    x_real = ms.mapmaking.pack_s2fft_to_real(sky_alm)

    beam_alm = sim.compute_beam_eq()
    beam_alm = cro.utils.reduce_lmax(beam_alm, lmax)
    phases = sim.phases

    expected_raw = sim.sim()
    # need to ground loss correct since mapmaker assumes it's done
    expected = cro.simulator.correct_ground_loss(
        expected_raw, sim.beam.compute_fgnd(), sim.Tgnd
    )

    # out is a column vector of shape (ntimes*nfreq, 1)
    out = ms.mapmaking._forward(x_real, beam_alm, phases)
    assert out.shape == (sim.freqs.size * sim.times_jd.size,)
    out2d = out.reshape(sim.times_jd.size, sim.freqs.size)
    assert np.allclose(out2d, expected)


def test_A(sim):
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    x_real = ms.mapmaking.pack_s2fft_to_real(sky_alm)

    Amat = ms.mapmaking.make_Amat(sim)
    ntimes = sim.times_jd.size
    nfreq = sim.freqs.size
    assert Amat.shape == (ntimes * nfreq, x_real.size)

    A_wfall_rav = Amat @ x_real
    A_wfall = A_wfall_rav.reshape(ntimes, nfreq)
    # compare with simulator
    sim_wfall = sim.sim()
    # matrix approach does not try to capture ground loss
    fgnd = sim.beam.compute_fgnd()
    sim_wfall = cro.simulator.correct_ground_loss(sim_wfall, fgnd, sim.Tgnd)

    assert A_wfall.shape == sim_wfall.shape
    assert jnp.allclose(A_wfall, sim_wfall)


def test_Alinear(sim):
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    x_real = ms.mapmaking.pack_s2fft_to_real(sky_alm)
    Amat = ms.mapmaking.make_Amat(sim)

    xzeros = jnp.zeros_like(x_real)
    yzeros = Amat @ xzeros
    assert jnp.allclose(yzeros, 0), "A @ 0 should be 0"

    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(x_real.shape)
    x2 = rng.standard_normal(x_real.shape)
    ysum = Amat @ (x1 + x2)
    y1 = Amat @ x1
    y2 = Amat @ x2
    assert jnp.allclose(ysum, y1 + y2)

    scalar = 42
    yscaled = Amat @ (scalar * x1)
    y1_scaled = scalar * (Amat @ x1)
    assert jnp.allclose(yscaled, y1_scaled)
    yscaled2 = Amat @ (scalar * x2)
    y2_scaled = scalar * (Amat @ x2)
    assert jnp.allclose(yscaled2, y2_scaled)


def test_adjoint(sim):
    """
    Ensures that Amat.rmatvec() correctly computes the conjugate transpose.
    """
    Amat = ms.mapmaking.make_Amat(sim)
    rng = np.random.default_rng(0)

    x = rng.standard_normal(Amat.shape[1])
    y = rng.standard_normal(Amat.shape[0])

    Ax = Amat.matvec(x)
    Ahy = Amat.rmatvec(y)

    inner1 = np.vdot(Ax, y)
    inner2 = np.vdot(x, Ahy)

    # Assert they are equal up to standard floating-point precision
    np.testing.assert_allclose(
        inner1.real,
        inner2.real,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Real parts of the inner products do not match!",
    )
    np.testing.assert_allclose(
        inner1.imag,
        inner2.imag,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Imaginary parts of the inner products do not match!",
    )


def test_stack_As_adjoint():
    """stack_As adjoint consistency with 1D and 2D inputs."""
    rng = np.random.default_rng(42)
    m1, m2, n = 20, 30, 10

    # Two dense operators
    M1 = rng.standard_normal((m1, n))
    M2 = rng.standard_normal((m2, n))
    A1 = sla.aslinearoperator(M1)
    A2 = sla.aslinearoperator(M2)
    Astack = ms.mapmaking.stack_As(A1, A2)

    # Reference: dense stacked matrix
    Mstack = np.vstack([M1, M2])

    x = rng.standard_normal(n)
    y = rng.standard_normal(m1 + m2)

    # Forward
    np.testing.assert_allclose(Astack.matvec(x), Mstack @ x)

    # Adjoint
    np.testing.assert_allclose(Astack.rmatvec(y), Mstack.T @ y)

    # Adjoint consistency: <Ax, y> == <x, A^H y>
    Ax = Astack.matvec(x)
    Ahy = Astack.rmatvec(y)
    np.testing.assert_allclose(
        np.dot(Ax, y),
        np.dot(x, Ahy),
        rtol=1e-10,
    )


def test_stack_As_2d_input():
    """stack_As handles 2D column-vector inputs from svds."""
    rng = np.random.default_rng(42)
    m1, m2, n = 20, 30, 10
    M1 = rng.standard_normal((m1, n))
    M2 = rng.standard_normal((m2, n))
    A1 = sla.aslinearoperator(M1)
    A2 = sla.aslinearoperator(M2)
    Astack = ms.mapmaking.stack_As(A1, A2)
    Mstack = np.vstack([M1, M2])

    # 2D column vector — this is what scipy passes internally
    x_2d = rng.standard_normal((n, 1))
    y_2d = rng.standard_normal((m1 + m2, 1))

    # Should not raise, and values must match dense reference
    fwd = np.asarray(Astack.matvec(x_2d)).ravel()
    adj = np.asarray(Astack.rmatvec(y_2d)).ravel()
    np.testing.assert_allclose(fwd, (Mstack @ x_2d).ravel())
    np.testing.assert_allclose(adj, (Mstack.T @ y_2d).ravel())


def test_stack_As_n_operators():
    """stack_As works with more than two operators."""
    rng = np.random.default_rng(42)
    ms_ = [10, 15, 20]
    n = 8
    matrices = [rng.standard_normal((m, n)) for m in ms_]
    ops = [sla.aslinearoperator(M) for M in matrices]
    Astack = ms.mapmaking.stack_As(*ops)

    Mstack = np.vstack(matrices)
    x = rng.standard_normal(n)
    y = rng.standard_normal(sum(ms_))

    np.testing.assert_allclose(Astack.matvec(x), Mstack @ x)
    np.testing.assert_allclose(Astack.rmatvec(y), Mstack.T @ y)


def test_Atilde_with_stacked_A():
    """Atilde works with a stacked A-matrix through svds."""
    rng = np.random.default_rng(42)
    m1, m2, n = 30, 40, 10
    M1 = rng.standard_normal((m1, n))
    M2 = rng.standard_normal((m2, n))
    A1 = sla.aslinearoperator(M1)
    A2 = sla.aslinearoperator(M2)
    Astack = ms.mapmaking.stack_As(A1, A2)

    Ndiag = np.abs(rng.standard_normal(m1 + m2)) + 0.1
    Sdiag = np.abs(rng.standard_normal(n)) + 0.1
    Atilde = ms.mapmaking.make_Atilde(Ndiag, Astack, Sdiag)

    # This is the operation that failed before the fix:
    # svds calls _rmatmat internally which passes 2D vectors.
    U, Sigma, Vh = sla.svds(Atilde, k=5)
    assert Sigma.shape == (5,)
    assert U.shape == (m1 + m2, 5)
    assert Vh.shape == (5, n)


# ------------------------------------------------------------------
# Pure JAX operator tests
# ------------------------------------------------------------------


def test_jax_operators(sim):
    """Pure JAX operators match scipy LinearOperator."""
    Amat = ms.mapmaking.make_Amat(sim)
    ops = ms.mapmaking.make_operators_jax(sim)

    rng = np.random.default_rng(42)
    x = rng.standard_normal(Amat.shape[1])
    y = rng.standard_normal(Amat.shape[0])

    # Forward
    fwd_scipy = Amat.matvec(x)
    fwd_jax = np.asarray(ops["forward_fn"](jnp.asarray(x)))
    np.testing.assert_allclose(fwd_jax, fwd_scipy.real, rtol=1e-10)

    # Adjoint
    adj_scipy = Amat.rmatvec(y)
    adj_jax = np.asarray(ops["adjoint_fn"](jnp.asarray(y)))
    np.testing.assert_allclose(adj_jax, adj_scipy.real, rtol=1e-10)


def test_jax_adjoint_consistency(sim):
    """JAX forward/adjoint satisfy <Ax, y> = <x, A^H y>."""
    ops = ms.mapmaking.make_operators_jax(sim)

    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.standard_normal(ops["shape"][1]))
    y = jnp.asarray(rng.standard_normal(ops["shape"][0]))

    Ax = ops["forward_fn"](x)
    Aty = ops["adjoint_fn"](y)

    inner1 = jnp.dot(Ax, y)
    inner2 = jnp.dot(x, Aty)
    np.testing.assert_allclose(float(inner1), float(inner2), rtol=1e-5)


def test_cg_wiener_vs_dense():
    """CG Wiener filter matches direct dense solve."""
    rng = np.random.default_rng(42)
    lmax = 4
    nalm = (lmax + 1) ** 2  # 25
    ndata = 50

    M = rng.standard_normal((ndata, nalm))
    Ndiag = np.abs(rng.standard_normal(ndata)) + 0.1
    Sdiag = np.abs(rng.standard_normal(nalm)) + 0.1

    y = rng.standard_normal(ndata)
    noise = rng.standard_normal(ndata) * 0.01

    # Dense reference solve
    Nm12 = 1.0 / np.sqrt(Ndiag)
    S12 = np.sqrt(Sdiag)
    Atilde_dense = np.diag(Nm12) @ M @ np.diag(S12)
    y_tilde = Nm12 * (y + noise)
    AtA = Atilde_dense.T @ Atilde_dense + np.eye(nalm)
    rhs = Atilde_dense.T @ y_tilde
    x_tilde_dense = np.linalg.solve(AtA, rhs)
    x_dense = S12 * x_tilde_dense
    x_dense_hp = np.asarray(ms.mapmaking.alm1d_to_hp(x_dense))

    # CG solve
    Mjax = jnp.asarray(M)

    def fwd(x):
        return Mjax @ x

    def adj(yv):
        return Mjax.T @ yv

    atilde_fwd, atilde_adj = ms.mapmaking.make_atilde_fns(
        Ndiag, fwd, adj, Sdiag
    )
    x_cg_hp, info = pipeline.wiener_filter_cg(
        atilde_fwd,
        atilde_adj,
        Ndiag,
        Sdiag,
        y,
        noise,
        tol=1e-12,
        maxiter=1000,
    )

    # JAX CG may return info=None; check solution directly
    np.testing.assert_allclose(np.asarray(x_cg_hp), x_dense_hp, rtol=1e-5)


def test_randomized_svd():
    """Randomized SVD singular values match numpy svd."""
    rng = np.random.default_rng(42)
    m, n = 200, 60
    k = 15

    # Matrix with rapidly decaying singular values
    # so top-k are well-captured by randomized SVD
    U0, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    true_sigma = np.exp(-np.arange(n) * 0.2)
    M = U0[:, :n] @ np.diag(true_sigma) @ V0.T
    Mjax = jnp.asarray(M)

    def fwd(x):
        return Mjax @ x

    def adj(y):
        return Mjax.T @ y

    _, Sigma_rsvd, _ = ms.mapmaking.randomized_svd_jax(
        fwd, adj, n, m, k, p=20, seed=42
    )

    Sigma_ref = true_sigma[:k]
    np.testing.assert_allclose(np.asarray(Sigma_rsvd), Sigma_ref, rtol=1e-3)


# ------------------------------------------------------------------
# Multi-frequency helpers
# ------------------------------------------------------------------


def test_resolve_freq_indices_single():
    """freq_index (singular) → single-element list."""
    cfg = {"sky": {"freq_index": 3, "freq_range": [25, 126]}}
    ix = pipeline._resolve_freq_indices(cfg)
    assert ix == [3]


def test_resolve_freq_indices_multi():
    """freq_indices (plural) → list."""
    cfg = {"sky": {"freq_indices": [0, 5, 10], "freq_range": [25, 126]}}
    ix = pipeline._resolve_freq_indices(cfg)
    assert ix == [0, 5, 10]


def test_resolve_freq_indices_all():
    """freq_indices: "all" → all frequencies."""
    cfg = {"sky": {"freq_indices": "all", "freq_range": [25, 30]}}
    ix = pipeline._resolve_freq_indices(cfg)
    assert ix == [0, 1, 2, 3, 4]


def test_resolve_freq_indices_default():
    """Neither key → all frequencies."""
    cfg = {"sky": {"freq_range": [25, 30]}}
    ix = pipeline._resolve_freq_indices(cfg)
    assert ix == [0, 1, 2, 3, 4]


def test_pad_and_stack():
    """Pad arrays of different lengths."""
    arrays = [np.array([1, 2, 3]), np.array([4, 5])]
    result = pipeline._pad_and_stack(arrays)
    expected = np.array([[1, 2, 3], [4, 5, 0]])
    np.testing.assert_array_equal(result, expected)


def test_forward_single_freq(sim):
    """_forward_single_freq matches _forward_jax for 1-freq."""
    sky_alm = sim.sky.compute_alm_eq(world="earth")
    lmax = cro.utils.lmax_from_shape(sky_alm.shape)
    x_real = ms.mapmaking.pack_s2fft_to_real(sky_alm)

    beam_alm = sim.compute_beam_eq()
    beam_alm = cro.utils.reduce_lmax(beam_alm, lmax)
    phases = sim.phases

    # Full multi-frequency forward over all frequencies
    nfreq = beam_alm.shape[0]
    x_per_freq = x_real.reshape(nfreq, -1)
    y_ref = ms.mapmaking._forward_jax(x_real, beam_alm, phases)

    # Per-frequency forward
    y_parts = []
    for f in range(nfreq):
        y_f = ms.mapmaking._forward_single_freq(
            x_per_freq[f], beam_alm[f], phases
        )
        y_parts.append(y_f)
    # Stack as (ntimes, nfreq) to match _forward_jax layout
    ntimes = sim.times_jd.size
    y_test = jnp.stack(y_parts, axis=1)  # (ntimes, nfreq)
    y_ref_2d = y_ref.reshape(ntimes, nfreq)

    np.testing.assert_allclose(
        np.asarray(y_test),
        np.asarray(y_ref_2d.real),
        rtol=1e-10,
    )
