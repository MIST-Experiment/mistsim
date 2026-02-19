"""Tests for the sim module."""

import astropy.units as u
import croissant.jax as crojax
import pytest
import numpy as np
import jax.numpy as jnp
import s2fft

from mistsim.beam import Beam
from mistsim.sim import Simulator, correct_ground_loss


class TestSimulatorInitialization:
    """Tests for Simulator class initialization."""

    @pytest.fixture
    def valid_beam(self):
        """Create a valid beam for testing."""
        nfreq = 2
        ntheta = 181
        nphi = 360
        data = jnp.ones((nfreq, ntheta, nphi))
        freqs = jnp.array([50.0, 100.0])
        lmax = 50
        return Beam(data=data, freqs=freqs, lmax=lmax)

    @pytest.fixture
    def valid_sky_alm(self):
        """Create valid sky alm for testing."""
        # For lmax=50, shape is (nfreq, lmax+1, 2*lmax+1)
        lmax = 50
        rng = np.random.default_rng(seed=42)
        sky_alm = s2fft.utils.signal_generator.generate_flm(
            rng, lmax + 1, reality=True
        )
        sky_alm2 = s2fft.utils.signal_generator.generate_flm(
            rng, lmax + 1, reality=True
        )
        sky_alm = jnp.array([sky_alm, sky_alm2])
        return sky_alm

    def test_simulator_initialization_basic(self, valid_beam, valid_sky_alm):
        """Test basic simulator initialization."""
        freqs = jnp.array([50.0, 100.0])
        times_jd = jnp.array([2459000.0, 2459000.5])
        lon = -122.0
        lat = 37.0

        sim = Simulator(
            beam=valid_beam,
            sky_alm=valid_sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=lon,
            lat=lat,
        )

        assert sim.lmax == 50
        assert sim.L == 51
        assert jnp.allclose(sim.freqs, freqs)
        assert jnp.allclose(sim.lon, lon)
        assert jnp.allclose(sim.lat, lat)

    def test_simulator_with_altitude(self, valid_beam, valid_sky_alm):
        """Test simulator initialization with altitude."""
        freqs = jnp.array([50.0, 100.0])
        times_jd = jnp.array([2459000.0])
        lon = -122.0
        lat = 37.0
        alt = 1000.0

        sim = Simulator(
            beam=valid_beam,
            sky_alm=valid_sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=lon,
            lat=lat,
            alt=alt,
        )

        assert jnp.isclose(sim.alt, alt)

    def test_simulator_with_ground_temperature(
        self, valid_beam, valid_sky_alm
    ):
        """Test simulator initialization with custom ground temperature."""
        freqs = np.array([50.0, 100.0])
        times_jd = np.array([2459000.0])
        lon = -122.0
        lat = 37.0
        Tgnd = 280.0

        sim = Simulator(
            beam=valid_beam,
            sky_alm=valid_sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=lon,
            lat=lat,
            Tgnd=Tgnd,
        )

        assert jnp.isclose(sim.Tgnd, Tgnd)

    def test_simulator_mismatched_frequencies_error(
        self, valid_beam, valid_sky_alm
    ):
        """Test that mismatched frequencies raise ValueError."""
        # Beam has frequencies [50.0, 100.0]
        # Provide different frequencies to simulator
        freqs = np.array([60.0, 110.0])
        times_jd = np.array([2459000.0])
        lon = -122.0
        lat = 37.0

        with pytest.raises(ValueError, match="frequencies do not match"):
            Simulator(
                beam=valid_beam,
                sky_alm=valid_sky_alm,
                times_jd=times_jd,
                freqs=freqs,
                lon=lon,
                lat=lat,
            )

    def test_simulator_mismatched_lmax_error(self, valid_beam):
        """Test that mismatched lmax values raise ValueError."""
        # valid_beam has lmax=50
        # Create sky_alm with different lmax
        nfreq = 2
        wrong_lmax = 30
        sky_alm = np.random.randn(
            nfreq, wrong_lmax + 1, 2 * wrong_lmax + 1
        ) + 1j * np.random.randn(nfreq, wrong_lmax + 1, 2 * wrong_lmax + 1)

        freqs = np.array([50.0, 100.0])
        times_jd = np.array([2459000.0])
        lon = -122.0
        lat = 37.0

        with pytest.raises(ValueError, match="different lmax values"):
            Simulator(
                beam=valid_beam,
                sky_alm=jnp.array(sky_alm),
                times_jd=times_jd,
                freqs=freqs,
                lon=lon,
                lat=lat,
            )

    def test_simulator_single_time(self, valid_beam, valid_sky_alm):
        """Test simulator with a single time point."""
        freqs = np.array([50.0, 100.0])
        times_jd = np.array([2459000.0])
        lon = 0.0
        lat = 0.0

        sim = Simulator(
            beam=valid_beam,
            sky_alm=valid_sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=lon,
            lat=lat,
        )

        assert sim.times_jd.shape == (1,)

    def test_simulator_multiple_times(self, valid_beam, valid_sky_alm):
        """Test simulator with multiple time points."""
        freqs = np.array([50.0, 100.0])
        times_jd = np.linspace(2459000.0, 2459001.0, 10)
        lon = 0.0
        lat = 0.0

        sim = Simulator(
            beam=valid_beam,
            sky_alm=valid_sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=lon,
            lat=lat,
        )

        assert sim.times_jd.shape == (10,)


class TestSimulatorMethods:
    """Tests for Simulator class methods."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator for testing."""
        nfreq = 2
        ntheta = 181
        nphi = 360
        lmax = 30

        # Create beam
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([50.0, 100.0])
        beam = Beam(data=data, freqs=freqs, lmax=lmax)

        # Create sky alm
        sky_alm = np.random.randn(
            nfreq, lmax + 1, 2 * lmax + 1
        ) + 1j * np.random.randn(nfreq, lmax + 1, 2 * lmax + 1)
        sky_alm = jnp.array(sky_alm)

        times_jd = np.array([2459000.0, 2459000.5])
        lon = -122.0
        lat = 37.0

        return Simulator(
            beam=beam,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=lon,
            lat=lat,
        )

    def test_compute_beam_eq(self, simulator):
        """Test compute_beam_eq method."""
        beam_eq_alm = simulator.compute_beam_eq()

        # Should have shape (nfreq, lmax+1, 2*lmax+1)
        expected_shape = (2, 31, 61)
        assert beam_eq_alm.shape == expected_shape
        assert jnp.iscomplexobj(beam_eq_alm)

    def test_compute_ground_contribution(self, simulator):
        """Test compute_ground_contribution method."""
        vis_gnd = simulator.compute_ground_contribution()

        # Should return one value per frequency
        assert vis_gnd.shape == (2,)
        # Ground contribution should be positive (temperature)
        assert jnp.all(vis_gnd > 0)
        # Should be less than or equal to ground temperature
        assert jnp.all(vis_gnd <= simulator.Tgnd)

    def test_sim_output_shape(self, simulator):
        """Test that sim() returns correct output shape."""
        vis = simulator.sim()

        # Output should have shape (ntimes, nfreq)
        expected_shape = (2, 2)
        assert vis.shape == expected_shape

    def test_sim_output_is_real(self, simulator):
        """Test that sim() returns real values."""
        vis = simulator.sim()
        assert not jnp.iscomplexobj(vis)

    def test_sim_output_finite(self, simulator):
        """Test that sim() returns finite values."""
        vis = simulator.sim()
        assert jnp.all(jnp.isfinite(vis))

    def test_sim_with_single_time(self):
        """Test simulation with a single time point."""
        nfreq = 1
        ntheta = 181
        nphi = 360
        lmax = 20

        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0])
        beam = Beam(data=data, freqs=freqs, lmax=lmax)

        sky_alm = np.random.randn(
            nfreq, lmax + 1, 2 * lmax + 1
        ) + 1j * np.random.randn(nfreq, lmax + 1, 2 * lmax + 1)
        sky_alm = jnp.array(sky_alm)

        times_jd = np.array([2459000.0])

        sim = Simulator(
            beam=beam,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=0.0,
            lat=0.0,
        )

        vis = sim.sim()
        assert vis.shape == (1, 1)

    def test_sim_different_times_different_results(self, simulator):
        """Test that different times produce different results."""
        vis = simulator.sim()
        # Results at different times should generally be different
        # (unless sky is completely uniform, which is unlikely with
        # random data)
        if vis.shape[1] > 1:
            # Check that not all time points are identical
            assert not jnp.allclose(vis[:, 0], vis[:, 1], rtol=1e-10)


class TestCorrectGroundLoss:
    """Tests for the correct_ground_loss function."""

    def test_correct_ground_loss_basic(self):
        """Test basic ground loss correction."""
        vis = jnp.array([100.0, 150.0])
        fgnd = jnp.array([0.5, 0.5])
        Tgnd = jnp.array(300.0)

        corrected = correct_ground_loss(vis, fgnd, Tgnd)

        # Check shape is preserved
        assert corrected.shape == vis.shape
        # Corrected values should be different from input
        assert not jnp.allclose(corrected, vis)

    def test_correct_ground_loss_zero_fgnd(self):
        """Test correction with zero ground fraction."""
        vis = jnp.array([100.0, 150.0])
        fgnd = jnp.array([0.0, 0.0])
        Tgnd = jnp.array(300.0)

        corrected = correct_ground_loss(vis, fgnd, Tgnd)

        # With zero ground fraction, result should equal input
        assert jnp.allclose(corrected, vis)

    def test_correct_ground_loss_mathematical_correctness(self):
        """Test mathematical correctness of ground loss correction."""
        vis = jnp.array([200.0])
        fgnd = jnp.array([0.25])
        Tgnd = jnp.array(300.0)

        corrected = correct_ground_loss(vis, fgnd, Tgnd)

        # Manual calculation
        fsky = 1 - fgnd
        expected = (vis - fgnd * Tgnd) / fsky

        assert jnp.allclose(corrected, expected)

    def test_correct_ground_loss_array_broadcast(self):
        """Test that function handles broadcasting correctly."""
        # Multiple frequencies and times
        vis = jnp.ones((3, 5)) * 100.0
        fgnd = jnp.array([0.1, 0.2, 0.3])[
            :, None
        ]  # Shape (3, 1) for broadcasting
        Tgnd = jnp.array(300.0)

        corrected = correct_ground_loss(vis, fgnd, Tgnd)

        # Shape should be preserved
        assert corrected.shape == vis.shape
        # Different frequencies should have different corrections
        assert not jnp.allclose(corrected[0], corrected[1])

    def test_correct_ground_loss_inverse_operation(self):
        """Test that correction can be reversed."""
        # Start with some sky signal
        vis_sky = jnp.array([150.0, 200.0])
        fgnd = jnp.array([0.3, 0.3])
        Tgnd = jnp.array(300.0)

        # Add ground contribution (reverse of correction)
        fsky = 1 - fgnd
        vis_with_ground = vis_sky * fsky + fgnd * Tgnd

        # Correct it back
        corrected = correct_ground_loss(vis_with_ground, fgnd, Tgnd)

        # Should recover original sky signal
        assert jnp.allclose(corrected, vis_sky, rtol=1e-6)


class TestIntegration:
    """Integration tests for the full simulation pipeline."""

    def test_full_pipeline(self):
        """Test the complete pipeline from beam to visibilities."""
        # Create a simple but realistic setup
        nfreq = 1
        ntheta = 181
        nphi = 360
        lmax = 20

        # Create beam
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0])
        beam = Beam(data=data, freqs=freqs, lmax=lmax)

        # Create sky alm
        rng = np.random.default_rng(seed=42)
        sky_alm = s2fft.utils.signal_generator.generate_flm(
            rng, lmax + 1, reality=True
        )
        sky_alm = jnp.array(sky_alm)[None, :, :]  # Add frequency axis

        # Create simulator
        times_jd = np.linspace(2459000.0, 2459000.5, 5)
        sim = Simulator(
            beam=beam,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=-122.0,
            lat=37.0,
            alt=0.0,
            Tgnd=300.0,
        )

        # Run simulation
        vis = sim.sim()

        # Check output
        assert vis.shape == (5, nfreq)
        assert jnp.all(jnp.isfinite(vis))
        assert not jnp.iscomplexobj(vis)

    def test_ground_loss_correction_pipeline(self):
        """Test full pipeline including ground loss correction."""
        nfreq = 1
        ntheta = 181
        nphi = 360
        lmax = 15

        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0])
        beam = Beam(data=data, freqs=freqs, lmax=lmax)

        sky_alm = np.random.randn(
            nfreq, lmax + 1, 2 * lmax + 1
        ) + 1j * np.random.randn(nfreq, lmax + 1, 2 * lmax + 1)
        sky_alm = jnp.array(sky_alm)

        times_jd = np.array([2459000.0])
        Tgnd = 300.0

        sim = Simulator(
            beam=beam,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=0.0,
            lat=0.0,
            Tgnd=Tgnd,
        )

        # Get visibilities
        vis = sim.sim()

        # Get ground fraction
        fgnd = beam.compute_fgnd()

        # Correct ground loss
        vis_corrected = correct_ground_loss(vis, fgnd, Tgnd)

        # Corrected visibilities should be different from original
        assert not jnp.allclose(vis, vis_corrected)
        assert jnp.all(jnp.isfinite(vis_corrected))

    def test_monopole_sky(self):
        """Test that a monopole sky produces expected visibilities."""
        nfreq = 2
        ntheta = 181
        nphi = 360
        lmax = 30

        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0, 150.0])

        # no horizon
        horizon_mask = np.ones((ntheta, nphi), dtype=bool)
        beam = Beam(data=data, freqs=freqs, lmax=lmax, horizon=horizon_mask)

        # Monopole sky: only alm[0,0] is nonzero
        sky_alm = jnp.zeros((nfreq, lmax + 1, 2 * lmax + 1), dtype=complex)
        sky_alm = sky_alm.at[:, 0, lmax].set(1.0)

        # should get the same visibility at all times and frequencies
        times_jd = np.linspace(2459000.0, 2459001.0, 10)
        sim = Simulator(
            beam=beam,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=30.0,
            lat=45.0,
        )

        vis = sim.sim()
        # All visibilities should be the same (monopole)
        assert jnp.allclose(vis, vis[0, 0])
        # the expected temp should be a00 * Y00 = 1 * sqrt(1/4pi) ~ 0.28209479
        expected_vis = 1.0 * np.sqrt(1 / (4 * np.pi))
        assert jnp.isclose(vis[0, 0], expected_vis, rtol=1e-6)

        # use default horizon (about half the sky visible)
        beam_default_horizon = Beam(data=data, freqs=freqs, lmax=lmax)
        sim_default_horizon = Simulator(
            beam=beam_default_horizon,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=30.0,
            lat=45.0,
            Tgnd=0,
        )
        vis_default_horizon = sim_default_horizon.sim()
        # With the default horizon, we should see about half the monopole
        assert jnp.isclose(
            vis_default_horizon[0, 0], expected_vis / 2, rtol=0.1
        )
        # correcting for ground loss should recover the full monopole
        fgnd = beam_default_horizon.compute_fgnd()
        vis_corrected = correct_ground_loss(vis_default_horizon, fgnd, Tgnd=0)
        assert jnp.isclose(vis_corrected[0, 0], expected_vis, rtol=1e-6)

        # non zero ground temp should increase the antenna temperature
        sim_with_ground = Simulator(
            beam=beam_default_horizon,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=30.0,
            lat=45.0,
            Tgnd=300.0,
        )
        vis_with_ground = sim_with_ground.sim()
        expected_vis_gnd = expected_vis * (1 - fgnd) + 300.0 * fgnd
        assert jnp.allclose(vis_with_ground[0, 0], expected_vis_gnd, rtol=1e-6)
        # correcting for ground loss should recover the full monopole
        vis_corrected_gnd = correct_ground_loss(
            vis_with_ground, fgnd, Tgnd=300.0
        )
        assert jnp.allclose(vis_corrected_gnd[0, 0], expected_vis, rtol=1e-6)

    def test_time_dependence(self):
        """Use a sky with cos(phi) dependence to test that visibilities change with time."""
        nfreq = 1
        ntheta = 181
        nphi = 360
        lmax = 30

        theta = np.linspace(0, np.pi, ntheta)
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        # ensure the beam has same dependence
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        beam_pattern = -np.sin(theta_grid) * np.cos(phi_grid)
        data = np.tile(beam_pattern[None, :, :], (nfreq, 1, 1))
        # add a monopole term to avoid bad normalization from zero mean
        data += 1.0
        freqs = np.array([100.0])
        horizon = np.ones((ntheta, nphi), dtype=bool)  # no horizon
        beam = Beam(data=data, freqs=freqs, lmax=lmax, horizon=horizon)

        sky_alm = jnp.zeros((nfreq, lmax + 1, 2 * lmax + 1), dtype=complex)
        # make all alms 0 expect l=1, m=1
        sky_alm = sky_alm.at[:, 1, lmax + 1].set(1.0)
        sky_alm = sky_alm.at[:, 1, lmax - 1].set(-1.0)  # reality condition

        times_jd = np.linspace(2459000.0, 2459001.0, 10)
        sim = Simulator(
            beam=beam,
            sky_alm=sky_alm,
            times_jd=times_jd,
            freqs=freqs,
            lon=0.0,
            lat=0.0,
        )

        vis = sim.sim()
        # Visibilities should change with time due to cos(phi) dependence
        assert not jnp.allclose(vis[:, 0], vis[0, 0], rtol=1e-10)

        # we should see beam_alm * sky_alm for the l=1, m=1 mode
        beam_alm = sim.compute_beam_eq()
        expected_vis = beam_alm[:, 1, lmax + 1] * sky_alm[:, 1, lmax + 1]
        expected_vis *= 2  # for symmetry of m=1 mode
        expected_vis = expected_vis.real / sim.beam.compute_norm()
        assert jnp.isclose(vis[0, 0], expected_vis[0], rtol=1e-6)

        # the dependence should be sinusoidal with period of 1 day (since m=1 mode)
        dtsec = sim.times_jd - sim.times_jd[0]
        dtsec = dtsec * 24 * 3600
        sim_phases = crojax.simulator.rot_alm_z(
            sim.lmax, times=dtsec, world="earth"
        )
        # only care about the m=1 mode
        phase_m1 = sim_phases[:, lmax + 1]
        assert jnp.allclose(sim_phases[:, lmax - 1], phase_m1.conj(), rtol=1e-10)

        sidereal_day_length = (1 * u.sday).to(u.day).value
        arg = -2 * np.pi * (times_jd - times_jd[0]) / sidereal_day_length

        assert jnp.allclose(phase_m1, jnp.exp(1j * arg), atol=1e-3)

        v0 = vis[0, 0]
        v1 = vis[1, 0]
        th1 = arg[1]
        # the visibility should follow v = v0 * cos(th) + v1 * sin(th) where th is the phase of the m=1 mode
        asin = (v0 * jnp.cos(th1) - v1) / jnp.sin(th1)
        expected_vis = v0 * jnp.cos(arg) - asin * jnp.sin(arg)

        assert jnp.allclose(
            vis[:, 0],
            expected_vis,
            rtol=1e-6,
        )
