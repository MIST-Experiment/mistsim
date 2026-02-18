"""Tests for the sim module."""

import pytest
import numpy as np
import jax.numpy as jnp

from mistsim.beam import Beam
from mistsim.sim import Simulator, correct_ground_loss

# Mark tests that require full astropy coordinate transforms
# These may fail in sandboxed environments without internet access
# due to SPICE kernel download requirements
requires_coords = pytest.mark.skipif(
    True,  # Skip in sandboxed environment
    reason="Requires astropy coordinate transforms with SPICE kernels",
)


class TestSimulatorInitialization:
    """Tests for Simulator class initialization."""

    @pytest.fixture
    def valid_beam(self):
        """Create a valid beam for testing."""
        nfreq = 2
        ntheta = 181
        nphi = 360
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([50.0, 100.0])
        lmax = 50
        return Beam(data=data, freqs=freqs, lmax=lmax)

    @pytest.fixture
    def valid_sky_alm(self):
        """Create valid sky alm for testing."""
        # For lmax=50, shape is (nfreq, lmax+1, 2*lmax+1)
        nfreq = 2
        lmax = 50
        sky_alm = np.random.randn(
            nfreq, lmax + 1, 2 * lmax + 1
        ) + 1j * np.random.randn(nfreq, lmax + 1, 2 * lmax + 1)
        return jnp.array(sky_alm)

    @requires_coords
    def test_simulator_initialization_basic(self, valid_beam, valid_sky_alm):
        """Test basic simulator initialization."""
        freqs = np.array([50.0, 100.0])
        times_jd = np.array([2459000.0, 2459000.5])
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

    @requires_coords
    def test_simulator_with_altitude(self, valid_beam, valid_sky_alm):
        """Test simulator initialization with altitude."""
        freqs = np.array([50.0, 100.0])
        times_jd = np.array([2459000.0])
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

    @requires_coords
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

    @requires_coords
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

    @requires_coords
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

    @requires_coords
    def test_compute_beam_eq(self, simulator):
        """Test compute_beam_eq method."""
        beam_eq_alm = simulator.compute_beam_eq()

        # Should have shape (nfreq, lmax+1, 2*lmax+1)
        expected_shape = (2, 31, 61)
        assert beam_eq_alm.shape == expected_shape
        assert jnp.iscomplexobj(beam_eq_alm)

    @requires_coords
    def test_compute_ground_contribution(self, simulator):
        """Test compute_ground_contribution method."""
        vis_gnd = simulator.compute_ground_contribution()

        # Should return one value per frequency
        assert vis_gnd.shape == (2,)
        # Ground contribution should be positive (temperature)
        assert jnp.all(vis_gnd > 0)
        # Should be less than or equal to ground temperature
        assert jnp.all(vis_gnd <= simulator.Tgnd)

    @requires_coords
    def test_sim_output_shape(self, simulator):
        """Test that sim() returns correct output shape."""
        vis = simulator.sim()

        # Output should have shape (nfreq, ntimes)
        expected_shape = (2, 2)
        assert vis.shape == expected_shape

    @requires_coords
    def test_sim_output_is_real(self, simulator):
        """Test that sim() returns real values."""
        vis = simulator.sim()
        assert not jnp.iscomplexobj(vis)

    @requires_coords
    def test_sim_output_finite(self, simulator):
        """Test that sim() returns finite values."""
        vis = simulator.sim()
        assert jnp.all(jnp.isfinite(vis))

    @requires_coords
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

    @requires_coords
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

    @requires_coords
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
        sky_alm = np.random.randn(
            nfreq, lmax + 1, 2 * lmax + 1
        ) + 1j * np.random.randn(nfreq, lmax + 1, 2 * lmax + 1)
        sky_alm = jnp.array(sky_alm)

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
        assert vis.shape == (nfreq, 5)
        assert jnp.all(jnp.isfinite(vis))
        assert not jnp.iscomplexobj(vis)

    @requires_coords
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
