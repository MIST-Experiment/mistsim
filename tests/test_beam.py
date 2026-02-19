"""Tests for the beam module."""

import pytest
import numpy as np
import jax.numpy as jnp

from mistsim.beam import Beam


class TestBeamInitialization:
    """Tests for Beam class initialization."""

    @pytest.fixture
    def valid_beam_data(self):
        """Create valid beam data for testing."""
        # mwss sampling with L=180 gives 181 theta points and 360 phi points
        nfreq = 3
        ntheta = 181
        nphi = 360
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([50.0, 100.0, 150.0])
        return data, freqs

    def test_beam_initialization_basic(self, valid_beam_data):
        """Test basic beam initialization with valid data."""
        data, freqs = valid_beam_data
        beam = Beam(data=data, freqs=freqs)

        assert beam.data.shape == data.shape
        assert jnp.allclose(beam.freqs, freqs)
        assert beam._lmax == 179
        assert beam.lmax == 179
        assert beam._L == 180

    def test_beam_initialization_with_lmax(self, valid_beam_data):
        """Test beam initialization with custom lmax."""
        data, freqs = valid_beam_data
        lmax = 50
        beam = Beam(data=data, freqs=freqs, lmax=lmax)

        assert beam.lmax == lmax
        assert beam._lmax == 179  # native lmax unchanged

    def test_beam_initialization_invalid_lmax(self, valid_beam_data):
        """Test that providing lmax > native lmax raises ValueError."""
        data, freqs = valid_beam_data
        with pytest.raises(ValueError, match="exceeds natural lmax"):
            Beam(data=data, freqs=freqs, lmax=200)

    def test_beam_initialization_with_horizon(self, valid_beam_data):
        """Test beam initialization with custom horizon mask."""
        data, freqs = valid_beam_data
        ntheta = 181
        nphi = 360
        horizon = np.ones((ntheta, nphi), dtype=bool)
        horizon[100:, :] = False  # Below theta=100 is below horizon

        beam = Beam(data=data, freqs=freqs, horizon=horizon)
        assert beam.horizon.shape == (ntheta, nphi)
        assert jnp.all(beam.horizon[:100])
        assert jnp.all(~beam.horizon[100:])

    def test_beam_initialization_default_horizon(self, valid_beam_data):
        """Test that default horizon is at theta=90 degrees."""
        data, freqs = valid_beam_data
        beam = Beam(data=data, freqs=freqs)

        # Default horizon should be theta <= 90
        expected_horizon = beam.theta <= jnp.radians(90.0)
        assert beam.horizon.shape == (181, 1)  # theta, 1
        assert jnp.allclose(beam.horizon[:, 0], expected_horizon)

    def test_beam_initialization_wrong_data_shape(self):
        """Test that wrong data shape raises ValueError."""
        data = np.ones((3, 100, 200))  # Wrong shape
        freqs = np.array([50.0, 100.0, 150.0])

        with pytest.raises(ValueError, match="does not match expected shape"):
            Beam(data=data, freqs=freqs)

    def test_beam_initialization_single_frequency(self):
        """Test beam initialization with a single frequency."""
        ntheta = 181
        nphi = 360
        data = np.ones((1, ntheta, nphi))
        freqs = np.array([100.0])

        beam = Beam(data=data, freqs=freqs)
        assert beam.data.shape == (1, ntheta, nphi)
        assert beam.freqs.shape == (1,)

    def test_beam_initialization_scalar_frequency(self):
        """Test beam initialization with scalar frequency."""
        ntheta = 181
        nphi = 360
        data = np.ones((1, ntheta, nphi))
        freqs = 100.0  # scalar

        beam = Beam(data=data, freqs=freqs)
        assert beam.freqs.shape == (1,)
        assert jnp.isclose(beam.freqs[0], 100.0)

    def test_beam_tilt_not_implemented(self, valid_beam_data):
        """Test that non-zero beam_tilt raises NotImplementedError."""
        data, freqs = valid_beam_data
        with pytest.raises(
            NotImplementedError, match="Beam tilt is not yet implemented"
        ):
            Beam(data=data, freqs=freqs, beam_tilt=10.0)

    def test_beam_az_rot_zero(self, valid_beam_data):
        """Test beam with zero azimuth rotation."""
        data, freqs = valid_beam_data
        beam = Beam(data=data, freqs=freqs, beam_az_rot=0.0)
        assert jnp.isclose(beam.beam_az_rot, 0.0)

    def test_beam_az_rot_nonzero(self, valid_beam_data):
        """Test beam with non-zero azimuth rotation."""
        data, freqs = valid_beam_data
        beam = Beam(data=data, freqs=freqs, beam_az_rot=45.0)
        assert jnp.isclose(beam.beam_az_rot, 45.0)


class TestBeamMethods:
    """Tests for Beam class methods."""

    @pytest.fixture
    def simple_beam(self):
        """Create a simple beam for testing."""
        nfreq = 2
        ntheta = 181
        nphi = 360
        # Create uniform beam
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([50.0, 100.0])
        return Beam(data=data, freqs=freqs)

    def test_compute_norm(self, simple_beam):
        """Test compute_norm method."""
        norm = simple_beam.compute_norm()

        # Should return one value per frequency
        assert norm.shape == (2,)
        # For uniform beam, norm should be 4π
        assert jnp.all(norm > 0)
        assert jnp.allclose(norm, 4 * jnp.pi)

    def test_compute_fgnd(self, simple_beam):
        """Test compute_fgnd method."""
        fgnd = simple_beam.compute_fgnd()

        # Should return one value per frequency
        assert fgnd.shape == (2,)
        # Ground fraction should be between 0 and 1
        assert jnp.all(fgnd >= 0)
        assert jnp.all(fgnd <= 1)
        # default horizon means fgnd should be 0.5 for uniform beam
        assert jnp.allclose(fgnd, 0.5, atol=1e-2)

    def test_compute_fgnd_consistency(self, simple_beam):
        """Test that fgnd is consistent with norm calculations."""
        fgnd = simple_beam.compute_fgnd()
        norm_total = simple_beam._compute_norm(use_horizon=False)
        norm_above = simple_beam._compute_norm(use_horizon=True)

        expected_fgnd = 1.0 - norm_above / norm_total
        assert jnp.allclose(fgnd, expected_fgnd)

    def test_compute_alm(self, simple_beam):
        """Test compute_alm method."""
        alm = simple_beam.compute_alm()

        # Should have shape (nfreq, lmax+1, 2*lmax+1)
        expected_shape = (2, 180, 359)
        assert alm.shape == expected_shape
        # alm should be complex
        assert jnp.iscomplexobj(alm)

        # should be the same for both frequencies since the beam is the same
        assert jnp.allclose(alm[0], alm[1])

    def test_compute_alm_no_horizon(self):
        """Test compute_alm with no horizon masking."""
        nfreq = 1
        ntheta = 181
        nphi = 360
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0])
        horizon = np.ones((ntheta, nphi), dtype=bool)

        beam = Beam(data=data, freqs=freqs, horizon=horizon)
        alm = beam.compute_alm()

        # should have shape (nfreq, lmax+1, 2*lmax+1)
        expected_shape = (1, 180, 359)
        assert alm.shape == expected_shape

        # should be 1/sqrt(4pi) for monopole and 0 for others
        monopole = alm[:, 0, 179]
        assert jnp.allclose(monopole, jnp.sqrt(4 * jnp.pi))
        assert jnp.allclose(alm[:, 1:, :], 0.0, atol=1e-3)

    def test_compute_alm_with_custom_lmax(self):
        """Test compute_alm with custom lmax."""
        nfreq = 1
        ntheta = 181
        nphi = 360
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0])
        lmax = 50

        beam = Beam(data=data, freqs=freqs, lmax=lmax)
        alm = beam.compute_alm()

        # Should have shape (nfreq, lmax+1, 2*lmax+1)
        expected_shape = (1, lmax + 1, 2 * lmax + 1)
        assert alm.shape == expected_shape

    def test_compute_alm_with_rotation(self):
        """Test compute_alm with beam rotation."""
        nfreq = 1
        ntheta = 181
        nphi = 360
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0])
        beam_az_rot = 90.0

        beam = Beam(data=data, freqs=freqs, beam_az_rot=beam_az_rot)
        alm = beam.compute_alm()

        # alm should be computed and have correct shape
        assert alm.shape == (1, 180, 359)
        assert jnp.iscomplexobj(alm)

        # should be related to the non-rotated case by a phase factor in m
        alm_no_rot = Beam(data=data, freqs=freqs).compute_alm()
        m = jnp.arange(-179, 180)
        phase_factor = jnp.exp(-1j * jnp.radians(beam_az_rot) * m)
        expected_alm = alm_no_rot[0] * phase_factor[None, :]
        assert jnp.allclose(alm[0], expected_alm, atol=1e-3)

    def test_horizon_masking_in_alm(self):
        """Test that horizon masking is applied in alm computation."""
        nfreq = 1
        ntheta = 181
        nphi = 360

        # Create identical beam data
        data = np.ones((nfreq, ntheta, nphi))
        freqs = np.array([100.0])

        # Create two different horizons
        horizon_all = np.ones((ntheta, nphi), dtype=bool)  # Everything visible
        horizon_half = np.ones((ntheta, nphi), dtype=bool)
        horizon_half[90:, :] = False  # Only upper hemisphere visible

        beam_all = Beam(data=data, freqs=freqs, horizon=horizon_all)
        beam_half = Beam(data=data, freqs=freqs, horizon=horizon_half)

        alm_all = beam_all.compute_alm()
        alm_half = beam_half.compute_alm()

        # The two beams should have different alm because of different masking
        # They should NOT be close since one has half the sphere masked
        assert not jnp.allclose(alm_all, alm_half, rtol=1e-3)

    def test_beam_norm_positive(self, simple_beam):
        """Test that beam normalization is always positive."""
        norm = simple_beam.compute_norm()
        assert jnp.all(norm > 0)

    def test_multiple_frequencies_consistency(self):
        """Test that methods work consistently across multiple frequencies."""
        nfreq = 5
        ntheta = 181
        nphi = 360
        data = np.random.rand(nfreq, ntheta, nphi)
        freqs = np.linspace(50, 250, nfreq)

        beam = Beam(data=data, freqs=freqs)

        norm = beam.compute_norm()
        fgnd = beam.compute_fgnd()
        alm = beam.compute_alm()

        assert norm.shape == (nfreq,)
        assert fgnd.shape == (nfreq,)
        assert alm.shape == (nfreq, 180, 359)
