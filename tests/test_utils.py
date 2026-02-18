"""Tests for the utils module."""

import pytest
import jax.numpy as jnp

from mistsim.utils import get_lmax


class TestGetLmax:
    """Tests for the get_lmax function."""

    def test_get_lmax_basic(self):
        """Test get_lmax with a basic alm array shape."""
        # For lmax=10, shape should be (11, 21) -> (lmax+1, 2*lmax+1)
        shape = (11, 21)
        assert get_lmax(shape) == 10

    def test_get_lmax_with_batch_dimension(self):
        """Test get_lmax with batch dimensions."""
        # For lmax=5, shape could be (3, 6, 11) with batch dimension
        shape = (3, 6, 11)
        assert get_lmax(shape) == 5

    def test_get_lmax_multiple_batch_dimensions(self):
        """Test get_lmax with multiple batch dimensions."""
        # For lmax=20, shape could be (2, 3, 4, 21, 41)
        shape = (2, 3, 4, 21, 41)
        assert get_lmax(shape) == 20

    def test_get_lmax_lmax_zero(self):
        """Test get_lmax when lmax is 0."""
        # For lmax=0, shape should be (1, 1)
        shape = (1, 1)
        assert get_lmax(shape) == 0

    def test_get_lmax_large_lmax(self):
        """Test get_lmax with a large lmax value."""
        # For lmax=179, shape should be (180, 359)
        shape = (180, 359)
        assert get_lmax(shape) == 179

    def test_get_lmax_2d_array(self):
        """Test get_lmax with a 2D array (no batch dimension)."""
        # For lmax=15, shape should be (16, 31)
        shape = (16, 31)
        assert get_lmax(shape) == 15

    def test_get_lmax_consistency(self):
        """Test that get_lmax is consistent across different batch sizes."""
        lmax = 7
        shape1 = (lmax + 1, 2 * lmax + 1)
        shape2 = (5, lmax + 1, 2 * lmax + 1)
        shape3 = (2, 3, lmax + 1, 2 * lmax + 1)
        
        assert get_lmax(shape1) == lmax
        assert get_lmax(shape2) == lmax
        assert get_lmax(shape3) == lmax
