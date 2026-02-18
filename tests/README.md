# mistsim Test Suite

This directory contains comprehensive tests for the mistsim package.

## Test Structure

- `test_utils.py`: Tests for utility functions (`get_lmax`)
- `test_beam.py`: Tests for the `Beam` class and its methods
- `test_sim.py`: Tests for the `Simulator` class, `correct_ground_loss` function, and integration tests
- `conftest.py`: Shared pytest configuration and fixtures

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run tests with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

To run a specific test file:
```bash
pytest tests/test_beam.py -v
```

## Test Coverage

Current coverage (as of last update):
- Overall: 84%
- `utils.py`: 100%
- `beam.py`: 100%
- `sim.py`: 66% (some methods require network access for coordinate transforms)

## Skipped Tests

Some tests are marked with `@requires_coords` and will be skipped in sandboxed environments without internet access. These tests require:
- Astropy coordinate system transforms
- SPICE kernel downloads from NASA servers

In environments with internet connectivity, these tests will run successfully. To run them, ensure you have internet access and remove the `skipif` condition in `test_sim.py`.

## Test Categories

### Unit Tests
- **Beam initialization**: Tests various initialization scenarios including error cases
- **Beam methods**: Tests `compute_norm()`, `compute_fgnd()`, and `compute_alm()`
- **Simulator validation**: Tests parameter validation (frequencies, lmax mismatches)
- **Ground loss correction**: Tests the `correct_ground_loss()` function

### Integration Tests
- **Full pipeline**: Tests the complete workflow from beam to visibilities
- **Ground loss correction pipeline**: Tests end-to-end ground loss handling

## Adding New Tests

When adding new tests:
1. Follow the existing structure using pytest classes
2. Add descriptive docstrings
3. Use fixtures for shared test data
4. Mark tests that require network access with `@requires_coords`
5. Ensure tests are independent and can run in any order
