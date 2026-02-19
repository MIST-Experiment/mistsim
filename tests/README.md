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

All tests run successfully without requiring any special configuration or environment variables. The test suite automatically handles any required data downloads when internet access is available.

## Test Coverage

Current test metrics:
- **Total tests**: 51
- **Passing**: 51 (100%)
- **Skipped**: 0
- **Code coverage**: High coverage across all modules
  - `utils.py`: 100%
  - `beam.py`: 100%
  - `sim.py`: Comprehensive coverage including coordinate transforms

## Test Categories

### Unit Tests

**Utils Module (7 tests)**:
- Tests for `get_lmax()` with various array shapes and edge cases

**Beam Module (21 tests)**:
- **Initialization tests**: Valid/invalid parameters, horizon masks, lmax validation, rotation angles
- **Method tests**: `compute_norm()`, `compute_fgnd()`, `compute_alm()` with various configurations
- **Validation tests**: Physical constants (norm = 4π, fgnd = 0.5), phase factors, monopole terms

**Simulator Module (23 tests)**:
- **Initialization tests**: Parameter validation, mismatched frequencies/lmax detection
- **Method tests**: Coordinate transformations, ground contributions, visibility generation
- **Ground loss correction**: Mathematical correctness, broadcasting, inverse operations
- **Integration tests**: Full pipeline validation, monopole sky behavior, time-dependent visibilities

### Integration Tests

The test suite includes comprehensive integration tests that validate:
- **Full simulation pipeline**: From beam definition to visibility computation
- **Monopole sky behavior**: Validates constant visibilities for monopole sky, horizon effects, and ground temperature contributions
- **Time-dependent visibilities**: Tests sidereal day periodicity and phase evolution using cos(phi) sky dependence
- **Ground loss correction**: End-to-end validation of ground contamination removal

## Key Features

### Bug Fixes Validated
- ✅ Horizon radians/degrees fix (default horizon now uses radians correctly)
- ✅ JAX static array fix (arrays no longer incorrectly marked as static)
- ✅ Time conversion fix (times correctly converted to seconds)
- ✅ Sky generation fix (using s2fft's generate_flm)

### Test Quality
- Uses realistic test data via s2fft utilities
- Validates physical constants and expected behavior
- Comprehensive assertions in each test
- Independent tests that can run in any order

### CI/CD Integration
Tests run automatically via GitHub Actions on:
- Every push and pull request
- Python versions 3.10, 3.11, and 3.12
- Includes linting checks

## Adding New Tests

When adding new tests:
1. Follow the existing structure using pytest classes
2. Add descriptive docstrings
3. Use fixtures for shared test data
4. Validate physical behavior, not just code execution
5. Ensure tests are independent and can run in any order
6. Include multiple assertions to validate different aspects
