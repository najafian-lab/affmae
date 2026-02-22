"""
README for the cluster attention test suite using pytest.

This directory contains comprehensive tests for comparing the new Triton-based
cluster attention implementation with the old CUDA-based implementation using
standard testing frameworks.
"""

# Cluster Attention Test Suite (pytest)

This test suite provides comprehensive testing for the cluster attention modules in the AFF backbone, comparing the new Triton-based implementation with the old CUDA-based implementation using pytest and torch.testing.

## Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Test data utilities and pytest fixtures
├── test_configs.py             # Test configuration fixtures
├── test_forward.py             # Forward pass comparison tests
├── test_backward.py            # Backward pass comparison tests
├── test_integration.py         # Integration and edge case tests
└── README.md                   # This file
```

## Test Configurations

The test suite includes three main configurations that meet all constraints:

1. **Config 1 (Small)**: `B=1, N=64, C=8, H=1, M=32`
2. **Config 2 (Medium)**: `B=16, N=128, C=16, H=2, M=48`
3. **Config 3 (Large)**: `B=32, N=256, C=256, H=8, M=64`

All configurations satisfy the constraints: B >= 1, N >= 32, C >= H, C % H == 0, and M < N.

## Running Tests

### Run All Tests
```bash
cd refined/affmae/pretraining/models/aff_backbone/tests
PYTHONPATH=. pytest
```

### Run Specific Test Files
```bash
PYTHONPATH=. pytest test_forward.py          # Forward pass tests only
PYTHONPATH=. pytest test_backward.py         # Backward pass tests only
PYTHONPATH=. pytest test_integration.py      # Integration tests only
```

### Run Specific Test Classes
```bash
PYTHONPATH=. pytest test_forward.py::TestForwardPass
PYTHONPATH=. pytest test_backward.py::TestBackwardPass
PYTHONPATH=. pytest test_integration.py::TestIntegration
```

### Run Specific Test Methods
```bash
PYTHONPATH=. pytest test_forward.py::TestForwardPass::test_local_attention_forward_pass
PYTHONPATH=. pytest test_backward.py::TestBackwardPass::test_local_attention_backward_pass
```

### Run with Verbose Output
```bash
PYTHONPATH=. pytest -v
```

### Run with Coverage
```bash
PYTHONPATH=. pytest --cov=cluster
```

### Run Specific Configuration
```bash
PYTHONPATH=. pytest -k "config_1_small"      # Run tests with config_1_small
PYTHONPATH=. pytest -k "config_2_large"      # Run tests with config_2_large
```

## Test Components

### 1. Test Data Utilities (`conftest.py`)
- Creates realistic test data for cluster attention
- Handles spatial clustering and neighbor indexing
- Generates position embeddings and masks
- Compatible with both old and new implementations
- Provides pytest fixtures for easy test setup

### 2. Forward Pass Tests (`test_forward.py`)
- Compares forward pass outputs between implementations
- Tests both local and global attention modes
- Uses `torch.testing.assert_close` for numerical comparisons
- Tests different data types (float32, float16)
- Validates output shapes

### 3. Backward Pass Tests (`test_backward.py`)
- Compares gradient computations between implementations
- Tests input gradients and parameter gradients
- Uses `torch.testing.assert_close` for gradient comparisons
- Tests gradient norms and numerical stability
- Validates gradient flow

### 4. Integration Tests (`test_integration.py`)
- Tests module initialization and parameter shapes
- Tests edge cases (empty batch, single token)
- Tests dropout behavior consistency
- Tests gradient flow through modules
- Validates module behavior in realistic scenarios

### 5. Test Configurations (`test_configs.py`)
- Defines test parameters and tolerances
- Provides pytest fixtures for configurations
- Manages different test environments
- Extensible for additional test scenarios

## Test Results

The test suite provides:
- **Pass/Fail status**: Clear test results with pytest
- **Detailed error messages**: Using torch.testing.assert_close
- **Coverage information**: When using pytest-cov
- **Parallel execution**: pytest can run tests in parallel
- **CI/CD integration**: Standard pytest output format

## Tolerances

Default tolerances (configurable via fixtures):
- Forward pass: 1e-4 (relative and absolute)
- Backward pass: 1e-3 (relative and absolute)
- Loss comparison: 1e-4 (relative and absolute)

## Dependencies

Required packages:
- PyTorch
- Triton
- PyKeOps
- pytest
- pytest-cov (optional, for coverage)

## Notes

- Tests require CUDA for optimal performance
- PyKeOps is used for efficient k-nearest neighbor computation
- Tests are designed to be deterministic (uses fixed random seeds)
- Both implementations must be available in the cluster module
- Uses standard pytest fixtures for test setup and teardown

## Troubleshooting

### Import Errors
Ensure all modules are properly installed and the Python path includes the parent directory.

### CUDA Errors
Make sure CUDA is available and PyTorch is compiled with CUDA support.

### Memory Issues
For large configurations, consider reducing batch size or using gradient checkpointing.

### Numerical Differences
Small numerical differences are expected due to different implementations. Adjust tolerances in fixtures if needed.

### Test Failures
Use `pytest -v` for verbose output and `pytest --tb=short` for shorter tracebacks.