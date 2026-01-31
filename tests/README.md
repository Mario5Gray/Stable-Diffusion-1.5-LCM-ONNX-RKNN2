# Tests

Functional tests for the Dream Lab dynamic model loading system.

## Test Coverage

### backends/
- **test_model_registry.py** - Tests for VRAM tracking and model registration
- **test_worker_factory.py** - Tests for model type detection and worker creation
- **test_worker_pool.py** - Tests for job queue, mode switching, and worker lifecycle

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_model_registry.py
pytest tests/test_worker_factory.py
pytest tests/test_worker_pool.py
```

### Run specific test class:
```bash
pytest tests/test_model_registry.py::TestModelRegistration
pytest tests/test_worker_pool.py::TestModeSwitching
```

### Run specific test:
```bash
pytest tests/test_model_registry.py::TestModelRegistration::test_register_single_model
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage report:
```bash
pytest --cov=backends --cov=server --cov-report=html
```

Then open `htmlcov/index.html` in a browser.

### Run functional tests only:
```bash
pytest -m functional
```

### Run excluding slow tests:
```bash
pytest -m "not slow"
```

## Test Structure

All tests follow the **Arrange-Act-Assert** pattern:

```python
def test_example(fixture):
    # Arrange - Set up test data
    registry = ModelRegistry()

    # Act - Perform the action
    registry.register_model("test", "/path", 1024**3, [])

    # Assert - Verify the result
    assert "test" in registry._models
```

## Fixtures

Common fixtures are defined at the top of test files:

- `mock_cuda` - Mocks CUDA functions for testing without GPU
- `registry` - Fresh ModelRegistry instance for each test
- `mock_mode_config` - Mocked mode configuration
- `mock_registry` - Mocked model registry
- `mock_worker_factory` - Mocked worker factory
- `worker_pool` - WorkerPool with mocked dependencies

## Test Categories

### Functional Tests
Test the behavior and functionality of components with mocking:
- Input/output behavior
- State management
- Error handling
- Edge cases

### Integration Tests (future)
Test interaction between real components:
- Actual model loading
- Real VRAM tracking
- End-to-end workflows

## Requirements

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-asyncio pytest-timeout
```

Or if using the project's requirements:

```bash
pip install -r requirements.txt  # Includes test dependencies
```

## Continuous Integration

These tests can be run in CI pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest --cov=backends --cov=server --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Writing New Tests

When adding new functionality:

1. **Create test file**: `tests/test_<module_name>.py`
2. **Organize by class**: Group related tests in `Test<Functionality>` classes
3. **Use descriptive names**: `test_<what_it_does>`
4. **Mock dependencies**: Use `@patch` for external dependencies
5. **Add markers**: Use `@pytest.mark.functional` or other markers
6. **Document**: Add docstrings explaining what's being tested

Example:

```python
class TestNewFeature:
    """Test new feature functionality."""

    def test_feature_basic_case(self, fixture):
        """Test feature works in basic case."""
        # Arrange
        obj = MyObject()

        # Act
        result = obj.do_something()

        # Assert
        assert result == expected_value
```

## Troubleshooting

### Import errors
Make sure you're running pytest from the project root:
```bash
cd /media/assets/AI/RKNN/dream-lab
pytest
```

### CUDA errors in tests
Tests mock CUDA functions. If you see real CUDA errors, check that mocks are active:
```python
@patch('torch.cuda.is_available', return_value=True)
def test_something(mock_cuda):
    # Test code
```

### Fixture not found
Make sure fixtures are defined in the same file or in `conftest.py`:
```python
@pytest.fixture
def my_fixture():
    return "test_value"
```
