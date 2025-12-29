# Claude Memory - I-JEPA Project

## Environment Setup

### Python Environment
- Python version: 3.11.9 (via pyenv)
- Python executable: `~/.pyenv/versions/3.11.9/bin/python`
- Run scripts: `~/.pyenv/versions/3.11.9/bin/python <script.py>`
- Run pytest: `~/.pyenv/versions/3.11.9/bin/python -m pytest`

### Dependencies
Install with:
```bash
~/.pyenv/versions/3.11.9/bin/pip install torch torchvision numpy tqdm pytest
```

## Unit Testing Protocol

### Test Location
- All tests go in `tests/` directory
- Test files are named `test_<module>.py`
- Each module being tested gets its own test file

### Test Structure
```python
"""
Docstring explaining what the tests verify.
"""

import pytest
import torch

from ijepa.models.<module> import <Class>


# Fixtures for common test data
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Group related tests in classes
class Test<ClassName>:
    """Tests for <ClassName>."""

    def test_<specific_behavior>(self, device, ...):
        """Clear description of what is being tested."""
        # Arrange
        model = <Class>(...).to(device)
        input_data = torch.randn(..., device=device)

        # Act
        output = model(input_data)

        # Assert
        assert output.shape == expected_shape
```

### Required Test Categories

1. **Shape Tests** - Verify output dimensions for various inputs
2. **Architecture Tests** - Check model attributes, parameter counts
3. **Numerical Stability** - No NaN/Inf in outputs
4. **Gradient Flow** - Ensure backprop works correctly
5. **Determinism** - Eval mode produces consistent outputs
6. **Integration Tests** - Components work together

### Running Tests
```bash
# Run all tests
~/.pyenv/versions/3.11.9/bin/python -m pytest tests/ -v

# Run specific test file
~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_vit.py -v

# Run with short traceback
~/.pyenv/versions/3.11.9/bin/python -m pytest tests/ -v --tb=short
```

### Test Conventions
- Use fixtures for reusable test data (device, batch_size, etc.)
- Test both CPU and GPU when `device` fixture is used
- Include edge cases (single item, empty, max size)
- Verify gradient flow with `requires_grad=True` and `.backward()`
- Check `torch.isnan()` and `torch.isinf()` for stability
- Test eval mode determinism with `model.eval()`

## Project Structure
```
ijepa/
├── models/
│   ├── vit.py          # PatchEmbed, TransformerBlock, ViTEncoder
│   ├── predictor.py    # Predictor with mask tokens
│   └── ijepa.py        # Full I-JEPA model (TODO)
├── data/
│   ├── masking.py      # MultiBlockMaskGenerator (TODO)
│   └── cifar10.py      # DataLoader setup (TODO)
├── training/
│   ├── train.py        # Training loop (TODO)
│   └── ema.py          # EMA utilities (TODO)
└── config.py           # Hyperparameters (TODO)

tests/
├── test_vit.py         # 37 tests for ViT components
└── test_predictor.py   # 22 tests for Predictor
```

## Implementation Status
- [x] `vit.py` - MLP, PatchEmbed, TransformerBlock, ViTEncoder
- [x] `predictor.py` - Predictor with mask token mechanism
- [ ] `masking.py` - MultiBlockMaskGenerator
- [ ] `ijepa.py` - Full I-JEPA model
- [ ] `cifar10.py` - Data loading
- [ ] `train.py` - Training loop
- [ ] `ema.py` - EMA update utilities
