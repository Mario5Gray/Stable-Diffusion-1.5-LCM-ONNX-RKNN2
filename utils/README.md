# Utils Package

Standalone utility scripts for model detection and verification.

## Contents

### model_detector.py

**Extensible model detection system** with plugin architecture.

**Usage:**
```bash
# Command line
python -m utils.model_detector /path/to/model.safetensors --json

# Python API
from utils.model_detector import detect_model
info = detect_model("/path/to/model.safetensors")
print(info.to_json())
```

**Features:**
- Detects SD1.5, SD2.x, SDXL variants
- Supports .safetensors, .ckpt, diffusers formats
- LoRA detection and classification
- Extensible plugin system for custom detectors
- JSON output for automation

**Documentation:** See [../docs/MODEL_DETECTOR_EXTENSIBLE.md](../docs/MODEL_DETECTOR_EXTENSIBLE.md)

### detect_model_type.py

**Legacy model detection tool** with directory scanning.

**Usage:**
```bash
# Single file
python -m utils.detect_model_type /path/to/model.safetensors

# Scan directory
python -m utils.detect_model_type --scan /models/ --recursive

# JSON output
python -m utils.detect_model_type model.safetensors --json
```

**Features:**
- Directory scanning with `--scan`
- Recursive search with `-r`
- Colored terminal output
- JSON output mode

**Note:** This is the older detection tool. For new code, use `model_detector.py` instead.

### custom_detector_example.py

**Examples of extending the model detector** with custom logic.

**Usage:**
```python
from utils.model_detector import ModelDetector
from utils.custom_detector_example import LCMDetector, TurboDetector

detector = ModelDetector()
detector.add_detector(LCMDetector())
detector.add_detector(TurboDetector())

info = detector.detect("/models/sdxl-lcm.safetensors")
```

**Included Examples:**
- `LCMDetector` - Detects Latent Consistency Models
- `TurboDetector` - Detects SDXL-Turbo variants
- `RefinerDetector` - Detects SDXL Refiner models
- `LicenseDetector` - Extracts license information
- `LoRAStrengthAnalyzer` - Suggests optimal LoRA strengths

### verify_cuda.py

**CUDA verification utility** for Docker containers.

**Usage:**
```bash
# Verify CUDA in Docker
docker run --rm --gpus all --privileged image:latest python -m utils.verify_cuda

# On host
python -m utils.verify_cuda
```

**Features:**
- Checks PyTorch installation
- Verifies CUDA availability
- Lists CUDA devices
- Tests basic CUDA operations
- Reports memory usage

Used by test scripts to verify GPU access before running tests.

## Integration

The `model_detector.py` is integrated with the server via `backends/worker_factory.py`:

```python
from utils.model_detector import detect_model

# Factory uses detector to auto-select worker
def detect_worker_type():
    info = detect_model(model_path)
    if info.cross_attention_dim == 2048:
        return "sdxl"
    else:
        return "sd15"
```

This allows the server to automatically detect whether a model is SD1.5 or SDXL without manual configuration.
