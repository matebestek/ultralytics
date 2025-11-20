# Ultralytics Copilot Instructions

This is the official Ultralytics YOLO repository implementing state-of-the-art object detection, segmentation, classification, pose estimation, and tracking models.

## Architecture & Structure

- **`ultralytics/`**: Main package with modular architecture
  - `models/`: YOLO11, YOLOv8, RTDETR, SAM model implementations
  - `nn/modules/`: Neural network building blocks and layer definitions  
  - `engine/`: Core training, validation, prediction engine
  - `utils/`: Utilities for metrics, plotting, torch operations
  - `cfg/`: Configuration files and YAML schemas
  - `data/`: Dataset handling and augmentation pipelines
- **`examples/`**: Production-ready integrations (ONNXRuntime, OpenVINO, TensorRT)
- **`tests/`**: Comprehensive test suite with CLI and Python API coverage

## Key Commands & Workflows

### CLI Interface (Primary)
The `yolo` command is the main entry point supporting all operations:

```bash
# Training: Always specify task, model, data, minimal epochs for testing
yolo train detect model=yolo11n.pt data=coco8.yaml epochs=1 imgsz=640

# Validation with outputs
yolo val detect model=yolo11n.pt data=coco8.yaml save_txt save_json

# Prediction with visualization
yolo detect predict model=yolo11n.pt source=path/to/images save save_crop visualize

# Export to deployment formats  
yolo export model=yolo11n.pt format=onnx imgsz=640
```

### Python API (Secondary)
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=1)
results = model("image.jpg")
```

### Testing & Development
```bash
# Run fast tests (excludes slow GPU tests)
pytest tests/ -v

# Include slow tests (requires CUDA)  
pytest tests/ -v --slow

# Run specific test modules
pytest tests/test_cli.py -v
```

## Project Conventions

- **Task-first approach**: Commands specify task (detect/segment/classify/pose/obb) before other args
- **YAML configs**: Data and model configurations use YAML files in `ultralytics/cfg/`
- **Modular exports**: Each format (ONNX, TensorRT, CoreML) has dedicated handling in `engine/exporter.py`
- **Consistent naming**: Models follow `yolo11{n,s,m,l,x}[-task].pt` pattern
- **Graceful fallbacks**: CPU/GPU detection with automatic device selection
- **Rich visualization**: Built-in plotting and annotation tools in `utils/plotting.py`

## Integration Points

- **HUB integration**: Ultralytics HUB for cloud training and deployment
- **Export targets**: ONNX, TensorRT, OpenVINO, CoreML, TFLite production deployment
- **Logging**: Native WandB, TensorBoard, MLflow integration via `utils/callbacks/`
- **Datasets**: Auto-download COCO, ImageNet, custom dataset support