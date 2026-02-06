# DepthPrior: Depth as Prior Knowledge for Object Detection

Official implementation of **"Depth as Prior Knowledge for Object Detection"**.

## Abstract

Detecting small and distant objects remains challenging for 2D detectors due to scale variation, low resolution, and background clutter. We introduce **DepthPrior**, a framework that uses depth as prior knowledge rather than as a fused feature, providing comparable benefits without modifying detector architectures.

DepthPrior consists of:
- **DLW (Depth-Based Loss Weighting)**: Weights each object's loss based on depth during training
- **DLS (Depth-Based Loss Stratification)**: Decomposes loss into close/distant components with separate weights
- **DCT (Depth-Aware Confidence Thresholding)**: Learns depth-dependent confidence thresholds via splines during inference

The only overhead is the initial cost of depth estimation. We validate DepthPrior across four benchmarks (KITTI, MS COCO, VisDrone, SUN RGB-D) and two detector architectures (EfficientDet, YOLOv11), achieving up to **+9% mAP_S** and **+7% mAR_S** for small objects.

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/mos-ks/DepthPrior.git
cd DepthPrior

# For YOLOv11 experiments
cd object_detection/yolov11
pip install -e .

# For EfficientDet experiments (see EfficientDet section below)
```

## Repository Structure

```
DepthPrior/
├── datasets/                    # Dataset preparation scripts
│   ├── KITTI/                   # KITTI dataset utilities
│   ├── MSCOCO/                  # MS COCO dataset utilities
│   ├── SUNRGBD/                 # SUN RGB-D dataset utilities
│   └── visdrone/                # VisDrone dataset utilities
├── object_detection/
│   ├── efficientdet/            # EfficientDet implementation
│   │   ├── train_lib.py         # Training library with DLW/DLS
│   │   ├── train_flags_depth.py # Training flags for depth experiments
│   │   ├── dataloader.py        # Data loading with depth support
│   │   ├── efficientdet_keras.py # Modified Keras model
│   │   ├── hparams_config.py    # Hyperparameter configuration
│   │   └── depth_map.py         # Depth map extraction
│   ├── yolov11/                 # YOLOv11 implementation
│   │   ├── depth/               # Depth-aware training modules
│   │   │   ├── loss.py          # DepthLoss with DLW/DLS strategies
│   │   │   ├── trainer.py       # Custom trainer with depth support
│   │   │   ├── model.py         # Model with depth loss
│   │   │   ├── validator.py     # COCO-style evaluation
│   │   │   └── dataset.py       # Depth dataset loader
│   │   ├── train.py             # Training entry point
│   │   ├── predict.py           # Inference script
│   │   └── configs/             # Training configurations
│   ├── depth_DCT.py             # DCT implementation
│   └── depth_DCT_util.py        # DCT utilities
└── README.md
```

## Datasets

We support four datasets with distinct depth distributions:

| Dataset | Domain | Depth Distribution |
|---------|--------|-------------------|
| KITTI | Automotive | Far-field concentration |
| VisDrone | Aerial | Mid-range |
| SUN RGB-D | Indoor | Linear growth |
| MS COCO | General | Bimodal |

### Dataset Preparation

1. **Download datasets** following their official instructions
2. **Generate depth maps** using [Depth-Anything](https://github.com/LiheYoung/Depth-Anything):
   ```bash
   # Example for KITTI
   python -m depth_anything.run --input /path/to/KITTI/images --output /path/to/KITTI/depth
   ```
3. **Create TFRecords (EfficientDet)** or **CSV files (YOLOv11)**:
   ```bash
   # For EfficientDet
   python datasets/KITTI/kitti_depth_tfrecord.py

   # For YOLOv11
   python object_detection/yolov11/scripts/prepare-depth-dataset-kitti.py
   ```

## Training

### YOLOv11

```bash
cd object_detection/yolov11

# Baseline training
python train.py --config configs/kitti.yaml

# DLW (Depth-Based Loss Weighting)
python train.py --config configs/kitti-depth.yaml \
    --depth_aware dlw \
    --alpha 1.0

# DLS (Depth-Based Loss Stratification)
python train.py --config configs/kitti-depth.yaml \
    --depth_aware dls \
    --far_weight 2.0 \
    --close_weight 1.0 \
    --depth_threshold 0.25
```

**Depth-aware strategies:**
- `dlw`: Depth-Based Loss Weighting
- `dls`: Depth-Based Loss Stratification 
- `image-mean`: Per-image mean depth weighting
- `batch-mean`: Per-batch mean depth weighting

### EfficientDet

EfficientDet implementation is based on [continental/uncertainty-detection-autolabeling](https://github.com/continental/uncertainty-detection-autolabeling).

**Setup:**
1. Clone the base repository
2. Replace these files with our modified versions from `object_detection/efficientdet/`:
   - `train_lib.py`
   - `train_flags_depth.py`
   - `dataloader.py`
   - `efficientdet_keras.py`
   - `hparams_config.py`

**Depth-aware strategies (activated via `model_dir` naming and flags):**

| Strategy | Activation | Description |
|----------|------------|-------------|
| **DLW** | Include `depth_detection` in `model_dir` | Exponential weighting: `1 + α·exp(depth)` |
| **DLS** | Include `depth_detection` in `model_dir` + `--two_losses_weight=[close,far]` | Threshold-based weighting with separate close/far weights |
| **L** | Include `depth_loss` in `model_dir` | Depth loss (box loss with depth image as input) |
| **CL** | Include `depth_loss` and `consistency` in `model_dir` | Consistency loss: MSE between RGB and depth predictions |
| **PD** | Include `depth_detection` in `model_dir` + `predict_distance` | Predict distance per object |
| **BW** | Include `depth_batch` in `model_dir` | Per-batch depth normalization |
| **IW** | Include `depth_image` in `model_dir` | Per-image depth normalization |

**Training commands:**

```bash
# DLW: Exponential weighting
python train_flags_depth.py \
    --train_file_pattern=/path/to/data/DepthPrior/datasets/KITTI/depth_anything_predictions/_train_depth.tfrecord \
    --val_file_pattern=/path/to/data/DepthPrior/datasets/KITTI/tf/_val.tfrecord \
    --model_dir=/path/to/models/KITTI_depth_detection_V0/ \
    --model_name=efficientdet-d0 \
    --batch_size=8 \
    --num_epochs=200

# DLS: Stratified loss with close/far weights
python train_flags_depth.py \
    --two_losses_weight=[1,5] \
    --train_file_pattern=/path/to/data/DepthPrior/datasets/KITTI/depth_anything_predictions/_train_depth.tfrecord \
    --model_dir=/path/to/models/KITTI_depth_detection_dls_V0/

# CL: Consistency loss
python train_flags_depth.py \
    --train_file_pattern=/path/to/data/DepthPrior/datasets/KITTI/depth_anything_predictions/_train_depth.tfrecord \
    --model_dir=/path/to/models/KITTI_depth_loss_consistency_V0/

# L: Depth loss
python train_flags_depth.py \
    --train_file_pattern=/path/to/data/DepthPrior/datasets/KITTI/depth_anything_predictions/_train_depth.tfrecord \
    --model_dir=/path/to/models/KITTI_depth_loss_V0/
```

**Notes:**
- DLS requires both `depth_detection` in `model_dir` AND non-zero `two_losses_weight`
- The `alpha` parameter for DLW can be set via `model_dir` name (e.g., `depth_detection_alpha1.5`)
- The `beta` parameter (split factor) for DLS defaults to 0.5; set via `model_dir` name (e.g., `depth_detection_factor0.3`)
- Use `inverse` in `model_dir` to invert depth values (close=1, far=0)

## Inference with DCT

DCT (Depth-Aware Confidence Thresholding) adjusts detection thresholds based on object depth at inference time.

```python
from object_detection.depth_DCT import DepthAwareThresholding

# Initialize DCT
dct = DepthAwareThresholding(
    reference_threshold=0.5,
    num_knots=10,
    gamma=1000,
    epsilon=0.1
)

# Optimize thresholds on validation set
dct.optimize(val_detections, val_depth_maps, val_annotations)

# Apply at inference
filtered_detections = dct.filter(detections, depth_map)
```

## Hyperparameters

### DLW
| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.0 | Controls far-object emphasis in exponential weighting |

### DLS
| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.5 | Close/distant boundary threshold |
| `lambda_close` | 1.0 | Weight for close objects |
| `lambda_distant` | 2.0 | Weight for distant objects |

### DCT
| Parameter | Default | Description |
|-----------|---------|-------------|
| `J` | 10 | Number of spline knots |
| `epsilon` | 0.1 | Allowed false positive tolerance |
| `gamma` | 1000 | Regularization for excessive false positives |
| `rho` | 0.1 | Minimum admissible threshold |

## Results

### EfficientDet on KITTI

| Method | mAP | mAP_S | mAR_S |
|--------|-----|-------|-------|
| Baseline | 53.5 | 19.7 | 38.5 |
| DLW | 54.7 (+1.2) | 23.7 (+4.0) | 41.3 (+2.8) |
| DLS | 55.1 (+1.6) | 28.4 (+8.7) | 45.0 (+6.5) |

### YOLOv11 on KITTI

| Method | mAP | mAP_S | mAR_S |
|--------|-----|-------|-------|
| Baseline | 48.4 | 31.0 | 38.1 |
| DLW | 49.4 (+1.0) | 33.3 (+2.3) | 40.0 (+1.9) |
| DLS | 47.4 (-1.0) | 34.4 (+3.4) | 41.2 (+3.1) |

## Citation

If you find this work useful, please cite:

```bibtex
@article{sbeyti2026depthprior,
  title={Depth as Prior Knowledge for Object Detection},
  author={Kassem~Sbeyti, Moussa and Klein, Nadja},
  journal={arXiv preprint arXiv:2602.05730},
  year={2026}
}
```

## Acknowledgments

- EfficientDet base implementation: [continental/uncertainty-detection-autolabeling](https://github.com/continental/uncertainty-detection-autolabeling)
- Depth estimation: [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- YOLOv11: [ultralytics](https://github.com/ultralytics/ultralytics)
- YOLOv11 depth integration: [nimalu](https://github.com/nimalu)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
