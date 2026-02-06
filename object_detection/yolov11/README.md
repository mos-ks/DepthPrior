# YOLO + depth information

This work-in-progress project focuses on the integration of object detection and depth estimation. It leverages Ultralytics' package for YOLO models and utilizes `uv` as the package manager. For containerization, `singularity` is employed.

The training process is as follows:

1. Build the Singularity container, which packages all the required dependencies.
2. Execute the training.
3. Important information can be found in the log files as well as in the corresponding directory under `./runs`.



## Repository structure
```bash
build/              # large build files, such as the singularity container
configs/            # configure the training
dataset-configs/
runs/               # logging
scripts/
main.py             # Starts training process
pyproject.toml      # lists dependencies
Singularity.def
training.py
```

## Build singularity container
```bash
mkdir -p build && singularity build -F build/yolo.sif Singularity.def
```

 ## Run training
```bash
mkdir -p runs
singularity exec \
    --bind ./runs:/runs \
    --bind /path/to/data:/lsdf \
    --nv \
    build/yolo.sif \
    python train.py \
    --config configs/kitti.yaml
```

## Open mlflow
```bash
uv run mlflow ui --backend-store-uri sqlite:///runs/mlruns.db
```

## Train on subset and make predictions

### 1. Prepare subset to train on

Ultralytics requires a dataset YAML file to specify the training data.  
Refer to `dataset-configs/kitti-subset.yaml` for an example.  
The `train` property in this file determines which samples are used for training.  
This property can point to either a directory containing the training images or, as in this example, a text file listing all sample paths. For this instance, edit `train/images-subset.txt` to adjust which samples are used for training.
Similarly, the `val` property specifies which samples are used for validation.

### 2. Train 

Another YAML file defines certain training parameters. For instance, this config determines which dataset to use and thus points to the corresponding dataset YAML file. 
The configuration file also determines where outputs such as model weights are saved. For example, in `kitti-subset.yaml`, the setting `runs_dir: "/runs/kitti-subset"` ensures that model weights are stored under `runs/kitti-subset/detect/train*/weights/best.pt`.

With the following command, the training can be started.

```bash
# inside the yolo directory
mkdir -p runs
singularity exec \
    --bind ./runs:/runs \
    --bind /path/to/data:/lsdf \
    --nv \
    build/yolo.sif \
    python train.py \
    --config configs/kitti-subset.yaml
```

### 3. Make predictions

The script `predict.py` loads the model weights and produces predictions based on the `image_list` provided. 
This paramter can point to either a directory containing the images or  a text file listing all image paths for which predictions should be made.

```bash
export MODEL_WEIGHTS=/runs/kitti-subset/detect/train/weights/best.pt
export SCORE_THRESHOLD=0.01
export TARGET_FILE=./predictions.jsonl
export IMAGE_LIST=/lsdf/DepthPrior/datasets/KITTI/yolo/val/images.txt

singularity exec \
    --bind ./runs:/runs \
    --bind /path/to/data:/lsdf \
    --nv \
    build/yolo.sif \
    python predict.py
```