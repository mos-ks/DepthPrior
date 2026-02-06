import pickle
from copy import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import de_parallel

from depth.dataset import YOLODepthDataset
from depth.model import DetectionModelWithDepthLoss
from depth.validator import CocoEvalDetectionValidator


class CustomDetectionTrainer(DetectionTrainer):
    """
    Custom trainer for object detection with optional depth-aware loss support.
    Overwritten Methods:
        get_model(cfg, weights, verbose):
            Returns a detection model, optionally with depth loss.
        get_validator():
            Returns a COCO-style detection validator for evaluation.
        build_dataset(img_path, mode="train", batch=None):
            Builds and returns a dataset for training or validation, supporting both standard and depth datasets.
        plot_training_samples(batch, ni):
            Plots and saves training batch samples, including depth maps if present.
    """

    def get_model(self, cfg, weights, verbose):
        if self.args.depth_aware:
            model = DetectionModelWithDepthLoss(cfg, nc=self.data["nc"], verbose=verbose)
        else:
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)

        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CocoEvalDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def build_dataset(self, img_path, mode="train", batch: int | None = None):
        cfg = self.args
        stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        assert not cfg.rect, "Rect is not yet supported for depth training."

        if not img_path.endswith(".csv"):
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=cfg.rect, stride=stride)

        return YOLODepthDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect,  # caution: build_dataset uses `or mode == "val"`
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        super().plot_training_samples(batch, ni)

        batch_cpu = obj_to_cpu(batch)

        # save as pickle
        with open(self.save_dir / f"train_batch{ni}.pkl", "wb") as f:
            pickle.dump(batch_cpu, f)

        if "depth_map" in batch:
            self._plot_depth_maps(batch_cpu, ni)

    def _plot_depth_maps(self, batch: dict, ni: int) -> None:
        depth_maps = np.array(batch["depth_map"])

        # replace padding fill value
        padding_mask = depth_maps == 114
        depth_maps[padding_mask] = 1.0

        n = len(depth_maps)
        ns = int(np.ceil(n ** 0.5))  # number of subplots per side (square grid)
        fig, axes = plt.subplots(ns, ns, figsize=(4 * ns, 4 * ns))
        for i in range(ns * ns):
            # Match ultralytics column-major layout: x = i // ns, y = i % ns
            col = i // ns  # column position (left to right)
            row = i % ns   # row position (top to bottom)
            ax = axes[row, col] if ns > 1 else axes
            if i < n:
                ax.imshow(depth_maps[i], cmap="gray")
                ax.axis("off")
            else:
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"train_batch{ni}_depth_maps.jpg")
        plt.close(fig)


def obj_to_cpu(obj: dict | list | tuple | torch.Tensor) -> dict:
    """
    Recursively moves all torch.Tensor objects within a nested structure (dict, list, tuple)
    to CPU and converts them to NumPy arrays. Non-tensor objects are returned unchanged.
    """
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: obj_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [obj_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(obj_to_cpu(v) for v in obj)
    else:
        return obj
