from typing import override
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8DetectionLoss,
)
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import colorstr

from depth.dataset import YOLODepthDataset


class CustomModel(DetectionModel):
    @override
    def init_criterion(self):
        return (
            E2EDetectLoss(self)
            if getattr(self, "end2end", False)
            else v8DetectionLoss(self)
        )


class DepthTrainer(DetectionTrainer):
    @override
    def get_model(self, cfg, weights, verbose):
        model = CustomModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    @override
    def build_dataset(self, img_path, mode="train", batch=None):
        cfg = self.args
        assert cfg.task == "detect", f"Task must be 'detect', not {cfg.task}."
        assert not cfg.cache, "Cache is not yet supported for depth training."
        assert not cfg.rect, "Rect is not yet supported for depth training."
        assert not cfg.single_cls, (
            "Single class is not yet supported for depth training."
        )
        assert not cfg.fraction or cfg.fraction == 1.0, (
            "Fraction is not yet supported for depth training."
        )
        return YOLODepthDataset(
            data=self.data,
            img_path=img_path,
            imgsz=cfg.imgsz,
            augment=mode == "train",
            hyp=cfg,
            prefix=colorstr(f"{mode}: "),
            batch_size=batch,
            stride=max(
                int(de_parallel(self.model).stride.max() if self.model else 0), 32
            ),
            pad=0.0 if mode == "train" else 0.5,
            classes=cfg.classes,
        )
