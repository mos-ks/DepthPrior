from ultralytics.nn.tasks import DetectionModel

from depth.loss import DepthLoss


class DetectionModelWithDepthLoss(DetectionModel):
    """
    DetectionModelWithDepthLoss is identical to DetectionModel, except it uses DepthLoss as its loss function.
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

    def init_criterion(self):
        return DepthLoss(self)
