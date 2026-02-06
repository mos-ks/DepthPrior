from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.models.yolo.detect import DetectionValidator


class CocoEvalDetectionValidator(DetectionValidator):
    def __init__(self, dataloader, save_dir, args: dict, _callbacks):
        # This needs to be set so that self.jdict is filled (required for coco evaluation)
        args.save_json = True
        super().__init__(dataloader, save_dir, args, _callbacks)

    def get_stats(self):
        stats_original = super().get_stats()
        stats_coco = self._compute_coco_metrics()
        return {**stats_original, **stats_coco}

    def _compute_coco_metrics(self):
        # check DetectionValidator.eval_json for reference

        ground_truth_file = self.data["path"] / "annotations_coco.json"
        ground_truth = COCO(str(ground_truth_file))
        predictions = self.jdict  # contains image_id, category_id, bbox, score

        if isinstance(predictions[0]["image_id"], str):
            # pycocotools requires integer image IDs
            id_mapping = {str(Path(x).stem): idx for idx, x in enumerate(self.dataloader.dataset.im_files)}
            img_ids = []
            for pred in predictions:
                image_id = id_mapping[pred["image_id"]]
                pred["image_id"] = image_id
                img_ids.append(image_id)
        else:
            img_ids = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]

        predictions = ground_truth.loadRes(predictions)

        val = COCOeval(ground_truth, predictions, "bbox")
        val.params.imgIds = img_ids
        val.evaluate()
        val.accumulate()

        # adapted from pycocotools.cocoeval.COCOeval
        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            params = val.params
            eval = val.eval

            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = f"{params.iouThrs[0]:0.2f}:{params.iouThrs[-1]:0.2f}" if iouThr is None else f"{iouThr:0.2f}"

            aind = [i for i, aRng in enumerate(params.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(params.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == params.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == params.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            mean_s = -1 if len(s[s > -1]) == 0 else np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        # Compute metrics
        stats = {}
        stats["ap50_95_maxdets100"] = _summarize(ap=1)
        stats["ap50_maxdets100"] = _summarize(ap=1, iouThr=0.5, maxDets=val.params.maxDets[2])
        stats["ap75_maxdets100"] = _summarize(ap=1, iouThr=0.75, maxDets=val.params.maxDets[2])
        stats["ap50_95_maxdets100_small"] = _summarize(ap=1, areaRng="small", maxDets=val.params.maxDets[2])
        stats["ap50_95_maxdets100_medium"] = _summarize(ap=1, areaRng="medium", maxDets=val.params.maxDets[2])
        stats["ap50_95_maxdets100_large"] = _summarize(ap=1, areaRng="large", maxDets=val.params.maxDets[2])
        stats["ap50_maxdets100_small"] = _summarize(ap=1, iouThr=0.5, areaRng="small", maxDets=val.params.maxDets[2])
        stats["ap50_maxdets100_medium"] = _summarize(ap=1, iouThr=0.5, areaRng="medium", maxDets=val.params.maxDets[2])
        stats["ap50_maxdets100_large"] = _summarize(ap=1, iouThr=0.5, areaRng="large", maxDets=val.params.maxDets[2])
        stats["ar50_95_maxdets1"] = _summarize(ap=0, maxDets=val.params.maxDets[0])
        stats["ar50_95_maxdets10"] = _summarize(ap=0, maxDets=val.params.maxDets[1])
        stats["ar50_95_maxdets100"] = _summarize(ap=0, maxDets=val.params.maxDets[2])
        stats["ar50_95_maxdets100_small"] = _summarize(ap=0, areaRng="small", maxDets=val.params.maxDets[2])
        stats["ar50_95_maxdets100_medium"] = _summarize(ap=0, areaRng="medium", maxDets=val.params.maxDets[2])
        stats["ar50_95_maxdets100_large"] = _summarize(ap=0, areaRng="large", maxDets=val.params.maxDets[2])
        stats["ar50_maxdets100_small"] = _summarize(ap=0, iouThr=0.5, areaRng="small", maxDets=val.params.maxDets[2])
        stats["ar50_maxdets100_medium"] = _summarize(ap=0, iouThr=0.5, areaRng="medium", maxDets=val.params.maxDets[2])
        stats["ar50_maxdets100_large"] = _summarize(ap=0, iouThr=0.5, areaRng="large", maxDets=val.params.maxDets[2])
        return {f"metrics/{k}": v for k, v in stats.items()}

    def init_metrics(self, model):
        super().init_metrics(model)

        # overwrite class_map
        # in super().init_metrics the indices are shifted
        self.class_map = list(range(len(model.names)))
