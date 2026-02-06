import torch
import torch.nn as nn
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import (
    TaskAlignedAssigner,
    bbox2dist,
    dist2bbox,
    make_anchors,
)

# See for ultralytics.data.augment.LetterBox
ORIGINAL_PADDING_VALUE = 114


class DepthLoss:
    """
    Computes a depth-weighted object detection loss.

    This loss function combines classification, bounding box regression (IoU), and Distribution Focal Loss (DFL),
    and applies additional weighting to each loss term based on the depth of objects in the scene.
    """

    def __init__(self, model, tal_topk=10):
        self.device = next(model.parameters()).device
        self.model = model
        self.hyp = model.args

        detect_module = model.model[-1]

        self.reg_max = detect_module.reg_max
        self.nc = detect_module.nc  # number of classes
        self.no = detect_module.nc + self.reg_max * 4
        self.stride = detect_module.stride

        # TaskAlignedAssigner selects positive anchors for each ground truth box.
        # topk: number of top candidate anchors per gt box.
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl_loss = DFLoss(self.reg_max)
        self.proj = torch.arange(detect_module.reg_max, dtype=torch.float, device=self.device)

    def __call__(self, preds, batch):
        # B:        batch size
        # NO:       number of outputs per anchor
        # H_i, W_i: spatial dimensions of feature map
        # N:        total number of anchor points across all feature maps
        # NC:       number of classes
        # reg_max:  number of bins for DFL
        # max_targets_per_image:
        #           maximum number of ground truth objects present
        #           in any single image within the current batch

        ################ Extract logits ##############################################
        features = preds[1] if isinstance(preds, tuple) else preds
        bbox_logits, class_logits = self._get_logits(features)
        batch_size = bbox_logits.shape[0]
        num_anchors = bbox_logits.shape[1]
        assert bbox_logits.shape == (batch_size, num_anchors, 4 * self.reg_max)
        assert class_logits.shape == (batch_size, num_anchors, self.nc)
        # features: list of detection block outputs
        #           each of shape (B, NO, H_i, W_i)
        # bbox_logits:   (B, N, 4*reg_max)
        # class_logits:  (B, N, NC)

        ####### Make anchor points (based on feature map sizes) ######################
        anchor_points, anchor_strides = make_anchors(features, self.stride, grid_cell_offset=0.5)
        assert anchor_points.shape == (num_anchors, 2)
        assert anchor_strides.shape == (num_anchors, 1)
        # anchor_points:  (N, 2) (feature map coordinates)
        # anchor_strides: (N, 1) (factor to convert to image coordinates)

        ####### Decode bbox logits ###################################################
        bboxes_pred = self.bbox_decode(anchor_points, bbox_logits)
        assert bboxes_pred.shape == (batch_size, num_anchors, 4)
        # bboxes_pred: (B, N, 4)
        # in feature map coordinates (xyxy)

        ####### Make targets #########################################################
        imgsz = self._infer_imgsz(features, class_logits.dtype)
        labels_gt, bboxes_gt, mask_gt = self._get_targets(batch, imgsz)
        max_targets_per_image = bboxes_gt.shape[1]
        assert labels_gt.shape == (batch_size, max_targets_per_image, 1)
        assert bboxes_gt.shape == (batch_size, max_targets_per_image, 4)
        assert mask_gt.shape == (batch_size, max_targets_per_image, 1)
        # labels_gt: (B, max_targets_per_image, 1)
        # bboxes_gt: (B, max_targets_per_image, 4)
        # mask_gt:   (B, max_targets_per_image, 1)
        # mask_gt is a tensor (BoolTensor) indicating valid targets

        ####### Assign ground-truth (gt) objects to anchors ##########################
        bboxes_target_img, class_logits_target, fg_mask, target_gt_idx = self._assign_targets(
            anchor_points, anchor_strides, labels_gt, bboxes_gt, mask_gt, class_logits, bboxes_pred
        )
        bboxes_target = bboxes_target_img / anchor_strides
        assert bboxes_target_img.shape == (batch_size, num_anchors, 4)
        assert class_logits_target.shape == (batch_size, num_anchors, self.nc)
        assert fg_mask.shape == (batch_size, num_anchors)
        # bboxes_target_img:    (B, N, 4) in image coordinates
        # bboxes_target:        (B, N, 4) in feature map coordinates
        # class_logits_target:  (B, N, NC)
        # fg_mask:              (B, N) boolean mask for positive anchors
        #                       some anchors may not be assigned to any gt box

        ####### Losses ###############################################################
        class_loss = self._class_loss(class_logits, class_logits_target)
        iou_loss = self._iou_loss(bboxes_pred, bboxes_target, fg_mask, class_logits_target)
        dfl_loss = self._dfl_loss(anchor_points, bboxes_target, bbox_logits, fg_mask, class_logits_target)
        # Stack all loss terms: (B, N, 3) (3 = IoU, Class, DFL)
        loss_all = torch.stack([iou_loss, class_loss, dfl_loss], dim=2)
        assert loss_all.shape == (batch_size, num_anchors, 3)

        ####### Depth-aware weighting ################################################
        if self.hyp.depth_aware == "dls":
            self._weigh_dls(loss_all, batch, features)
        elif self.hyp.depth_aware == "dlw":
            self._weigh_dlw(loss_all, batch, features)
        elif self.hyp.depth_aware == "image-mean":
            self._weigh_image_mean(loss_all, batch, bboxes_gt, mask_gt)
        elif self.hyp.depth_aware == "batch-mean":
            self._weigh_batch_mean(loss_all, batch, bboxes_gt, mask_gt)
        else:
            raise ValueError(f"Unknown depth awareness strategy: {self.hyp.depth_aware}")

        # aggregate losses
        loss_all = loss_all.sum(dim=(0, 1))  # (box, cls, dfl)

        # Apply global loss weights from hyperparameters
        loss_all[0] *= self.hyp.box  # box loss
        loss_all[1] *= self.hyp.cls  # class loss
        loss_all[2] *= self.hyp.dfl  # distribution focal loss
        batch_size = bbox_logits.shape[0]
        return loss_all * batch_size, loss_all.detach()

    def _class_loss(self, class_logits, class_logits_target):
        class_logits_target_sum = max(class_logits_target.sum(), 1)
        class_loss = self.bce(class_logits, class_logits_target.to(class_logits.dtype))
        class_loss /= class_logits_target_sum
        class_loss = class_loss.sum(dim=-1)
        # (B, N, NC) -> (B, N)
        return class_loss

    def _iou_loss(self, bboxes_pred, bboxes_target, fg_mask, class_logits_target):
        class_logits_target_sum = max(class_logits_target.sum(), 1)
        weight = class_logits_target.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(bboxes_pred[fg_mask], bboxes_target[fg_mask], xywh=False, CIoU=True)
        iou_loss_shape = bboxes_pred.shape[:-1] + (1,)  # (B, N, 1)
        iou_loss = torch.zeros(iou_loss_shape, device=self.device)
        iou_loss[fg_mask] = ((1.0 - iou) * weight) / class_logits_target_sum  # (BN, N_fg)
        iou_loss = iou_loss.sum(dim=-1)  # (B, N)
        return iou_loss

    def _dfl_loss(self, anchor_points, bboxes_target, bbox_logits, fg_mask, class_logits_target):
        weight = class_logits_target.sum(-1)[fg_mask].unsqueeze(-1)

        target_ltrb = bbox2dist(anchor_points, bboxes_target, self.dfl_loss.reg_max - 1)
        dfl_loss_shape = bboxes_target.shape[:-1] + (1,)  # (B, N, 1)
        bbox_dfl_loss = torch.zeros(dfl_loss_shape, device=self.device)  # (B, N, 1)
        bbox_dfl_loss[fg_mask] = (
            self.dfl_loss(
                bbox_logits[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            )
            * weight
        )

        class_logits_target_sum = max(class_logits_target.sum(), 1)
        bbox_dfl_loss[fg_mask] = bbox_dfl_loss[fg_mask] / class_logits_target_sum
        bbox_dfl_loss = bbox_dfl_loss.sum(dim=-1)  # (B, N)
        return bbox_dfl_loss

    def _assign_targets(self, anchor_points, anchor_strides, labels_gt, bboxes_gt, mask_gt, class_logits, pred_bboxes):
        pred_bboxes_img = (pred_bboxes.detach() * anchor_strides).type(bboxes_gt.dtype)
        anchor_points_img = anchor_points * anchor_strides
        _, target_bboxes_img, class_scores_target, fg_mask, target_gt_idx = self.assigner(
            class_logits.detach().sigmoid(),
            pred_bboxes_img,
            anchor_points_img,
            labels_gt,
            bboxes_gt,
            mask_gt,
        )
        return target_bboxes_img, class_scores_target, fg_mask, target_gt_idx

    def _weigh_dls(self, loss_all, batch, features):
        """
        DLS (Depth-Based Loss Stratification): Applies per-anchor weights based on depth threshold.

        Each depth map is normalized independently (ignoring padding), resized to every detection
        head, and compared against `self.hyp.depth_threshold` (default 0.5). Anchors below the
        threshold are treated as "close" and scaled by `close_weight`, while anchors at/above the
        threshold are treated as "far" and scaled by `far_weight`.

        Args:
            loss_all: (B, N, 3) tensor containing loss components for each anchor
            batch: Contains depth maps in 'depth_map'
            features: List of feature maps from detection heads
        """
        depth_maps = torch.stack(batch["depth_map"]).to(self.device)

        padding_mask = depth_maps == ORIGINAL_PADDING_VALUE
        depth_maps[padding_mask] = torch.nan
        depth_maps = self._minmax_normalize_depth_maps(depth_maps, mode="image")
        depth_maps[padding_mask] = 0

        depth_values = self._resize_to_feature_map(depth_maps, features)
        assert depth_values.shape == (loss_all.shape[0], loss_all.shape[1])

        # Threshold: points with depth < threshold are 'far' (True), >= threshold are 'close' (False)
        far_masks = depth_values < self.hyp.depth_threshold  # (B, N)

        # Apply depth-based weighting
        loss_all[far_masks, :] *= self.hyp.far_weight
        loss_all[~far_masks, :] *= self.hyp.close_weight

    def _weigh_dlw(self, loss_all, batch, features):
        """
        DLW (Depth-Based Loss Weighting): Applies smooth, per-anchor exponential weighting.

        Depth maps are resized to every detection head and jointly min-max normalized across the
        whole batch so that all anchors share a consistent [0, 1] range. The normalized depth values
        (optionally flipped when `depth_inverse` is True) are fed through `1 + alpha * exp(depth)`
        and directly multiply the IoU/class/DFL terms.

        Args:
            loss_all: (B, N, 3) tensor containing loss components for each anchor
            batch: Contains depth maps in 'depth_map'
            features: List of feature maps from detection heads
        """
        depth_maps = torch.stack(batch["depth_map"]).to(self.device)
        padding_mask = depth_maps == ORIGINAL_PADDING_VALUE
        depth_maps[padding_mask] = torch.nan
        depth_maps = self._minmax_normalize_depth_maps(depth_maps, mode="batch")
        depth_maps[padding_mask] = 0

        depth_values = self._resize_to_feature_map(depth_maps, features)
        assert depth_values.shape == (loss_all.shape[0], loss_all.shape[1])

        if self.hyp.depth_inverse:
            depth_values = 1.0 - depth_values

        # map depth values to weights
        weights = 1 + self.hyp.alpha * torch.exp(depth_values)

        # Apply depth-based weighting
        loss_all *= weights.unsqueeze(-1)

    def _weigh_image_mean(self, loss_all, batch, bboxes_gt, mask_gt):
        """
        Applies a single weight per image derived from batch-normalized box means.

        Raw depth maps are averaged inside every valid ground-truth box, the resulting per-box
        means are min-max normalized across the whole batch (optionally inverted via `depth_inverse`),
        and then averaged per image. The final values pass through `1 + alpha * exp(depth_mean)` and
        are broadcast to all anchors in the corresponding image.

        Args:
            loss_all: (B, N, 3) tensor containing loss components for each anchor
            batch: Contains depth maps in 'depth_map'
            bboxes_gt: (B, max_targets_per_image, 4) ground truth bounding boxes
            mask_gt: (B, max_targets_per_image, 1) boolean mask for valid targets
        """
        # load depth maps
        depth_maps = torch.stack(batch["depth_map"]).to(self.device)
        # (B, H_full, W_full), e.g. (128, 640, 640)

        padding_mask = depth_maps == ORIGINAL_PADDING_VALUE
        depth_maps[padding_mask] = torch.nan

        # Calculate mean depth for each bbox
        depth_values = self._calculate_mean_depth_per_bbox(depth_maps, bboxes_gt, mask_gt)
        assert depth_values.shape == (bboxes_gt.shape[0], bboxes_gt.shape[1])
        if torch.isnan(depth_values).any():
            num_nan = torch.isnan(depth_values).sum().item()
            print(f"Warning: {num_nan} NaN values found in depth_values")
        # (B, max_targets)

        # normalize depth values to [0, 1]
        # set invalid boxes (mask_gt) to nan before normalization
        depth_values = depth_values.masked_fill(~mask_gt.squeeze(-1).bool(), float("nan"))
        depth_values = min_max_normalize(depth_values, dimension=1)
        assert depth_values.shape == (bboxes_gt.shape[0], bboxes_gt.shape[1])

        if self.hyp.depth_inverse:
            depth_values = 1.0 - depth_values

        # compute mean depth per image
        n_boxes = mask_gt.sum(dim=(1, 2))  # (B,)
        assert n_boxes.shape == (loss_all.shape[0],)
        depth_per_image = torch.nansum(depth_values, dim=1) / (n_boxes + 1e-8)
        assert depth_per_image.shape == (loss_all.shape[0],)
        # (B,)

        # map depth values to weights
        weights = 1 + self.hyp.alpha * torch.exp(depth_per_image)
        assert weights.shape == (loss_all.shape[0],)
        # (B,)

        # check if any weights are negative
        if (weights < 0).any():
            raise ValueError(
                "Negative weights encountered in depth-based weighting. Adjust far_weight and close_weight."
            )

        # Apply depth-based weighting
        # loss_all: (B, N, 3)
        loss_all *= weights.unsqueeze(-1).unsqueeze(-1)

    def _weigh_batch_mean(self, loss_all, batch, bboxes_gt, mask_gt):
        """
        Applies a single global weight derived from batch-wide normalized depths.

        Mean depth is computed inside every valid ground-truth box, normalized to [0, 1] across the
        entire batch (optionally inverted via `depth_inverse`), and then averaged to obtain one
        scalar per image and finally per batch. The resulting `1 + alpha * exp(mean_depth)` value
        multiplies every anchor in the mini-batch.

        Args:
            loss_all: (B, N, 3) tensor containing loss components for each anchor
            batch: Contains depth maps in 'depth_map'
            bboxes_gt: (B, max_targets_per_image, 4) ground truth bounding boxes
            mask_gt: (B, max_targets_per_image, 1) boolean mask for valid targets
        """
        # load depth maps
        depth_maps = torch.stack(batch["depth_map"]).to(self.device)
        # (B, H_full, W_full), e.g. (128, 640, 640)

        padding_mask = depth_maps == ORIGINAL_PADDING_VALUE
        depth_maps[padding_mask] = torch.nan

        # Calculate mean depth for each bbox
        depth_values = self._calculate_mean_depth_per_bbox(depth_maps, bboxes_gt, mask_gt)
        assert depth_values.shape == (bboxes_gt.shape[0], bboxes_gt.shape[1])
        if torch.isnan(depth_values).any():
            num_nan = torch.isnan(depth_values).sum().item()
            print(f"Warning: {num_nan} NaN values found in depth_values")
        # (B, max_targets)

        # normalize depth values to [0, 1]
        # set invalid boxes (mask_gt) to nan before normalization
        depth_values = depth_values.masked_fill(~mask_gt.squeeze(-1).bool(), float("nan"))
        depth_values = min_max_normalize(depth_values, dimension=1)
        assert depth_values.shape == (bboxes_gt.shape[0], bboxes_gt.shape[1])

        if self.hyp.depth_inverse:
            depth_values = 1.0 - depth_values

        # compute mean depth per image
        n_boxes = mask_gt.sum(dim=(1, 2))  # (B,)
        assert n_boxes.shape == (loss_all.shape[0],)
        depth_per_image = torch.nansum(depth_values, dim=1) / (n_boxes + 1e-8)
        assert depth_per_image.shape == (loss_all.shape[0],)
        # (B,)

        # mean depth across batch
        mean_depth = depth_per_image.sum() / depth_per_image.shape[0]
        assert isinstance(mean_depth.item(), float)

        weight = 1 + self.hyp.alpha * torch.exp(mean_depth)
        assert isinstance(weight.item(), float)

        # check if weight is negative
        if weight < 0:
            raise ValueError(
                "Negative weights encountered in depth-based weighting. Adjust far_weight and close_weight."
            )

        # Apply depth-based weighting
        loss_all *= weight

    def _get_logits(self, features):
        """
        Reshapes and splits the model output features into predicted distributions and class scores.

        Args:
            features: List of feature maps from detection heads.

        Returns:
            Tuple of (box_logits, class_logits)
        """
        features_reshaped = [xi.view(features[0].shape[0], self.no, -1) for xi in features]
        # list of (B, NO, Hi*Wi)
        features_concatenated = torch.cat(features_reshaped, dim=2)
        # (B, NO, N)
        box_logits, class_logits = features_concatenated.split((self.reg_max * 4, self.nc), dim=1)
        # box_logits: (B, 4*reg_max, N)
        # class_logits: (B, NC, N)
        class_logits = class_logits.permute(0, 2, 1).contiguous()
        # (B, N, NC)
        box_logits = box_logits.permute(0, 2, 1).contiguous()
        # (B, N, 4*reg_max)
        return box_logits, class_logits

    def _calculate_mean_depth_per_bbox(self, depth_maps, bboxes_gt, mask_gt):
        # bboxes_gt: (B, max_targets, 4) in image coordinates (xyxy)

        batch_size = depth_maps.shape[0]
        max_targets = bboxes_gt.shape[1]

        # Create coordinate tensors for bbox indexing
        x1y1 = bboxes_gt[..., :2].long()  # (B, max_targets, 2)
        x2y2 = bboxes_gt[..., 2:].long()  # (B, max_targets, 2)

        depth_values = torch.zeros(batch_size, max_targets, device=self.device)
        valid_boxes = mask_gt.squeeze(-1)  # (B, max_targets)

        # For valid boxes, compute mean depth
        for b in range(batch_size):
            valid_idx = valid_boxes[b].bool()
            if valid_idx.any():
                x1, y1 = x1y1[b, valid_idx].T
                x2, y2 = x2y2[b, valid_idx].T

                depth_values[b, valid_idx] = torch.stack(
                    [torch.nanmean(depth_maps[b, y1i:y2i, x1i:x2i]) for y1i, y2i, x1i, x2i in zip(y1, y2, x1, x2)]
                )
        return depth_values

    def _minmax_normalize_depth_maps(self, depth_maps: torch.Tensor, mode="image"):
        batch_size = depth_maps.shape[0]

        if mode == "image":
            # min-max normalization per depth map
            # nanquantile to ignore NaNs from padding
            depth_maps_flat = depth_maps.view(batch_size, -1)
            depth_maps_flat = min_max_normalize(depth_maps_flat, dimension=1)
            depth_maps = depth_maps_flat.view(depth_maps.shape)
        elif mode == "batch":
            # min-max normalization across the batch
            depth_flat = depth_maps.view(-1)
            depth_flat = min_max_normalize(depth_flat, dimension=0)
            depth_maps = depth_flat.view(depth_maps.shape)

        return depth_maps

    def _resize_to_feature_map(self, depth_maps, features):
        # depth_maps: (B, H_full, W_full)
        # Resize to feature map
        batch_size = depth_maps.shape[0]

        depth_values = []
        for i in range(len(features)):
            h, w = features[i].shape[2:]
            # Interpolate depth maps to match feature map size
            interpolated = nn.functional.interpolate(
                depth_maps.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            # (B, h, w)
            depth_values.append(interpolated.view(batch_size, -1))

        depth_values = torch.cat(depth_values, dim=1)  # (B, N)
        return depth_values

    def _infer_imgsz(self, feats, dtype):
        """
        Infers the image size from the feature maps and strides.
        """
        return torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

    def _get_targets(self, batch, imgsz):
        """
        Prepares ground truth labels and bounding boxes for loss computation.
        """
        batch_size = batch["img"].shape[0]
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        return gt_labels, gt_bboxes, mask_gt

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocess targets by converting to tensor format and scaling coordinates.
        """
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.
        """
        b, a, c = pred_dist.shape  # batch, anchors, channels
        # Softmax over reg_max bins, then project to continuous values
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)


def min_max_normalize(tensor, dimension):
    min_val = torch.nanquantile(tensor, 0.0, dim=dimension, keepdim=True)
    max_val = torch.nanquantile(tensor, 1.0, dim=dimension, keepdim=True)
    normalized = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized
