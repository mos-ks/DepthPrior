# Original Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

"""Modified.
Training related libraries."""


import glob
import math
import os
import re

import anchors
import coco_metric
import efficientdet_keras
import iou_utils
import label_util
import neural_structured_learning as nsl
import numpy as np
import postprocess
import sklearn.metrics
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import tensorflow_hub as hub
import utils
import utils_infer
import utils_keras
from absl import logging
from matplotlib import pyplot as plt
from utils_class import stable_softmax
from utils_extra import mc_eval, mtplt_to_img, plot_conf_matrix, plot_roc


def update_learning_rate_schedule_parameters(params):
    """Updates params that are related to the learning rate schedule."""
    batch_size = params["batch_size"]
    # Learning rate is proportional to the batch size
    params["adjusted_learning_rate"] = params["learning_rate"] * batch_size / 64
    steps_per_epoch = params["steps_per_epoch"]
    params["lr_warmup_step"] = int(params["lr_warmup_epoch"] * steps_per_epoch)
    params["first_lr_drop_step"] = int(params["first_lr_drop_epoch"] * steps_per_epoch)
    params["second_lr_drop_step"] = int(
        params["second_lr_drop_epoch"] * steps_per_epoch
    )
    params["total_steps"] = int(params["num_epochs"] * steps_per_epoch)


class StepwiseLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Stepwise learning rate schedule."""

    def __init__(
        self,
        adjusted_lr: float,
        lr_warmup_init: float,
        lr_warmup_step: int,
        first_lr_drop_step: int,
        second_lr_drop_step: int,
    ):
        """Build a StepwiseLrSchedule.

        Args:
          adjusted_lr: `float`, The initial learning rate.
          lr_warmup_init: `float`, The warm up learning rate.
          lr_warmup_step: `int`, The warm up step.
          first_lr_drop_step: `int`, First lr decay step.
          second_lr_drop_step: `int`, Second lr decay step.
        """
        super().__init__()
        logging.info("LR schedule method: stepwise")
        self.adjusted_lr = adjusted_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.first_lr_drop_step = first_lr_drop_step
        self.second_lr_drop_step = second_lr_drop_step

    def __call__(self, step):
        linear_warmup = self.lr_warmup_init + (
            tf.cast(step, dtype=tf.float32)
            / self.lr_warmup_step
            * (self.adjusted_lr - self.lr_warmup_init)
        )
        learning_rate = tf.where(
            step < self.lr_warmup_step, linear_warmup, self.adjusted_lr
        )
        lr_schedule = [
            [1.0, self.lr_warmup_step],
            [0.1, self.first_lr_drop_step],
            [0.01, self.second_lr_drop_step],
        ]
        for mult, start_global_step in lr_schedule:
            learning_rate = tf.where(
                step < start_global_step, learning_rate, self.adjusted_lr * mult
            )
        return learning_rate


class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Cosine learning rate schedule."""

    def __init__(
        self,
        adjusted_lr: float,
        lr_warmup_init: float,
        lr_warmup_step: int,
        total_steps: int,
    ):
        """Build a CosineLrSchedule.

        Args:
          adjusted_lr: `float`, The initial learning rate.
          lr_warmup_init: `float`, The warm up learning rate.
          lr_warmup_step: `int`, The warm up step.
          total_steps: `int`, Total train steps.
        """
        super().__init__()
        logging.info("LR schedule method: cosine")
        self.adjusted_lr = adjusted_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

    def __call__(self, step):
        linear_warmup = self.lr_warmup_init + (
            tf.cast(step, dtype=tf.float32)
            / self.lr_warmup_step
            * (self.adjusted_lr - self.lr_warmup_init)
        )
        cosine_lr = (
            0.5
            * self.adjusted_lr
            * (1 + tf.cos(math.pi * tf.cast(step, tf.float32) / self.decay_steps))
        )
        return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)


class PolynomialLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Polynomial learning rate schedule."""

    def __init__(
        self,
        adjusted_lr: float,
        lr_warmup_init: float,
        lr_warmup_step: int,
        power: float,
        total_steps: int,
    ):
        """Build a PolynomialLrSchedule.

        Args:
          adjusted_lr: `float`, The initial learning rate.
          lr_warmup_init: `float`, The warm up learning rate.
          lr_warmup_step: `int`, The warm up step.
          power: `float`, power.
          total_steps: `int`, Total train steps.
        """
        super().__init__()
        logging.info("LR schedule method: polynomial")
        self.adjusted_lr = adjusted_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.power = power
        self.total_steps = total_steps

    def __call__(self, step):
        linear_warmup = self.lr_warmup_init + (
            tf.cast(step, dtype=tf.float32)
            / self.lr_warmup_step
            * (self.adjusted_lr - self.lr_warmup_init)
        )
        polynomial_lr = self.adjusted_lr * tf.pow(
            1 - (tf.cast(step, dtype=tf.float32) / self.total_steps), self.power
        )
        return tf.where(step < self.lr_warmup_step, linear_warmup, polynomial_lr)


def learning_rate_schedule(params):
    """Learning rate schedule based on global step."""
    update_learning_rate_schedule_parameters(params)
    lr_decay_method = params["lr_decay_method"]
    if lr_decay_method == "stepwise":
        return StepwiseLrSchedule(
            params["adjusted_learning_rate"],
            params["lr_warmup_init"],
            params["lr_warmup_step"],
            params["first_lr_drop_step"],
            params["second_lr_drop_step"],
        )

    if lr_decay_method == "cosine":
        return CosineLrSchedule(
            params["adjusted_learning_rate"],
            params["lr_warmup_init"],
            params["lr_warmup_step"],
            params["total_steps"],
        )

    if lr_decay_method == "polynomial":
        return PolynomialLrSchedule(
            params["adjusted_learning_rate"],
            params["lr_warmup_init"],
            params["lr_warmup_step"],
            params["poly_lr_power"],
            params["total_steps"],
        )

    raise ValueError("unknown lr_decay_method: {}".format(lr_decay_method))


def get_optimizer(params):
    """Get optimizer."""
    learning_rate = learning_rate_schedule(params)
    momentum = params["momentum"]
    if params["optimizer"].lower() == "sgd":
        logging.info("Use SGD optimizer")
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=momentum)
    elif params["optimizer"].lower() == "adam":
        logging.info("Use Adam optimizer")
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=momentum)
    else:
        raise ValueError("optimizers should be adam or sgd")

    moving_average_decay = params["moving_average_decay"]
    if moving_average_decay:
        from tensorflow_addons import \
            optimizers as tfa_optimizers  # pylint: disable=g-import-not-at-top

        optimizer = tfa_optimizers.MovingAverage(
            optimizer, average_decay=moving_average_decay, dynamic_decay=True
        )
    precision = utils.get_precision(params["strategy"], params["mixed_precision"])
    if precision == "mixed_float16" and params["loss_scale"]:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, initial_scale=params["loss_scale"]
        )
    return optimizer


class EarlyStoppingCustomCallback(tf.keras.callbacks.Callback):
    "Checks validation loss for early stopping"

    def __init__(self, monitor="val_loss", patience=10):
        super(EarlyStoppingCustomCallback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = float("inf") if "loss" in monitor else float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            raise ValueError(
                f"Metric '{self.monitor}' not found in logs. Available metrics are: {','.join(logs.keys())}"
            )

        if (self.monitor == "val_loss" and current_metric < self.best_metric) or (
            self.monitor != "val_loss" and current_metric > self.best_metric
        ):
            self.best_metric = current_metric
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)


class COCOCallback(tf.keras.callbacks.Callback):
    """A utility for COCO eval callback.
    Also plots the confusion matrix, mAP/IoU and ROC curve based on validation dataset.
    """

    def __init__(self, test_dataset, update_freq=None, conf_matrix=False):
        """Constructs the necessary attributes for calculating the metrics
        Args:
          test_dataset (tensor): Validation dataset to test the model on
          update_freq (int): Epoch frequency at which results are calculated
          conf_matrix (bool): Plot confusion matrix and ROC curve
        """
        super().__init__()
        self.test_dataset = test_dataset
        self.update_freq = update_freq

        if conf_matrix:
            self.conf_matrix = conf_matrix
            self.gather_classes = []
            self.gather_logits = []
            self.gather_boxes = []
            self.gather_gtboxes = []

    def set_model(self, model: tf.keras.Model):
        self.model = model
        config = model.config
        self.config = config
        label_map = label_util.get_label_map(config.label_map)
        # Extract class names
        self.class_names = np.asarray([label_map[key] for key in label_map])

        log_dir = os.path.join(config.model_dir, "coco")
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.evaluator = coco_metric.EvaluationMetric(
            filename=config.val_json_file, label_map=label_map
        )

        # Get AP for each class
        if self.config.label_map:
            for i in range(len(self.class_names)):
                name = "AP_/%s" % self.class_names[i]
                self.evaluator.metric_names.append(name)

    def _plot_map(self, aps, iou_thrs, per_cls=False):
        """Plot mAP curve for IoU thresholds from 0.05 to 0.95 for each and all classes

        Args:
          aps (array): mAPs
          iou_thrs (array): IoU thresholds
          per_cls: Select per-class AP
        """
        if per_cls:
            title = "for each class"
        else:
            title = "for all classes"
        fig = plt.figure()
        for i in range(len(aps)):
            plt.plot(iou_thrs, aps[i], label=self.class_names[i])
        plt.ylabel("Average Precision at IoU")
        plt.xlabel("IoU Threshold")
        if per_cls:
            plt.legend()
        plt.title("AP Curve " + title)
        return fig

    def _gather_pred_classes(self, pred_classes):
        self.gather_classes = np.append(self.gather_classes, pred_classes)

    def _gather_pred_logits(self, pred_logits):
        self.gather_logits = np.append(self.gather_logits, pred_logits)

    def _gather_pred_boxes(self, pred_boxes):
        self.gather_boxes = np.append(self.gather_boxes, [pred_boxes])

    def _gather_gt_boxes(self, gt_boxes):
        self.gather_gtboxes = np.append(self.gather_gtboxes, [gt_boxes])

    @tf.function
    def _get_detections(self, images, labels):
        if self.config.mc_dropout:
            cls_outputs, box_outputs = mc_eval(self.model, images, self.config)
        else:
            cls_outputs, box_outputs = self.model(images, training=False)

        detections = postprocess.generate_detections(
            self.config,
            cls_outputs,
            box_outputs,
            labels["image_scales"],
            labels["source_ids"],
        )

        if self.conf_matrix:
            if self.config.enable_softmax:
                tf.numpy_function(
                    self._gather_pred_classes,
                    [detections[:, :, -(self.config.num_classes + 1)]],
                    [],
                )
                tf.numpy_function(
                    self._gather_pred_logits,
                    [detections[:, :, -self.config.num_classes :]],
                    [],
                )

            else:
                tf.numpy_function(self._gather_pred_classes, [detections[:, :, -1]], [])

            tf.numpy_function(self._gather_pred_boxes, [detections[:, :, 1:5]], [])
            tf.numpy_function(
                self._gather_gt_boxes, [labels["groundtruth_data"][:, :, :4]], []
            )

        tf.numpy_function(
            self.evaluator.update_state,
            [labels["groundtruth_data"], postprocess.transform_detections(detections)],
            [],
        )

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if self.update_freq and epoch % self.update_freq == 0:
            logging.info("Starting COCO eval")
            self.evaluator.reset_states()
            strategy = tf.distribute.get_strategy()
            count = self.config.eval_samples // self.config.batch_size
            dataset = self.test_dataset.take(count)
            dataset = strategy.experimental_distribute_dataset(dataset)
            for images, labels in dataset:
                strategy.run(self._get_detections, (images, labels))
            metrics, all_maps = self.evaluator.result()

            # Calculate mIoU
            if self.conf_matrix:
                from utils_box import calc_iou_np
                from utils_extra import gt_box_assigner

                indices = np.where(self.gather_gtboxes.reshape([-1, 4])[:, -1] > -1)[0]
                gtbboxes = self.gather_gtboxes.reshape([self.config.batch_size, -1, 4])
                gtbboxes = np.swapaxes(
                    np.swapaxes(
                        [
                            gtbboxes[:, :, 1],
                            gtbboxes[:, :, 0],
                            gtbboxes[:, :, 3],
                            gtbboxes[:, :, 2],
                        ],
                        0,
                        2,
                    ),
                    0,
                    1,
                )
                bboxes = self.gather_boxes.reshape([self.config.batch_size, -1, 4])

                valid_indices = np.all(gtbboxes > -1, axis=-1)
                best_indices = [
                    list(
                        map(
                            lambda j: gt_box_assigner("IoU", gtbboxes[i], bboxes[i], j),
                            np.where(valid_indices[i])[0],
                        )
                    )
                    for i in range(len(gtbboxes))
                ]

                filtered_bboxes = np.concatenate(
                    [bboxes[i][best_indices[i]] for i in range(len(bboxes))]
                )
                gtbboxes = gtbboxes.reshape([-1, 4])[indices]
                miou = np.mean(calc_iou_np(gtbboxes, filtered_bboxes))
                tf.summary.scalar("mIou", miou, step=epoch)

            # Plot mAP per IoU threshold
            iou_thrs = np.linspace(
                0.05, 0.95, int(np.round((0.95 - 0.05) / 0.05)) + 1, endpoint=True
            )
            map_plots = []
            aps_prothr = [np.mean(all_maps[i, :, :]) for i in range(len(all_maps))]
            map_fig = self._plot_map([aps_prothr], iou_thrs)
            map_plots.append(mtplt_to_img(map_fig))
            try:
                aps_prothr_percls = [
                    [np.mean(all_maps[i, :, j]) for i in range(len(all_maps))]
                    for j in range(self.config.num_classes)
                ]
                map_fig = self._plot_map(aps_prothr_percls, iou_thrs, per_cls=True)
                map_plots.append(mtplt_to_img(map_fig))
            except IndexError:  # Pop from empty list error
                pass
            with self.file_writer.as_default():
                tf.summary.image(
                    "AP/IOU Curves",
                    tf.concat(map_plots, axis=0),
                    step=epoch,
                    max_outputs=1000,
                )

            # Get coco metrics
            eval_results = {}
            with self.file_writer.as_default(), tf.summary.record_if(True):
                for i, name in enumerate(self.evaluator.metric_names):
                    tf.summary.scalar(name, metrics[i], step=epoch)
                    eval_results[name] = metrics[i]

            # Plot confusion matrix and ROC plot
            if self.conf_matrix:
                try:
                    labels = list(map(lambda x: x[1], self.test_dataset))
                    gt_classes_filter = []
                    for i in range(len(labels)):
                        gt_classes_filter.append(
                            labels[i]["groundtruth_data"][:, :, -1]
                        )  # [y1, x1, y2, x2, is_crowd, area, class]
                    gt_classes_filter = np.reshape(gt_classes_filter, [-1])
                    gt_classes = gt_classes_filter[gt_classes_filter >= 0]
                    pred_classes = self.gather_classes[gt_classes_filter >= 0]

                    # Calculate confusion matrix
                    cm = sklearn.metrics.confusion_matrix(gt_classes, pred_classes)
                    figure = plot_conf_matrix(cm, class_names=self.class_names)
                    cm_image = mtplt_to_img(figure)

                    if self.config.enable_softmax:
                        pred_softmax = stable_softmax(
                            self.gather_logits.reshape(-1, self.config.num_classes)[
                                gt_classes_filter >= 0
                            ]
                        )
                        roc_fig = plot_roc(gt_classes - 1, pred_softmax, self.class_names)
                        roc_image = mtplt_to_img(roc_fig)
                        with self.file_writer.as_default():
                            tf.summary.image("ROC Curve", roc_image, step=epoch)
                    with self.file_writer.as_default():
                        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

                    self.gather_classes = []
                    self.gather_logits = []
                    self.gather_boxes = []
                    self.gather_gtboxes = []
                except Exception as e:
                    logging.error("Error while plotting confusion matrix or ROC: %s", e)
            return eval_results


class DisplayCallback(tf.keras.callbacks.Callback):
    """Display inference result on test images for different score and IoU thresholds"""

    def __init__(self, image_names, update_freq=None):
        """Constructs the necessary attributes for saving the detections on test images

        Args:
          image_names (array): Images names including path
          update_freq (int): Epoch frequency at which results are calculated
        """
        super().__init__()
        self.sample_images = [
            tf.image.decode_jpeg(tf.io.read_file(im), channels=3) for im in image_names
        ]
        self.update_freq = update_freq

    def set_model(self, model: tf.keras.Model):
        self.model = model
        config = model.config
        log_dir = os.path.join(config.model_dir, "test_images")
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if self.update_freq and epoch % self.update_freq == 0:
            self._draw_inference(epoch)

    def _draw_inference(self, step):
        """Plot the detections on their images for different IoU and score thresholds"""
        self.model.__class__ = efficientdet_keras.EfficientDetModel
        orig_iou_thresh = self.model.config.nms_configs.iou_thresh
        orig_score_thresh = self.model.config.nms_configs.score_thresh

        self.iou_thrs = np.linspace(0, 1, 11)[0:10]
        self.score_thrs = np.linspace(0, 1, 11)[0:10]
        self.max_boxes_to_draw = None

        for i in range(len(self.iou_thrs)):
            self.model.config.override(
                "nms_configs.iou_thresh=" + str(self.iou_thrs[i])
            )
            for j in range(len(self.score_thrs)):
                self.model.config.override(
                    "nms_configs.score_thresh=" + str(self.score_thrs[j])
                )
                images = []
                for k in range(len(self.sample_images)):
                    predictions = self.model(
                        tf.expand_dims(self.sample_images[k], axis=0), training=False
                    )
                    if self.model.config.enable_softmax:
                        boxes, scores, classes, valid_len, _ = tf.nest.map_structure(
                            np.array, predictions
                        )
                    else:
                        boxes, scores, classes, valid_len = tf.nest.map_structure(
                            np.array, predictions
                        )
                    filter = valid_len[0]
                    image = utils_infer.visualize_image(
                        self.sample_images[k],
                        boxes[0][:filter],
                        classes[0].astype(int)[:filter],
                        scores[0][:filter],
                        label_map=self.model.config.label_map,
                        min_score_thresh=0,
                        max_boxes_to_draw=self.max_boxes_to_draw,
                    )
                    images.append(image)
                with self.file_writer.as_default():
                    for k in range(len(images)):
                        tf.summary.image(
                            name="Test image for threshold, iou: "
                            + str("%.1f" % self.iou_thrs[i])
                            + ", score: "
                            + str("%.1f" % self.score_thrs[j])
                            + "/"
                            + str(k),
                            data=tf.expand_dims(images[k], axis=0),
                            step=step,
                            max_outputs=10000000,
                        )
        # Reset
        self.model.__class__ = EfficientDetNetTrain
        self.model.config.override("nms_configs.iou_thresh=" + str(orig_iou_thresh))
        self.model.config.override("nms_configs.score_thresh=" + str(orig_score_thresh))


class KeepLastNCheckpoints(tf.keras.callbacks.Callback):
    """Keeps only the last N checkpoints"""

    def __init__(self, directory, num_to_keep=3):
        super(KeepLastNCheckpoints, self).__init__()
        self.directory = directory
        self.num_to_keep = num_to_keep

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 3:
            checkpoint_files = glob.glob(
                os.path.join(self.directory, "best_ckpt-*.index")
            )
            checkpoint_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

            if len(checkpoint_files) > self.num_to_keep:
                for file_to_remove in checkpoint_files[: -self.num_to_keep]:
                    os.remove(file_to_remove)
                    os.remove(
                        file_to_remove.split(".index")[0] + ".data-00000-of-00001"
                    )


def get_callbacks(params, val_dataset=None):
    """Get callbacks for given params."""
    if params["moving_average_decay"]:
        from tensorflow_addons import \
            callbacks as tfa_callbacks  # pylint: disable=g-import-not-at-top

        avg_callback = tfa_callbacks.AverageModelCheckpoint(
            filepath=os.path.join(params["model_dir"], "emackpt-{epoch:d}"),
            verbose=params["verbose"],
            save_freq=int(params["steps_per_epoch"] * params["save_freq"]),
            save_weights_only=True,
            update_weights=False,
        )
        callbacks = [avg_callback]
    else:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(params["model_dir"], "ckpt-{epoch:d}"),
            verbose=params["verbose"],
            save_freq=int(params["steps_per_epoch"] * params["save_freq"]),
            save_weights_only=True,
        )

        # best_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        #     os.path.join(params["model_dir"] + "/best_ckpts/", "best_ckpt-{epoch:d}"),
        #     monitor="val_loss",
        #     save_best_only=True,
        #     mode="min",
        #     save_weights_only=True,
        # )
        # keep_last_n_checkpoints = KeepLastNCheckpoints(
        #     directory=params["model_dir"] + "/best_ckpts/", num_to_keep=3
        # )
        # callbacks = [best_ckpt_callback, keep_last_n_checkpoints, ckpt_callback]
        callbacks = [ckpt_callback]

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=params["model_dir"],
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=True,
        update_freq=params["steps_per_execution"],
        profile_batch=2 if params["profile"] else 0,
    )
    callbacks.append(tb_callback)
    # Save detections on image
    if params.get("sample_image", None):
        display_callback = DisplayCallback(
            params.get("sample_images", None), params["sample_images_freq"]
        )
        callbacks.append(display_callback)
    # AP, confusion matrix, ROC curve
    if params.get("map_freq", None) and val_dataset and params["strategy"] != "tpu":
        coco_callback = COCOCallback(val_dataset, params["map_freq"], conf_matrix=True)
        callbacks.append(coco_callback)
    # Early stopping callback
    if params["early_stopping_patience"] > 0:
        callbacks.append(
            EarlyStoppingCustomCallback(patience=params["early_stopping_patience"])
        )
    return callbacks


class AdversarialLoss(tf.keras.losses.Loss):
    """Adversarial keras loss wrapper."""

    # TODO(fsx950223): WIP
    def __init__(self, adv_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_config = adv_config
        self.model = None
        self.loss_fn = None
        self.tape = None
        self.built = False

    def build(self, model, loss_fn, tape):
        self.model = model
        self.loss_fn = loss_fn
        self.tape = tape
        self.built = True

    def call(self, features, y, y_pred, labeled_loss):
        return self.adv_config.multiplier * nsl.keras.adversarial_loss(
            features,
            y,
            self.model,
            self.loss_fn,
            predictions=y_pred,
            labeled_loss=self.labeled_loss,
            gradient_tape=self.tape,
        )


class FocalLoss(tf.keras.losses.Loss):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    """

    def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
        """Initialize focal loss.

        Args:
          alpha: A float32 scalar multiplying alpha to the loss from positive
            examples and (1-alpha) to the loss from negative examples.
          gamma: A float32 scalar modulating loss from hard and easy examples.
          label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
          **kwargs: other params.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    @tf.autograph.experimental.do_not_convert
    def call(self, y, y_pred):
        """Compute focal loss for y and y_pred.

        Args:
          y: A tuple of (normalizer, y_true), where y_true is the target class.
          y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

        Returns:
          the focal loss.
        """

        if isinstance(y_pred, list):
            pseudo_scores = y_pred[1]
            y_pred = y_pred[0]
        else:
            pseudo_scores = None
        normalizer, pseudo, y_true = y
        pred_prob = tf.sigmoid(y_pred)
        alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
        gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)
        # compute focal loss multipliers before label smoothing, such that it will
        # not blow up the loss.
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = (1.0 - p_t) ** gamma

        # apply label smoothing for cross_entropy for each entry.
        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # use pseudo_scores to weight the loss
        if pseudo_scores is not None:
            ce = tf.multiply(
                ce,
                tf.expand_dims(
                    tf.expand_dims(tf.expand_dims(pseudo_scores, axis=1), axis=1),
                    axis=1,
                ),
            )
        # compute the final loss and return
        return alpha_factor * modulating_factor * ce / normalizer


class BoxLoss(tf.keras.losses.Loss):
    """L2 box regression loss."""

    def __init__(
        self, delta=0.1, loss_att=False, loss_depth=False, loss_type="huber", **kwargs
    ):
        """Initialize box loss.

        Args:
          delta (float): Point where the huber loss function changes from a
            quadratic to linear. It is typically around the mean value of regression
            target. For instances, the regression targets of 512x512 input with 6
            anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
          loss_att (bool): Check if loss attenuation enabled
          loss_type: Defines if MSE or Huber loss
          **kwargs: other params.
        """
        super().__init__(**kwargs)
        self.huber = tf.keras.losses.Huber(
            delta, reduction=tf.keras.losses.Reduction.NONE
        )
        self.loss_att = loss_att
        self.loss_depth = loss_depth
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.ls_type = loss_type

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, box_outputs):
        if isinstance(box_outputs, list):
            if self.loss_depth:
                return_depth_loss = box_outputs[-1]
            else:
                return_depth_loss = False
            pseudo_scores = box_outputs[1]
            box_outputs = box_outputs[0]
        else:
            pseudo_scores = None
            return_depth_loss = False
        if self.loss_att:
            # Split regression output
            loss_att_outputs = box_outputs[:, :, :, int(box_outputs.shape[3] / 2) :]
            box_outputs = box_outputs[:, :, :, : int(box_outputs.shape[3] / 2)]

            # Add sigma^2/2 to the width and height for compensation of post-processing decoding
            factor = (
                tf.math.square(
                    loss_att_outputs[:, :, :, int(loss_att_outputs.shape[3] / 2) :]
                )
                / 2.0
            )  # Take only uncert of width and height
            yx = box_outputs[:, :, :, : int(loss_att_outputs.shape[3] / 2)]
            hw = box_outputs[:, :, :, int(loss_att_outputs.shape[3] / 2) :]
            box_outputs = tf.concat([yx, tf.add(hw, factor)], axis=-1)

        if self.loss_depth:
            depth_output = box_outputs[:, :, :, -1:]
            box_outputs = box_outputs[:, :, :, :-1]
        num_positives, box_targets = y_true
        normalizer = num_positives * 4.0
        mask = tf.cast(
            box_targets != 0.0, box_outputs.dtype
        )  # number of acutal bbs (1,0,1,1,1,0)
        box_targets = tf.expand_dims(box_targets, axis=-1)
        box_outputs = tf.expand_dims(box_outputs, axis=-1)

        if self.ls_type == "huber":
            box_loss = tf.cast(self.huber(box_targets, box_outputs), box_outputs.dtype)
        else:
            box_loss = tf.cast(self.mse(box_targets, box_outputs), box_outputs.dtype)

        if self.loss_depth and return_depth_loss and pseudo_scores is not None:
            depth_map = tf.tile(
                pseudo_scores, multiples=[1, 1, 1, depth_output.shape[3]]
            )
            depth_loss = tf.cast(self.mse(depth_output, depth_map), box_outputs.dtype)
            depth_loss = tf.tile(
                tf.expand_dims(depth_loss, axis=-1), multiples=[1, 1, 1, mask.shape[3]]
            )
            return tf.reduce_sum(depth_loss * mask) / normalizer *100
        # use pseudo_scores to weight the loss
        if pseudo_scores is not None and not self.loss_depth:
            if isinstance(pseudo_scores, (list, tuple)) and len(pseudo_scores) == 2:
                twolosses_weight=pseudo_scores[0][0]
                pseudo_scores[0]=pseudo_scores[0][1]
                weight_map_box_close = tf.tile(
                    pseudo_scores[0], multiples=[1, 1, 1, box_loss.shape[3]]
                )
                weight_map_box_far = tf.tile(
                    pseudo_scores[1], multiples=[1, 1, 1, box_loss.shape[3]]
                )
                box_loss = twolosses_weight[0] * box_loss * weight_map_box_close  + twolosses_weight[1] * box_loss * weight_map_box_far 
            elif len(pseudo_scores.shape) < 4:
                box_loss = tf.multiply(
                    box_loss,
                    tf.expand_dims(
                        tf.expand_dims(tf.expand_dims(pseudo_scores, axis=1), axis=1),
                        axis=1,
                    ),
                )
            else:
                weight_map_box = tf.tile(
                    pseudo_scores, multiples=[1, 1, 1, box_loss.shape[3]]
                )
                box_loss = tf.multiply(box_loss, weight_map_box)
        if self.loss_att:
            box_loss = tf.reshape(box_loss, [-1, 4])
            loss_att_outputs = tf.math.square(tf.reshape(loss_att_outputs, [-1, 4]))
            mask = tf.reshape(mask, [-1, 4])
            return (
                0.25
                * tf.reduce_sum(
                    (box_loss / loss_att_outputs + tf.math.log(1 + loss_att_outputs))
                    * mask
                )
                / normalizer
            )
        else:
            return tf.reduce_sum(box_loss * mask) / normalizer


class BoxIouLoss(tf.keras.losses.Loss):
    """Box iou loss."""

    def __init__(
        self,
        iou_loss_type,
        min_level,
        max_level,
        num_scales,
        aspect_ratios,
        anchor_scale,
        image_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.iou_loss_type = iou_loss_type
        self.input_anchors = anchors.Anchors(
            min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size
        )

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, box_outputs):
        anchor_boxes = tf.tile(
            self.input_anchors.boxes,
            [box_outputs.shape[0] // self.input_anchors.boxes.shape[0], 1],
        )
        num_positives, box_targets = y_true
        normalizer = num_positives * 4.0
        mask = tf.cast(box_targets != 0.0, box_outputs.dtype)
        box_outputs = anchors.decode_box_outputs(box_outputs, anchor_boxes) * mask
        box_targets = anchors.decode_box_outputs(box_targets, anchor_boxes) * mask
        box_iou_loss = iou_utils.iou_loss(box_outputs, box_targets, self.iou_loss_type)
        box_iou_loss = tf.reduce_sum(box_iou_loss) / normalizer
        return box_iou_loss


class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """Compute the precision, recall and f1 scores for each class  = Confusion matrix"""

    def __init__(self, num_classes, label_map, **kwargs):
        """Constructs the necessary attributes for calculating the classification metrics

        Args:
          num_classes (int): Number of classes
          label_map (dict): Class labels
        """
        super(ConfusionMatrixMetric, self).__init__(
            name="confusion_matrix_metric", **kwargs
        )
        self.num_classes = num_classes  # Necessary for when label_map is None
        label_map = label_util.get_label_map(label_map)
        if label_map is not None:
            self.class_names = np.asarray([label_map[key] for key in label_map])
        else:
            self.class_names = np.arange(1, num_classes + 1)
        self.total_conf_matrix = self.add_weight(
            "total", shape=(self.num_classes, self.num_classes), initializer="zeros"
        )

    def reset_state(self):
        for vr in self.variables:
            vr.assign(tf.zeros(shape=vr.shape))

    def update_state(self, y_true, y_pred):
        conf_matrix = self.confusion_matrix(y_true, y_pred)
        if not tf.reduce_all(tf.math.is_nan(conf_matrix)):
            self.total_conf_matrix.assign_add(conf_matrix)

    def result(self):
        return self._calc_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=1)
        conf_matrix = tf.math.confusion_matrix(
            y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes
        )
        return conf_matrix

    def _calc_confusion_matrix(self):
        conf_matrix = self.total_conf_matrix
        diags = tf.linalg.diag_part(conf_matrix)
        precision = tf.math.divide_no_nan(diags, tf.reduce_sum(conf_matrix, 0))
        recall = tf.math.divide_no_nan(diags, tf.reduce_sum(conf_matrix, 1))
        f1 = 2 * tf.math.divide_no_nan(
            tf.multiply(precision, recall), tf.add(precision, recall)
        )
        return precision, recall, f1

    def extract_cm_metrics(self, output):
        results = self.result()
        for i in range(self.num_classes):
            output["precision_{}".format(self.class_names[i])] = results[0][i]
            output["recall_{}".format(self.class_names[i])] = results[1][i]
            output["F1_{}".format(self.class_names[i])] = results[2][i]


class RmseMetric(tf.keras.metrics.Metric):
    """Computes mean RMSE between the offsets"""

    def __init__(self, **kwargs):
        super(RmseMetric, self).__init__(name="rmse", **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def reset_state(self):
        self.mse_sum.assign(tf.zeros(shape=self.mse_sum.shape))
        self.total_samples.assign(tf.zeros(shape=self.total_samples.shape))

    def update_state(self, y_true, y_pred):
        y_pred = y_pred[:, :4]
        squared_errors = tf.square(y_true - y_pred)
        mse = tf.reduce_sum(squared_errors)
        num_samples = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.mse_sum.assign_add(mse)
        self.total_samples.assign_add(num_samples)

    def result(self):
        mean_mse = tf.math.divide_no_nan(self.mse_sum, self.total_samples)
        rmse = tf.sqrt(mean_mse)
        return rmse


class MeanUncertMetric(tf.keras.metrics.Metric):
    """Compute mean uncertainty"""

    def __init__(self, **kwargs):
        super(MeanUncertMetric, self).__init__(name="mUncert", **kwargs)
        self.uncert = self.add_weight("uncert", initializer="zeros")
        self.n = self.add_weight("n", initializer="zeros")

    def reset_state(self):
        self.uncert.assign(tf.zeros(shape=self.uncert.shape))
        self.n.assign(tf.zeros(shape=self.n.shape))

    def update_state(self, y_true, y_pred):
        current_uncert = tf.reduce_mean(y_pred[:, 4:])
        if not tf.math.is_nan(current_uncert):
            self.uncert.assign_add(current_uncert)
            self.n.assign_add(tf.constant(1.0))

    def result(self):
        return tf.math.divide_no_nan(self.uncert, self.n)


class StdUncertMetric(tf.keras.metrics.Metric):
    """Compute the standard deviation of the uncertainty"""

    def __init__(self, **kwargs):
        super(StdUncertMetric, self).__init__(name="stdUncert", **kwargs)
        self.uncert = self.add_weight("uncert", initializer="zeros")
        self.n = self.add_weight("n", initializer="zeros")

    def reset_state(self):
        self.uncert.assign(tf.zeros(shape=self.uncert.shape))
        self.n.assign(tf.zeros(shape=self.n.shape))

    def update_state(self, y_true, y_pred):
        current_uncert = tf.math.reduce_std(y_pred[:, 4:])
        if not tf.math.is_nan(current_uncert):
            self.uncert.assign_add(current_uncert)
            self.n.assign_add(tf.constant(1.0))

    def result(self):
        return tf.math.divide_no_nan(self.uncert, self.n)


def apply_boolean_mask(batch):
    """Apply boolean mask to the scores and return the mean value per image."""
    scores, mask = batch
    filtered = tf.boolean_mask(scores, mask)
    
    # Handle empty mask case - return 0 instead of NaN
    return tf.cond(
        tf.size(filtered) > 0,
        lambda: tf.reduce_mean(filtered),
        lambda: tf.constant(0.0, dtype=scores.dtype)
    )


def plot_mask(mask, title):
    """Creates a matplotlib figure with a 2D 'jet' colormap."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if mask.shape[-1] == 3:
        cax = ax.imshow(tf.image.rgb_to_grayscale(mask[0]))
    else:
        cax = ax.imshow(mask[0], cmap="jet")
    ax.set_title(title)
    fig.colorbar(cax)
    plt.tight_layout()

    return fig


import io


def figure_to_image(figure):
    """Converts the matplotlib figure to a PNG image and returns a TensorFlow image tensor."""
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


@tf.function
def log_images(tf_writer, object_mask, class_mask, depth_maps, images, step):
    """Logs object depth mask, class mask scores, and imscores to TensorBoard every 5 epochs."""
    def np_plot_and_convert(obj_mask, cls_mask, depth_map, image):
        """Helper function to handle NumPy conversion and plotting."""

        fig_depth = plot_mask(obj_mask, "Object Depth Mask")
        fig_class = plot_mask(cls_mask, "Class Mask Scores")
        fig_dmap = plot_mask(depth_map, "Depth Map")
        fig_image = plot_mask(image, "Image")

        img_depth = figure_to_image(fig_depth)
        img_class = figure_to_image(fig_class)
        img_dmap = figure_to_image(fig_dmap)
        img_image = figure_to_image(fig_image)

        return img_depth, img_class, img_dmap, img_image

    # Run the plotting and conversion outside the computational graph
    img_depth, img_class, img_dmap, img_image = tf.numpy_function(
        np_plot_and_convert,
        [object_mask, class_mask, depth_maps, images],
        [tf.uint8, tf.uint8, tf.uint8, tf.uint8],
    )

    with tf_writer.as_default():
        tf.summary.image("Object Depth Mask", img_depth, step=step)
        tf.summary.image("Class Mask Scores", img_class, step=step)
        tf.summary.image("Depth Map", img_dmap, step=step)
        tf.summary.image("Image", img_image, step=step)


class EfficientDetNetTrain(efficientdet_keras.EfficientDetNet):
    """A customized trainer for EfficientDet.
    see https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log_dir = os.path.join(self.config.model_dir, "train_images")
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        # Define mtrics
        if self.config.loss_attenuation:
            self.metrics_collect = {
                "class": [
                    ConfusionMatrixMetric(
                        self.config.num_classes, self.config.label_map
                    )
                ],
                "box": [RmseMetric(), MeanUncertMetric(), StdUncertMetric()],
            }
        else:
            self.metrics_collect = {
                "class": [
                    ConfusionMatrixMetric(
                        self.config.num_classes, self.config.label_map
                    )
                ],
                "box": [RmseMetric()],
            }

    @staticmethod
    def _split_output_labels(cls_outputs, box_outputs, labels, start_val, end_val):
        """Split the output and labels based on the indices and return subset"""
        sup_indices = tf.range(start=start_val, limit=end_val)
        split_cls = [tf.gather(co, sup_indices, axis=0) for co in cls_outputs]
        split_box = [tf.gather(bo, sup_indices, axis=0) for bo in box_outputs]
        split_labels = {
            dict_key: tf.gather(labels[dict_key], sup_indices, axis=0)
            for dict_key in labels.keys()
        }
        return split_cls, split_box, split_labels

    def _freeze_vars(self):
        if self.config.var_freeze_expr:
            return [
                v
                for v in self.trainable_variables
                if not re.match(self.config.var_freeze_expr, v.name)
            ]
        return self.trainable_variables

    def _reg_l2_loss(self, weight_decay, regex=r".*(kernel|weight):0$"):
        """Return regularization l2 loss loss."""
        var_match = re.compile(regex)
        return weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in self._freeze_vars() if var_match.match(v.name)]
        )

    def _get_class_data(self, labels, cls_outputs):
        """Extracts [N, num_classes] for predictions and [N,] for ground truth of the classes
            Removes all value "-1" belonging to background

        Args:
          labels (tensor): Ground truth classes
          cls_outputs (list): Predicted logits for each feature map
        Returns:
          y_true (tensor): Aggregated ground truth classes
          y_pred (tensor): Aggregated predicted logits
          mask (tensor): Foreground ground truth objects only
        """
        levels = range(len(cls_outputs))
        y_true = []
        y_pred = []
        # Join all feature maps
        for level in levels:
            y_pred_level = tf.reshape(cls_outputs[level], [-1, self.config.num_classes])
            y_true_level = tf.cast(
                tf.reshape(
                    labels["cls_targets_%d" % (level + self.config.min_level)], [-1]
                ),
                dtype=y_pred_level.dtype,
            )
            y_true.append(y_true_level)
            y_pred.append(y_pred_level)
        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)

        mask = tf.where(y_true >= 0)  # Remove background
        y_true = tf.reshape(tf.gather(y_true, mask, axis=0), [-1])
        y_pred = tf.reshape(
            tf.gather(y_pred, mask, axis=0), [-1, self.config.num_classes]
        )
        return y_true, y_pred, mask

    def _get_box_data(self, labels, box_outputs, mask):
        """Extracts [N, 4] bounding boxes for predictions and ground truth
            Removes all value "-1" belonging to background

        Args:
          labels (tensor): Ground truth bounding boxes
          box_outputs (list): Predicted bounding boxes for each feature map
          mask (tensor): Foreground ground truth objects only
        Returns:
          y_true (tensor): Aggregated ground truth bounding boxes
          y_pred (tensor): Aggregated predicted bounding boxes
        """
        levels = range(len(box_outputs))
        y_true = []
        y_pred = []
        y_uncert = []
        for level in levels:
            if self.config.loss_attenuation:
                box_pred = box_outputs[level][
                    :, :, :, : int(box_outputs[0].shape[-1] / 2)
                ]
                uncert_pred = box_outputs[level][
                    :, :, :, int(box_outputs[0].shape[-1] / 2) :
                ]
                y_uncert_level = tf.reshape(uncert_pred, [-1, 4])
                y_uncert.append(y_uncert_level)
            else:
                box_pred = box_outputs[level]
            y_pred_level = tf.reshape(box_pred, [-1, 4])
            y_true_level = tf.cast(
                tf.reshape(
                    labels["box_targets_%d" % (level + self.config.min_level)], [-1, 4]
                ),
                dtype=y_pred_level.dtype,
            )
            y_pred.append(y_pred_level)
            y_true.append(y_true_level)

        y_true = tf.reshape(tf.gather(tf.concat(y_true, axis=0), mask, axis=0), [-1, 4])
        y_pred = tf.reshape(tf.gather(tf.concat(y_pred, axis=0), mask, axis=0), [-1, 4])

        if self.config.loss_attenuation:
            y_uncert = tf.reshape(
                tf.gather(tf.concat(y_uncert, axis=0), mask, axis=0), [-1, 4]
            )
            y_pred = tf.concat([y_pred, y_uncert], axis=1)
        return y_true, y_pred

    def _get_metrics(self, labels, cls_outputs, box_outputs):
        """Collect metrics per training iteration

        Args:
            labels (dict): Ground truth labels
            cls_outputs (list): Predicted logits for each feature map
            box_outputs (list): Predicted bounding boxes for each feature map

        Returns:
            metrics_output (dict): Collected results
        """
        y_true_cls, y_pred_cls, mask_pos = self._get_class_data(labels, cls_outputs)
        [
            class_metric.update_state(y_true_cls, y_pred_cls)
            for class_metric in self.metrics_collect["class"]
        ]
        y_true_box, y_pred_box = self._get_box_data(labels, box_outputs, mask_pos)
        [
            box_metric.update_state(y_true_box, y_pred_box)
            for box_metric in self.metrics_collect["box"]
        ]
        metrics_output = {}
        for i in range(len(self.metrics_collect["class"])):
            m = self.metrics_collect["class"][i]
            if m.name == "confusion_matrix_metric":
                m.extract_cm_metrics(metrics_output)
            else:
                metrics_output[m.name] = m.result()

        for i in range(len(self.metrics_collect["box"])):
            m = self.metrics_collect["box"][i]
            metrics_output[m.name] = m.result()
        return metrics_output

    def _clip_uncert(self, box_outputs):
        """Clips the uncertainty

        Args:
          box_outputs (list): Predicted bounding boxes for each feature map
        Returns:
          box_outputs (list): Predicted bounding boxes for each feature map with clipped uncertainty
        """

        for level in range(len(box_outputs)):
            loss_att_values = tf.clip_by_value(
                t=box_outputs[level][:, :, :, int(box_outputs[0].shape[-1] / 2) :],
                clip_value_min=self.config.clip_min_uncert,
                clip_value_max=self.config.clip_max_uncert,
            )
            box_outputs[level] = tf.concat(
                [
                    box_outputs[level][:, :, :, : int(box_outputs[0].shape[-1] / 2)],
                    loss_att_values,
                ],
                axis=-1,
            )
        return box_outputs

    def _detection_loss(
        self,
        cls_outputs,
        box_outputs,
        labels,
        loss_vals,
        pseudo_scores=None,
        pseudo=False,
        box_only=False,
    ):
        """Computes total detection loss.
        Computes total detection loss including box and class loss from all levels.
        Args:
          cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].
          box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width,
            num_anchors * 4].
          labels: the dictionary that returned from dataloader that includes
            groundtruth targets.
          loss_vals: A dict of loss values.
          pseudo_score: Weights each image with a confidence/uncertainty score. Defaults to None.
          pseudo: Changes loss name to pseudo. Defaults to False.
        Returns:
          total_loss: an integer tensor representing total loss reducing from
            class and box losses from all levels.
          cls_loss: an integer tensor representing total class loss.
          box_loss: an integer tensor representing total box regression loss.
          box_iou_loss: an integer tensor representing total box iou loss.
        """
        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        dtype = cls_outputs[0].dtype
        num_positives_sum = tf.reduce_sum(labels["mean_num_positives"]) + 1.0
        positives_momentum = self.config.positives_momentum or 0
        if positives_momentum > 0:
            # normalize the num_positive_examples for training stability.
            moving_normalizer_var = tf.Variable(
                0.0,
                name="moving_normalizer",
                dtype=dtype,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
            )
            num_positives_sum = tf.keras.backend.moving_average_update(
                moving_normalizer_var,
                num_positives_sum,
                momentum=self.config.positives_momentum,
            )
        elif positives_momentum < 0:
            num_positives_sum = utils.cross_replica_mean(num_positives_sum)
        num_positives_sum = tf.cast(num_positives_sum, dtype)
        levels = range(len(cls_outputs))
        cls_losses = []
        box_losses = []
        depth_losses = []
        for level in levels:
            if self.config.get("depth_detection"):
                object_mask = None
                class_mask = None
                # weight_map: Tensor of shape [batch, h, w, 1] with per-cell weights.
                # Each cell's weight is computed by first resizing the depth map
                # using bilinear interpolation and then converting the depth to a weight.

                # Resize depth_map from (256,512) to the feature map spatial dimensions (h, w)
                weight_map = tf.image.resize(
                    labels["depth_array"],
                    size=labels[
                        "cls_targets_%d" % (level + self.config.min_level)
                    ].shape[1:-1],
                    method="bilinear",
                )
                
                if self.depth_detection_distance_mask or self.config.predict_distance:
                    # normalize as it is intra image not inter image
                    x_min = tf.reduce_min(weight_map, axis=[1, 2, 3], keepdims=True)
                    x_max = tf.reduce_max(weight_map, axis=[1, 2, 3], keepdims=True)
                    # Avoid division by zero
                    weight_map = (weight_map - x_min) / (x_max - x_min + 1e-8)
                    weight_map = tf.clip_by_value(weight_map, 0.0, 1.0)
                else:
                    if "minmax" in self.config.model_dir:
                        # Normalize depth values to [0, 1] range based on known min/max depth
                        min_depth = tf.reduce_min(weight_map)
                        max_depth = tf.reduce_max(weight_map)
                        weight_map = (weight_map - min_depth) / (max_depth - min_depth)
                        weight_map = tf.clip_by_value(weight_map, 0.0, 1.0)
                    # reverse: close =0, far=1
                    if "inverse" in self.config.model_dir:
                        weight_map = 1.0 - weight_map
                    if "expo" in self.config.model_dir:
                        weight_map = 1 + self.depth_alpha * tf.math.exp(weight_map)
                    if "quadratic" in self.config.model_dir:
                        weight_map = self.depth_alpha * tf.math.square(weight_map)
                    if "linear" in self.config.model_dir:
                        weight_map = self.depth_alpha * weight_map

                # weight_map /= 255.0  # for brightness values
                loss_vals["depth_weights_norm_mean"] = tf.reduce_mean(weight_map)
                # exp_weight = 1 + self.depth_alpha * tf.math.exp(weight_map)

                # Optionally, incorporate a ramp-up factor over training steps:
                weight_map = (
                    1.0 - self.gamma_at_step
                ) * 1.0 + self.gamma_at_step * weight_map
                height = tf.shape(labels["depth_array"])[1]
                width = tf.shape(labels["depth_array"])[2]
                if self.depth_detection_distance_mask:
                    # Define depth thresholds for close, medium, far
                    split_factor = 0.5 if "factor" not in self.config.model_dir else float(self.config.model_dir.split("factor")[-1].split("_")[0])
                    logging.info(f"Using split factor: {split_factor}")
                    far_mask = tf.cast(
                        weight_map < (split_factor*tf.reduce_max(weight_map)), tf.float32
                    )  # far objects
                    close_mask = tf.cast(weight_map >= (split_factor*tf.reduce_max(weight_map)), tf.float32)  # close objects
                    weight_map = [close_mask, far_mask]
                    

                    loss_vals["close_mask"] = tf.reduce_sum(close_mask)
                    loss_vals["far_mask"] =  tf.reduce_sum(far_mask)
                    
                    loss_vals["close_mask_max"] = tf.reduce_max(close_mask)
                    loss_vals["far_mask_max"] =  tf.reduce_max(far_mask)
                    
                    loss_vals["close_mask_min"] = tf.reduce_min(close_mask)
                    loss_vals["far_mask_min"] =  tf.reduce_min(far_mask)
                    
                    # if self.config.rampup_det_losses:
                    #     # Training progress: goes from 0 to 1 over time
                    #     progress = tf.cast(self.optimizer.iterations, tf.float32) / (self.config.steps_per_epoch * self.config.num_epochs)
                    #     progress = tf.clip_by_value(progress, 0.0, 1.0)

                    #     # # Scale progress so it reaches 1.0 at 70% of training
                    #     scaled_progress = tf.clip_by_value(progress / 0.6, 0.0, 1.0)

                    #     # # Update the weight with linear ramp-up
                    #     initial_weight = self.two_losses_weight[0]  # Assuming this is a float like 0.1 or 0.2
                    #     # current_weight = initial_weight + (1.0 - initial_weight) * scaled_progress

                    #     # Reverse the ramp-up: start with 1.0 and decrease to initial_weight over time
                    #     current_weight = 1.0 - (1.0 - initial_weight) * scaled_progress
                    #     # Use current_weight wherever needed
                    #     self.two_losses_weight[0] = current_weight

                    loss_vals["close_loss_weight"] = self.two_losses_weight[0]
                    loss_vals["far_loss_weight"] =  self.two_losses_weight[1]

                    # if True or (
                    #     self.optimizer.iterations % (1 * self.config.steps_per_epoch) == 0
                    # ):
                    #     log_images(
                    #         self.summary_writer,
                    #         object_mask=close_mask,
                    #         class_mask=medium_mask,
                    #         depth_maps=far_mask,
                    #         images=self.images_save,
                    #         step=self.optimizer.iterations,
                    #     )

                if (
                    self.depth_detection_object_mask
                ):  # Ensure current_step and total_steps are tensors, not Python scalars
                    current_step = tf.cast(self.optimizer.iterations, tf.float32)
                    total_steps = tf.cast(
                        self.config.steps_per_epoch * self.config.num_epochs, tf.float32
                    )

                    # 1) Compute training progress fraction [1 → 0] (decreasing over time)
                    fraction = 1.0 - (current_step / total_steps)
                    fraction = tf.clip_by_value(
                        fraction, 0.0, 1.0
                    )  # Ensure within valid range

                    # 2) Define dynamic threshold (starts at 1.0, decreases to 0.0)
                    max_brightness = (
                        0.5  # Depth map uses brightness: 1 = close, 0 = far
                    )
                    min_brightness = 0.0  # Final state: allow all objects

                    current_threshold = min_brightness + fraction * (
                        max_brightness - min_brightness
                    )

                    # 3) Create a binary mask:
                    #    - Pixels with brightness ABOVE the threshold → 1 (included in loss)
                    #    - Pixels with brightness BELOW the threshold → 0 (ignored)
                    binary_depth_mask = tf.cast(
                        weight_map >= (current_threshold * tf.reduce_max(weight_map)),
                        tf.float32,
                    )
                    weight_map *= binary_depth_mask

                    loss_vals["depth_weights_norm_min"] = tf.reduce_min(weight_map)
                    loss_vals["depth_weights_norm_max"] = tf.reduce_max(weight_map)
                    loss_vals["current_threshold"] = current_threshold
                    loss_vals["binary_depth_mask_mean"] = tf.reduce_mean(
                        binary_depth_mask
                    )
                    loss_vals["binary_depth_mask_min"] = tf.reduce_min(
                        binary_depth_mask
                    )
                    loss_vals["binary_depth_mask_max"] = tf.reduce_max(
                        binary_depth_mask
                    )

                if self.depth_detection_class_mask:

                    def build_pixel_weights(cls_targets, class_weight_dict):
                        """
                        Produces a pixel-wise weight map from EfficientDet anchor labels.

                        Args:
                        cls_targets: A tensor of shape [batch, height, width, num_anchors].
                            Each anchor’s class is in:
                            -2 => ignore anchor
                            -1 => background anchor
                            0..(K-1) => (real_class - 1)
                        class_weight_dict: A dict that maps unshifted class labels to a float weight.
                            * After unshifting, background=0, real class=1..K, etc.
                            * Example:
                                {
                                0: 1.0,   # background
                                1: 1.0,   # real class=1
                                2: 2.0,   # real class=2
                                3: 3.0,   # real class=3
                                }

                        Returns:
                        A float32 tensor of shape [batch, height, width, 1],
                        where each pixel’s value is chosen from class_weight_dict
                        based on the highest assigned anchor’s real class.
                        """
                        # 1) Treat ignore (-2) anchors the same as background (-1).
                        merged = tf.where(cls_targets == -2, -1, cls_targets)

                        # 2) Take the max label across the anchor dim => shape [b, h, w].
                        max_label = tf.reduce_max(merged, axis=-1)

                        # 3) If max_label < 0, that pixel is effectively background => keep it at -1.
                        max_label = tf.where(
                            max_label < 0, tf.fill(tf.shape(max_label), -1), max_label
                        )

                        # 4) Unshift by +1 so stored=0 => real class=1, stored=1 => real class=2, etc.
                        unshifted = max_label + 1
                        #   -1 => +1 => 0 => background
                        #    0 => +1 => 1 => class=1
                        #    1 => +1 => 2 => class=2, etc.

                        # 5) Build a lookup table over [min_key..max_key].
                        min_key = min(class_weight_dict.keys())
                        max_key = max(class_weight_dict.keys())
                        lookup_array = [
                            class_weight_dict[i] for i in range(min_key, max_key + 1)
                        ]
                        lookup_tensor = tf.constant(lookup_array, dtype=tf.float32)

                        # 6) Shift the unshifted indices into [0..(len(lookup_array)-1)] range.
                        offset = -min_key if min_key < 0 else 0
                        shifted_idx = tf.cast(unshifted + offset, tf.int32)

                        # 7) Gather the corresponding weight => shape [b, h, w].
                        weight_map = tf.gather(lookup_tensor, shifted_idx)

                        # 8) Expand dims => shape [b, h, w, 1].
                        weight_map = weight_map[..., tf.newaxis]
                        return weight_map

                    class_weight_dict = {
                        -1: 1.0,
                        0: 1.0,
                        1: 1.0,
                        2: 200.0,
                        3: 200.0,
                        4: 200.0,
                        5: 300.0,
                        6: 200.0,
                        7: 300.0,
                    }

                    cls_map = labels["cls_targets_%d" % (level + self.config.min_level)]
                    class_mask = build_pixel_weights(cls_map, class_weight_dict)
                    # Then update your weight_map accordingly.
                    weight_map = class_mask * weight_map + (1.0 - class_mask)
                # loss_vals["class_targets"] = tf.math.reduce_std(
                #     tf.cast(
                #         labels[("cls_targets_%d" % (level + self.config.min_level))],
                #         tf.float32,
                #     )
                # )

                # loss_vals["class_mask"] = tf.reduce_mean(class_mask)
                # loss_vals["min_class_mask"] = tf.reduce_min(class_mask)
                # loss_vals["max_class_mask"] = tf.reduce_max(class_mask)
                loss_vals["depth_weights_ramped"] = tf.reduce_mean(weight_map)
                loss_vals["depth_array"] = tf.reduce_mean(labels["depth_array"])
                loss_vals["min_depth_array"] = tf.reduce_min(labels["depth_array"])
                loss_vals["max_depth_array"] = tf.reduce_max(labels["depth_array"])

                # if True or (
                #     self.optimizer.iterations % (1 * self.config.steps_per_epoch) == 0
                # ):
                #     log_images(
                #         self.summary_writer,
                #         object_mask=tf.reduce_mean(box_outputs[level][:, :, :, int(box_outputs[level].shape[-1] -1) :], axis=-1),
                #         class_mask=weight_map,
                #         depth_maps=labels["depth_array"],
                #         images=self.images_save,
                #         step=self.optimizer.iterations,
                #     )

            # Onehot encoding for classification labels.
            cls_targets_at_level = tf.one_hot(
                labels["cls_targets_%d" % (level + self.config.min_level)],
                self.config.num_classes,
                dtype=dtype,
            )

            if self.config.data_format == "channels_first":
                bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
                cls_targets_at_level = tf.reshape(
                    cls_targets_at_level, [bs, -1, width, height]
                )
            else:
                bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
                cls_targets_at_level = tf.reshape(
                    cls_targets_at_level, [bs, width, height, -1]
                )

            class_loss_layer = self.loss.get(FocalLoss.__name__, None)
            if class_loss_layer:

                if (
                    pseudo_scores is not None
                    and self.config.activate_im_score
                    or self.config.get("depth_image")
                ):
                    cls_loss = class_loss_layer(
                        [num_positives_sum, pseudo, cls_targets_at_level],
                        [cls_outputs[level], pseudo_scores],
                    )
                else:
                    cls_loss = class_loss_layer(
                        [num_positives_sum, pseudo, cls_targets_at_level],
                        cls_outputs[level],
                    )
                if self.config.data_format == "channels_first":
                    cls_loss = tf.reshape(
                        cls_loss, [bs, -1, width, height, self.config.num_classes]
                    )
                else:
                    cls_loss = tf.reshape(
                        cls_loss, [bs, width, height, -1, self.config.num_classes]
                    )

                cls_loss *= tf.cast(
                    tf.expand_dims(
                        tf.not_equal(
                            labels["cls_targets_%d" % (level + self.config.min_level)],
                            -2,
                        ),
                        -1,
                    ),
                    dtype,
                )
                if self.config.get("depth_detection") and not self.config.predict_distance:
                    if self.depth_detection_distance_mask:
                        weight_map_close = tf.tile(
                            tf.expand_dims(weight_map[0], axis=-1),
                            multiples=[1, 1, 1, cls_loss.shape[3], cls_loss.shape[4]],
                        )
                        weight_map_far = tf.tile(
                            tf.expand_dims(weight_map[1], axis=-1),
                            multiples=[1, 1, 1, cls_loss.shape[3], cls_loss.shape[4]],
                        )
                        cls_loss = self.two_losses_weight[0] * cls_loss * weight_map_close + self.two_losses_weight[1] *cls_loss * weight_map_far
                    else:
                        weight_map_cls = tf.expand_dims(
                            weight_map, axis=-1
                        )  # Now shape: [batch, h, w, 1, 1]
                        weight_map_cls = tf.tile(
                            weight_map_cls,
                            multiples=[1, 1, 1, cls_loss.shape[3], cls_loss.shape[4]],
                        )
                        cls_loss = cls_loss * weight_map_cls
                cls_loss_sum = tf.reduce_sum(cls_loss)
                cls_losses.append(tf.cast(cls_loss_sum, dtype))

            if self.config.box_loss_weight and self.loss.get(BoxLoss.__name__, None):
                box_targets_at_level = labels[
                    "box_targets_%d" % (level + self.config.min_level)
                ]
                box_loss_layer = self.loss[BoxLoss.__name__]

                if (
                    pseudo_scores is not None
                    and self.config.activate_im_score
                    or self.config.get("depth_image")
                ):
                    box_loss = box_loss_layer(
                        [num_positives_sum, box_targets_at_level],
                        [box_outputs[level], pseudo_scores],
                    )
                elif self.config.get("depth_detection"):
                    if self.config.predict_distance:
                        depth_box_loss = box_loss_layer(
                            [num_positives_sum, box_targets_at_level],
                            [box_outputs[level], weight_map, True],
                        )
                        depth_losses.append(depth_box_loss)

                        box_loss = box_loss_layer(
                            [num_positives_sum, box_targets_at_level],
                            [box_outputs[level], weight_map, False],
                        )
                    else:
                        if self.depth_detection_distance_mask: weight_map[0] = [self.two_losses_weight,weight_map[0]]
                        box_loss = box_loss_layer(
                            [num_positives_sum, box_targets_at_level],
                            [box_outputs[level], weight_map],
                        )
                else:
                    box_loss = box_loss_layer(
                        [num_positives_sum, box_targets_at_level], box_outputs[level]
                    )

                box_losses.append(box_loss)

        if self.config.iou_loss_type:
            box_outputs = tf.concat(
                [tf.reshape(v, [-1, 4]) for v in box_outputs], axis=0
            )
            box_targets = tf.concat(
                [
                    tf.reshape(
                        labels["box_targets_%d" % (level + self.config.min_level)],
                        [-1, 4],
                    )
                    for level in levels
                ],
                axis=0,
            )
            box_iou_loss_layer = self.loss[BoxIouLoss.__name__]
            box_iou_loss = box_iou_loss_layer(
                [num_positives_sum, box_targets], box_outputs
            )

            loss_vals["box_iou_loss"] = box_iou_loss
        else:
            box_iou_loss = 0

        cls_loss = tf.add_n(cls_losses) if cls_losses else 0
        if self.config.loss_attenuation:
            box_loss = tf.reduce_mean(box_losses) if box_losses else 0
        else:
            box_loss = tf.add_n(box_losses) if box_losses else 0
        if self.config.predict_distance:
            depth_loss = tf.add_n(depth_losses) if depth_losses else 0
            # current_step = self.optimizer.iterations	
            # current_epoch = current_step // self.config.steps_per_epoch
            # # Set gamma: 0 for epochs < 60, 1 for epochs >= 60.
            # depth_loss *= tf.cast(current_epoch >= 60, tf.float32)
        else:
            depth_loss = 0
        if not self.config.get("depth_additional_parameter"):
            self.config.depth_additional_parameter = 0.0
        if box_only:
            total_loss = self.config.box_loss_weight * box_loss

        else:
            total_loss = (
                cls_loss
                + self.config.box_loss_weight * box_loss
                + self.config.iou_loss_weight * box_iou_loss
                + self.config.depth_additional_parameter * depth_loss
            )

        if pseudo:
            add_name = "pseudo_"
        else:
            add_name = ""
        loss_vals[add_name + "mdepth_loss"] = depth_loss
        loss_vals[add_name + "det_loss"] = total_loss
        loss_vals[add_name + "cls_loss"] = cls_loss
        loss_vals[add_name + "box_loss"] = box_loss
        return total_loss

    def _weight_scheduling(self):
        # Calculate weight scheduling for CSD
        num_iter = self.config.steps_per_epoch * self.config.num_epochs
        current_iteration = self.optimizer.iterations
        ramp_up_end = int(32000 * num_iter / 120000)
        ramp_down_start = int(100000 * num_iter / 120000)
        ramp_up = tf.cast(current_iteration < ramp_up_end, tf.bool)
        ramp_down = tf.cast(current_iteration > ramp_down_start, tf.bool)
        second_iteration = tf.cast(current_iteration > 0, tf.bool)
        ramp_up_factor = tf.math.exp(
            -5 * tf.math.pow((1 - current_iteration / ramp_up_end), 2)
        )
        ramp_down_factor = tf.math.exp(
            -12.5
            * tf.math.pow(
                (1 - (num_iter - current_iteration) / int(20000 * num_iter / 120000)), 2
            )
        )
        ramp_weight = tf.cond(
            ramp_up,
            lambda: ramp_up_factor,
            lambda: tf.cond(
                ramp_down,
                lambda: ramp_down_factor,
                lambda: tf.constant(1, dtype=ramp_down_factor.dtype),
            ),
        )

        ramp_weight = tf.cond(
            second_iteration,
            lambda: ramp_weight,
            lambda: tf.constant(0, dtype=ramp_weight.dtype),
        )
        return ramp_weight

    def consistency_loss_tf(
        self,
        labels,
        cls_outputs,
        box_outputs,  # List or dict of [N,4] float tensors for predicted RGB boxes per level.
        boxes_depth_levels,  # List or dict of [M,4] float tensors for predicted Depth boxes per level.
        iou_threshold=0.5,
        score_thr=0.4,
        reduction="mean",
    ):
        """
        Computes consistency loss per feature level.
        Simple MSE between RGB and depth box predictions.
        """
        dtype = cls_outputs[0].dtype
        levels = range(len(cls_outputs))
        box_losses = []

        for level in levels:            
            # Class loss
            max_score_anchors = tf.math.reduce_max(cls_outputs[level], axis=-1)
            score_mask = tf.math.sigmoid(max_score_anchors)
            positive_mask = tf.expand_dims(
                tf.cast(
                    tf.math.greater(
                        score_mask,
                        score_thr * tf.math.reduce_max(score_mask),
                    ),
                    dtype,
                ),
                axis=-1,
            )
            # Box Loss - simple MSE between RGB and depth predictions
            box_output_level = tf.reshape(
                box_outputs[level] * positive_mask,
                box_outputs[level].shape[:3]
                + [int(box_outputs[level].shape[3] / 4), 4],
            )  # y, x, h, w

            depth_output_level = tf.reshape(
                boxes_depth_levels[level] * positive_mask,
                box_outputs[level].shape[:3]
                + [int(box_outputs[level].shape[3] / 4), 4],
            )  # y, x, h, w

            mse_loss_fn = self.loss[tf.keras.losses.MeanAbsoluteError.__name__]
            box_loss_level = tf.reduce_mean(
                mse_loss_fn(box_output_level, depth_output_level)
            )
            box_losses.append(tf.cast(box_loss_level, dtype))

        box_loss = tf.reduce_mean(box_losses) if box_losses else 0
        return box_loss

    def _CSD_detection_loss(
        self, cls_outputs, box_outputs, cls_outputs_flipped, box_outputs_flipped
    ):
        """Computes detection consistency loss based on:

        Jeong, Jisoo, et al. "Consistency-based semi-supervised learning for object detection."
        Advances in neural information processing systems 32 (2019).

        Args:
          cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].
          box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width,
            num_anchors * 4].
          cls_outputs_flipped: for the flipped image, an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].
          box_outputs_flipped: for the flipped image, an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width,
            num_anchors * 4].
        Returns:
          total_loss: an integer tensor representing total loss reducing from
            class and box losses from all levels.
        """
        dtype = cls_outputs[0].dtype
        levels = range(len(cls_outputs))
        cls_losses = []
        box_losses = []
        for level in levels:
            # Class loss
            # Reshape to get softmax to calculate JSD
            max_score_anchors = tf.math.reduce_max(cls_outputs[level], axis=-1)
            if self.config.csd_BE:
                logging.info("BE activated with " + str(self.config.csd_BE_thr))
                score_mask = tf.math.sigmoid(max_score_anchors)
                positive_mask = tf.expand_dims(
                    tf.cast(
                        tf.math.greater(
                            score_mask,
                            self.config.csd_BE_thr * tf.math.reduce_max(score_mask),
                        ),
                        dtype,
                    ),
                    axis=-1,
                )
            else:
                positive_mask = tf.expand_dims(tf.ones_like(max_score_anchors), axis=-1)

            cls_output_level = tf.nn.softmax(
                tf.reshape(
                    cls_outputs[level] * positive_mask,
                    cls_outputs[level].shape[:3]
                    + [
                        int(cls_outputs[level].shape[3] / self.config.num_classes),
                        self.config.num_classes,
                    ],
                ),
                axis=-1,
            )
            cls_outputs_flipped[level] = tf.image.flip_left_right(
                cls_outputs_flipped[level]
            )  # Flip back output so anchors match each other
            flipped_cls_output_level = tf.nn.softmax(
                tf.reshape(
                    cls_outputs_flipped[level] * positive_mask,
                    cls_outputs_flipped[level].shape[:3]
                    + [
                        int(cls_outputs[level].shape[3] / self.config.num_classes),
                        self.config.num_classes,
                    ],
                ),
                axis=-1,
            )
            jsd_loss = self.loss[tf.keras.losses.KLDivergence.__name__]
            cls_loss_level = tf.reduce_sum(
                jsd_loss(cls_output_level, flipped_cls_output_level)
            ) * (1.0 / self.config.batch_size)
            cls_losses.append(tf.cast(cls_loss_level, dtype))

            # Box Loss
            box_output_level = tf.reshape(
                box_outputs[level] * positive_mask,
                box_outputs[level].shape[:3]
                + [int(box_outputs[level].shape[3] / 4), 4],
            )  # y, x, h, w
            box_outputs_flipped[level] = tf.image.flip_left_right(
                box_outputs_flipped[level]
            )  # Flip back output so anchors match each other
            flipped_box_output_level = tf.reshape(
                box_outputs_flipped[level] * positive_mask,
                box_outputs_flipped[level].shape[:3]
                + [int(box_outputs_flipped[level].shape[3] / 4), 4],
            )  # y, x, h, w
            loss_y = tf.reduce_mean(
                tf.pow(
                    (
                        box_output_level[:, :, :, :, 0]
                        - flipped_box_output_level[:, :, :, :, 0]
                    ),
                    2,
                )
            )
            loss_x = tf.reduce_mean(
                tf.pow(
                    (
                        box_output_level[:, :, :, :, 1]
                        + flipped_box_output_level[:, :, :, :, 1]
                    ),
                    2,
                )
            )  # Since offset x is flipped during prediction
            loss_h = tf.reduce_mean(
                tf.pow(
                    (
                        box_output_level[:, :, :, :, 2]
                        - flipped_box_output_level[:, :, :, :, 2]
                    ),
                    2,
                )
            )
            loss_w = tf.reduce_mean(
                tf.pow(
                    (
                        box_output_level[:, :, :, :, 3]
                        - flipped_box_output_level[:, :, :, :, 3]
                    ),
                    2,
                )
            )

            box_loss_level = tf.divide(loss_x + loss_y + loss_w + loss_h, 4)
            box_losses.append(tf.cast(box_loss_level, dtype))

        cls_loss = tf.reduce_mean(cls_losses) if cls_losses else 0
        box_loss = tf.reduce_mean(box_losses) if box_losses else 0
        return cls_loss, box_loss

    def train_step(self, images, labels):
        """Train step.
        Args:
          images: Image tensor with shape [batch_size,
            height, width, 3]. The height and width are fixed and equal.
          labels: Input labels in a dictionary. The labels include class targets and box targets which
            are dense label maps. The labels are generated from get_input_fn
            function in data/dataloader.py.
        Returns:
          A dict record loss info.
        """
        # If using unlabeled/pseudo-labeled data
        if "unlabeled_start" in self.config.keys():
            unlabeled_start = self.config.unlabeled_start
            if self.config.ssl_method == "STAC":
                stac_activate = True
                csd_activate = False
            else:
                stac_activate = False                      
                csd_activate = True
        else:
            unlabeled_start = self.config.batch_size
            csd_activate = False
            stac_activate = False

        if self.config.get("depth_batch", None) is not None: # if using train_flags_depth.py
            activate_depth = True
        else:
            activate_depth = False

        if self.config.save_train_images:
            with self.summary_writer.as_default():
                tf.summary.image("input_image", images, max_outputs=5)
        self.two_losses_weight = [float(w) for w in self.config.two_losses_weight] if "two_losses_weight" in self.config.keys() else [0.0, 0.0]

        with tf.GradientTape() as tape:
            labels = utils_keras.fp16_to_fp32_nested(labels)
            if labels["groundtruth_data"].shape[-1] > 7:  # Use image scores
                im_scores = labels["groundtruth_data"][:, :, -1]
                im_pos_mask = im_scores >= 0
                self.depth_alpha = 1.0 if "alpha" not in self.config.model_dir else float(self.config.model_dir.split("alpha")[-1].split("_")[0])
                self.images_save = images
                
                self.depth_detection_distance_mask = True if self.two_losses_weight != [0.0, 0.0] else False

                if self.config.depth_batch:  
                    logging.info("Activated depth_batch.")
                elif self.config.depth_image:  
                    logging.info("Activated depth_image.")
                elif self.config.depth_loss and not "consistency" in self.config.model_dir:  
                    logging.info("Activated depth_loss (box loss with image as input to the model).")
                elif self.config.depth_loss and "consistency" in self.config.model_dir:
                    logging.info("Activated depth_loss based on consistency (box loss with image as input to the model).")
                elif self.config.depth_detection and self.config.depth_additional_parameter > 0.0 and self.config.predict_distance:
                    logging.info(f"Activated depth model predicting extra distance per object with weight {self.config.depth_additional_parameter}.")
                elif self.config.depth_detection and not self.config.predict_distance and not self.depth_detection_distance_mask:  
                    logging.info(f"Activated DWL with alpha (1 + alpha*exp(x)) {self.depth_alpha} and Ramp up {self.config.rampup_det_losses}")
                elif self.config.depth_detection and not self.config.predict_distance and self.depth_detection_distance_mask:  
                    logging.info(
                        f"Activated DSL with weights {self.two_losses_weight} and Ramp up {self.config.rampup_det_losses}"
                    )
                # logging.info(
                #     f"Activated depth detection with alpha (1 + alpha*exp(x)) {self.depth_alpha}"
                # )
                self.depth_detection_object_mask = False
                # logging.info(
                #     f"Activated depth detection with object mask {self.depth_detection_object_mask}"
                # )
                self.depth_detection_class_mask = False
                # logging.info(
                #     f"Activated depth detection with class mask {self.depth_detection_class_mask}"
                # )
                if activate_depth:
                    rampup_steps = self.config.steps_per_epoch * self.config.num_epochs
                    current_step = self.optimizer.iterations
                    if self.config.rampup_det_losses:
                        if "linear" in self.config.model_dir:                     
                            # linear:
                            self.gamma_at_step = tf.clip_by_value(
                            tf.cast(current_step, tf.float32) / rampup_steps, 0.0, 1.0
                            )
                            logging.info(f"Using linear rampup")
                        else: 
                            constant_steps = float(self.config.model_dir.split("rampup_step")[-1].split("_")[0])
                            early_const = int(
                                constant_steps * rampup_steps
                            )  # First and last constant steps
                            cut_range = int(
                                (1 - 2 * constant_steps) * rampup_steps
                            )  # Middle dynamic range
                            steps = tf.range(0, rampup_steps, dtype=tf.float32)

                            # Compute ramp-up factor for the middle region
                            middle_gamma = tf.clip_by_value(
                                steps[:cut_range] / ((1 - 2 * constant_steps) * rampup_steps),
                                0.0,
                                1.0,
                            )

                            # Create constant tensors for the first and last 20% regions
                            early_gamma = tf.zeros([early_const], dtype=tf.float32)
                            late_gamma = tf.ones([early_const], dtype=tf.float32)

                            # Concatenate to form the final gamma tensor
                            gamma = tf.concat([early_gamma, middle_gamma, late_gamma], axis=0)

                            # Get the gamma value for the current step
                            self.gamma_at_step = tf.gather(gamma, current_step)
                            logging.info(f"Using stepwise rampup with {constant_steps}% constant steps")
                    else: self.gamma_at_step = 1.0


                    # Normalize depth values to [0, 1] range based on known min/max depth
                    min_score = tf.reduce_min(im_scores)
                    max_score = tf.reduce_max(im_scores)
                    im_scores = (im_scores - min_score) / (max_score - min_score)
                    # reverse: close =0, far=1
                    if "inverse" in self.config.model_dir:
                        im_scores = 1.0 - im_scores
                    im_scores = tf.clip_by_value(im_scores, 0.0, 1.0)
                    im_scores = tf.map_fn(
                        apply_boolean_mask,
                        (im_scores, im_pos_mask),
                        dtype=tf.float32,
                    )
                    
                    if self.config.depth_image:
                        if "expo" in self.config.model_dir:
                            im_scores = 1 + self.depth_alpha * tf.math.exp(im_scores)
                        im_scores = (
                            1.0 - self.gamma_at_step
                        ) * 1.0 + self.gamma_at_step * im_scores
                    self.depth_scores = im_scores
                    if self.config.depth_batch:
                        avg_batch_score = tf.reduce_mean(im_scores)
                        if "expo" in self.config.model_dir:
                            avg_batch_score = 1 + self.depth_alpha * tf.math.exp(avg_batch_score)
                        avg_batch_score = (
                            1.0 - self.gamma_at_step
                        ) * 1.0 + self.gamma_at_step * avg_batch_score
                    else:
                        avg_batch_score = 1.0
                else:
                    im_scores = tf.map_fn(
                        apply_boolean_mask,
                        (im_scores, im_pos_mask),
                        dtype=tf.float32,
                    )
                    avg_batch_score = tf.reduce_mean(im_scores[:unlabeled_start])
            else:
                avg_batch_score = 1.0
                im_scores = None

            if len(self.config.heads) == 2:
                cls_outputs, box_outputs, seg_outputs = utils_keras.fp16_to_fp32_nested(
                    self(images, training=True)
                )
                loss_dtype = cls_outputs[0].dtype
            elif "object_detection" in self.config.heads:
                cls_outputs, box_outputs = utils_keras.fp16_to_fp32_nested(
                    self(images, training=True)
                )
                loss_dtype = cls_outputs[0].dtype
            elif "segmentation" in self.config.heads:
                (seg_outputs,) = utils_keras.fp16_to_fp32_nested(
                    self(images, training=True)
                )
                loss_dtype = seg_outputs.dtype
            else:
                raise ValueError("No valid head found: {}".format(self.config.heads))
            if self.config.loss_attenuation:
                box_outputs = self._clip_uncert(box_outputs)
            total_loss = 0
            loss_vals = {}            
            if csd_activate:
                # Save example image
                # tf.io.write_file("saved_image.jpg",  tf.image.encode_jpeg(tf.cast(tf.clip_by_value((images[0] + 1.0) * 127.5, 0, 255), tf.uint8)))
                aug_images = tf.image.flip_left_right(images)
                cls_outputs_aug, box_outputs_aug = utils_keras.fp16_to_fp32_nested(
                    self(aug_images, training=True)
                )
                if self.config.loss_attenuation:
                    box_outputs = [
                        box_outputs[level][:, :, :, : int(box_outputs[0].shape[-1] / 2)]
                        for level in range(len(box_outputs))
                    ]
                    box_outputs_aug = [
                        box_outputs_aug[level][
                            :, :, :, : int(box_outputs_aug[0].shape[-1] / 2)
                        ]
                        for level in range(len(box_outputs_aug))
                    ]

                supervised_cls_outputs, supervised_box_outputs, supervised_labels = (
                    self._split_output_labels(
                        cls_outputs, box_outputs, labels, 0, unlabeled_start
                    )
                )
                # Supervized loss on the labeled only
                sup_det_loss = self._detection_loss(
                    supervised_cls_outputs,
                    supervised_box_outputs,
                    supervised_labels,
                    loss_vals,
                )
                # Unsupervized loss on all data
                unsup_cls_loss, unsup_box_loss = self._CSD_detection_loss(
                    cls_outputs,
                    box_outputs,
                    cls_outputs_aug,
                    box_outputs_aug,
                )
                unsup_det_loss = unsup_cls_loss + unsup_box_loss
                if self.config.csd_ramp:
                    csd_ramping_weight = tf.cast(self._weight_scheduling(), loss_dtype)
                else:
                    csd_ramping_weight = 1.0
                loss_vals["ramp_w"] = csd_ramping_weight
                total_loss += sup_det_loss + tf.math.multiply(
                    csd_ramping_weight, unsup_det_loss
                )

            elif stac_activate:
                supervised_cls_outputs, supervised_box_outputs, supervised_labels = (
                    self._split_output_labels(
                        cls_outputs, box_outputs, labels, 0, unlabeled_start
                    )
                )

                (
                    unsupervised_cls_outputs,
                    unsupervised_box_outputs,
                    unsupervised_labels,
                ) = self._split_output_labels(
                    cls_outputs,
                    box_outputs,
                    labels,
                    unlabeled_start,
                    len(cls_outputs[0]),
                )

                if im_scores is not None:  # use Pseudo-Scores
                    pseudo_scores = im_scores[unlabeled_start:]
                    avg_pseudo_score = tf.reduce_mean(pseudo_scores)
                else:
                    avg_pseudo_score = 1.0
                    pseudo_scores = None
                # Supervized loss on both labeled and unlabeled
                sup_det_loss = self._detection_loss(
                    supervised_cls_outputs,
                    supervised_box_outputs,
                    supervised_labels,
                    loss_vals,
                )
                pseudo_det_loss = self._detection_loss(
                    unsupervised_cls_outputs,
                    unsupervised_box_outputs,
                    unsupervised_labels,
                    loss_vals,
                    pseudo_scores=pseudo_scores,
                    pseudo=True,
                )
                if im_scores is not None:  # score for labeled images
                    avg_batch_score = tf.reduce_mean(im_scores[:unlabeled_start])
                total_loss += tf.math.multiply(
                    sup_det_loss, avg_batch_score
                ) + tf.math.multiply(
                    tf.math.multiply(self.config.stac_lambda, pseudo_det_loss),
                    avg_pseudo_score,
                )

            else:
                if "object_detection" in self.config.heads:
                    optimize_loss_weights=False
                    loss_vals['best_weights_min'] = tf.reduce_min(self.two_losses_weight)
                    loss_vals['best_weights_max'] = tf.reduce_max(self.two_losses_weight)
                    if optimize_loss_weights and (self.optimizer.iterations % (1 * self.config.steps_per_epoch) == 0):
                        original_weights = self.two_losses_weight
                        # TensorFlow-only optimization using a grid search or small loop
                        import optuna

                        # Simple ranges to search over [0.1, 0.5, 1.0]
                        def _f_x(params):
                            self.two_losses_weight = params

                            # Compute loss with this weight setting
                            opt_loss = tf.get_static_value(self._detection_loss(cls_outputs, box_outputs, labels, loss_vals))
                            return opt_loss

                        def _f(x):
                            """Wraps _f_x()"""
                            x = x.values
                            n_particles = x.shape[0]
                            j = [_f_x(x[i]) for i in range(n_particles)]
                            return np.asarray(j)

                        study = optuna.create_study(direction="minimize")
                        min_value = float("inf")
                        unchanged_count = 0
                        max_unchanged_epochs = (
                            50  # Set the desired number of epochs for convergence
                        )
                        for epoch in range(1, 100):
                            trial = study.ask()
                            suggested_params = [
                                trial.suggest_float(f"param_{i}", 0.1, 1)
                                for i in range(1, 4)
                            ]
                            value = _f_x(suggested_params)
                            study.tell(trial, value)
                            logging.info(f"Epoch {epoch}, Value {value}, Best Value {min_value}")

                            # Check for convergence
                            if value is not None and value < min_value:
                                min_value = value
                                unchanged_count = 0
                            else:
                                unchanged_count += 1

                            if unchanged_count >= max_unchanged_epochs:
                                logging.info(
                                    f"Convergence reached. Minimum value has not changed for {max_unchanged_epochs} epochs."
                                )
                                break
                        if value is not None:
                            # Assign best weights
                            self.two_losses_weight = [
                                study.best_params[f"param_{i}"] for i in range(1, 4)
                            ]
                            logging.info(
                                f"Optimized weights with {self.two_losses_weight}"
                            )   
                        loss_vals['best_weights_min'] = tf.reduce_min(self.two_losses_weight)
                        loss_vals['best_weights_max'] = tf.reduce_max(self.two_losses_weight)
                    if hasattr(self, "depth_scores"):
                        det_loss = self._detection_loss(
                            cls_outputs,
                            box_outputs,
                            labels,
                            loss_vals,
                            pseudo_scores=self.depth_scores,
                        )
                    else:
                        det_loss = self._detection_loss(
                            cls_outputs, box_outputs, labels, loss_vals
                        )
                    total_loss += det_loss

            if activate_depth:
                loss_vals["gamma_at_step"] = self.gamma_at_step
                if self.config.depth_loss:
                    depth_images = labels["depth_array"]
                    depth_images = tf.tile(depth_images, multiples=[1, 1, 1, 3])

                    cls_outputs_depth, box_outputs_depth = utils_keras.fp16_to_fp32_nested(
                        self(depth_images, training=True)
                    )

                    
                    # Unsupervized loss on all data
                    if "consistency" in self.config.model_dir:
                        det_depth_loss = self.consistency_loss_tf(
                            labels,
                            cls_outputs,
                            box_outputs,
                            box_outputs_depth,
                        )
                    else:
                        det_depth_loss = self._detection_loss(
                            cls_outputs_depth,
                            box_outputs_depth,
                            labels,
                            loss_vals,
                            pseudo=True,
                            box_only=True,
                        )

                    total_loss += det_depth_loss
                    loss_vals["depth_det_loss"] = det_depth_loss
                    loss_vals["gamma_depth_det_loss"] = self.gamma_at_step

            loss_vals["avg_batch_score"] = avg_batch_score
            if csd_activate:
                loss_vals["unsup_det_loss"] = unsup_det_loss
                loss_vals["unsup_cls_loss"] = unsup_cls_loss
                loss_vals["unsup_box_loss"] = unsup_box_loss
            if stac_activate:
                loss_vals["avg_pseudo_score"] = avg_pseudo_score

            if "segmentation" in self.config.heads:
                seg_loss_layer = self.loss[
                    tf.keras.losses.SparseCategoricalCrossentropy.__name__
                ]
                seg_loss = seg_loss_layer(labels["image_masks"], seg_outputs)
                total_loss += seg_loss
                loss_vals["seg_loss"] = seg_loss

            reg_l2_loss = self._reg_l2_loss(self.config.weight_decay)
            loss_vals["reg_l2_loss"] = reg_l2_loss
            total_loss += tf.cast(reg_l2_loss, loss_dtype)

            total_loss = tf.multiply(total_loss, avg_batch_score)
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)
                optimizer = self.optimizer.inner_optimizer
            else:
                scaled_loss = total_loss
                optimizer = self.optimizer
        loss_vals["loss"] = total_loss
        loss_vals["learning_rate"] = optimizer.learning_rate(optimizer.iterations)
        trainable_vars = self._freeze_vars()
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients
        if self.config.clip_gradients_norm > 0:
            clip_norm = abs(self.config.clip_gradients_norm)
            gradients = [
                tf.clip_by_norm(g, clip_norm) if g is not None else None
                for g in gradients
            ]
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
            loss_vals["gradient_norm"] = tf.linalg.global_norm(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Calculate metrics
        if self.config.predict_distance:
            box_outputs = [
                box_outputs[level][:, :, :, :-1]
                for level in range(len(box_outputs))
            ]
        metrics_output = self._get_metrics(labels, cls_outputs, box_outputs)
        for key in metrics_output:
            loss_vals[key] = metrics_output[key]
        return loss_vals

    @property
    def metrics(self):
        return self.metrics_collect["class"] + self.metrics_collect["box"]

    def test_step(self, images, labels):
        """Test step.
        Args:
          images: Image tensor with shape [batch_size,
            height, width, 3]. The height and width are fixed and equal.
          labels: Input labels
            in a dictionary. The labels include class targets and box targets which
            are dense label maps. The labels are generated from get_input_fn
            function in data/dataloader.py.
        Returns:
          A dict record loss info.
        """

        if self.config.get("depth_batch", None) is not None:
            orig_depth_batch = self.config.depth_batch
            orig_depth_image = self.config.depth_image
            orig_depth_detection = self.config.depth_detection
            orig_depth_loss = self.config.depth_loss

            self.config.depth_batch = False
            self.config.depth_image = False
            self.config.depth_detection = False
            self.config.depth_loss = False
        # If using unlabeled data
        labels = utils_keras.fp16_to_fp32_nested(labels)
        if len(self.config.heads) == 2:
            cls_outputs, box_outputs, seg_outputs = utils_keras.fp16_to_fp32_nested(
                self(images, training=False)
            )
            loss_dtype = cls_outputs[0].dtype
        elif "object_detection" in self.config.heads:
            cls_outputs, box_outputs = utils_keras.fp16_to_fp32_nested(
                self(images, training=False)
            )
            loss_dtype = cls_outputs[0].dtype
        elif "segmentation" in self.config.heads:
            (seg_outputs,) = utils_keras.fp16_to_fp32_nested(
                self(images, training=False)
            )
            loss_dtype = seg_outputs.dtype
        else:
            raise ValueError("No valid head found: {}".format(self.config.heads))
        if self.config.loss_attenuation:
            box_outputs = self._clip_uncert(box_outputs)
        total_loss = 0
        loss_vals = {}
        if "object_detection" in self.config.heads:
            det_loss = self._detection_loss(cls_outputs, box_outputs, labels, loss_vals)
            total_loss += det_loss
        if "segmentation" in self.config.heads:
            seg_loss_layer = self.loss[
                tf.keras.losses.SparseCategoricalCrossentropy.__name__
            ]
            seg_loss = seg_loss_layer(labels["image_masks"], seg_outputs)
            total_loss += seg_loss
            loss_vals["seg_loss"] = seg_loss
        reg_l2_loss = self._reg_l2_loss(self.config.weight_decay)
        loss_vals["reg_l2_loss"] = reg_l2_loss
        loss_vals["loss"] = total_loss + tf.cast(reg_l2_loss, loss_dtype)

        # Calculate metrics
        if self.config.predict_distance:
            box_outputs = [
                box_outputs[level][:, :, :, :-1]
                for level in range(len(box_outputs))
            ]
        metrics_output = self._get_metrics(labels, cls_outputs, box_outputs)
        for key in metrics_output:
            loss_vals[key] = metrics_output[key]

        if self.config.get("depth_batch", None) is not None:
            self.config.depth_batch = orig_depth_batch
            self.config.depth_image = orig_depth_image
            self.config.depth_detection = orig_depth_detection
            self.config.depth_loss = orig_depth_loss
        return loss_vals


class CollectEpochLoss:
    """Compute the loss per epoch"""

    def __init__(self):
        self.epochend = {}
        self.n_train = 0
        self.n_val = 0

    def reset(self):
        self.__init__()

    def update(self, loss_vals, val=False):
        if val:
            val = "val_"
            self.n_val += 1
        else:
            val = ""
            self.n_train += 1
        self.epochend = {
            val
            + i: (
                self.epochend[val + i] + loss_vals[i]
                if val + i in self.epochend
                else loss_vals[i]
            )
            for i in loss_vals
        }

    def result(self, val=False):
        if val:
            norm = self.n_val
        else:
            norm = self.n_train
        return {i: (self.epochend[i] / norm).astype("f2") for i in self.epochend}


class EfficientDetNetTrainHub(EfficientDetNetTrain):
    """EfficientDetNetTrain for Hub module."""

    def __init__(self, config, hub_module_url, name=""):
        super(efficientdet_keras.EfficientDetNet, self).__init__(name=name)
        self.config = config
        self.hub_module_url = hub_module_url
        self.base_model = hub.KerasLayer(hub_module_url, trainable=True)

        # class/box output prediction network.
        num_anchors = len(config.aspect_ratios) * config.num_scales

        conv2d_layer = efficientdet_keras.ClassNet.conv2d_layer(
            config.separable_conv, config.data_format
        )
        self.classes = efficientdet_keras.ClassNet.classes_layer(
            conv2d_layer,
            config.num_classes,
            num_anchors,
            name="class_net/class-predict",
        )

        self.boxes = efficientdet_keras.BoxNet.boxes_layer(
            config.separable_conv,
            num_anchors,
            config.data_format,
            name="box_net/box-predict",
        )

        log_dir = os.path.join(self.config.model_dir, "train_images")
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def call(self, inputs, training):
        cls_outputs, box_outputs = self.base_model(inputs, training=training)
        for i in range(self.config.max_level - self.config.min_level + 1):
            cls_outputs[i] = self.classes(cls_outputs[i])
            box_outputs[i] = self.boxes(box_outputs[i])
        return (cls_outputs, box_outputs)
