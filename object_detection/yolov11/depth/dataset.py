import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_file_speeds


class YOLODepthDataset(YOLODataset):
    """
    Custom YOLO dataset class for images with corresponding depth maps.

    Dataset CSV Format:
        Each line in the CSV file should contain two comma-separated paths:
            image_path,depth_path
        - image_path: Relative or absolute path to the image file.
        - depth_path: Relative or absolute path to the depth map file (expected as .npy).

    The dataset loads both the image and its corresponding depth map,
    stacking the depth map as a fourth channel (resulting in a 4-channel image: RGBD).
    """

    def __init__(self, *args, img_path: str, **kwargs):
        self.img_path = img_path
        self.read_dataset_definition()
        super().__init__(*args, img_path=img_path, **kwargs)

    def __getitem__(self, index):
        # Applies transforms, including extraction of the depth map as a separate label
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index) -> dict:
        """
        Loads the image and its corresponding depth map, stacking them as a 4-channel image.
        """
        image_and_label = super().get_image_and_label(index)
        depth_map, orig_shape, new_shape = self.load_depth_map(index)

        # Ensure the original and resized shapes match between image and depth map
        assert image_and_label["ori_shape"] == orig_shape, (
            f"Shape mismatch between image and depth map ({image_and_label['ori_shape']} vs. {orig_shape})"
        )
        assert image_and_label["resized_shape"] == new_shape, (
            f"Shape mismatch between image and depth map ({image_and_label['resized_shape']} vs. {new_shape})"
        )

        # Stack depth map as the fourth channel (RGBD)
        image_and_label["img"] = np.dstack((image_and_label["img"], depth_map))
        return image_and_label

    def load_depth_map(self, index: int, rect_mode=True):
        """
        Loads and preprocesses the depth map for the given index.

        Preprocessing steps:
            - Loads the depth map from a .npy file.
            - Resizes to match the image shape (rectangular or square).
            - Adds a channel dimension if needed.

        Returns:
            (np.ndarray): Depth map (H, W, 1).
            (Tuple[int, int]): Original image dimensions (height, width).
            (Tuple[int, int]): Resized image dimensions (height, width).
        """
        f = self.depth_files[index]
        im = np.load(f)

        # original dimensions
        h0, w0 = im.shape[:2]
        if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (
                    min(math.ceil(w0 * r), self.imgsz),
                    min(math.ceil(h0 * r), self.imgsz),
                )
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]

        return im, (h0, w0), im.shape[:2]

    def read_dataset_definition(self):
        """
        Reads the dataset definition from the specified CSV file.

        Expects each line in the CSV to be:
            image_path,depth_path

        Populates:
            self.im_files: List of image file paths.
            self.depth_files: List of depth map file paths.
        """
        with open(self.img_path) as file:
            lines = file.readlines()
            rows = [line.strip().split(",") for line in lines]
            self.im_files = [row[0] for row in rows]
            self.depth_files = [row[1] for row in rows]

        self.im_files = [str(Path(self.img_path).parent / f) for f in self.im_files]
        self.depth_files = [str(Path(self.img_path).parent / f) for f in self.depth_files]
        assert len(self.im_files) == len(self.depth_files), "Number of images and depth maps do not match"

    def get_img_files(self, img_path):
        # Overridden to prevent im_files from being overwritten by the base class
        check_file_speeds(self.im_files, prefix=self.prefix)
        return self.im_files

    def build_transforms(self, hyp=None):
        """
        Builds the transform pipeline.

        The last transform extracts the depth map from the 4-channel image
        and moves it to a separate label entry ("depth_map").
        """
        transforms = super().build_transforms(hyp)
        format = transforms[-1]
        transforms[-1] = DepthMapExtractor()
        transforms.append(format)
        return transforms

    def set_rectangle(self) -> None:
        """See YOLODataset.set_rectangle for docstring. Overwritten to handle depth data."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.depth_files = [self.depth_files[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image


class DepthMapExtractor:
    """
    Transform to extract the depth map from the 4-channel image.

    Purpose:
        - Separates the depth map (last channel) from the image tensor.
        - Stores the depth map in the "depth_map" label entry as a torch.Tensor.
    """

    def __call__(self, labels):
        img = labels["img"]
        # Extract the depth map (last channel)
        depth_map = img[:, :, -1]
        # Keep only RGB channels
        labels["img"] = img[:, :, :-1]
        # Store depth map as a torch tensor
        labels["depth_map"] = torch.from_numpy(depth_map).float()
        return labels
