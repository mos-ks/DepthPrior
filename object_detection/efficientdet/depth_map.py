import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(5, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth_paper.depth_anything.dpt import DepthAnything
from depth_paper.depth_anything.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from torchvision.transforms import Compose
from tqdm import tqdm

# install hugginface_hub, torchvision


def depth_calc(filenames):
    encoder = "vitl"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    depth_anything = (
        DepthAnything.from_pretrained("LiheYoung/depth_anything_{}14".format(encoder))
        .to(DEVICE)
        .eval()
    )

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    depth_maps = []
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        depth_maps.append(depth)
    return depth_maps
