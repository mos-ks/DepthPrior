import contextlib
import csv
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

CLASS_LIST = ["Pedestrian", "Car", "Van", "Cyclist", "Truck", "Tram", "Person_sitting"]

KITTI_ROOT = Path("/path/to/data/DepthPrior/datasets/KITTI")
KITTI_YOLO_ROOT = KITTI_ROOT / "yolo"
KITTI_YOLO_ROOT_RELATIVE = Path("/path/to/data/DepthPrior/datasets/KITTI/yolo")


def main():
    # get set of validation indices
    validation_indices = get_validation_indices()

    # create train and val directories
    labels_path_yolo = KITTI_YOLO_ROOT / "labels"
    labels_path_yolo.mkdir(parents=True, exist_ok=True)

    # create symlinks to images directory
    with contextlib.suppress(FileExistsError):
        (KITTI_YOLO_ROOT / "images").symlink_to((Path("..") / "training" / "image_2"), target_is_directory=True)

    # create images.txt files for train and val
    # and convert the labels to YOLO format
    image_paths = list((KITTI_ROOT / "training" / "image_2").glob("*.png"))
    with (
        open(KITTI_YOLO_ROOT / "training.txt", "w") as ti,
        open(KITTI_YOLO_ROOT / "validation.txt", "w") as vi,
        open(KITTI_YOLO_ROOT / "full.txt", "w") as fi,
    ):
        for image_path in tqdm(image_paths, desc="Processing images"):
            label_kitti_path = (KITTI_ROOT / "training" / "label_2") / f"{image_path.stem}.txt"
            is_validation = get_image_id(image_path) in validation_indices
            yolo_labels = parse_label(label_kitti_path, image_path)
            if len(yolo_labels) == 0:
                continue

            images_file = vi if is_validation else ti
            label_path = labels_path_yolo / f"{image_path.stem}.txt"

            images_file.write(f"./images/{image_path.name}\n")
            fi.write(f"./images/{image_path.name}\n")

            with open(label_path, "w") as label_file:
                for label in yolo_labels:
                    repr = " ".join([str(f) for f in label])
                    label_file.write(f"{repr}\n")

    # create kitti.yaml file
    dataset_obj = {
        "path": str(KITTI_YOLO_ROOT_RELATIVE),
        "train": "training.txt",
        "val": "validation.txt",
        "names": {i: c for i, c in enumerate(CLASS_LIST)},
    }
    with open(Path(__file__).parent.parent / "dataset-configs" / "kitti.yaml", "w") as dataset_file:
        yaml.dump(dataset_obj, dataset_file)


def get_image_id(image_path: Path):
    return int(image_path.stem)


def parse_label(label_path, image_path):
    yolo_labels = []
    with open(label_path) as label_file:
        reader = csv.DictReader(
            label_file,
            fieldnames=[
                "type",
                "truncated",
                "occluded",
                "alpha",
                "bbox2_left",
                "bbox2_top",
                "bbox2_right",
                "bbox2_bottom",
                "bbox3_height",
                "bbox3_width",
                "bbox3_length",
                "bbox3_x",
                "bbox3_y",
                "bbox3_z",
                "bbox3_yaw",
                "score",
            ],
            delimiter=" ",
        )
        for row in reader:
            if row["type"] not in CLASS_LIST:
                continue
            clazz_number = CLASS_LIST.index(row["type"])
            size = Image.open(image_path).size  # (1242, 375)
            bbox = (
                float(row["bbox2_left"]),
                float(row["bbox2_right"]),
                float(row["bbox2_top"]),
                float(row["bbox2_bottom"]),
            )
            yolo_bbox = to_yolo_bbox(bbox, size)
            # <object-class> <x> <y> <width> <height>.
            yolo_label = (clazz_number,) + yolo_bbox
            yolo_labels.append(yolo_label)
    return yolo_labels


def to_yolo_bbox(bbox, size):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def get_validation_indices():
    # Read vaL_index_list.txt and extract validation indices
    with open(KITTI_ROOT / "vaL_index_list.txt") as file:
        validation_indices = file.readlines()
        validation_indices = [int(s) for s in validation_indices]
        validation_indices = set(validation_indices)
    return validation_indices


if __name__ == "__main__":
    main()
