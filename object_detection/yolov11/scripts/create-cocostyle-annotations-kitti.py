import csv
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

KITTI_ROOT = Path("/path/to/data/DepthPrior/datasets/KITTI")
KITTI_YOLO_ROOT = KITTI_ROOT / "yolo"
KITTI_YOLO_ROOT_RELATIVE = Path("/path/to/data/DepthPrior/datasets/KITTI/yolo")
IMAGES_LIST_IN = KITTI_YOLO_ROOT / "validation.txt"
ANNOTATIONS_FILE_OUT = KITTI_YOLO_ROOT / "annotations_coco.json"
CLASS_LIST = ["Pedestrian", "Car", "Van", "Cyclist", "Truck", "Tram", "Person_sitting"]

CSV_FIELD_NAMES = [
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
    "dim_height",
    "dim_width",
    "dim_length",
    "loc_x",
    "loc_y",
    "loc_z",
    "rotation_y",
]


def main():
    with open(IMAGES_LIST_IN) as f:
        annotations = []
        images = []
        for image_path in tqdm(f, desc="Processing images"):
            image_path = Path(image_path.strip())
            image_id = image_path.stem

            image = Image.open(KITTI_ROOT / "training" / "image_2" / image_path.name)
            width, height = image.size
            images.append(
                {
                    "id": int(image_id),
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )

            label_yolo_path = KITTI_YOLO_ROOT / "labels" / f"{image_id}.txt"
            label_kitti_path = (KITTI_ROOT / "training" / "label_2") / f"{image_path.stem}.txt"

            with open(label_yolo_path) as label_file:
                lines = label_file.readlines()

            class_ids = [int(line.split(" ")[0]) for line in lines]

            with open(label_kitti_path) as label_file:
                bboxes = []
                reader = csv.DictReader(
                    label_file,
                    fieldnames=CSV_FIELD_NAMES,
                    delimiter=" ",
                )
                for row in reader:
                    if row["type"] not in CLASS_LIST:
                        continue
                    bbox = [
                        float(row["bbox_left"]),
                        float(row["bbox_top"]),
                        float(row["bbox_right"]) - float(row["bbox_left"]),
                        float(row["bbox_bottom"]) - float(row["bbox_top"]),
                    ]
                    bboxes.append(bbox)

            assert len(class_ids) == len(bboxes), (
                f"Number of class IDs ({len(class_ids)}) does not match "
                f"number of bounding boxes ({len(bboxes)}) for image {image_id}"
            )

            for i, (class_id, bbox) in enumerate(zip(class_ids, bboxes)):
                width = bbox[2]
                height = bbox[3]
                area = width * height

                annotations.append(
                    {
                        "image_id": int(image_id),
                        "id": int(image_id) * 100 + i,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }
                )

    categories = []
    for i, class_name in enumerate(CLASS_LIST):
        categories.append({"id": i, "name": class_name, "supercategory": class_name})

    # Save the annotations to a JSON file
    with open(ANNOTATIONS_FILE_OUT, "w") as f:
        json.dump({"annotations": annotations, "images": images, "categories": categories}, f)


if __name__ == "__main__":
    main()
