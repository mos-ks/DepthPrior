import csv
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SUNRGBD_ROOT = Path("/path/to/data/DepthPrior/datasets/SUNRGBD")
CLASS_LIST = [
    "bed",
    "table",
    "sofa",
    "chair",
    "toilet",
    "desk",
    "dresser",
    "night_stand",
    "bookshelf",
    "bathtub",
    "box",
    "books",
    "bottle",
    "bag",
    "pillow",
    "monitor",
    "television",
    "lamp",
    "garbage_bin",
]

images = list((SUNRGBD_ROOT / "val" / "images").glob("*.jpg"))

images_coco = []
annotations_coco = []
for image_idx, image_path in tqdm(enumerate(images), desc="Processing images", total=len(images)):
    image_id = image_path.stem

    image = Image.open(image_path)
    bbox_width, bbox_height = image.size
    images_coco.append(
        {
            "id": image_idx,
            "file_name": image_path.name,
            "width": bbox_width,
            "height": bbox_height,
        }
    )

    label_path = SUNRGBD_ROOT / "val" / "labels" / f"{image_path.stem}.txt"
    with open(label_path) as label_file:
        bboxes = []
        class_ids = []
        reader = csv.DictReader(
            label_file,
            fieldnames=[
                "class",
                "x_center",
                "y_center",
                "width",
                "height",
            ],
            delimiter=" ",
        )
        for row in reader:
            bbox = [
                (float(row["x_center"]) - float(row["width"]) / 2) * bbox_width,
                (float(row["y_center"]) - float(row["height"]) / 2) * bbox_height,
                float(row["width"]) * bbox_width,
                float(row["height"]) * bbox_height,
            ]
            bboxes.append(bbox)
            class_ids.append(int(row["class"]))

        assert len(class_ids) == len(bboxes), (
            f"Number of class IDs ({len(class_ids)}) does not match "
            f"number of bounding boxes ({len(bboxes)}) for image {image_id}"
        )

        for i, (class_id, bbox) in enumerate(zip(class_ids, bboxes)):
            bbox_width = bbox[2]
            bbox_height = bbox[3]
            area = bbox_width * bbox_height

            annotations_coco.append(
                {
                    "image_id": image_idx,
                    "id": image_idx * 100 + i,
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
with open(SUNRGBD_ROOT / "annotations_coco.json", "w") as f:
    json.dump({"annotations": annotations_coco, "images": images_coco, "categories": categories}, f)
