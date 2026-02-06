import json
import os
from pathlib import Path


def convert_coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create category mapping (COCO IDs are not sequential)
    category_mapping = {}
    for i, cat in enumerate(coco_data['categories']):
        category_mapping[cat['id']] = i
    print(f"Category mapping: {category_mapping}")
    # Process annotations
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Convert each image
    for img in coco_data['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_width = img['width']
        img_height = img['height']
        
        # Create YOLO label file
        label_file = Path(output_dir) / f"{Path(img_name).stem}.txt"
        
        if img_id in image_annotations:
            with open(label_file, 'w') as f:
                for ann in image_annotations[img_id]:
                    # Convert COCO bbox to YOLO format
                    x, y, w, h = ann['bbox']
                    
                    # Convert to center coordinates and normalize
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Get class ID
                    class_id = category_mapping[ann['category_id']]
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        else:
            # Create empty label file for images without annotations
            label_file.touch()

# Run conversion
convert_coco_to_yolo('/path/to/data/DepthPrior/datasets/MSCOCO/annotations/instances_train2017.json', '/path/to/data/DepthPrior/datasets/MSCOCO/labels/train2017')
convert_coco_to_yolo('/path/to/data/DepthPrior/datasets/MSCOCO/annotations/instances_val2017.json', '/path/to/data/DepthPrior/datasets/MSCOCO/labels/val2017')