#!/usr/bin/env python3
"""
Process COCO annotations with depth information.
"""

import json
import os

import numpy as np

general_path = "/path/to/data/DepthPrior/"

def get_depth_from_bbox(depth_map, bbox):
    """Extract depth value from bounding box region."""
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Clip to valid range
    h_img, w_img = depth_map.shape[:2]
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img, y2))
    
    if x2 <= x1 or y2 <= y1:
        return -1.0
    
    depth_values = depth_map[y1:y2, x1:x2]
    return float(np.mean(depth_values)) if depth_values.size > 0 else -1.0


# Directories
train_annotations_file = os.path.join(general_path, "datasets/MSCOCO/annotations/instances_train2017.json")
train_depth_dir = os.path.join(general_path, "datasets/MSCOCO/depth_anything_predictions")
train_output_file = os.path.join(general_path, "datasets/MSCOCO/annotations/instances_train2017_with_depth.json")

# Load COCO annotations
print("Loading COCO annotations...")
with open(train_annotations_file, 'r') as f:
    coco_data = json.load(f)

print(f"Processing {len(coco_data['annotations'])} annotations with depth...")

image_ids = [img['id'] for img in coco_data['images']]
# Process each annotation
processed_count = 0
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    id_location = image_ids.index(image_id) if image_id in image_ids else None
    image_info = coco_data['images'][id_location]  # Assuming images are 1-indexed, but list is 0-indexed? Wait, images is list, need to find by id
    
    # Find image info
    img_info = None
    for img in coco_data['images']:
        if img['id'] == image_id:
            img_info = img
            break
    if img_info is None:
        continue
    
    img_name_without_ext = os.path.splitext(img_info['file_name'])[0]
    depth_map_path = os.path.join(train_depth_dir, f"{img_name_without_ext}_depth.npy")
    
    if not os.path.exists(depth_map_path):
        ann['depth'] = -1.0
        continue
    
    # Load depth map
    try:
        depth_map = np.load(depth_map_path).astype(np.float32)
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]  # Take first channel if needed
    except Exception as e:
        print(f"Error loading depth {depth_map_path}: {e}")
        ann['depth'] = -1.0
        continue
    
    # Compute depth for bbox
    bbox = ann['bbox']  # [x, y, w, h]
    depth_value = get_depth_from_bbox(depth_map, bbox)
    ann['depth'] = depth_value
    
    processed_count += 1
    if processed_count % 10000 == 0:
        print(f"Processed {processed_count}/{len(coco_data['annotations'])} annotations")

# Save modified annotations
print(f"Saving modified annotations to {train_output_file}")
with open(train_output_file, 'w') as f:
    json.dump(coco_data, f)

print(f"\nCompleted! Processed {processed_count} annotations")
print(f"Output file: {train_output_file}")