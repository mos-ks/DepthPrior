#!/usr/bin/env python3
"""
Process SUN RGB-D annotations with depth information.
Adds depth value to each bounding box annotation.
"""

import os
from pathlib import Path

import cv2
import numpy as np

general_path = "/path/to/data/DepthPrior/"

def get_depth_from_bbox(depth_map, bbox, width, height):
    """
    Extract depth value from bounding box region.
    Args:
        depth_map: 2D numpy array of depth values
        bbox: YOLO format [x_center, y_center, box_width, box_height] (normalized)
        width: image width
        height: image height
    """
    # Convert YOLO to pixel coordinates
    x_center, y_center, box_width, box_height = bbox
    
    x_center_px = x_center * width
    y_center_px = y_center * height
    box_width_px = box_width * width
    box_height_px = box_height * height
    
    x1 = int(x_center_px - box_width_px / 2)
    y1 = int(y_center_px - box_height_px / 2)
    x2 = int(x_center_px + box_width_px / 2)
    y2 = int(y_center_px + box_height_px / 2)
    
    # Clip to valid range
    h, w = depth_map.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    
    if x2 <= x1 or y2 <= y1:
        return -1.0
    
    depth_values = depth_map[y1:y2, x1:x2]
    # Filter out invalid depth values (0 or very large values)
    valid_depths = depth_values[(depth_values > 0) & (depth_values < 10000)]
    
    if valid_depths.size == 0:
        return -1.0
    
    return float(np.mean(valid_depths))


# Directories
train_labels_dir = os.path.join(general_path, "datasets/SUNRGBD/train/labels")
train_depth_dir = os.path.join(general_path, "datasets/SUNRGBD/depth_anything_predictions") #datasets/SUNRGBD/train/depth
train_images_dir = os.path.join(general_path, "datasets/SUNRGBD/train/images")
train_output_dir = os.path.join(general_path, "datasets/SUNRGBD/depth_anything_predictions")

# Create output directory
os.makedirs(train_output_dir, exist_ok=True)

# Get list of label files
label_files = [f for f in os.listdir(train_labels_dir) if f.endswith(".txt")]
train_files = sorted([f.replace(".txt", "") for f in label_files])

print(f"Processing {len(train_files)} training annotations with depth...")

processed_count = 0
skipped_count = 0

for img_name in train_files:
    label_path = os.path.join(train_labels_dir, f"{img_name}.txt")
    depth_path = os.path.join(train_depth_dir, f"{img_name}_depth.npy")
    image_path = os.path.join(train_images_dir, f"{img_name}.jpg")
    output_label_path = os.path.join(train_output_dir, f"{img_name}.txt")
    
    # Check files exist
    if not os.path.exists(label_path):
        continue
    
    if not os.path.exists(depth_path):
        print(f"Missing depth: {depth_path}")
        skipped_count += 1
        # Copy original label without depth
        with open(label_path, 'r') as f:
            lines = f.readlines()
        with open(output_label_path, 'w') as f:
            for line in lines:
                line = line.strip()
                if line:
                    f.write(f"{line} -1.000\n")
        continue
    
    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        skipped_count += 1
        continue
    
    # Load depth map
    try:
        depth_map = np.load(depth_path)
        if depth_map is None:
            print(f"Error loading depth {depth_path}")
            skipped_count += 1
            continue
        
        depth_map = depth_map.astype(np.float32)
            
    except Exception as e:
        print(f"Error loading depth {depth_path}: {e}")
        skipped_count += 1
        continue
    
    # Get image dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image {image_path}")
        skipped_count += 1
        continue
    
    height, width = img.shape[:2]
    
    # Process label file
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # YOLO format: class_id x_center y_center box_width box_height
        parts = line.split()
        if len(parts) < 5:
            continue
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
        except (ValueError, IndexError):
            continue
        
        depth_value = get_depth_from_bbox(depth_map, (x_center, y_center, box_width, box_height), width, height)
        
        # Append depth (space-separated after YOLO coordinates)
        modified_lines.append(f"{line} {depth_value:.3f}\n")
    
    # Save modified labels
    with open(output_label_path, "w") as f:
        f.writelines(modified_lines)
    
    processed_count += 1
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{len(train_files)} files")

print(f"\nCompleted! Processed {processed_count} files, skipped {skipped_count} files")
print(f"Output directory: {train_output_dir}")
