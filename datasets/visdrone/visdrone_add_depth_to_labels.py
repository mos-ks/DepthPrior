#!/usr/bin/env python3
"""
Process VisDrone annotations with depth information and create TFRecords.
"""

import os

import numpy as np

general_path = "/path/to/data/DepthPrior/"

def get_depth_from_bbox(depth_map, bbox):
    """Extract depth value from bounding box region."""
    left, top, width, height = bbox
    x1, y1 = int(left), int(top)
    x2, y2 = int(left + width), int(top + height)
    
    # Clip to valid range
    h, w = depth_map.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    
    if x2 <= x1 or y2 <= y1:
        return -1.0
    
    depth_values = depth_map[y1:y2, x1:x2]
    return float(np.mean(depth_values)) if depth_values.size > 0 else -1.0


# Directories
train_annotations_dir = os.path.join(general_path, "datasets/visdrone/VisDrone2019-DET-train/annotations")
train_depth_dir = os.path.join(general_path, "datasets/visdrone/depth_anything_predictions")
train_output_dir = train_depth_dir#os.path.join(general_path, "datasets/visdrone/annotations_with_depth")

# Create output directory
os.makedirs(train_output_dir, exist_ok=True)

# Get list of depth files
depth_files = [f for f in os.listdir(train_depth_dir) if f.endswith("_depth.npy")]
train_files = sorted([f.replace("_depth.npy", "") for f in depth_files])

print(f"Processing {len(train_files)} training annotations with depth...")

processed_count = 0
for img_name in train_files:
    depth_map_path = os.path.join(train_depth_dir, f"{img_name}_depth.npy")
    label_path = os.path.join(train_annotations_dir, f"{img_name}.txt")
    output_label_path = os.path.join(train_output_dir, f"{img_name}.txt")
    
    # Check files exist
    if not os.path.exists(depth_map_path):
        print(f"Missing depth: {depth_map_path}")
        continue
    if not os.path.exists(label_path):
        print(f"Missing annotation: {label_path}")
        continue
    
    # Load depth map
    try:
        depth_map = np.load(depth_map_path).astype(np.float32)
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]  # Take first channel if needed
    except Exception as e:
        print(f"Error loading depth {depth_map_path}: {e}")
        continue
    
    # Process label file
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # VisDrone format: left,top,width,height,score,category,truncation,occlusion
        parts = line.split(',')
        if len(parts) < 8:
            continue
        
        try:
            left = float(parts[0])
            top = float(parts[1])
            width = float(parts[2])
            height = float(parts[3])
        except (ValueError, IndexError):
            continue
        
        depth_value = get_depth_from_bbox(depth_map, (left, top, width, height))
        
        # Append depth (space-separated after comma-separated values)
        modified_lines.append(f"{line} {depth_value:.3f}\n")
    
    # Save modified labels
    with open(output_label_path, "w") as f:
        f.writelines(modified_lines)
    
    processed_count += 1
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{len(train_files)} files")

print(f"\nCompleted! Processed {processed_count} files")
print(f"Output directory: {train_output_dir}")
