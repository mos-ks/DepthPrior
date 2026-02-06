import os

import numpy as np

general_path = "/path/to/data/DepthPrior/"

# List of indices for processing
indices_i = sorted([int(f.split("_")[0]) for f in os.listdir(general_path+"/datasets/KITTI/depth_anything_predictions/") if f.endswith(".npy")])
def get_depth_from_bbox(depth_map, bbox):
    """Extract depth value from bounding box region."""
    x1, y1, x2, y2 = map(int, bbox)
    depth_values = depth_map[y1:y2, x1:x2]
    return np.mean(depth_values) if depth_values.size > 0 else -1  # Return -1 if no valid depth values

for i in indices_i:
    depth_map_path = general_path + f"/datasets/KITTI/depth_anything_predictions/{str(i).zfill(6)}_depth.npy"
    label_path = general_path + f"/datasets/KITTI/training/label_2/{str(i).zfill(6)}.txt"
    output_label_path = general_path + f"/datasets/KITTI/depth_anything_predictions/{str(i).zfill(6)}.txt"
    
    # Load depth map
    if not os.path.exists(depth_map_path) or not os.path.exists(label_path):
        continue
    
    depth_map = np.load(depth_map_path).astype(np.float32)
    
    # Process label file
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue  # Skip invalid lines
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(float, parts[4:8])
        depth_value = get_depth_from_bbox(depth_map, (x1, y1, x2, y2))
        
        # Append depth to the label line
        modified_lines.append(" ".join(parts) + f" {depth_value:.3f}\n")
    
    # Save modified labels
    with open(output_label_path, "w") as f:
        f.writelines(modified_lines)
    
    print(f"Processed: {output_label_path}")

#/path/to/data/DepthPrior//datasets/KITTI/depth_anything_predictions/007479_label_depth.txt
import os

# Define the base directory
output_label_path = os.path.join(general_path, "datasets/KITTI/depth_anything_predictions")

# Iterate through all files in the directory
for filename in os.listdir(output_label_path):
    if "_label_depth" in filename:
        new_filename = filename.replace("_label_depth", "")
        old_file = os.path.join(output_label_path, filename)
        new_file = os.path.join(output_label_path, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

print("Renaming completed.")
# Renamed: 006236_label_depth.txt -> 006236.txt
os.chdir(general_path)
from datasets.KITTI.kitti_tf_creator import kitti_active_tfrecords

classes_to_use = [
                "car",
                "van",
                "truck",
                "pedestrian",
                "person_sitting",
                "cyclist",
                "tram",
            ]  # Not capital
kitti_active_tfrecords(
    general_path + "datasets/KITTI/training",
    general_path + "datasets/KITTI/depth_anything_predictions",
    classes_to_use,
    general_path + "datasets/KITTI/kitti.pbtxt",
    indices_i,
    "depth",
    train=True,
    pseudo=general_path + "datasets/KITTI/depth_anything_predictions",
)

# # Create TFRecord for 10% of KITTI data as in _train_init_V0.txt
# with open(general_path + "datasets/KITTI/stac/num_labeled_10/V0/_train_init_V0.txt", "r") as f:
#     indices_10 = [int(x) for x in f.read().strip().split()]

# kitti_active_tfrecords(
#     general_path + "datasets/KITTI/training",
#     general_path + "datasets/KITTI/depth_anything_predictions",
#     classes_to_use,
#     general_path + "datasets/KITTI/kitti.pbtxt",
#     indices_10,
#     "depth10",
#     train=True,
#     pseudo=general_path + "datasets/KITTI/depth_anything_predictions",
# )

# def kitti_active_tfrecords(
#     data_dir,
#     output_path,
#     classes_to_use,
#     label_map_path,
#     train_indices,
#     current_iteration,
#     train=True,
#     pseudo=None,