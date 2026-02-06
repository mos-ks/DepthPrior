"""
Convert SUN RGB-D dataset to TFRecord with depth support.
Follows VisDrone structure.
"""

from __future__ import absolute_import, division, print_function

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import PIL.Image as pil
import tensorflow as tf
from object_detection.utils import dataset_util

# SUN RGB-D 2D class mapping
SUNRGBD2D_CLASSES = {
    'bed': 1, 'table': 2, 'sofa': 3, 'chair': 4, 'toilet': 5,
    'desk': 6, 'dresser': 7, 'night_stand': 8, 'bookshelf': 9, 'bathtub': 10,
    'box': 11, 'books': 12, 'bottle': 13, 'bag': 14, 'pillow': 15,
    'monitor': 16, 'television': 17, 'lamp': 18, 'garbage_bin': 19
}

def label_map_extractor(label_map_path):
    """Extract a dictionary with class labels and IDs from pbtxt file"""
    ids = []
    names = []
    with open(label_map_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "name" in line:
                names.append(line.split(":")[1].strip().strip("'\""))
            elif "id" in line:
                ids.append(int(line.split(":")[1].strip()))
    
    label_map = {}
    for i in range(len(ids)):
        label_map[names[i]] = ids[i]
    return label_map


def read_sunrgbd_annotation(filename):
    """
    Reads a SUN RGB-D YOLO annotation file with optional depth.
    Format: class_id x_center y_center box_width box_height [depth]
    """
    if not os.path.exists(filename):
        return {
            'class_ids': np.array([]),
            'x_centers': np.array([]),
            'y_centers': np.array([]),
            'box_widths': np.array([]),
            'box_heights': np.array([]),
            'depth': np.array([])
        }
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        # Check for depth (6th value)
        if len(parts) >= 6:
            depth = float(parts[5])
        else:
            depth = -1.0
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
            
            annotations.append([class_id, x_center, y_center, box_width, box_height, depth])
        except ValueError:
            continue
    
    if not annotations:
        return {
            'class_ids': np.array([]),
            'x_centers': np.array([]),
            'y_centers': np.array([]),
            'box_widths': np.array([]),
            'box_heights': np.array([]),
            'depth': np.array([])
        }
    
    annotations = np.array(annotations)
    
    return {
        'class_ids': annotations[:, 0].astype(int),
        'x_centers': annotations[:, 1],
        'y_centers': annotations[:, 2],
        'box_widths': annotations[:, 3],
        'box_heights': annotations[:, 4],
        'depth': annotations[:, 5]
    }


def filter_annotations(annotations, classes_to_use):
    """Filter annotations to keep only specified classes and valid boxes."""
    if len(annotations['class_ids']) == 0:
        return annotations
    
    # Get class names from IDs (YOLO uses 0-based indexing)
    id_to_name = {v-1: k for k, v in SUNRGBD2D_CLASSES.items()}
    
    # Filter valid classes
    valid_mask = np.array([
        id_to_name.get(int(class_id), None) in classes_to_use 
        for class_id in annotations['class_ids']
    ])
    
    # Filter valid bounding boxes
    bbox_mask = (annotations['box_widths'] > 0) & (annotations['box_heights'] > 0)
    
    final_mask = valid_mask & bbox_mask
    
    filtered_anno = {}
    for key in annotations.keys():
        filtered_anno[key] = annotations[key][final_mask]
    
    return filtered_anno


def prepare_example(image_path, annotations, label_map_dict, depth_map_path=None, image_id=0):
    """Create TFRecord example with optional depth."""
    
    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_image = fid.read()
    
    # Load full depth map if provided
    depth_bytes = None
    depth_map_shape = None
    if depth_map_path is not None and os.path.exists(depth_map_path):
        try:
            depth_map = np.load(depth_map_path)
            if depth_map is not None:
                depth_map = depth_map.astype(np.float32)
                depth_map_shape = depth_map.shape
                depth_bytes = depth_map.tobytes()
        except Exception as e:
            print(f"Error loading depth map {depth_map_path}: {e}")
    
    # Get image dimensions
    encoded_image_io = io.BytesIO(encoded_image)
    image = pil.open(encoded_image_io)
    image = np.asarray(image)
    height = int(image.shape[0])
    width = int(image.shape[1])
    
    key = hashlib.sha256(encoded_image).hexdigest()
    
    # Convert YOLO format to normalized bbox coordinates
    if len(annotations['class_ids']) > 0:
        # YOLO: x_center, y_center, width, height (normalized) -> xmin, ymin, xmax, ymax (normalized)
        xmin_norm = np.clip(annotations['x_centers'] - annotations['box_widths']/2, 0.0, 1.0)
        ymin_norm = np.clip(annotations['y_centers'] - annotations['box_heights']/2, 0.0, 1.0)
        xmax_norm = np.clip(annotations['x_centers'] + annotations['box_widths']/2, 0.0, 1.0)
        ymax_norm = np.clip(annotations['y_centers'] + annotations['box_heights']/2, 0.0, 1.0)
        
        # Get class names from YOLO class IDs
        id_to_name = {v-1: k for k, v in SUNRGBD2D_CLASSES.items()}
        class_names = [id_to_name[int(class_id)] for class_id in annotations['class_ids']]
        
        difficult_obj = [0] * len(class_names)
        depths = annotations['depth'].tolist()
        
    else:
        xmin_norm = np.array([])
        ymin_norm = np.array([])
        xmax_norm = np.array([])
        ymax_norm = np.array([])
        class_names = []
        difficult_obj = []
        depths = []

    image_format = image_path.split('.')[-1].lower()
    if image_format == 'jpg':
        image_format = 'jpeg'
    
    # Build feature dictionary
    feature_dict = {
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(os.path.basename(image_path).encode("utf8")),
        "image/source_id": dataset_util.bytes_feature(str(image_id).encode("utf8")),
        "image/key/sha256": dataset_util.bytes_feature(key.encode("utf8")),
        "image/encoded": dataset_util.bytes_feature(encoded_image),
        "image/format": dataset_util.bytes_feature(image_format.encode("utf8")),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmin_norm),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmax_norm),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymin_norm),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymax_norm),
        "image/object/class/text": dataset_util.bytes_list_feature(
            [x.encode("utf8") for x in class_names]
        ),
        "image/object/class/label": dataset_util.int64_list_feature(
            [label_map_dict[x] for x in class_names]
        ),
        "image/object/difficult": dataset_util.int64_list_feature(difficult_obj),
    }
    # Add depth as pseudo_score (following VisDrone pattern)
    if depths:
        feature_dict["image/object/pseudo_score"] = (dataset_util.float_list_feature(depths),
        )
    
    if depth_bytes is not None:
        feature_dict["image/encoded_depth"] = dataset_util.bytes_feature(depth_bytes)
        feature_dict["image/depth_shape"] = dataset_util.int64_list_feature(depth_map_shape)
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def sunrgbd_to_tfrecords(
    data_dir,
    output_path,
    classes_to_use,
    label_map_path,
    train_indices=None,
    current_iteration="depth",
    use_labels_with_depth=True,
):
    """
    Create TFRecords from SUN RGB-D dataset.
    
    Args:
        data_dir: Root directory of SUN RGB-D dataset
        output_path: Output path prefix for TFRecord files
        classes_to_use: List of class names to include
        label_map_path: Path to label map .pbtxt file
        train_indices: List of train image filenames (without extension)
        val_indices: List of val image filenames (without extension)
        current_iteration: Suffix for output filename
        use_labels_with_depth: Use labels with depth annotations
    """
    label_map_dict = label_map_extractor(label_map_path)
    data_dir = Path(data_dir)
    
    # Determine label directories
    if use_labels_with_depth:
        train_labels_dir = data_dir / "depth_anything_predictions"
    else:
        train_labels_dir = data_dir / "train" / "labels"
    
    train_images_dir = data_dir / "train" / "images"
    train_depth_dir = data_dir / "depth_anything_predictions"
    
    # Get file lists
    if train_indices is None:
        train_indices = [f.stem for f in train_labels_dir.glob("*.txt")]
    
    print(f"Train images: {len(train_indices)}")
    print(f"Using labels with depth: {use_labels_with_depth}")
    print(f"Train labels dir: {train_labels_dir}")
    # Create TFRecord writers
    train_output_file = f"{output_path}_train_{current_iteration}.tfrecord"
    
    train_writer = tf.io.TFRecordWriter(train_output_file)
    def process_split(indices, writer, split_name, images_dir, labels_dir, depth_dir):
        count = 0
        for idx, img_name in enumerate(indices):
            image_path = images_dir / f"{img_name}.jpg"
            label_path = labels_dir / f"{img_name}.txt"
            depth_path = depth_dir / f"{img_name}_depth.npy"
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            if not label_path.exists():
                print(f"Warning: Label not found: {label_path}")
                continue
            
            # Check for depth
            if not depth_path.exists():
                depth_path = None
            
            # Read and filter annotations
            annotations = read_sunrgbd_annotation(str(label_path))
            filtered_annotations = filter_annotations(annotations, classes_to_use)
            
            # Skip if no valid annotations
            if len(filtered_annotations['class_ids']) == 0:
                continue
            
            # Create example
            try:
                example = prepare_example(
                    str(image_path), filtered_annotations, label_map_dict, 
                    str(depth_path) if depth_path else None, idx
                )
                writer.write(example.SerializeToString())
                count += 1
                
                if count % 500 == 0:
                    print(f"Processed {count}/{len(indices)} {split_name} images")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return count
    
    # Process train split
    print("\nProcessing train split...")
    train_count = process_split(train_indices, train_writer, "train", 
                               train_images_dir, train_labels_dir, train_depth_dir)
    
    # Close writers
    train_writer.close()
    
    print(f"\n{'='*50}")
    print("Finished creating TFRecords!")
    print(f"Training images: {train_count}")
    print(f"Train output: {train_output_file}")
    print(f"{'='*50}")
    
    return train_count


if __name__ == "__main__":
    general_path = "/path/to/data/DepthPrior/"
    data_dir = os.path.join(general_path, "datasets/SUNRGBD")
    output_path = os.path.join(general_path, "datasets/SUNRGBD/depth_anything_predictions/")
    label_map_path = os.path.join(general_path, "datasets/SUNRGBD/sunrgbd2d.pbtxt")
    
    classes_to_use = [
        'bed', 'table', 'sofa', 'chair', 'toilet',
        'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub',
        'box', 'books', 'bottle', 'bag', 'pillow',
        'monitor', 'television', 'lamp', 'garbage_bin'
    ]
    
    # Create TFRecords with depth
    sunrgbd_to_tfrecords(
        data_dir=data_dir,
        output_path=output_path,
        classes_to_use=classes_to_use,
        label_map_path=label_map_path,
        train_indices=None,  # Use all files
        current_iteration="depth",
        use_labels_with_depth=True,
    )

print("\nAll done!")