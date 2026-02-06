"""
Convert VisDrone detection dataset to TFRecord for object_detection.
Based on KITTI converter but adapted for VisDrone format.
"""

from __future__ import absolute_import, division, print_function

import hashlib
import io
import os
import random

import numpy as np
import PIL.Image as pil
import tensorflow as tf
from object_detection.utils import dataset_util

# VisDrone class mapping
VISDRONE_CLASSES = {
    1: 'pedestrian',
    2: 'people', 
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor'
}

def label_map_extractor(label_map_path):
    """Extract a dictionary with class labels and IDs from txt file"""
    ids = []
    names = []
    label_map = {}
    with open(label_map_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "name" in line:
                names.append(line.split(":")[1].strip().strip("'\""))
            elif "id" in line:
                ids.append(int(line.split(":")[1].strip()))
    for i in range(len(ids)):
        label_map[names[i]] = ids[i]
    return label_map


def read_visdrone_annotation(filename):
    """
    Reads a VisDrone annotation file.
    
    Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    Args:
        filename: path to the annotation txt file
        
    Returns:
        anno: Dictionary with converted annotation information
    """
    if not os.path.exists(filename):
        # Return empty annotation for missing files
        return {
            'bbox_left': np.array([]),
            'bbox_top': np.array([]),
            'bbox_width': np.array([]),
            'bbox_height': np.array([]),
            'score': np.array([]),
            'object_category': np.array([]),
            'truncation': np.array([]),
            'occlusion': np.array([])
        }
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        # Return empty annotation for empty files
        return {
            'bbox_left': np.array([]),
            'bbox_top': np.array([]),
            'bbox_width': np.array([]),
            'bbox_height': np.array([]),
            'score': np.array([]),
            'object_category': np.array([]),
            'truncation': np.array([]),
            'occlusion': np.array([])
        }
    
    # Parse each line
    annotations = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 8:
            annotations.append([float(x) for x in parts[:8]])
    
    if not annotations:
        # Return empty annotation
        return {
            'bbox_left': np.array([]),
            'bbox_top': np.array([]),
            'bbox_width': np.array([]),
            'bbox_height': np.array([]),
            'score': np.array([]),
            'object_category': np.array([]),
            'truncation': np.array([]),
            'occlusion': np.array([])
        }
    
    annotations = np.array(annotations)
    
    anno = {
        'bbox_left': annotations[:, 0],
        'bbox_top': annotations[:, 1], 
        'bbox_width': annotations[:, 2],
        'bbox_height': annotations[:, 3],
        'score': annotations[:, 4],
        'object_category': annotations[:, 5].astype(int),
        'truncation': annotations[:, 6].astype(int),
        'occlusion': annotations[:, 7].astype(int)
    }
    
    return anno


def filter_visdrone_annotations(annotations, classes_to_use):
    """
    Filter annotations to keep only specified classes and valid boxes.
    
    Args:
        annotations: Dictionary from read_visdrone_annotation
        classes_to_use: List of class names to keep
        
    Returns:
        Filtered annotations dictionary
    """
    if len(annotations['object_category']) == 0:
        return annotations
    
    # Map class names to VisDrone category IDs
    name_to_id = {v: k for k, v in VISDRONE_CLASSES.items()}
    valid_category_ids = [name_to_id[cls] for cls in classes_to_use if cls in name_to_id]
    
    # Filter by class
    valid_mask = np.isin(annotations['object_category'], valid_category_ids)
    
    # Filter by score (keep only score > 0 for groundtruth)
    score_mask = annotations['score'] > 0
    
    # Filter by valid bounding box dimensions
    bbox_mask = (annotations['bbox_width'] > 0) & (annotations['bbox_height'] > 0)
    
    # Combine all filters
    final_mask = valid_mask & score_mask & bbox_mask
    
    # Apply filter to all fields
    filtered_anno = {}
    for key in annotations.keys():
        filtered_anno[key] = annotations[key][final_mask]
    
    return filtered_anno


def prepare_visdrone_example(image_path, annotations, label_map_dict, depth_path=None, image_id=0):
    """
    Converts VisDrone annotations to tf.Example proto.
    
    Args:
        image_path: Path to the image file
        annotations: Dictionary with VisDrone annotations
        label_map_dict: Mapping from class names to label IDs
        depth_path: Optional path to depth file
        
    Returns:
        tf.Example proto
    """
    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_image = fid.read()
    
    # Handle depth if provided
    if depth_path is not None and os.path.exists(depth_path):
        depth_array = np.load(depth_path)
        encoded_depth_bytes = depth_array.tobytes()
    else:
        encoded_depth_bytes = None
        depth_array = None
    
    # Load image to get dimensions
    encoded_image_io = io.BytesIO(encoded_image)
    image = pil.open(encoded_image_io)
    image = np.asarray(image)
    width = int(image.shape[1])
    height = int(image.shape[0])
    
    # Generate key
    key = hashlib.sha256(encoded_image).hexdigest()
    
    # Convert VisDrone bbox format to normalized coordinates
    if len(annotations['bbox_left']) > 0:
        # VisDrone: left, top, width, height -> normalized xmin, ymin, xmax, ymax
        xmin_norm = annotations['bbox_left'] / float(width)
        ymin_norm = annotations['bbox_top'] / float(height)
        xmax_norm = (annotations['bbox_left'] + annotations['bbox_width']) / float(width)
        ymax_norm = (annotations['bbox_top'] + annotations['bbox_height']) / float(height)
        
        # Clip to valid range [0, 1]
        xmin_norm = np.clip(xmin_norm, 0.0, 1.0)
        ymin_norm = np.clip(ymin_norm, 0.0, 1.0)
        xmax_norm = np.clip(xmax_norm, 0.0, 1.0)
        ymax_norm = np.clip(ymax_norm, 0.0, 1.0)
        
        # Convert category IDs to class names
        class_names = [VISDRONE_CLASSES[cat_id] for cat_id in annotations['object_category']]
        
        # Set all objects as not difficult
        difficult_obj = [0] * len(class_names)
        
    else:
        # No annotations
        xmin_norm = np.array([])
        ymin_norm = np.array([])
        xmax_norm = np.array([])
        ymax_norm = np.array([])
        class_names = []
        difficult_obj = []

    # Determine image format
    image_format = image_path.split('.')[-1].lower()
    if image_format == 'jpg':
        image_format = 'jpeg'
    
    # Create feature dictionary
    features_dict = {
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(os.path.basename(image_path).encode("utf8")),
        "image/source_id": dataset_util.bytes_feature(
            str(image_id).encode("utf8")
        ),
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
    
    # Add depth information if available
    if encoded_depth_bytes is not None:
        features_dict["image/encoded_depth"] = dataset_util.bytes_feature(encoded_depth_bytes)
        features_dict["image/depth_shape"] = dataset_util.int64_list_feature(depth_array.shape)
    
    example = tf.train.Example(features=tf.train.Features(feature=features_dict))
    return example


def convert_visdrone_to_tfrecords(data_dir, output_path, classes_to_use, label_map_path, 
                                 train_images_txt=None, include_depth=False):
    """
    Convert VisDrone dataset to TFRecords.
    
    Args:
        data_dir: Path to VisDrone dataset root directory
        output_path: Output path prefix for TFRecord files  
        classes_to_use: List of class names to include
        label_map_path: Path to label map file
        train_images_txt: Optional path to txt file with train image names (one per line)
        include_depth: Whether to include depth information
    """
    
    label_map_dict = label_map_extractor(label_map_path)
    
    # VisDrone has separate train/val directories
    train_images_dir = os.path.join(data_dir, "VisDrone2019-DET-train", "images")
    train_annotations_dir = os.path.join(data_dir, "VisDrone2019-DET-train", "annotations")
    val_images_dir = os.path.join(data_dir, "VisDrone2019-DET-val", "images") 
    val_annotations_dir = os.path.join(data_dir, "VisDrone2019-DET-val", "annotations")
    
    # Check directories exist
    for dir_path, name in [(train_images_dir, "train images"), 
                          (train_annotations_dir, "train annotations"),
                          (val_images_dir, "val images"), 
                          (val_annotations_dir, "val annotations")]:
        if not os.path.exists(dir_path):
            raise ValueError(f"{name} directory not found: {dir_path}")
    
    # Get train files - from txt file if provided, otherwise all files in train directory
    if train_images_txt and os.path.exists(train_images_txt):
        with open(train_images_txt, 'r') as f:
            train_files = [line.strip().split("/")[-1] for line in f.readlines() if line.strip()]
        print(f"Loaded {len(train_files)} training images from {train_images_txt}")
    else:
        train_files = [f for f in os.listdir(train_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        train_files = sorted(train_files)
        print(f"Found {len(train_files)} training images in {train_images_dir}")
    
    # Get ALL val files from directory
    val_files = [f for f in os.listdir(val_images_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    val_files = sorted(val_files)
    print(f"Found {len(val_files)} validation images in {val_images_dir}")
    
    print(f"Final split - Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create TFRecord writers
    train_writer = tf.io.TFRecordWriter(f"{output_path}_train.tfrecord")
    val_writer = tf.io.TFRecordWriter(f"{output_path}_val.tfrecord")
    
    def process_split(files, writer, split_name, images_dir, annotations_dir):
        count = 0
        for img_file in files:
            img_name_without_ext = os.path.splitext(img_file)[0]
            
            # Paths
            image_path = os.path.join(images_dir, img_file)
            annotation_path = os.path.join(annotations_dir, f"{img_name_without_ext}.txt")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            if include_depth:
                depth_path = os.path.join(os.path.dirname(images_dir), "depth", f"{img_name_without_ext}.npy")
            else:
                depth_path = None
            
            # Read and filter annotations
            annotations = read_visdrone_annotation(annotation_path)
            filtered_annotations = filter_visdrone_annotations(annotations, classes_to_use)
            
            # Create example
            example = prepare_visdrone_example(image_path, filtered_annotations, 
                                             label_map_dict, depth_path, count)
            
            writer.write(example.SerializeToString())
            count += 1
            
            if count % 1000 == 0:
                print(f"Processed {count} {split_name} images")
        
        return count
    
    # Process train split (from txt file or all train images)
    train_count = process_split(train_files, train_writer, "train", 
                               train_images_dir, train_annotations_dir)
    
    # Process val split (all images in val folder)
    val_count = process_split(val_files, val_writer, "val", 
                             val_images_dir, val_annotations_dir)
    
    # Close writers
    train_writer.close()
    val_writer.close()
    
    print("Finished creating TFRecords")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")


if __name__ == "__main__":
    # Configuration
    data_dir = "/path/to/data/DepthPrior/datasets/visdrone"  # Update this path
    output_path = "/path/to/data/DepthPrior/datasets/visdrone/tfrecords/"  # Update this path
    label_map_path = "/path/to/data/DepthPrior/datasets/visdrone/visdrone.pbtxt"  # Update this path
    train_images_txt = None #"/path/to/data/DepthPrior/datasets/visdrone/VisDrone2019-DET-train/image_subset_10.txt"  # Optional, set to None if not using a subset
    classes_to_use = [
        'pedestrian',
        'people', 
        'bicycle',
        'car',
        'van',
        'truck',
        'tricycle',
        'awning-tricycle',
        'bus',
        'motor'
    ]
    
    convert_visdrone_to_tfrecords(
        data_dir=data_dir,
        output_path=output_path,
        classes_to_use=classes_to_use,
        label_map_path=label_map_path,
        train_images_txt=train_images_txt,  # Set path to your txt file or None for all train images
        include_depth=False  # Set to True if you have depth data
    )
