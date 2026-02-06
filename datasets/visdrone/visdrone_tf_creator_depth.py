"""
Convert VisDrone detection dataset to TFRecord with depth support.
"""

from __future__ import absolute_import, division, print_function

import hashlib
import io
import os

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


def read_visdrone_annotation(filename):
    """
    Reads a VisDrone annotation file with optional depth.
    Format: left,top,width,height,score,category,truncation,occlusion [depth]
    """
    if not os.path.exists(filename):
        return {
            'bbox_left': np.array([]),
            'bbox_top': np.array([]),
            'bbox_width': np.array([]),
            'bbox_height': np.array([]),
            'score': np.array([]),
            'object_category': np.array([]),
            'truncation': np.array([]),
            'occlusion': np.array([]),
            'depth': np.array([])
        }
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Split by comma first
        parts = line.split(',')
        if len(parts) < 8:
            continue
        
        # Check for depth (space-separated after last comma field)
        depth_parts = parts[-1].split()
        if len(depth_parts) == 2:
            parts[-1] = depth_parts[0]  # occlusion
            depth = float(depth_parts[1])
        else:
            depth = -1.0
        
        try:
            annotation = [float(x) for x in parts[:8]]
            annotation.append(depth)
            annotations.append(annotation)
        except ValueError:
            continue
    
    if not annotations:
        return {
            'bbox_left': np.array([]),
            'bbox_top': np.array([]),
            'bbox_width': np.array([]),
            'bbox_height': np.array([]),
            'score': np.array([]),
            'object_category': np.array([]),
            'truncation': np.array([]),
            'occlusion': np.array([]),
            'depth': np.array([])
        }
    
    annotations = np.array(annotations)
    
    return {
        'bbox_left': annotations[:, 0],
        'bbox_top': annotations[:, 1], 
        'bbox_width': annotations[:, 2],
        'bbox_height': annotations[:, 3],
        'score': annotations[:, 4],
        'object_category': annotations[:, 5].astype(int),
        'truncation': annotations[:, 6].astype(int),
        'occlusion': annotations[:, 7].astype(int),
        'depth': annotations[:, 8]
    }


def filter_annotations(annotations, classes_to_use):
    """Filter annotations to keep only specified classes and valid boxes."""
    if len(annotations['object_category']) == 0:
        return annotations
    
    name_to_id = {v: k for k, v in VISDRONE_CLASSES.items()}
    valid_category_ids = [name_to_id[cls] for cls in classes_to_use if cls in name_to_id]
    
    valid_mask = np.isin(annotations['object_category'], valid_category_ids)
    score_mask = annotations['score'] > 0
    bbox_mask = (annotations['bbox_width'] > 0) & (annotations['bbox_height'] > 0)
    
    final_mask = valid_mask & score_mask & bbox_mask
    
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
    if depth_map_path is not None and os.path.exists(depth_map_path):
        try:
            depth_map = np.load(depth_map_path).astype(np.float32)
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
    
    # Convert to normalized coordinates
    if len(annotations['bbox_left']) > 0:
        xmin_norm = np.clip(annotations['bbox_left'] / float(width), 0.0, 1.0)
        ymin_norm = np.clip(annotations['bbox_top'] / float(height), 0.0, 1.0)
        xmax_norm = np.clip((annotations['bbox_left'] + annotations['bbox_width']) / float(width), 0.0, 1.0)
        ymax_norm = np.clip((annotations['bbox_top'] + annotations['bbox_height']) / float(height), 0.0, 1.0)
        
        class_names = [VISDRONE_CLASSES[cat_id] for cat_id in annotations['object_category']]
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
    
    # Add depth if available
    if depths:
        feature_dict["image/object/pseudo_score"] = (dataset_util.float_list_feature(depths),
        )
    
    if depth_bytes is not None:
        feature_dict["image/encoded_depth"] = dataset_util.bytes_feature(depth_bytes)
        feature_dict["image/depth_shape"] = dataset_util.int64_list_feature(depth_map_shape)
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def visdrone_active_tfrecords(
    data_dir,
    output_path,
    classes_to_use,
    label_map_path,
    train_indices,
    current_iteration,
    train=True,
    pseudo=None,
):
    """
    Create TFRecords from VisDrone dataset.
    
    Args:
        data_dir: Directory with images
        output_path: Output path for TFRecord file
        classes_to_use: List of class names to include
        label_map_path: Path to label map .pbtxt file
        train_indices: List of image filenames to process
        current_iteration: Suffix for output filename
        train: If True, create train tfrecord, else val
        pseudo: Directory with annotations (with or without depth)
    """
    label_map_dict = label_map_extractor(label_map_path)
    
    # Determine annotation directory
    if pseudo is not None:
        annotation_dir = pseudo
    else:
        annotation_dir = os.path.join(os.path.dirname(data_dir), "annotations")
    
    # Check for depth directory
    depth_dir = annotation_dir.replace("annotations_with_depth", "depth_anything_predictions")
    if not os.path.exists(depth_dir):
        depth_dir = None
    
    # Create TFRecord writer
    if train:
        output_file = f"{output_path}_train_{current_iteration}.tfrecord"
    else:
        output_file = f"{output_path}_val_{current_iteration}.tfrecord"
    
    print(f"Creating TFRecord: {output_file}")
    print(f"Image directory: {data_dir}")
    print(f"Annotation directory: {annotation_dir}")
    print(f"Depth directory: {depth_dir}")
    
    writer = tf.io.TFRecordWriter(output_file)
    
    count = 0
    for idx, img_name in enumerate(train_indices):
        img_name_without_ext = os.path.splitext(img_name)[0]
        image_path = os.path.join(data_dir, img_name)
        annotation_path = os.path.join(annotation_dir, f"{img_name_without_ext}.txt")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get depth map path if available
        depth_map_path = None
        if depth_dir is not None:
            depth_map_path = os.path.join(depth_dir, f"{img_name_without_ext}_depth.npy")
            if not os.path.exists(depth_map_path):
                depth_map_path = None
        # Read and filter annotations
        annotations = read_visdrone_annotation(annotation_path)
        filtered_annotations = filter_annotations(annotations, classes_to_use)
        
        # Create example
        example = prepare_example(
            image_path, filtered_annotations, label_map_dict, depth_map_path, idx
        )
        
        writer.write(example.SerializeToString())
        count += 1
        
        if count % 500 == 0:
            print(f"Processed {count}/{len(train_indices)} images")
    
    writer.close()
    print(f"\nFinished creating TFRecord!")
    print(f"Total images: {count}")
    print(f"Output: {output_file}")
    
    return count


if __name__ == "__main__":
    data_dir = "/path/to/data/DepthPrior/datasets/visdrone"
    output_path = "/path/to/data/DepthPrior/datasets/visdrone/depth_anything_predictions/"
    label_map_path = "/path/to/data/DepthPrior/datasets/visdrone/visdrone.pbtxt"
    
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
    # Get all training images
    train_images_dir = os.path.join(data_dir, "VisDrone2019-DET-train/images")
    train_files = sorted([f for f in os.listdir(train_images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Create TFRecord with depth
    visdrone_active_tfrecords(
        train_images_dir,
        output_path,
        classes_to_use,
        label_map_path,
        train_files,
        "depth",
        train=True,
        pseudo=output_path,
    )

print("\nAll done!")
