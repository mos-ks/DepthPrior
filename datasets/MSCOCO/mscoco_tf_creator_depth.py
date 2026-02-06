"""
Convert VisDrone detection dataset to TFRecord with depth support.
"""

from __future__ import absolute_import, division, print_function

import hashlib
import io
import json
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf
from object_detection.utils import dataset_util

# COCO class mapping (80 classes)
COCO_CLASSES = {
    1: 'person',
    2: 'bicycle', 
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
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

def load_coco_annotations(annotation_file):
    """
    Load COCO annotations from JSON file.
    
    Args:
        annotation_file: Path to COCO annotation JSON file
        
    Returns:
        Dictionary with image info and annotations organized by image_id
    """
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from category_id to category name
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create mapping from image_id to image info
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    return {
        'categories': categories,
        'images': images,
        'annotations': annotations_by_image
    }

def filter_coco_annotations(annotations, categories, classes_to_use):
    """
    Filter COCO annotations to keep only specified classes and valid boxes.
    
    Args:
        annotations: List of COCO annotation dictionaries for an image
        categories: Dictionary mapping category_id to category name
        classes_to_use: List of class names to keep
        
    Returns:
        List of filtered annotations
    """
    if not annotations:
        return []
    
    filtered_annotations = []
    
    for ann in annotations:
        # Get category name
        category_name = categories.get(ann['category_id'])
        
        # Check if class should be included
        if category_name not in classes_to_use:
            continue
            
        # Check if annotation has valid bbox
        bbox = ann.get('bbox', [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
            
        # Check if annotation is not crowd (iscrowd = 0)
        if ann.get('iscrowd', 0) == 1:
            continue
            
        filtered_annotations.append(ann)
    
    return filtered_annotations

def prepare_coco_example(image_info, annotations, categories, label_map_dict, 
                        images_dir, count, depth_path=None):
    """
    Converts COCO annotations to tf.Example proto.
    
    Args:
        image_info: COCO image info dictionary
        annotations: List of COCO annotation dictionaries
        categories: Dictionary mapping category_id to category name
        label_map_dict: Mapping from class names to label IDs
        images_dir: Directory containing images
        depth_path: Optional path to depth file
        
    Returns:
        tf.Example proto
    """
    image_path = os.path.join(images_dir, image_info['file_name'])
    
    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_image = fid.read()
    
    # Handle depth if provided
    depth_bytes = None
    depth_shape = None
    if depth_path is not None and os.path.exists(depth_path):
        try:
            depth_map = np.load(depth_path).astype(np.float32)
            depth_bytes = depth_map.tobytes()
            depth_shape = depth_map.shape
        except Exception as e:
            print(f"Error loading depth map {depth_path}: {e}")
    
    # Get image dimensions from COCO data
    width = int(image_info['width'])
    height = int(image_info['height'])
    
    # Generate key
    key = hashlib.sha256(encoded_image).hexdigest()
    
    # Process annotations
    if annotations:
        xmin_norm = []
        ymin_norm = []
        xmax_norm = []
        ymax_norm = []
        class_names = []
        difficult_obj = []
        depths = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height] (top-left corner)
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to normalized coordinates
            xmin = x / float(width)
            ymin = y / float(height)
            xmax = (x + w) / float(width)
            ymax = (y + h) / float(height)
            
            # Clip to valid range [0, 1]
            xmin = max(0.0, min(1.0, xmin))
            ymin = max(0.0, min(1.0, ymin))
            xmax = max(0.0, min(1.0, xmax))
            ymax = max(0.0, min(1.0, ymax))
            
            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue
                
            xmin_norm.append(xmin)
            ymin_norm.append(ymin)
            xmax_norm.append(xmax)
            ymax_norm.append(ymax)
            
            # Get class name
            category_name = categories[ann['category_id']]
            class_names.append(category_name)
            
            # Set difficulty (COCO doesn't have difficulty, so set to 0)
            difficult_obj.append(0)
            
            # Compute depth for this bbox
            if 'depth' in ann:
                depths.append(ann['depth'])
            elif depth_bytes is not None:
                depth_map = np.frombuffer(depth_bytes, dtype=np.float32).reshape(depth_shape)
                depth_value = get_depth_from_bbox(depth_map, bbox)
                depths.append(depth_value)
            else:
                depths.append(-1.0)
        
        # Convert to numpy arrays
        xmin_norm = np.array(xmin_norm)
        ymin_norm = np.array(ymin_norm)
        xmax_norm = np.array(xmax_norm)
        ymax_norm = np.array(ymax_norm)
        
    else:
        # No annotations
        xmin_norm = np.array([])
        ymin_norm = np.array([])
        xmax_norm = np.array([])
        ymax_norm = np.array([])
        class_names = []
        difficult_obj = []
        depths = []
    
    # Determine image format
    image_format = image_info['file_name'].split('.')[-1].lower()
    if image_format == 'jpg':
        image_format = 'jpeg'
    
    # Create feature dictionary
    features_dict = {
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(image_info['file_name'].encode("utf8")),
        "image/source_id": dataset_util.bytes_feature(
            str(count).encode("utf8")  # Use sequential count instead of COCO ID
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
    
    # Add pseudo_score if depths available
    if depths:
        features_dict["image/object/pseudo_score"] = dataset_util.float_list_feature(depths)
    
    # Add depth information if available
    if depth_bytes is not None:
        features_dict["image/encoded_depth"] = dataset_util.bytes_feature(depth_bytes)
        features_dict["image/depth_shape"] = dataset_util.int64_list_feature(depth_shape)
    example = tf.train.Example(features=tf.train.Features(feature=features_dict))
    return example

def mscoco_active_tfrecords(
    data_dir,
    output_path,
    classes_to_use,
    label_map_path,
    train_subset_txt=None,
    train=True,
):
    """
    Create TFRecords from MSCOCO dataset with depth.
    
    Args:
        data_dir: Path to COCO dataset root directory
        output_path: Output path prefix for TFRecord file
        classes_to_use: List of class names to include
        label_map_path: Path to label map file
        train_subset_txt: Optional path to txt file with train image names subset
        train: If True, create train tfrecord, else val
    """
    
    label_map_dict = label_map_extractor(label_map_path)
    
    # COCO directory structure
    if train:
        images_dir = os.path.join(data_dir, "train2017/images")
        annotations_file = os.path.join(data_dir, "annotations", "instances_train2017_with_depth.json")
        if not os.path.exists(annotations_file):
            annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
        split_name = "train"
    else:
        images_dir = os.path.join(data_dir, "val2017/images")
        annotations_file = os.path.join(data_dir, "annotations", "instances_val2017_with_depth.json")
        if not os.path.exists(annotations_file):
            annotations_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
        split_name = "val"
    
    # Check files exist
    if not os.path.exists(images_dir):
        raise ValueError(f"{split_name} images not found: {images_dir}")
    if not os.path.exists(annotations_file):
        raise ValueError(f"{split_name} annotations not found: {annotations_file}")
    
    # Load annotations
    print(f"Loading {split_name} COCO annotations...")
    data = load_coco_annotations(annotations_file)
    
    # Get images
    if train and train_subset_txt and os.path.exists(train_subset_txt):
        with open(train_subset_txt, 'r') as f:
            subset_filenames = set(os.path.basename(line.strip()) for line in f.readlines() if line.strip())
        images_dict = {img_id: img_info for img_id, img_info in data['images'].items() if img_info['file_name'] in subset_filenames}
        print(f"Using {len(images_dict)} {split_name} images from subset file")
    else:
        images_dict = data['images']
        print(f"Using all {len(images_dict)} {split_name} images")
    
    # Create TFRecord writer
    output_file = f"{output_path}_{split_name}_depth.tfrecord"
    print(f"Creating TFRecord: {output_file}")
    print(f"Image directory: {images_dir}")
    print(f"Depth directory: {os.path.join(data_dir, 'depth_anything_predictions')}")
    
    writer = tf.io.TFRecordWriter(output_file)
    
    count = 0
    for image_id, image_info in images_dict.items():
        
        # Get annotations for this image
        annotations = data['annotations'].get(image_id, [])
        
        # Filter annotations
        filtered_annotations = filter_coco_annotations(annotations, data['categories'], classes_to_use)
        
        # Prepare depth path
        img_name_without_ext = os.path.splitext(image_info['file_name'])[0]
        depth_path = os.path.join(data_dir, "depth_anything_predictions", f"{img_name_without_ext}_depth.npy")
        # Create example
        try:
            example = prepare_coco_example(image_info, filtered_annotations, data['categories'],
                                         label_map_dict, images_dir, count, depth_path)
            
            writer.write(example.SerializeToString())
            count += 1
            
            if count % 1000 == 0:
                print(f"Processed {count} {split_name} images")
                
        except Exception as e:
            print(f"Error processing {image_info['file_name']}: {e}")
            continue
    
    writer.close()
    print(f"\nFinished creating TFRecord!")
    print(f"Total images: {count}")
    print(f"Output: {output_file}")
    
    return count

if __name__ == "__main__":
    # Configuration
    data_dir = "/path/to/data/DepthPrior/datasets/MSCOCO"
    output_path = "/path/to/data/DepthPrior/datasets/MSCOCO/depth_anything_predictions/"
    label_map_path = "/path/to/data/DepthPrior/datasets/MSCOCO/coco.pbtxt"
    train_subset_txt = None  # Optional: None or path to subset file
    
    # All 80 COCO classes
    classes_to_use = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Create train TFRecord
    mscoco_active_tfrecords(
        data_dir=data_dir,
        output_path=output_path,
        classes_to_use=classes_to_use,
        label_map_path=label_map_path,
        train_subset_txt=train_subset_txt,
        train=True,
    )
    
    # # Create val TFRecord
    # mscoco_active_tfrecords(
    #     data_dir=data_dir,
    #     output_path=output_path,
    #     classes_to_use=classes_to_use,
    #     label_map_path=label_map_path,
    #     train_subset_txt=None,  # No subset for val
    #     train=False,
    # )

print("\nAll done!")