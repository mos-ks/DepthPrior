#!/usr/bin/env python3
"""
Complete SUN RGB-D 2D dataset setup for object detection.
Downloads, converts to YOLO format, and creates TFRecords.
"""

import hashlib
import io
import json
import os
import pickle
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import tensorflow as tf
from object_detection.utils import dataset_util
from tqdm import tqdm

# SUN RGB-D 2D class mapping (19 object classes for detection)
SUNRGBD2D_CLASSES = {
    'bed': 1, 'table': 2, 'sofa': 3, 'chair': 4, 'toilet': 5,
    'desk': 6, 'dresser': 7, 'night_stand': 8, 'bookshelf': 9, 'bathtub': 10,
    'box': 11, 'books': 12, 'bottle': 13, 'bag': 14, 'pillow': 15,
    'monitor': 16, 'television': 17, 'lamp': 18, 'garbage_bin': 19
}

def download_sunrgbd2d(download_dir):
    """Download SUN RGB-D 2D dataset"""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("SUN RGB-D 2D Download Instructions:")
    print("=" * 50)
    print("1. Go to: http://rgbd.cs.princeton.edu/")
    print("2. Navigate to the SUN RGB-D section")
    print("3. Download the following files:")
    print("   - SUNRGBD.zip (RGB images and depth)")
    print("   - SUNRGBDMeta2DBB_v2.mat (2D bounding box annotations)")
    print("   - Or download from:")
    print("   - http://www.cs.princeton.edu/~shurans/SUNRGBDMeta2DBB_v2.mat")
    print(f"4. Place files in: {download_dir}")
    print("=" * 50)
    
    # Check if files exist
    required_files = [
        'SUNRGBD.zip',
        'SUNRGBDMeta2DBB_v2.mat'
    ]
    
    missing_files = []
    for file in required_files:
        if not (download_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        print("Please download them manually from the links above.")
        return False
    
    # Extract SUNRGBD.zip if not already extracted
    sunrgbd_dir = download_dir / "SUNRGBD"
    if not sunrgbd_dir.exists():
        print("Extracting SUNRGBD.zip...")
        with zipfile.ZipFile(download_dir / "SUNRGBD.zip", 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        print("Extraction complete!")
    
    return True

def load_sunrgbd2d_annotations(download_dir):
    """Load SUN RGB-D 2D annotations"""
    download_dir = Path(download_dir)
    
    # Load 2D bounding box annotations
    mat_file = download_dir / "SUNRGBDMeta2DBB_v2.mat"
    
    print(f"Loading 2D annotations from: {mat_file}")
    
    try:
        # Load the .mat file
        mat_data = sio.loadmat(str(mat_file))
        
        # Find the main data key (excluding internal matlab keys)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print("Available keys in .mat file:", data_keys)
        
        if len(data_keys) == 1:
            annotations = mat_data[data_keys[0]]
        else:
            # Try common names
            possible_keys = ['SUNRGBDMeta2DBB', 'SUNRGBDMeta', 'meta2DBB', 'data']
            annotations = None
            for key in possible_keys:
                if key in mat_data:
                    annotations = mat_data[key]
                    break
            
            if annotations is None:
                print(f"Could not find annotations in keys: {data_keys}")
                return None
        
        print(f"Loaded annotations with shape: {annotations.shape}")
        
        # Extract the actual annotation entries
        if annotations.shape[0] == 1:
            # If shape is (1, N), get the actual entries
            annotations = annotations[0]  # Now shape should be (10335,)
            print(f"Extracted annotations with shape: {annotations.shape}")
        
        return annotations
        
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return None

def parse_sunrgbd2d_annotation(annotation_entry):
    """Parse a single annotation entry from the .mat file"""
    try:
        result = {
            'image_path': '',
            'depth_path': '',
            'bboxes': [],
            'camera_matrix': None
        }
        
        # Extract paths based on the actual structure we found
        sequence_name = str(annotation_entry['sequenceName'][0])
        rgb_name = str(annotation_entry['rgbname'][0])
        depth_name = str(annotation_entry['depthname'][0])
        
        # Clean up the sequence name - remove the full path prefix
        if 'SUNRGBD/' in sequence_name:
            sequence_name = sequence_name.split('SUNRGBD/')[-1]
        
        result['image_path'] = f"{sequence_name}/image/{rgb_name}"
        result['depth_path'] = f"{sequence_name}/depth/{depth_name}"
        
        # Extract 2D bounding boxes from groundtruth2DBB
        groundtruth2DBB = annotation_entry['groundtruth2DBB'][0]
        
        if len(groundtruth2DBB) > 0:
            for bbox_entry in groundtruth2DBB:
                try:
                    # Each bbox_entry is a tuple with 4 elements:
                    # (instance_id, bbox_coords, class_name, truncation)
                    instance_id = bbox_entry[0]  # Not needed
                    bbox_coords = bbox_entry[1][0]  # Get the actual coordinates
                    class_name = str(bbox_entry[2][0]).lower()  # Get class name
                    truncation = bbox_entry[3]  # Not needed for now
                    
                    # bbox_coords should be [x1, y1, x2, y2] or [x, y, width, height]
                    if len(bbox_coords) >= 4:
                        x1, y1, x2, y2 = bbox_coords[:4]
                        
                        # Check if it's in [x, y, width, height] format
                        # If x2 and y2 are small values, they're likely width/height
                        if x2 < x1 or y2 < y1:
                            # Convert from [x, y, width, height] to [x1, y1, x2, y2]
                            width, height = x2, y2
                            x2 = x1 + width
                            y2 = y1 + height
                        
                        # Map class names to our detection classes
                        # Some SUN RGB-D classes need mapping:
                        class_mapping = {
                            'night_stand': 'night_stand',
                            'dresser_mirror': 'dresser',  # Map mirror to dresser
                            'ottoman': 'chair',  # Map ottoman to chair
                            'dresser': 'dresser',
                            'bed': 'bed',
                            'lamp': 'lamp',
                            'pillow': 'pillow',
                            'table': 'table',
                            'sofa': 'sofa',
                            'chair': 'chair',
                            'toilet': 'toilet',
                            'desk': 'desk',
                            'bookshelf': 'bookshelf',
                            'bathtub': 'bathtub',
                            'box': 'box',
                            'books': 'books',
                            'bottle': 'bottle',
                            'bag': 'bag',
                            'monitor': 'monitor',
                            'television': 'television',
                            'garbage_bin': 'garbage_bin'
                        }
                        
                        mapped_class = class_mapping.get(class_name, class_name)
                        
                        # Only keep classes that are in our detection set
                        if mapped_class in SUNRGBD2D_CLASSES:
                            result['bboxes'].append({
                                'class': mapped_class,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)]
                            })
                
                except Exception as e:
                    print(f"Error parsing bbox entry: {e}")
                    continue
        
        return result if result['image_path'] and len(result['bboxes']) > 0 else None
        
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        return None

def debug_mat_structure(download_dir):
    """Debug function to understand the .mat file structure"""
    download_dir = Path(download_dir)
    mat_file = download_dir / "SUNRGBDMeta2DBB_v2.mat"
    
    mat_data = sio.loadmat(str(mat_file))
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    print("=== MAT FILE STRUCTURE DEBUG ===")
    for key in data_keys:
        data = mat_data[key]
        print(f"Key: {key}")
        print(f"  Type: {type(data)}")
        print(f"  Shape: {data.shape}")
        
        if data.shape[0] == 1 and data.shape[1] > 0:
            first_entry = data[0][0]
            print(f"  First entry type: {type(first_entry)}")
            
            if hasattr(first_entry, 'dtype'):
                print(f"  First entry dtype: {first_entry.dtype}")
                if hasattr(first_entry.dtype, 'names') and first_entry.dtype.names:
                    print(f"  Available fields: {first_entry.dtype.names}")
                    
                    # Show sample values for each field
                    for field in first_entry.dtype.names:
                        try:
                            field_value = first_entry[field]
                            print(f"    {field}: {type(field_value)} - {field_value}")
                        except:
                            print(f"    {field}: [Error reading field]")
        
        break  # Only show first key for now

def convert_sunrgbd2d_to_yolo(download_dir, output_dir):
    """Convert SUN RGB-D 2D to YOLO format"""
    download_dir = Path(download_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val']:
        for subdir in ['images', 'labels', 'depth']:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations = load_sunrgbd2d_annotations(download_dir)
    if annotations is None:
        print("Failed to load annotations")
        return 0, 0
    
    # Path to SUNRGBD data
    sunrgbd_data_path = download_dir / "SUNRGBD"
    
    train_count = 0
    val_count = 0
    
    print(f"Processing {len(annotations)} annotations...")
    
    for i, annotation_entry in enumerate(tqdm(annotations, desc="Converting to YOLO")):
        try:
            # Parse annotation
            parsed = parse_sunrgbd2d_annotation(annotation_entry)
            if parsed is None:
                continue
            
            # Find image file
            img_relative_path = parsed['image_path']
            img_path = sunrgbd_data_path / img_relative_path
            
            if not img_path.exists():
                # Try alternative path structures
                possible_paths = [
                    sunrgbd_data_path / img_relative_path,
                    sunrgbd_data_path / img_relative_path.replace('SUNRGBD/', ''),
                ]
                
                img_path = None
                for path in possible_paths:
                    if path.exists():
                        img_path = path
                        break
                
                if img_path is None:
                    print(f"Image not found: {img_relative_path}")
                    continue
            
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
                
            height, width = img.shape[:2]
            
            # Determine train/val split (80/20)
            is_train = i % 5 != 0
            split = "train" if is_train else "val"
            
            # Generate output filename
            scene_name = img_path.parent.parent.name  # Get the scene directory name
            img_filename = f"{scene_name}_{img_path.stem}.jpg"
            
            # Copy image
            dst_img_path = output_dir / split / "images" / img_filename
            shutil.copy2(img_path, dst_img_path)
            
            # Look for depth image
            depth_relative_path = parsed['depth_path']
            depth_path = sunrgbd_data_path / depth_relative_path
            if not depth_path.exists():
                depth_path = sunrgbd_data_path / depth_relative_path.replace('SUNRGBD/', '')
            
            if depth_path.exists():
                dst_depth_path = output_dir / split / "depth" / f"{scene_name}_{img_path.stem}.png"
                shutil.copy2(depth_path, dst_depth_path)
            
            # Convert bounding boxes to YOLO format
            label_filename = f"{scene_name}_{img_path.stem}.txt"
            label_path = output_dir / split / "labels" / label_filename
            
            with open(label_path, 'w') as f:
                for bbox_info in parsed['bboxes']:
                    class_name = bbox_info['class'].lower()
                    
                    # Map to our detection classes
                    if class_name not in SUNRGBD2D_CLASSES:
                        continue
                    
                    class_id = SUNRGBD2D_CLASSES[class_name] - 1  # YOLO uses 0-based indexing
                    
                    # Convert bbox to YOLO format
                    x1, y1, x2, y2 = bbox_info['bbox']
                    
                    # Ensure valid bounding box
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Convert to normalized center coordinates
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    
                    # Clip to valid range [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    box_width = max(0.0, min(1.0, box_width))
                    box_height = max(0.0, min(1.0, box_height))
                    
                    # Write YOLO annotation
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            if is_train:
                train_count += 1
            else:
                val_count += 1
                
        except Exception as e:
            print(f"Error processing annotation {i}: {e}")
            continue
    
    print(f"Converted {train_count} training and {val_count} validation images")
    
    # Create YOLO config and label map
    create_sunrgbd2d_yaml(output_dir)
    create_sunrgbd2d_pbtxt(output_dir)
    
    return train_count, val_count

def create_sunrgbd2d_yaml(output_dir):
    """Create YOLO dataset configuration"""
    yaml_content = f"""# SUN RGB-D 2D dataset configuration for YOLO
path: {output_dir.absolute()}
train: train/images
val: val/images

# Number of classes
nc: {len(SUNRGBD2D_CLASSES)}

# Class names
names:
"""
    
    for class_name, class_id in sorted(SUNRGBD2D_CLASSES.items(), key=lambda x: x[1]):
        yaml_content += f"  {class_id-1}: {class_name}\n"
    
    yaml_path = output_dir / "sunrgbd2d.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YOLO config: {yaml_path}")

def create_sunrgbd2d_pbtxt(output_dir):
    """Create TensorFlow label map"""
    pbtxt_content = ""
    
    for class_name, class_id in sorted(SUNRGBD2D_CLASSES.items(), key=lambda x: x[1]):
        pbtxt_content += f"""item {{
  id: {class_id}
  name: '{class_name}'
}}
"""
    
    pbtxt_path = output_dir / "sunrgbd2d.pbtxt"
    with open(pbtxt_path, 'w') as f:
        f.write(pbtxt_content)
    
    print(f"Created label map: {pbtxt_path}")

def convert_sunrgbd2d_to_tfrecords(data_dir, output_path, include_depth=True, train_subset_txt=None):
    """Convert SUN RGB-D 2D to TFRecords"""
    
    def label_map_extractor(label_map_path):
        """Extract label mapping from pbtxt"""
        label_map = {}
        with open(label_map_path, "r") as file:
            lines = file.readlines()
            current_id = None
            current_name = None
            
            for line in lines:
                line = line.strip()
                if "id:" in line:
                    current_id = int(line.split(":")[1].strip())
                elif "name:" in line:
                    current_name = line.split(":")[1].strip().strip("'\"")
                    if current_id is not None and current_name is not None:
                        label_map[current_name] = current_id
                        current_id = None
                        current_name = None
        
        return label_map
    
    def create_tfrecord_example(image_path, label_path, depth_path, label_map, image_id):
        """Create TFRecord example"""
        
        # Read image
        with tf.io.gfile.GFile(str(image_path), "rb") as fid:
            encoded_image = fid.read()
        
        # Get image dimensions
        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]
        
        # Handle depth
        encoded_depth_bytes = None
        depth_array = None
        if depth_path and depth_path.exists():
            depth_array = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_array is not None:
                encoded_depth_bytes = depth_array.tobytes()
        
        # Generate key
        key = hashlib.sha256(encoded_image).hexdigest()
        
        # Parse YOLO labels
        xmin_norm = []
        ymin_norm = []
        xmax_norm = []
        ymax_norm = []
        class_names = []
        class_ids = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_idx, x_center, y_center, box_width, box_height = map(float, parts)
                        
                        # Convert YOLO to normalized bbox
                        xmin = max(0.0, x_center - box_width/2)
                        ymin = max(0.0, y_center - box_height/2)
                        xmax = min(1.0, x_center + box_width/2)
                        ymax = min(1.0, y_center + box_height/2)
                        
                        if xmax > xmin and ymax > ymin:
                            xmin_norm.append(xmin)
                            ymin_norm.append(ymin)
                            xmax_norm.append(xmax)
                            ymax_norm.append(ymax)
                            
                            # Get class name
                            class_name = None
                            for name, id_val in label_map.items():
                                if id_val == int(class_idx) + 1:
                                    class_name = name
                                    break
                            
                            if class_name:
                                class_names.append(class_name)
                                class_ids.append(label_map[class_name])
        
        # Create features
        features_dict = {
            "image/height": dataset_util.int64_feature(height),
            "image/width": dataset_util.int64_feature(width),
            "image/filename": dataset_util.bytes_feature(image_path.name.encode("utf8")),
            "image/source_id": dataset_util.bytes_feature(str(image_id).encode("utf8")),
            "image/key/sha256": dataset_util.bytes_feature(key.encode("utf8")),
            "image/encoded": dataset_util.bytes_feature(encoded_image),
            "image/format": dataset_util.bytes_feature(b"jpeg"),
            "image/object/bbox/xmin": dataset_util.float_list_feature(xmin_norm),
            "image/object/bbox/xmax": dataset_util.float_list_feature(xmax_norm),
            "image/object/bbox/ymin": dataset_util.float_list_feature(ymin_norm),
            "image/object/bbox/ymax": dataset_util.float_list_feature(ymax_norm),
            "image/object/class/text": dataset_util.bytes_list_feature(
                [x.encode("utf8") for x in class_names]
            ),
            "image/object/class/label": dataset_util.int64_list_feature(class_ids),
            "image/object/difficult": dataset_util.int64_list_feature([0] * len(class_names)),
        }
        
        # Add depth
        if encoded_depth_bytes is not None:
            features_dict["image/encoded_depth"] = dataset_util.bytes_feature(encoded_depth_bytes)
            features_dict["image/depth_shape"] = dataset_util.int64_list_feature(depth_array.shape)
        
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        return example
    
    # Setup
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load label map
    label_map = label_map_extractor(data_dir / "sunrgbd2d.pbtxt")
    
    # Get train files - from txt file if provided, otherwise all files in train directory
    train_images_dir = data_dir / "train" / "images"
    train_labels_dir = data_dir / "train" / "labels"
    train_depth_dir = data_dir / "train" / "depth" if include_depth else None
    
    if train_subset_txt and Path(train_subset_txt).exists():
        with open(train_subset_txt, 'r') as f:
            # Extract just the filename from each line (like VisDrone does)
            train_files = [line.strip().split("/")[-1] for line in f.readlines() if line.strip()]
        print(f"Loaded {len(train_files)} training images from {train_subset_txt}")
    else:
        # Use all files in train directory
        train_files = [f.name for f in train_images_dir.glob("*.jpg")]
        train_files = sorted(train_files)
        print(f"Found {len(train_files)} training images in {train_images_dir}")
    
    # Get val files (always use all val files)
    val_images_dir = data_dir / "val" / "images"
    val_labels_dir = data_dir / "val" / "labels"
    val_depth_dir = data_dir / "val" / "depth" if include_depth else None
    
    val_files = [f.name for f in val_images_dir.glob("*.jpg")]
    val_files = sorted(val_files)
    print(f"Found {len(val_files)} validation images in {val_images_dir}")
    
    print(f"Final split - Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create writers
    train_writer = tf.io.TFRecordWriter(str(output_path / "sunrgbd2d_train.tfrecord"))
    val_writer = tf.io.TFRecordWriter(str(output_path / "sunrgbd2d_val.tfrecord"))
    
    def process_split(files, writer, split_name, images_dir, labels_dir, depth_dir):
        count = 0
        for img_file in files:
            img_name_without_ext = Path(img_file).stem
            
            # Paths
            image_path = images_dir / img_file
            label_path = labels_dir / f"{img_name_without_ext}.txt"
            depth_path = depth_dir / f"{img_name_without_ext}.png" if depth_dir else None
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            try:
                example = create_tfrecord_example(image_path, label_path, depth_path, label_map, count)
                writer.write(example.SerializeToString())
                count += 1
                
                if count % 1000 == 0:
                    print(f"Processed {count} {split_name} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return count
    
    # Process train split (from txt file or all train images)
    train_count = process_split(train_files, train_writer, "train", 
                               train_images_dir, train_labels_dir, train_depth_dir)
    
    # Process val split (all images in val folder)
    val_count = process_split(val_files, val_writer, "val", 
                             val_images_dir, val_labels_dir, val_depth_dir)
    
    # Close writers
    train_writer.close()
    val_writer.close()
    
    print("Finished creating TFRecords")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
def main():
    """Main function"""
    download_dir = "/path/to/data/DepthPrior/datasets/SUNRGBD/"
    train_subset_txt = None # "/DepthPrior/datasets/SUNRGBD/train/image_subset_10.txt"  # Path to the subset file  
    # download_sunrgbd2d(download_dir)
    # debug_mat_structure(download_dir)
    # convert_sunrgbd2d_to_yolo(download_dir, download_dir)
    
    tfrecord_output = Path(download_dir) / "tfrecords"
    convert_sunrgbd2d_to_tfrecords(download_dir, tfrecord_output, False, train_subset_txt)

if __name__ == "__main__":
    main()