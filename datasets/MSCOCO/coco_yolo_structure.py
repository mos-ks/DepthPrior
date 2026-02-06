#!/usr/bin/env python3
"""
Reorganize MSCOCO dataset to proper YOLO format.

Current structure:
MSCOCO/
├── labels/
│   ├── train2017/  (label files)
│   └── val2017/    (label files)
├── train2017/      (image files directly)
└── val2017/        (image files directly)

Target structure:
MSCOCO/
├── train2017/
│   ├── images/     (image files)
│   └── labels/     (label files)
└── val2017/
    ├── images/     (image files)
    └── labels/     (label files)
"""

import os
import shutil
from pathlib import Path

from tqdm import tqdm


def reorganize_coco_dataset(base_path):
    """
    Reorganize COCO dataset from current structure to YOLO format.
    
    Args:
        base_path (str): Path to MSCOCO directory
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")
    
    print(f"Reorganizing COCO dataset at: {base_path}")
    
    # Define paths
    current_labels_dir = base_path / "labels"
    train_images_current = base_path / "train2017"
    val_images_current = base_path / "val2017"
    
    # Check current structure
    if not current_labels_dir.exists():
        raise ValueError(f"Labels directory not found: {current_labels_dir}")
    if not train_images_current.exists():
        raise ValueError(f"Train images directory not found: {train_images_current}")
    if not val_images_current.exists():
        raise ValueError(f"Val images directory not found: {val_images_current}")
    
    # Create temporary directory to avoid conflicts
    temp_dir = base_path / "temp_reorganize"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Move current train2017 and val2017 to temp
        print("Step 1: Moving current directories to temp...")
        temp_train_images = temp_dir / "train2017_images"
        temp_val_images = temp_dir / "val2017_images"
        temp_labels = temp_dir / "labels"
        
        shutil.move(str(train_images_current), str(temp_train_images))
        shutil.move(str(val_images_current), str(temp_val_images))
        shutil.move(str(current_labels_dir), str(temp_labels))
        
        # Step 2: Create new directory structure
        print("Step 2: Creating new directory structure...")
        
        # Create new train2017 structure
        new_train_dir = base_path / "train2017"
        new_train_images = new_train_dir / "images"
        new_train_labels = new_train_dir / "labels"
        
        new_train_dir.mkdir(exist_ok=True)
        new_train_images.mkdir(exist_ok=True)
        new_train_labels.mkdir(exist_ok=True)
        
        # Create new val2017 structure
        new_val_dir = base_path / "val2017"
        new_val_images = new_val_dir / "images"
        new_val_labels = new_val_dir / "labels"
        
        new_val_dir.mkdir(exist_ok=True)
        new_val_images.mkdir(exist_ok=True)
        new_val_labels.mkdir(exist_ok=True)
        
        # Step 3: Move train images
        print("Step 3: Moving train images...")
        train_image_files = list(temp_train_images.glob("*.jpg"))
        print(f"Found {len(train_image_files)} train images")
        
        for img_file in tqdm(train_image_files, desc="Moving train images"):
            dest = new_train_images / img_file.name
            shutil.move(str(img_file), str(dest))
        
        # Step 4: Move val images
        print("Step 4: Moving val images...")
        val_image_files = list(temp_val_images.glob("*.jpg"))
        print(f"Found {len(val_image_files)} val images")
        
        for img_file in tqdm(val_image_files, desc="Moving val images"):
            dest = new_val_images / img_file.name
            shutil.move(str(img_file), str(dest))
        
        # Step 5: Move train labels
        print("Step 5: Moving train labels...")
        train_label_source = temp_labels / "train2017"
        if train_label_source.exists():
            train_label_files = list(train_label_source.glob("*.txt"))
            print(f"Found {len(train_label_files)} train label files")
            
            for label_file in tqdm(train_label_files, desc="Moving train labels"):
                dest = new_train_labels / label_file.name
                shutil.move(str(label_file), str(dest))
        else:
            print("Warning: No train labels found")
        
        # Step 6: Move val labels
        print("Step 6: Moving val labels...")
        val_label_source = temp_labels / "val2017"
        if val_label_source.exists():
            val_label_files = list(val_label_source.glob("*.txt"))
            print(f"Found {len(val_label_files)} val label files")
            
            for label_file in tqdm(val_label_files, desc="Moving val labels"):
                dest = new_val_labels / label_file.name
                shutil.move(str(label_file), str(dest))
        else:
            print("Warning: No val labels found")
        
        # Step 7: Clean up temp directory
        print("Step 7: Cleaning up temporary files...")
        shutil.rmtree(str(temp_dir))
        
        print("\n✅ Reorganization complete!")
        
        # Step 8: Verify final structure
        print("Final structure verification:")
        print(f"📁 {base_path}/")
        
        train_img_count = len(list(new_train_images.glob("*.jpg")))
        train_lbl_count = len(list(new_train_labels.glob("*.txt")))
        print(f"├── 📁 train2017/")
        print(f"│   ├── 📁 images/ ({train_img_count} files)")
        print(f"│   └── 📁 labels/ ({train_lbl_count} files)")
        
        val_img_count = len(list(new_val_images.glob("*.jpg")))
        val_lbl_count = len(list(new_val_labels.glob("*.txt")))
        print(f"└── 📁 val2017/")
        print(f"    ├── 📁 images/ ({val_img_count} files)")
        print(f"    └── 📁 labels/ ({val_lbl_count} files)")
        
        print(f"\nTotal images: {train_img_count + val_img_count}")
        print(f"Total labels: {train_lbl_count + val_lbl_count}")
        
    except Exception as e:
        print(f"❌ Error during reorganization: {e}")
        print("Attempting to restore original structure...")
        
        # Try to restore original structure
        try:
            if temp_dir.exists():
                if (temp_dir / "train2017_images").exists():
                    shutil.move(str(temp_dir / "train2017_images"), str(base_path / "train2017"))
                if (temp_dir / "val2017_images").exists():
                    shutil.move(str(temp_dir / "val2017_images"), str(base_path / "val2017"))
                if (temp_dir / "labels").exists():
                    shutil.move(str(temp_dir / "labels"), str(base_path / "labels"))
                shutil.rmtree(str(temp_dir))
                print("✅ Original structure restored")
        except Exception as restore_error:
            print(f"❌ Failed to restore original structure: {restore_error}")
            print(f"Manual cleanup may be required in: {temp_dir}")
        
        raise e

if __name__ == "__main__":
    reorganize_coco_dataset("/path/to/data/DepthPrior/datasets/MSCOCO")
