import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def read_visdrone_annotation(filename):
    """Read VisDrone annotation file and return number of valid objects"""
    if not os.path.exists(filename):
        return 0
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return 0
    
    valid_objects = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        if len(parts) >= 8:
            try:
                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_width = float(parts[2])
                bbox_height = float(parts[3])
                score = float(parts[4])
                object_category = int(parts[5])
                
                # Filter valid objects (same as your TFRecord converter)
                if (score > 0 and bbox_width > 0 and bbox_height > 0 and 
                    object_category in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):  # Valid VisDrone classes
                    valid_objects += 1
            except:
                continue
    
    return valid_objects

def analyze_visdrone_dataset(data_dir, classes_to_use=None):
    """
    Analyze VisDrone dataset to find maximum instances per image.
    
    Args:
        data_dir: Path to VisDrone dataset root directory
        classes_to_use: List of class names to filter (optional)
    """
    
    # VisDrone class mapping
    VISDRONE_CLASSES = {
        1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
        6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
    }
    
    if classes_to_use:
        name_to_id = {v: k for k, v in VISDRONE_CLASSES.items()}
        valid_category_ids = [name_to_id[cls] for cls in classes_to_use if cls in name_to_id]
    else:
        valid_category_ids = list(VISDRONE_CLASSES.keys())
    
    # Analyze both train and val sets
    datasets = {
        'train': {
            'annotations_dir': os.path.join(data_dir, "VisDrone2019-DET-train", "annotations"),
            'images_dir': os.path.join(data_dir, "VisDrone2019-DET-train", "images")
        },
        'val': {
            'annotations_dir': os.path.join(data_dir, "VisDrone2019-DET-val", "annotations"),
            'images_dir': os.path.join(data_dir, "VisDrone2019-DET-val", "images")
        }
    }
    
    all_instance_counts = []
    per_class_stats = defaultdict(list)
    dataset_stats = {}
    
    for split_name, paths in datasets.items():
        annotations_dir = paths['annotations_dir']
        images_dir = paths['images_dir']
        
        if not os.path.exists(annotations_dir):
            print(f"Warning: {split_name} annotations directory not found: {annotations_dir}")
            continue
            
        print(f"\nAnalyzing {split_name} set...")
        
        # Get all annotation files
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
        split_instance_counts = []
        
        for i, ann_file in enumerate(annotation_files):
            ann_path = os.path.join(annotations_dir, ann_file)
            
            # Count objects per class
            class_counts = defaultdict(int)
            total_objects = 0
            
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 8:
                        try:
                            bbox_width = float(parts[2])
                            bbox_height = float(parts[3])
                            score = float(parts[4])
                            object_category = int(parts[5])
                            
                            # Apply same filters as TFRecord converter
                            if (score > 0 and bbox_width > 0 and bbox_height > 0 and 
                                object_category in valid_category_ids):
                                class_counts[VISDRONE_CLASSES[object_category]] += 1
                                total_objects += 1
                        except:
                            continue
            
            split_instance_counts.append(total_objects)
            all_instance_counts.append(total_objects)
            
            # Track per-class statistics
            for class_name, count in class_counts.items():
                per_class_stats[class_name].append(count)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} files...")
        
        # Calculate statistics for this split
        if split_instance_counts:
            dataset_stats[split_name] = {
                'count': len(split_instance_counts),
                'max': max(split_instance_counts),
                'min': min(split_instance_counts),
                'mean': np.mean(split_instance_counts),
                'median': np.median(split_instance_counts),
                'std': np.std(split_instance_counts),
                'percentiles': {
                    '90': np.percentile(split_instance_counts, 90),
                    '95': np.percentile(split_instance_counts, 95),
                    '99': np.percentile(split_instance_counts, 99),
                    '99.5': np.percentile(split_instance_counts, 99.5)
                }
            }
    
    # Overall statistics
    if all_instance_counts:
        print("\n" + "="*60)
        print("OVERALL DATASET STATISTICS")
        print("="*60)
        print(f"Total images analyzed: {len(all_instance_counts)}")
        print(f"Maximum instances per image: {max(all_instance_counts)}")
        print(f"Minimum instances per image: {min(all_instance_counts)}")
        print(f"Mean instances per image: {np.mean(all_instance_counts):.2f}")
        print(f"Median instances per image: {np.median(all_instance_counts):.2f}")
        print(f"Standard deviation: {np.std(all_instance_counts):.2f}")
        
        print(f"\nPercentile Analysis:")
        for p in [90, 95, 99, 99.5, 99.9]:
            print(f"  {p}th percentile: {np.percentile(all_instance_counts, p):.0f}")
        
        # Per-split statistics
        print("\n" + "="*60)
        print("PER-SPLIT STATISTICS")
        print("="*60)
        for split_name, stats in dataset_stats.items():
            print(f"\n{split_name.upper()} SET:")
            print(f"  Images: {stats['count']}")
            print(f"  Max instances: {stats['max']}")
            print(f"  Mean instances: {stats['mean']:.2f}")
            print(f"  99th percentile: {stats['percentiles']['99']:.0f}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR max_instances_per_image")
        print("="*60)
        max_instances = max(all_instance_counts)
        p99 = np.percentile(all_instance_counts, 99)
        p995 = np.percentile(all_instance_counts, 99.5)
        
        print(f"Conservative (covers all images): {max_instances}")
        print(f"Balanced (covers 99.5% of images): {int(np.ceil(p995))}")
        print(f"Efficient (covers 99% of images): {int(np.ceil(p99))}")
        
        # Check how many images would be affected
        images_over_100 = sum(1 for x in all_instance_counts if x > 100)
        images_over_200 = sum(1 for x in all_instance_counts if x > 200)
        
        print(f"\nImages with >100 instances: {images_over_100} ({100*images_over_100/len(all_instance_counts):.2f}%)")
        print(f"Images with >200 instances: {images_over_200} ({100*images_over_200/len(all_instance_counts):.2f}%)")
        
        # Create histogram
        plt.figure(figsize=(12, 6))
        plt.hist(all_instance_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(p99, color='red', linestyle='--', label=f'99th percentile ({p99:.0f})')
        plt.axvline(p995, color='orange', linestyle='--', label=f'99.5th percentile ({p995:.0f})')
        plt.axvline(max_instances, color='green', linestyle='--', label=f'Maximum ({max_instances})')
        plt.xlabel('Number of Instances per Image')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Instances per Image in VisDrone Dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(data_dir, 'visdrone_instances_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nHistogram saved to: {output_path}")
        
        return max_instances, int(np.ceil(p99)), int(np.ceil(p995))
    
    else:
        print("No valid annotation files found!")
        return None, None, None

if __name__ == "__main__":
    # Update this path to your VisDrone dataset
    data_dir = "/path/to/data/DepthPrior/datasets/visdrone"
    
    # Specify which classes you're using (same as in TFRecord converter)
    classes_to_use = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    max_val, p99_val, p995_val = analyze_visdrone_dataset(data_dir, classes_to_use)
    
    if max_val:
        print(f"\n🎯 RECOMMENDED max_instances_per_image VALUES:")
        print(f"   For 99% coverage: {p99_val}")
        print(f"   For 99.5% coverage: {p995_val}")
        print(f"   For 100% coverage: {max_val}")