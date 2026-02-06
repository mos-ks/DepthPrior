#!/usr/bin/env python3
"""
Calculate the average resolution and standard deviation of all images in a folder.
"""

import math
import os
from pathlib import Path

from PIL import Image


def get_average_resolution(folder_path):
    """
    Calculate the average width, height, and standard deviation of all images in a folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        
    Returns:
        tuple: (average_width, average_height, std_width, std_height, image_count)
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    total_width = 0
    total_height = 0
    image_count = 0
    widths = []
    heights = []
    
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    print(f"Scanning folder: {folder_path}\n")
    
    # Iterate through all files in the folder
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    total_width += width
                    total_height += height
                    widths.append(width)
                    heights.append(height)
                    image_count += 1
                    print(f"✓ {file_path.name}: {width}x{height}")
            except Exception as e:
                print(f"✗ Error reading {file_path.name}: {e}")
        if image_count == 1000: 
            print("\nReached 1000 images, stopping further processing.")
            break
    
    if image_count == 0:
        print("\nNo images found in the folder.")
        return None, None, None, None, 0
    
    avg_width = total_width / image_count
    avg_height = total_height / image_count
    
    # Calculate standard deviation
    std_width = math.sqrt(sum((w - avg_width) ** 2 for w in widths) / image_count)
    std_height = math.sqrt(sum((h - avg_height) ** 2 for h in heights) / image_count)
    
    return avg_width, avg_height, std_width, std_height, image_count


def main():
    """Main function to run the script."""
    print("=" * 60)
    print("Average Image Resolution Calculator")
    print("=" * 60)
    
    folder_path = input("\nEnter the folder path: ").strip()
    
    try:
        avg_width, avg_height, std_width, std_height, count = get_average_resolution(folder_path)
        
        if count > 0:
            print("\n" + "=" * 60)
            print(f"Results:")
            print(f"  Total images processed: {count}")
            print(f"  Average width: {avg_width:.2f} pixels")
            print(f"  Average height: {avg_height:.2f} pixels")
            print(f"  Standard deviation (width): {std_width:.2f} pixels")
            print(f"  Standard deviation (height): {std_height:.2f} pixels")
            print(f"  Average resolution: {avg_width:.2f}x{avg_height:.2f}")
            print("=" * 60)
    
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()