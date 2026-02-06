import os
import random

subset_file="/path/to/data/DepthPrior/datasets/SUNRGBD/image_subset_10.txt"
images_folder="/path/to/data/DepthPrior/datasets/SUNRGBD/train/images"
output_file="/path/to/data/DepthPrior/datasets/SUNRGBD/train/remaining_images_subset_10.txt"

# Check if files/folders exist
if not os.path.exists(subset_file):
    print(f"Error: Subset file '{subset_file}' does not exist!")

if not os.path.exists(images_folder):
    print(f"Error: Images folder '{images_folder}' does not exist!")

# Read the subset file to get selected images
selected_images = set()
with open(subset_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            # Extract just the filename from the path
            filename = os.path.basename(line)
            selected_images.add(filename)

print(f"Found {len(selected_images)} images in subset file")

# Get all image files from the folder
image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
all_files = os.listdir(images_folder)
all_images = [f for f in all_files if os.path.splitext(f.lower())[1] in image_extensions]

print(f"Found {len(all_images)} total images in folder")

# Find remaining images (not in subset)
remaining_images = [img for img in all_images if img not in selected_images]
remaining_images.sort()  # Sort for consistent output

# Write remaining images to output file
with open(output_file, 'w') as f:
    for image in remaining_images:
        f.write(f"{images_folder}/{image}\n")

print(f"Created '{output_file}' with {len(remaining_images)} remaining images")
print(f"Breakdown:")
print(f"  - Total images: {len(all_images)}")
print(f"  - In subset: {len(selected_images)}")
print(f"  - Remaining: {len(remaining_images)}")

# Show first few remaining images
if remaining_images:
    print(f"\nFirst few remaining images:")
    for image in remaining_images[:10]:
        print(f"  {images_folder}/{image}")
    if len(remaining_images) > 10:
        print(f"  ... and {len(remaining_images) - 10} more")