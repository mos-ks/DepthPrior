import os
import random

images_folder="/path/to/data/DepthPrior/datasets/SUNRGBD/train/images"
output_file="/path/to/data/DepthPrior/datasets/SUNRGBD/image_subset_10.txt"
percentage=0.1
    
# Check if the images folder exists
if not os.path.exists(images_folder):
    print(f"Error: Folder '{images_folder}' does not exist!")

# Get all files from the images folder
all_files = os.listdir(images_folder)

# Filter only image files (common extensions)
image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
image_files = [f for f in all_files if os.path.splitext(f.lower())[1] in image_extensions]

if not image_files:
    print(f"No image files found in '{images_folder}'!")

# Calculate number of images to select (at least 1)
num_to_select = max(1, int(len(image_files) * percentage))

# Randomly select the subset
selected_images = random.sample(image_files, num_to_select)

# Sort the selected images for consistent output
selected_images.sort()

# Write to output file
with open(output_file, 'w') as f:
    for image in selected_images:
        f.write(f"{images_folder}/{image}\n")

print(f"Generated '{output_file}' with {num_to_select} images out of {len(image_files)} total images ({percentage*100:.1f}%)")
print(f"Selected images:")
for image in selected_images[:10]:  # Show first 10
    print(f"  {images_folder}/{image}")
if len(selected_images) > 10:
    print(f"  ... and {len(selected_images) - 10} more")