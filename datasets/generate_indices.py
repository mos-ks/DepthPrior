import os


def generate_image_indices(images_folder, output_file, image_names_file=None):
    """
    Generate a text file with indices of image files.
    
    Args:
        images_folder: Path to the folder containing images
        output_file: Path to output text file
        image_names_file: Optional path to file containing specific image paths to find indices for.
                         If None, generates indices for all files (0 to N-1)
    """
    
    # Check if the images folder exists
    if not os.path.exists(images_folder):
        print(f"Error: Folder '{images_folder}' does not exist!")
        return
    
    # Get all files from the images folder
    all_files = os.listdir(images_folder)
    
    # Filter only image files (common extensions)
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in all_files if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"No image files found in '{images_folder}'!")
        return
    
    # Sort the image files for consistent indexing
    image_files.sort()
    
    print(f"Found {len(image_files)} image files in folder")
    
    # Generate indices
    if image_names_file is None:
        # Generate all indices from 0 to N-1
        indices = list(range(len(image_files)))
        print(f"Generating indices for all {len(image_files)} files")
    else:
        # Read specific image names from file
        if not os.path.exists(image_names_file):
            print(f"Error: Image names file '{image_names_file}' does not exist!")
            return
            
        with open(image_names_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        # Extract just the filename from each path
        image_names_list = []
        for path in image_paths:
            # Handle different path formats: /images/file.jpg, ./images/file.jpg, file.jpg
            filename = os.path.basename(path)
            image_names_list.append(filename)
        
        print(f"Read {len(image_names_list)} image names from file")
        
        # Find indices for specific image names
        indices = []
        not_found = []
        
        for image_name in image_names_list:
            try:
                index = image_files.index(image_name)
                indices.append(index)
            except ValueError:
                not_found.append(image_name)
        
        if not_found:
            print(f"Warning: {len(not_found)} image names not found:")
            for name in not_found[:5]:  # Show first 5
                print(f"  {name}")
            if len(not_found) > 5:
                print(f"  ... and {len(not_found) - 5} more")
        
        print(f"Found indices for {len(indices)} out of {len(image_names_list)} requested images")
    
    # Write indices to output file
    with open(output_file, 'w') as f:
        for index in indices:
            f.write(f"{index}\n")
    
    print(f"Generated '{output_file}' with {len(indices)} indices")
    print(f"Sample indices (first 10):")
    for i, index in enumerate(indices[:10]):
        print(f"  {index} -> {image_files[index]}")
    if len(indices) > 10:
        print(f"  ... and {len(indices) - 10} more")

if __name__ == "__main__":
    # Configuration
    images_folder = "/path/to/data/DepthPrior/datasets/SUNRGBD/val/images"
    output_file = "/path/to/data/DepthPrior/datasets/SUNRGBD/val/validation_indices.txt"

    # Path to file containing specific image paths (one per line)
    image_names_file = None# "/path/to/data/DepthPrior/datasets/SUNRGBD/remaining_images_subset_10.txt" #None # 

    # Check if specific images file exists
    if image_names_file is not None:
        print("=== Generating indices for SPECIFIC files from file ===")
        generate_image_indices(images_folder, output_file, image_names_file)
    else:
        print("=== Generating indices for ALL files ===")
        generate_image_indices(images_folder, output_file)
