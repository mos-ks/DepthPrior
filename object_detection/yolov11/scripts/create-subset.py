from pathlib import Path

KITTI_YOLO_ROOT = Path("/path/to/data/DepthPrior/datasets/KITTI/yolo")


def main():
    subset_name = "subset_01"

    subset_size_counter = 0
    with (
        open(KITTI_YOLO_ROOT / "full.txt") as f,
        open(KITTI_YOLO_ROOT / f"{subset_name}.txt", "w") as subset_file,
    ):
        for line in f.readlines():
            image_path = Path(line.strip())
            image_id = get_image_id(image_path)
            if not is_part_of_subset(subset_name, image_id):
                continue

            subset_size_counter += 1
            subset_file.write(line)
            if subset_size_counter % 100 == 0:
                print(f"Processed {subset_size_counter} images for subset '{subset_name}'")

    print(f"Created subset '{subset_name}' with {subset_size_counter} images.")


def is_part_of_subset(subset_name: str, image_id: int):
    # Define the logic to determine if an image ID is part of the subset
    if subset_name == "subset_01":
        # Example logic for subset_01: include images with IDs 1 to 1000
        return 1 <= image_id <= 1000
    elif subset_name == "subset_02":
        pass
    return False


def get_image_id(image_file: Path) -> int:
    return int(image_file.stem)


if __name__ == "__main__":
    main()
