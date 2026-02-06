from pathlib import Path

from tqdm import tqdm

KITTI_ROOT = Path("/path/to/data/DepthPrior/datasets/KITTI")
KITTI_YOLO_ROOT = KITTI_ROOT / "yolo"
KITTI_YOLO_ROOT_RELATIVE = Path("/path/to/data/DepthPrior/datasets/KITTI/yolo")
IMAGES_LIST_IN = KITTI_YOLO_ROOT / "training.txt"
IMAGES_LIST_OUT = KITTI_YOLO_ROOT / "training.csv"
NPY_DIR = "depth_anything_predictions"


def main():
    not_found_samples = []
    with open(IMAGES_LIST_IN) as f, open(IMAGES_LIST_OUT, "w") as depth_file:
        for image_path in tqdm(f, desc="Processing images"):
            image_path = Path(image_path.strip())
            image_id = image_path.stem
            depth_path = f"../{NPY_DIR}/{image_id}_depth.npy"

            if not (KITTI_ROOT / NPY_DIR / f"{image_id}_depth.npy").exists():
                not_found_samples.append(image_id)
                continue

            depth_file.write(f"{image_path},{depth_path}\n")

    if len(not_found_samples) > 0:
        print(f"Warning: {len(not_found_samples)} samples were not found in the depth predictions.")
        print("These samples will be skipped in the depth dataset preparation.")
        print(f"Example IDs of not found samples: {not_found_samples[:20]}")


if __name__ == "__main__":
    main()
