from pathlib import Path

from tqdm import tqdm

SUNRGBD_ROOT = Path("/path/to/data/DepthPrior/datasets/SUNRGBD")


with open(SUNRGBD_ROOT / "train" / "training.csv", "w") as f:
    not_found_samples = []
    for img_file in tqdm((SUNRGBD_ROOT / "train" / "images").glob("*.jpg")):
        img_id = img_file.stem
        depth_file = SUNRGBD_ROOT / "train" / "depth_anything_predictions" / f"{img_id}_depth.npy"
        if not depth_file.exists():
            not_found_samples.append(depth_file)
            continue

        f.write(f"./images/{img_file.name},./depth_anything_predictions/{depth_file.name}\n")

    if len(not_found_samples) > 0:
        print(f"Warning: {len(not_found_samples)} samples were not found in the depth predictions.")
        print("These samples will be skipped in the depth dataset preparation.")
        print(f"Example IDs of not found samples: {not_found_samples[:20]}")


with open(SUNRGBD_ROOT / "val" / "validation.csv", "w") as f:
    not_found_samples = []
    for img_file in tqdm((SUNRGBD_ROOT / "val" / "images").glob("*.jpg")):
        img_id = img_file.stem
        depth_file = SUNRGBD_ROOT / "val" / "depth_validation" / f"{img_id}_depth.npy"
        if not depth_file.exists():
            not_found_samples.append(depth_file)
            continue

        f.write(f"./images/{img_file.name},./depth_validation/{depth_file.name}\n")

    if len(not_found_samples) > 0:
        print(f"Warning: {len(not_found_samples)} samples were not found in the depth predictions.")
        print("These samples will be skipped in the depth dataset preparation.")
        print(f"Example IDs of not found samples: {not_found_samples[:20]}")
