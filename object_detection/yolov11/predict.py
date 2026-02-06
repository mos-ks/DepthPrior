from pathlib import Path
from ultralytics import YOLO
import os
import json

model_weights = os.environ.get("MODEL_WEIGHTS", "build/yolo11s.pt")
score_threshold = float(os.environ.get("SCORE_THRESHOLD", 0.01))
target_file = os.environ.get("TARGET_FILE", "predictions.jsonl")
image_list = os.environ.get("IMAGE_LIST", "/path/to/data/DepthPrior/datasets/KITTI/yolo/val/images.txt")

# Load a model
print(f"Loading model from {model_weights}")
model = YOLO(model_weights)
class_names = model.names

# Run batched inference on a list of images
print(f"Making predictions on images listed in {image_list} with score threshold {score_threshold}")
image_list = Path(image_list).expanduser()
results = model.predict(image_list, conf=score_threshold, batch=32, stream=True)


with open(target_file, "w") as f:
    for result in results:
        # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        image_path = result.path
        filename = Path(image_path).name

        print(f"Processing image: {image_path}")

        boxes = result.boxes
        coordinates = boxes.xyxy.cpu().numpy().tolist()
        confidences = boxes.conf.cpu().numpy().tolist()
        classes = boxes.cls.cpu().numpy().astype(int).tolist()
        classes = [class_names[int(cls)] for cls in classes]

        for coord, conf, cls in zip(coordinates, confidences, classes):
            result_dict = {
                "image_name": filename,
                "score_thresh": score_threshold,
                "det_score": conf,
                "bbox": coord,
                "class": cls,
            }
            f.write(json.dumps(result_dict) + "\n")
