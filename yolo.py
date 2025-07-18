import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO  # YOLOv8+ interface includes YOLOv9


def run_yolo(input_path, output_path, mask_path, csv_path, model=None):
    """
    Runs YOLOv9 inference on the input image and saves results.

    Args:
        input_path (str): Path to input image.
        output_path (str): Annotated image path.
        mask_path (str): Output binary mask image path.
        csv_path (str): Output CSV with bbox info.
        model (YOLO, optional): Preloaded YOLO model instance.
    """
    # Load image
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"[ERROR] Image not found at {input_path}")
        return

    # Initialize model if not provided
    if model is None:
        print("[INFO] Loading YOLOv9 model dynamically (fallback)...")
        model = YOLO('yolov9c.pt')  # You can replace with yolov9e.pt / yolov9s.pt etc.

    # Run inference
    results = model.predict(img_bgr, verbose=False)[0]
    boxes = results.boxes
    names = model.names

    # Create mask and annotated image
    annotated = img_bgr.copy()
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    bbox_data = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{names[cls_id]} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Fill binary mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Save box info
        bbox_data.append({
            'name': names[cls_id],
            'confidence': round(conf, 4),
            'xmin': x1,
            'ymin': y1,
            'xmax': x2,
            'ymax': y2
        })

    # Save results
    cv2.imwrite(output_path, annotated)
    cv2.imwrite(mask_path, mask)
    pd.DataFrame(bbox_data).to_csv(csv_path, index=False)

    print(f"[INFO] YOLOv9 detection complete on {input_path}")
