import json
from pathlib import Path

import cv2
import numpy as np


def make_synthetic_scene(shape: str, shift: int = 0) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    if shape == "rectangle":
        cv2.rectangle(img, (40 + shift, 60), (180 + shift, 190), (255, 255, 255), -1)
    elif shape == "circle":
        cv2.circle(img, (120 + shift, 130), 60, (255, 255, 255), -1)
    else:
        raise ValueError("shape must be 'rectangle' or 'circle'")
    return img


def iou(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def bbox_from_threshold(gray):
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(th)
    if points is None:
        raise ValueError("Aucun pixel détecté après seuillage ; vérifier l'image ou le seuil.")
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)


img_gt = make_synthetic_scene("rectangle", 0)
img_pred = make_synthetic_scene("rectangle", 12)

gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)

box_gt = bbox_from_threshold(gray_gt)
box_pred = bbox_from_threshold(gray_pred)

metrics = {"iou_score": float(iou(box_pred, box_gt)), "bbox_gt": box_gt, "bbox_pred": box_pred}
Path("outputs/jour1").mkdir(parents=True, exist_ok=True)
Path("outputs/jour1/metrics_minimal.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
