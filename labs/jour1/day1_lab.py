import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


def ensure_dirs() -> tuple[Path, Path]:
    # Create output folders used by the lab.
    # - outputs/jour1: metrics and artifacts root
    # - outputs/jour1/figures: generated visual summaries
    out_dir = Path("outputs/jour1")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, figures_dir


def make_synthetic_scene(shape: str, shift: int = 0) -> np.ndarray:
    # Build a simple synthetic scene: black background + white geometric object.
    # shift allows controlled translation to simulate prediction offset.
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    if shape == "rectangle":
        # Filled rectangle: used as the main reference object.
        cv2.rectangle(img, (40 + shift, 60), (180 + shift, 190), (255, 255, 255), -1)
    elif shape == "circle":
        # Filled circle: used as a clearly different shape for comparison.
        cv2.circle(img, (120 + shift, 130), 60, (255, 255, 255), -1)
    else:
        raise ValueError("shape must be 'rectangle' or 'circle'")
    return img


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    # Compute Intersection over Union between two boxes (x1, y1, x2, y2).
    # IoU quantifies overlap quality for detection.
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        # No overlap.
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union


def bbox_from_threshold(img_gray: np.ndarray) -> tuple[int, int, int, int]:
    # Segment bright object and infer its bounding box automatically.
    _, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)


def hog_features(gray_img: np.ndarray) -> np.ndarray:
    # Compute global HOG descriptor on a fixed window size.
    # Fixed size is required so feature dimensions are comparable.
    resized = cv2.resize(gray_img, (128, 64), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(resized)


def sift_features(gray_img: np.ndarray) -> tuple[list, np.ndarray | None]:
    # Detect local keypoints and descriptors with SIFT.
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray_img, None)


def run() -> dict:
    # 1) Prepare folders for artifacts.
    out_dir, figures_dir = ensure_dirs()

    # 2) Build synthetic images:
    # - gt: reference object
    # - pred: same object but translated (simulates imperfect localization)
    # - other: different shape (negative comparison)
    img_gt = make_synthetic_scene("rectangle", shift=0)
    img_pred = make_synthetic_scene("rectangle", shift=12)
    img_other = make_synthetic_scene("circle", shift=0)

    # 3) Convert to grayscale for classical feature extraction.
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    gray_other = cv2.cvtColor(img_other, cv2.COLOR_BGR2GRAY)

    # 4) Estimate boxes and evaluate overlap quality with IoU.
    box_gt = bbox_from_threshold(gray_gt)
    box_pred = bbox_from_threshold(gray_pred)
    iou_score = iou(box_pred, box_gt)

    # 5) Compute HOG descriptors and compare distances.
    # Lower distance => more similar global structure.
    hog_gt = hog_features(gray_gt)
    hog_pred = hog_features(gray_pred)
    hog_other = hog_features(gray_other)

    # 6) Compute SIFT keypoints/descriptors for local matching.
    kp_gt, desc_gt = sift_features(gray_gt)
    kp_pred, desc_pred = sift_features(gray_pred)
    kp_other, desc_other = sift_features(gray_other)

    # 7) Match descriptors with KNN (k=2) for Lowe's ratio test.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_similar = bf.knnMatch(desc_gt, desc_pred, k=2)
    matches_different = bf.knnMatch(desc_gt, desc_other, k=2)

    def ratio_count(matches: list, ratio: float = 0.75) -> int:
        # Keep "good" matches according to Lowe ratio criterion.
        good = 0
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good += 1
        return good

    good_similar = ratio_count(matches_similar)
    good_different = ratio_count(matches_different)

    # 8) Generate a quick visual report image.
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Scene GT")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Scene Pred")
    axs[1].axis("off")

    kp_img = cv2.drawKeypoints(gray_gt, kp_gt, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axs[2].imshow(kp_img, cmap="gray")
    axs[2].set_title("SIFT keypoints")
    axs[2].axis("off")

    plt.tight_layout()
    figure_path = figures_dir / "jour1_overview.png"
    plt.savefig(figure_path, dpi=130)
    plt.close(fig)

    # 9) Consolidate all measurable outputs for reproducibility.
    results = {
        "iou_score": float(iou_score),
        "bbox_gt": box_gt,
        "bbox_pred": box_pred,
        "hog_dimension": int(hog_gt.shape[0]),
        "hog_shifted_l2": float(np.linalg.norm(hog_gt - hog_pred)),
        "hog_different_l2": float(np.linalg.norm(hog_gt - hog_other)),
        "sift_kp_gt": len(kp_gt),
        "sift_kp_pred": len(kp_pred),
        "sift_kp_other": len(kp_other),
        "sift_good_matches_similar": good_similar,
        "sift_good_matches_different": good_different,
        "figure_path": str(figure_path),
    }

    # 10) Persist metrics to disk for downstream checks/reporting.
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    metrics = run()
    print(json.dumps(metrics, indent=2))
