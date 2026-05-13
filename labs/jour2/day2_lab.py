#!/usr/bin/env python3
"""
Lab Jour 2 — CNN et Faster R-CNN
Construire un CNN simple + utiliser Faster R-CNN pré-entraîné
"""

import json
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


REAL_IMAGE_PATH = "labs/shared/assets/coco_dog.jpg"
SYNTHETIC_IMAGE_PATH = "labs/jour2/assets/test_detection.png"
REAL_DOG_GT_BOX = (50, 35, 645, 555)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def generate_dataset(num_samples=200, img_size=64):
    """Génère un dataset synthétique : rectangles, cercles, triangles."""
    X = []
    y = []
    rng = np.random.RandomState(42)
    for i in range(num_samples):
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        label = i % 3
        x1 = int(rng.randint(5, 20))
        y1 = int(rng.randint(5, 20))
        x2 = int(rng.randint(40, 59))
        y2 = int(rng.randint(40, 59))
        color = tuple(int(c) for c in rng.randint(100, 256, 3))

        if label == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif label == 1:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = min(x2 - x1, y2 - y1) // 2
            cv2.circle(img, (cx, cy), r, color, -1)
        else:
            pts = np.array([[(x1+x2)//2, y1], [x1, y2], [x2, y2]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)

        X.append(img)
        y.append(label)

    X = np.array(X, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    y = torch.tensor(y, dtype=torch.long)
    return torch.tensor(X), y


def train_cnn(model, X, y, epochs=15, lr=0.001, batch_size=32):
    """Entraîne le CNN et retourne l'historique des pertes."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    n = len(X)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X[idx]
            batch_y = y[idx]
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        avg_loss = epoch_loss / n
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    accuracy = evaluate_cnn(model, X, y)
    print(f"  Précision finale : {accuracy:.3f}")
    return losses, accuracy


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Sépare aléatoirement le dataset en sous-ensembles train/test."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(X), generator=generator)
    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def evaluate_cnn(model, X, y):
    """Évalue la précision d'un CNN sur un sous-ensemble donné."""
    model.eval()
    with torch.no_grad():
        out = model(X)
        preds = out.argmax(dim=1)
        return (preds == y).float().mean().item()


def run_faster_rcnn_detection(img_path, score_thresh=0.5):
    """Exécute Faster R-CNN sur une image et retourne les détections."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_thresh)
    model.eval()

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée : {img_path}")

    img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        predictions = model([img_tensor])

    return predictions[0], img


def draw_detections(img, boxes, labels, scores):
    """Dessine les boîtes détectées sur l'image."""
    COCO_COLORS = {
        1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
        16: (255, 255, 0), 17: (255, 0, 255), 18: (0, 255, 255),
        19: (200, 200, 0), 44: (200, 0, 200), 62: (0, 200, 200),
    }
    for box, label, score in zip(boxes, labels, scores):
        color = COCO_COLORS.get(int(label), (128, 128, 128))
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{int(label)}:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def compute_iou(box_a, box_b):
    """Calcule l'IoU entre deux boîtes."""
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


def prepare_detection_image():
    """Utilise l'image réelle COCO-like si elle existe, sinon crée une image synthétique."""
    if os.path.exists(REAL_IMAGE_PATH):
        return REAL_IMAGE_PATH, [REAL_DOG_GT_BOX], "real_coco_dog"

    os.makedirs(os.path.dirname(SYNTHETIC_IMAGE_PATH), exist_ok=True)
    test_img = np.zeros((400, 500, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 60), (200, 220), (255, 255, 255), -1)
    cv2.circle(test_img, (350, 200), 70, (200, 200, 0), -1)
    cv2.imwrite(SYNTHETIC_IMAGE_PATH, test_img)
    return SYNTHETIC_IMAGE_PATH, [(40, 50, 210, 230), (270, 120, 430, 280)], "synthetic_shapes"


def detection_metrics_at_threshold(boxes, scores, gt_boxes, score_thresh, iou_thresh=0.5):
    """Calcule précision/rappel pour un seuil de score donné."""
    selected = [(box, score) for box, score in zip(boxes, scores) if score >= score_thresh]
    matched_gt = set()
    tp = 0
    fp = 0

    for pred_box, _ in selected:
        best_iou = 0.0
        best_idx = None
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou_val = compute_iou((pred_box[0], pred_box[1], pred_box[2], pred_box[3]), gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx
        if best_iou >= iou_thresh and best_idx is not None:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": score_thresh, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def save_precision_recall_curve(boxes, scores, gt_boxes):
    """Trace la précision et le rappel selon le seuil de score."""
    thresholds = [0.05, 0.1, 0.15, 0.25, 0.5]
    rows = [detection_metrics_at_threshold(boxes, scores, gt_boxes, t) for t in thresholds]

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, [r["precision"] for r in rows], marker="o", label="Précision")
    plt.plot(thresholds, [r["recall"] for r in rows], marker="o", label="Rappel")
    plt.xlabel("Seuil de score")
    plt.ylabel("Valeur")
    plt.ylim(-0.05, 1.05)
    plt.title("Précision/Rappel selon le seuil Faster R-CNN")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = "outputs/jour2/figures/precision_recall.png"
    plt.savefig(path, dpi=130)
    plt.close()
    return rows, path


def save_feature_maps(model, X):
    """Visualise les premières cartes d'activation du CNN entraîné."""
    model.eval()
    with torch.no_grad():
        activations = model.features[:2](X[:1]).squeeze(0).cpu().numpy()

    n_maps = min(8, activations.shape[0])
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    for idx, ax in enumerate(axs.ravel()):
        if idx < n_maps:
            ax.imshow(activations[idx], cmap="viridis")
            ax.set_title(f"Filtre {idx + 1}")
        ax.axis("off")
    plt.suptitle("Premières feature maps du CNN")
    plt.tight_layout()
    path = "outputs/jour2/figures/feature_maps.png"
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    os.makedirs("outputs/jour2/figures", exist_ok=True)

    # --- CNN ---
    print("=" * 50)
    print("PARTIE 1 : CNN — Entraînement sur données synthétiques")
    print("=" * 50)

    X, y = generate_dataset(num_samples=360, img_size=64)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    model = SimpleCNN(num_classes=3)
    losses, train_accuracy = train_cnn(model, X_train, y_train, epochs=15)
    test_accuracy = evaluate_cnn(model, X_test, y_test)
    print(f"  Précision test : {test_accuracy:.3f}")
    feature_maps_path = save_feature_maps(model, X_test)
    print(f"  Feature maps sauvegardées : {feature_maps_path}")

    # Courbe de perte
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)+1), losses, marker="o", linewidth=2, color="steelblue")
    plt.title("Perte d'entraînement du CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/jour2/figures/cnn_training.png", dpi=130)
    plt.close()
    print(f"  Courbe sauvegardée : outputs/jour2/figures/cnn_training.png")

    # --- Faster R-CNN ---
    print("\n" + "=" * 50)
    print("PARTIE 2 : Faster R-CNN — Détection et évaluation")
    print("=" * 50)

    test_img_path, gt_boxes, image_source = prepare_detection_image()
    print(f"  Image de test : {test_img_path} ({image_source})")

    pred, img_bgr = run_faster_rcnn_detection(test_img_path, score_thresh=0.1)
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    print(f"  Détections : {len(boxes)}")
    for box, label, score in zip(boxes, labels, scores):
        print(f"    Classe {label}: score={score:.3f}, box=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")

    img_result = draw_detections(img_bgr.copy(), boxes, labels, scores)
    result_path = "outputs/jour2/figures/detection_result.png"
    cv2.imwrite(result_path, img_result)
    print(f"  Résultat sauvegardé : {result_path}")

    ious = []
    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in boxes:
            iou_val = compute_iou(
                (pred_box[0], pred_box[1], pred_box[2], pred_box[3]),
                gt_box
            )
            best_iou = max(best_iou, iou_val)
        ious.append(best_iou)

    avg_iou = float(np.mean(ious)) if ious else 0.0
    pr_rows, pr_path = save_precision_recall_curve(boxes, scores, gt_boxes)
    print(f"  Courbe précision-rappel sauvegardée : {pr_path}")

    metrics = {
        "cnn_final_loss": round(losses[-1], 4),
        "dataset_split": {"train_samples": int(len(X_train)), "test_samples": int(len(X_test)), "test_ratio": 0.2},
        "cnn_train_accuracy": round(train_accuracy, 4),
        "cnn_test_accuracy": round(test_accuracy, 4),
        "frcnn_num_detections": int(len(boxes)),
        "image_source": image_source,
        "gt_boxes": [list(box) for box in gt_boxes],
        "frcnn_detections": [
            {"label": int(l), "score": round(float(s), 3),
             "box": [round(float(b), 1) for b in box]}
            for box, l, s in zip(boxes, labels, scores)
        ],
        "avg_iou": round(avg_iou, 4),
        "iou_per_gt": [round(float(i), 4) for i in ious],
        "precision_recall": [
            {
                "threshold": round(float(row["threshold"]), 3),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
            }
            for row in pr_rows
        ],
        "figures": {
            "cnn_training": "outputs/jour2/figures/cnn_training.png",
            "detection_result": result_path,
            "precision_recall": pr_path,
            "feature_maps": feature_maps_path,
        },
    }

    with open("outputs/jour2/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Métriques sauvegardées : outputs/jour2/metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
