#!/usr/bin/env python3
"""
Lab Jour 3 — YOLOv8, comparaison et optimisation
Comparer Faster R-CNN et YOLO sur les mêmes images

Ce lab illustre le compromis vitesse/précision : Faster R-CNN représente une
approche two-stage plus lourde, tandis que YOLOv8n représente une approche
one-stage adaptée au temps réel.
"""

import json
import os
import time
import numpy as np
import cv2
import torch
import matplotlib
# Backend non interactif : les graphiques sont sauvegardés dans `outputs/` sans
# nécessiter de fenêtre graphique.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


REAL_IMAGE_PATH = "labs/shared/assets/coco_dog.jpg"

# Boîte vérité terrain approximative du chien dans l'image réelle. Elle est
# utilisée pour comparer l'IoU de Faster R-CNN et YOLOv8n.
REAL_DOG_GT_BOX = (50, 35, 645, 555)
RANDOM_SEED = 42


def set_reproducible_seed(seed=RANDOM_SEED):
    """Fixe les graines pseudo-aléatoires pour stabiliser le lab."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_test_image(path, seed=42):
    """Crée une image de test avec des formes variées."""
    rng = np.random.RandomState(seed)

    # Fallback historique : il garde le lab exécutable même si l'image réelle
    # est absente. Les formes simulent grossièrement des objets à détecter.
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Fond dégradé : rend l'image moins triviale qu'un fond uniforme.
    for y in range(480):
        img[y, :] = [int(30 + 20 * y / 480), int(30 + 15 * y / 480), int(50 + 25 * y / 480)]

    # Formes de tailles différentes pour discuter petits/moyens/grands objets.
    cv2.rectangle(img, (80, 100), (200, 350), (180, 120, 80), -1)
    cv2.circle(img, (450, 250), 80, (200, 150, 0), -1)
    cv2.rectangle(img, (300, 300), (500, 420), (100, 100, 100), -1)
    cv2.rectangle(img, (50, 50), (120, 80), (220, 220, 220), -1)

    cv2.imwrite(path, img)
    return img


def load_detection_image(path):
    """Charge l'image réelle si disponible, sinon génère l'image synthétique historique."""
    # L'image réelle est privilégiée car les modèles pré-entraînés sur COCO
    # répondent beaucoup mieux à des objets naturels qu'à des formes abstraites.
    if os.path.exists(REAL_IMAGE_PATH):
        img = cv2.imread(REAL_IMAGE_PATH)
        if img is None:
            raise FileNotFoundError(f"Image non lisible : {REAL_IMAGE_PATH}")
        return img, REAL_IMAGE_PATH, [REAL_DOG_GT_BOX], "real_coco_dog"

    img = create_test_image(path)
    gt_boxes = [(70, 90, 210, 360), (360, 160, 540, 340), (290, 290, 510, 430), (40, 40, 130, 90)]
    return img, path, gt_boxes, "synthetic_shapes"


def compute_iou(box_a, box_b):
    """Calcule l'IoU entre deux boîtes."""
    # Fonction autonome pour éviter une dépendance au lab Jour 1. Le format est
    # toujours x1, y1, x2, y2.
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


def run_frcnn(img, model, score_thresh=0.25):
    """Exécute Faster R-CNN."""
    # Conversion OpenCV BGR -> RGB puis HWC -> CHW, comme dans le Jour 2.
    tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        pred = model([tensor])[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    # Filtrage par score pour contrôler le compromis précision/rappel.
    mask = scores >= score_thresh
    return boxes[mask], scores[mask], labels[mask]


def run_yolo(img, model, conf_thresh=0.25):
    """Exécute YOLOv8."""
    # Ultralytics accepte directement une image NumPy OpenCV. Le seuil `conf`
    # joue le même rôle que `score_thresh` côté Faster R-CNN.
    results = model(img, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, labels


def benchmark(name, run_fn, img, num_runs=5):
    """Mesure le temps d'inférence."""
    # On répète plusieurs fois pour lisser les variations CPU ponctuelles. Le
    # premier passage peut inclure des coûts de cache/initialisation.
    times = []
    for _ in range(num_runs):
        start = time.time()
        run_fn(img)
        times.append(time.time() - start)
    return {"mean": float(np.mean(times)), "std": float(np.std(times))}


def precision_recall_at_threshold(boxes, scores, gt_boxes, score_thresh, iou_thresh=0.5):
    """Calcule précision/rappel pour un seuil de score donné."""
    # Sélection des prédictions conservées à ce seuil.
    selected = [(box, score) for box, score in zip(boxes, scores) if score >= score_thresh]
    matched_gt = set()
    tp = 0
    fp = 0

    for pred_box, _ in selected:
        best_iou = 0.0
        best_idx = None

        # Recherche du meilleur objet GT non encore associé à cette prédiction.
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou_val = compute_iou((pred_box[0], pred_box[1], pred_box[2], pred_box[3]), gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx
        if best_iou >= iou_thresh and best_idx is not None:
            # La prédiction localise suffisamment un objet réel.
            tp += 1
            matched_gt.add(best_idx)
        else:
            # La prédiction ne correspond à aucun objet GT : faux positif.
            fp += 1

    # Les GT restantes sont les objets manqués par le détecteur.
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": score_thresh, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def average_precision_from_pr_rows(pr_rows):
    """Approximation pédagogique de l'AP à partir des points précision/rappel."""
    # Comme au Jour 2, ce calcul sert à comprendre l'idée AP/mAP sans mettre en
    # place toute la procédure COCO officielle.
    by_recall = {}
    for row in pr_rows:
        recall = float(row["recall"])
        precision = float(row["precision"])
        by_recall[recall] = max(by_recall.get(recall, 0.0), precision)

    points = sorted(by_recall.items(), key=lambda item: item[0])
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in points:
        delta = max(0.0, recall - prev_recall)
        ap += precision * delta
        prev_recall = max(prev_recall, recall)
    return ap


def draw_detections(img, boxes, scores, labels, color, title):
    """Dessine les détections sur l'image."""
    # L'image est copiée pour conserver une version brute réutilisable.
    out = img.copy()
    for box, score, label in zip(boxes, scores, labels):
        # Seul le score est affiché pour éviter de surcharger la visualisation.
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return out


def best_ious(pred_boxes, gt_boxes):
    """Calcule le meilleur IoU pour chaque GT."""
    # Pour chaque objet réel, on cherche la prédiction qui le recouvre le mieux.
    # Cette métrique simple est très lisible pour comparer deux détecteurs.
    ious_per_gt = []
    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou_val = compute_iou(
                (pred_box[0], pred_box[1], pred_box[2], pred_box[3]),
                gt_box
            )
            best_iou = max(best_iou, iou_val)
        ious_per_gt.append(best_iou)
    return ious_per_gt


def main():
    # Orchestration complète : chargement modèles, benchmark, IoU, figures,
    # sweep de seuil et rapport JSON.
    set_reproducible_seed()
    os.makedirs("outputs/jour3/figures", exist_ok=True)

    print("Chargement des modèles...")

    # Faster R-CNN : modèle lourd mais précis, chargé depuis torchvision.
    model_frcnn = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=0.25
    )
    model_frcnn.eval()

    # YOLOv8n : version nano, plus rapide et adaptée à la démonstration CPU.
    model_yolo = YOLO("yolov8n.pt")

    img_path = "labs/jour3/test_image.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    # On charge l'image réelle si elle existe, sinon le fallback synthétique.
    img, img_path, gt_boxes, image_source = load_detection_image(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Image de test : {img_path} ({img.shape[1]}x{img.shape[0]}, {image_source})")

    print("\nBenchmark Faster R-CNN...")

    # Benchmark et prédictions finales sont séparés : le benchmark mesure le
    # temps moyen, puis on réutilise une prédiction pour les métriques.
    frcnn_time = benchmark("Faster R-CNN", lambda im: run_frcnn(im, model_frcnn), img, num_runs=3)
    frcnn_boxes, frcnn_scores, frcnn_labels = run_frcnn(img, model_frcnn)
    print(f"  Temps : {frcnn_time['mean']:.3f}s ± {frcnn_time['std']:.3f}s")
    print(f"  Détections : {len(frcnn_boxes)}")

    print("\nBenchmark YOLOv8n...")
    yolo_time = benchmark("YOLOv8n", lambda im: run_yolo(im, model_yolo), img, num_runs=3)
    yolo_boxes, yolo_scores, yolo_labels = run_yolo(img, model_yolo)
    print(f"  Temps : {yolo_time['mean']:.3f}s ± {yolo_time['std']:.3f}s")
    print(f"  Détections : {len(yolo_boxes)}")

    speedup = frcnn_time["mean"] / yolo_time["mean"] if yolo_time["mean"] > 0 else float("inf")
    print(f"\nYOLOv8n est {speedup:.1f}x plus rapide que Faster R-CNN")

    # Figure 1 : comparaison des temps d'inférence.
    plt.figure(figsize=(8, 4))
    models = ["Faster R-CNN", "YOLOv8n"]
    times = [frcnn_time["mean"], yolo_time["mean"]]
    stds = [frcnn_time["std"], yolo_time["std"]]
    colors = ["steelblue", "coral"]
    bars = plt.bar(models, times, yerr=stds, color=colors, capsize=5, width=0.5)
    plt.ylabel("Temps d'inférence (s)")
    plt.title(f"Comparaison de vitesse (x{speedup:.1f} plus rapide avec YOLO)")
    plt.grid(axis="y", alpha=0.3)
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{t:.3f}s", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/jour3/figures/speed_comparison.png", dpi=130)
    plt.close()

    frcnn_ious = best_ious(frcnn_boxes, gt_boxes)
    yolo_ious = best_ious(yolo_boxes, gt_boxes)

    print(f"\nIoU par GT (Faster R-CNN) : {[f'{i:.3f}' for i in frcnn_ious]}")
    print(f"IoU par GT (YOLOv8n)        : {[f'{i:.3f}' for i in yolo_ious]}")
    print(f"IoU moyen Faster R-CNN : {np.mean(frcnn_ious):.3f}")
    print(f"IoU moyen YOLOv8n        : {np.mean(yolo_ious):.3f}")

    # Figure 2 : comparaison IoU par objet GT.
    plt.figure(figsize=(8, 4))
    x = np.arange(len(gt_boxes))
    width = 0.35
    plt.bar(x - width/2, frcnn_ious, width, label="Faster R-CNN", color="steelblue")
    plt.bar(x + width/2, yolo_ious, width, label="YOLOv8n", color="coral")
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Seuil 0.5")
    plt.xlabel("Objet GT")
    plt.ylabel("Meilleur IoU")
    plt.title("Comparaison IoU par objet")
    plt.xticks(x, [f"GT {i+1}" for i in range(len(gt_boxes))])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/jour3/figures/iou_comparison.png", dpi=130)
    plt.close()

    # Figure 3 : visualisation côte à côte des détections.
    frcnn_vis = draw_detections(img, frcnn_boxes, frcnn_scores, frcnn_labels,
                                (255, 0, 0), f"Faster R-CNN ({len(frcnn_boxes)} detections)")
    yolo_vis = draw_detections(img, yolo_boxes, yolo_scores, yolo_labels,
                               (0, 255, 0), f"YOLOv8n ({len(yolo_boxes)} detections)")

    combined = np.hstack([frcnn_vis, yolo_vis])
    overlay_path = "outputs/jour3/figures/detection_overlay.png"
    cv2.imwrite(overlay_path, combined)
    print(f"\nVisualisation sauvegardée : {overlay_path}")

    print("\nOptimisation — Variation du seuil de confiance (YOLOv8n)...")
    thresholds = [0.1, 0.25, 0.5, 0.75]
    threshold_results = []
    for thresh in thresholds:
        # Le sweep montre comment le seuil modifie le nombre de détections et
        # donc le compromis entre faux positifs et faux négatifs.
        boxes, scores, labels = run_yolo(img, model_yolo, conf_thresh=thresh)
        ious = best_ious(boxes, gt_boxes)
        avg_iou = float(np.mean(ious)) if ious else 0.0
        pr = precision_recall_at_threshold(boxes, scores, gt_boxes, thresh)
        threshold_results.append({
            "threshold": thresh,
            "num_detections": len(boxes),
            "avg_iou": round(avg_iou, 4),
            "precision": round(float(pr["precision"]), 4),
            "recall": round(float(pr["recall"]), 4),
        })
        print(f"  Seuil {thresh}: {len(boxes)} détections, IoU moyen={avg_iou:.3f}")

    yolo_ap50 = average_precision_from_pr_rows(threshold_results)

    # Figure 4 : nombre de détections et IoU moyen en fonction du seuil.
    plt.figure(figsize=(8, 4))
    xs = [row["threshold"] for row in threshold_results]
    counts = [row["num_detections"] for row in threshold_results]
    avg_ious = [row["avg_iou"] for row in threshold_results]
    ax1 = plt.gca()
    ax1.plot(xs, counts, marker="o", color="steelblue", label="Détections")
    ax1.set_xlabel("Seuil de confiance")
    ax1.set_ylabel("Nombre de détections", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, avg_ious, marker="s", color="coral", label="IoU moyen")
    ax2.set_ylabel("IoU moyen", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.set_ylim(-0.05, 1.05)
    plt.title("Impact du seuil de confiance YOLOv8n")
    plt.tight_layout()
    threshold_path = "outputs/jour3/figures/threshold_sweep.png"
    plt.savefig(threshold_path, dpi=130)
    plt.close()
    print(f"  Courbe de seuil sauvegardée : {threshold_path}")

    metrics = {
        # Rapport JSON : structure pensée pour être relue dans un compte rendu
        # étudiant ou par un script de validation.
        "speed": {
            "faster_rcnn_mean_s": round(frcnn_time["mean"], 4),
            "yolov8n_mean_s": round(yolo_time["mean"], 4),
            "speedup_yolo_vs_frcnn": round(speedup, 2),
        },
        "image_source": image_source,
        "gt_boxes": [list(box) for box in gt_boxes],
        "iou": {
            "faster_rcnn_avg": round(float(np.mean(frcnn_ious)), 4),
            "yolov8n_avg": round(float(np.mean(yolo_ious)), 4),
            "faster_rcnn_per_gt": [round(float(i), 4) for i in frcnn_ious],
            "yolov8n_per_gt": [round(float(i), 4) for i in yolo_ious],
        },
        "num_detections": {
            "faster_rcnn": int(len(frcnn_boxes)),
            "yolov8n": int(len(yolo_boxes)),
        },
        "threshold_sweep": threshold_results,
        "map50_simplified": round(float(yolo_ap50), 4),
        "metric_note": "mAP@0.5 simplifié pour usage pédagogique sur une seule classe et quelques seuils, pas une évaluation COCO officielle.",
        "figures": {
            "speed_comparison": "outputs/jour3/figures/speed_comparison.png",
            "iou_comparison": "outputs/jour3/figures/iou_comparison.png",
            "detection_overlay": overlay_path,
            "threshold_sweep": threshold_path,
        },
    }

    with open("outputs/jour3/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétriques sauvegardées : outputs/jour3/metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
