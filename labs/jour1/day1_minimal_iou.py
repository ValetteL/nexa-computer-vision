"""Version minimale du Jour 1 : calculer une IoU de bout en bout.

Ce script volontairement court isole la compétence centrale : localiser deux
objets par seuillage, construire leurs boîtes englobantes, puis mesurer leur
recouvrement avec l'Intersection over Union.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def make_synthetic_scene(shape: str, shift: int = 0) -> np.ndarray:
    """Construit une scène simple avec une forme blanche sur fond noir."""
    # Image BGR 256x256. Même si l'objet est blanc/noir, on garde 3 canaux pour
    # rester cohérent avec les images OpenCV couleur des autres labs.
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    if shape == "rectangle":
        # Le décalage horizontal simule une prédiction imparfaite.
        cv2.rectangle(img, (40 + shift, 60), (180 + shift, 190), (255, 255, 255), -1)
    elif shape == "circle":
        # Le cercle est prévu pour comparaison/extension, même si ce script
        # minimal utilise surtout le rectangle.
        cv2.circle(img, (120 + shift, 130), 60, (255, 255, 255), -1)
    else:
        raise ValueError("shape must be 'rectangle' or 'circle'")
    return img


def iou(box_a, box_b):
    """Calcule l'Intersection over Union entre deux boîtes x1, y1, x2, y2."""
    # Coin haut-gauche de l'intersection : maximum des deux boîtes.
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])

    # Coin bas-droit de l'intersection : minimum des deux boîtes.
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # Si les coordonnées se croisent dans le mauvais sens, il n'y a pas de zone
    # commune : l'IoU vaut 0.
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # Aire commune divisée par l'aire totale couverte par les deux boîtes.
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def bbox_from_threshold(gray):
    """Seuillage puis extraction de la boîte englobante des pixels blancs."""
    # Tout pixel au-dessus de 127 devient blanc. Les formes synthétiques étant
    # blanches sur fond noir, cela isole directement l'objet.
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(th)
    if points is None:
        raise ValueError("Aucun pixel détecté après seuillage ; vérifier l'image ou le seuil.")

    # `boundingRect` retourne x, y, largeur, hauteur. On convertit en format
    # x1, y1, x2, y2 pour le calcul d'IoU.
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)


# Scène de référence : rectangle sans décalage.
img_gt = make_synthetic_scene("rectangle", 0)

# Scène prédite : même rectangle, décalé de 12 pixels vers la droite.
img_pred = make_synthetic_scene("rectangle", 12)

# Conversion en niveaux de gris : le seuillage travaille sur une seule intensité
# par pixel, pas sur les trois canaux BGR.
gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)

# Extraction automatique des deux boîtes, comme dans un pipeline de détection.
box_gt = bbox_from_threshold(gray_gt)
box_pred = bbox_from_threshold(gray_pred)

# Sauvegarde JSON : format lisible par un humain et réutilisable dans un rapport.
metrics = {"iou_score": float(iou(box_pred, box_gt)), "bbox_gt": box_gt, "bbox_pred": box_pred}
Path("outputs/jour1").mkdir(parents=True, exist_ok=True)
Path("outputs/jour1/metrics_minimal.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
