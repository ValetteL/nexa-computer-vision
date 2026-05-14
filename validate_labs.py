#!/usr/bin/env python3
"""Validation technique rapide des labs du module.

Ce script évite de lancer les modèles lourds. Il vérifie les imports,
les fonctions critiques et la présence des assets attendus.
"""

from pathlib import Path

import cv2
import numpy as np

from labs.jour1.day1_lab import bbox_from_threshold, iou, ratio_match_count
from labs.jour2.day2_lab import average_precision_from_pr_rows, train_test_split


def check_assets() -> list[str]:
    required = [
        "labs/shared/assets/coco_dog.jpg",
        "requirements.txt",
        "JOUR-01.md",
        "JOUR-02.md",
        "JOUR-03.md",
    ]
    return [path for path in required if not Path(path).exists()]


def check_iou() -> None:
    score = iou((10, 10, 50, 50), (30, 30, 70, 70))
    assert 0 < score < 1, f"IoU inattendue: {score}"


def check_bbox_error() -> None:
    empty = np.zeros((32, 32), dtype=np.uint8)
    try:
        bbox_from_threshold(empty)
    except ValueError:
        return
    raise AssertionError("bbox_from_threshold doit signaler une image sans objet")


def check_sift_empty_descriptors() -> None:
    good = ratio_match_count(None, None)
    assert good == 0, f"ratio_match_count doit retourner 0, reçu {good}"


def check_train_test_split() -> None:
    import torch

    x = torch.arange(30).reshape(10, 3)
    y = torch.arange(10)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_ratio=0.2, seed=42)
    assert len(x_train) == len(y_train) == 8
    assert len(x_test) == len(y_test) == 2


def check_ap() -> None:
    rows = [
        {"recall": 0.5, "precision": 1.0},
        {"recall": 1.0, "precision": 0.5},
    ]
    ap = average_precision_from_pr_rows(rows)
    assert 0 <= ap <= 1, f"AP hors bornes: {ap}"


def main() -> None:
    missing = check_assets()
    if missing:
        raise FileNotFoundError(f"Assets manquants: {missing}")

    assert cv2.__version__, "OpenCV non disponible"
    check_iou()
    check_bbox_error()
    check_sift_empty_descriptors()
    check_train_test_split()
    check_ap()
    print("Validation rapide OK")


if __name__ == "__main__":
    main()
