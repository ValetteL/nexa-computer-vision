#!/usr/bin/env python3
"""Validation technique rapide des labs du module.

Ce script évite de lancer les modèles lourds. Il vérifie les imports,
les fonctions critiques et la présence des assets attendus.

Il sert de garde-fou avant une session de cours : si ce script passe, les
éléments de base du dépôt sont en place et les fonctions pédagogiques clés se
comportent comme prévu.
"""

from pathlib import Path

import cv2
import numpy as np

from labs.jour1.day1_lab import bbox_from_threshold, iou, ratio_match_count
from labs.jour2.day2_lab import average_precision_from_pr_rows, train_test_split


def check_assets() -> list[str]:
    """Retourne la liste des fichiers indispensables absents du projet."""
    # On ne vérifie pas ici les modèles lourds téléchargeables automatiquement
    # afin de garder cette validation rapide et utilisable hors ligne.
    required = [
        "labs/shared/assets/coco_dog.jpg",
        "labs/projet_face/face_actor_demo.py",
        "PROJET-RECONNAISSANCE-FACIALE.md",
        "requirements.txt",
        "JOUR-01.md",
        "JOUR-02.md",
        "JOUR-03.md",
    ]
    return [path for path in required if not Path(path).exists()]


def check_iou() -> None:
    """Vérifie que l'IoU retourne une valeur intermédiaire cohérente."""
    # Les deux boîtes se recouvrent partiellement : le score doit donc être
    # strictement entre 0 et 1.
    score = iou((10, 10, 50, 50), (30, 30, 70, 70))
    assert 0 < score < 1, f"IoU inattendue: {score}"


def check_bbox_error() -> None:
    """Vérifie qu'une image vide produit une erreur explicite."""
    # Une image noire ne contient aucun pixel blanc après seuillage. Le lab doit
    # signaler ce cas au lieu de provoquer une erreur OpenCV peu lisible.
    empty = np.zeros((32, 32), dtype=np.uint8)
    try:
        bbox_from_threshold(empty)
    except ValueError:
        return
    raise AssertionError("bbox_from_threshold doit signaler une image sans objet")


def check_sift_empty_descriptors() -> None:
    """Vérifie que le matching SIFT accepte l'absence de descripteurs."""
    # SIFT peut retourner None sur des images trop simples. Le code pédagogique
    # doit traiter ce cas comme 0 bon match, pas comme un crash.
    good = ratio_match_count(None, None)
    assert good == 0, f"ratio_match_count doit retourner 0, reçu {good}"


def check_train_test_split() -> None:
    """Vérifie le découpage train/test utilisé pour la compétence C3.2."""
    import torch

    # Petit tenseur factice : 10 échantillons, test_ratio=0.2 => 8 train / 2 test.
    x = torch.arange(30).reshape(10, 3)
    y = torch.arange(10)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_ratio=0.2, seed=42)
    assert len(x_train) == len(y_train) == 8
    assert len(x_test) == len(y_test) == 2


def check_ap() -> None:
    """Vérifie que l'AP pédagogique reste bornée entre 0 et 1."""
    # Deux points précision/rappel suffisent pour tester le calcul sans lancer
    # de modèle de détection.
    rows = [
        {"recall": 0.5, "precision": 1.0},
        {"recall": 1.0, "precision": 0.5},
    ]
    ap = average_precision_from_pr_rows(rows)
    assert 0 <= ap <= 1, f"AP hors bornes: {ap}"


def check_face_apis() -> None:
    """Vérifie que l'installation OpenCV expose YuNet et SFace."""
    # Ces APIs sont nécessaires au projet bonus de reconnaissance faciale.
    assert hasattr(cv2, "FaceDetectorYN"), "OpenCV FaceDetectorYN indisponible"
    assert hasattr(cv2, "FaceRecognizerSF"), "OpenCV FaceRecognizerSF indisponible"


def main() -> None:
    """Exécute toutes les validations rapides du dépôt."""
    missing = check_assets()
    if missing:
        raise FileNotFoundError(f"Assets manquants: {missing}")

    # Validation minimale de l'import OpenCV avant de tester les fonctions qui
    # en dépendent.
    assert cv2.__version__, "OpenCV non disponible"
    check_iou()
    check_bbox_error()
    check_sift_empty_descriptors()
    check_train_test_split()
    check_ap()
    check_face_apis()
    print("Validation rapide OK")


if __name__ == "__main__":
    main()
