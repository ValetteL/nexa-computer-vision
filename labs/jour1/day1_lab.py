"""Lab complet du Jour 1 : OpenCV, IoU, HOG et SIFT.

Le script construit une mini-chaîne de vision classique : génération d'images,
extraction de boîtes par seuillage, mesure IoU, calcul de descripteurs HOG et
matching SIFT. Les résultats sont sauvegardés en JSON et en figure pour être
analysés dans le cours.
"""

import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Backend non interactif : nécessaire pour générer des figures sur serveur,
# SSH ou environnement sans écran graphique.
matplotlib.use("Agg")


def ensure_dirs() -> tuple[Path, Path]:
    """Crée les dossiers de sortie utilisés par le lab."""
    # `outputs/jour1` regroupe les métriques JSON et les artefacts du jour.
    out_dir = Path("outputs/jour1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Les figures sont séparées pour garder une arborescence lisible.
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, figures_dir


def make_synthetic_scene(shape: str, shift: int = 0) -> np.ndarray:
    """Construit une scène synthétique contrôlée pour comparer des formes."""
    # Fond noir + objet blanc : ce choix rend le seuillage très fiable, afin de
    # concentrer le lab sur le pipeline et les métriques plutôt que sur le bruit.
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    if shape == "rectangle":
        # Rectangle de référence. `shift` simule une détection décalée.
        cv2.rectangle(img, (40 + shift, 60), (180 + shift, 190), (255, 255, 255), -1)
    elif shape == "circle":
        # Forme différente utilisée pour vérifier que les descripteurs changent.
        cv2.circle(img, (120 + shift, 130), 60, (255, 255, 255), -1)
    else:
        raise ValueError("shape must be 'rectangle' or 'circle'")
    return img


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Calcule l'Intersection over Union entre deux boîtes x1, y1, x2, y2."""
    # L'intersection commence au maximum des coordonnées haut-gauche.
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])

    # L'intersection finit au minimum des coordonnées bas-droite.
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        # Les boîtes ne se recouvrent pas : l'intersection est nulle.
        return 0.0

    # IoU = aire intersection / aire union. C'est la métrique standard pour
    # juger si une boîte détectée localise correctement un objet.
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union


def bbox_from_threshold(img_gray: np.ndarray) -> tuple[int, int, int, int]:
    """Segmente l'objet clair et retourne sa boîte englobante."""
    # Le seuil fixe 127 est suffisant ici car l'objet est blanc et le fond noir.
    _, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(th)
    if points is None:
        raise ValueError("Aucun pixel détecté après seuillage ; vérifier l'image ou le seuil.")
    # OpenCV retourne x, y, w, h. On convertit en x1, y1, x2, y2 pour être
    # cohérent avec les métriques de détection utilisées aux Jours 2 et 3.
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)


def hog_features(gray_img: np.ndarray) -> np.ndarray:
    """Calcule un descripteur HOG global sur une fenêtre fixe."""
    # HOG exige une taille d'entrée stable. Sans redimensionnement, deux images
    # de tailles différentes produiraient des vecteurs incomparables.
    resized = cv2.resize(gray_img, (128, 64), interpolation=cv2.INTER_AREA)

    # Paramètres classiques : fenêtre 128x64, cellules 8x8, blocs 16x16,
    # 9 orientations. Le vecteur final a 3780 composantes.
    hog = cv2.HOGDescriptor(
        _winSize=(128, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(resized)


def sift_features(gray_img: np.ndarray) -> tuple[list, np.ndarray | None]:
    """Détecte les points clés SIFT et leurs descripteurs locaux."""
    # SIFT peut retourner `None` pour les descripteurs si l'image est trop
    # simple. Le code de matching gère ce cas pour éviter un crash.
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray_img, None)


def ratio_match_count(desc_a: np.ndarray | None, desc_b: np.ndarray | None, ratio: float = 0.75) -> int:
    """Compte les bons matches SIFT selon le ratio test de Lowe."""
    # Une absence de descripteurs n'est pas une erreur : formes géométriques
    # pleines et peu texturées peuvent produire très peu de points clés.
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) < 2:
        return 0

    # BFMatcher compare chaque descripteur au plus proche voisin dans l'autre
    # image. k=2 est nécessaire pour appliquer le ratio de Lowe.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_a, desc_b, k=2)
    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        # Le match est considéré fiable si le meilleur voisin est nettement
        # meilleur que le deuxième. Cela filtre les correspondances ambiguës.
        if m.distance < ratio * n.distance:
            good += 1
    return good


def run() -> dict:
    """Exécute tout le lab Jour 1 et retourne les métriques calculées."""
    # 1) Préparer les dossiers de sortie.
    out_dir, figures_dir = ensure_dirs()

    # 2) Construire trois images contrôlées :
    # - GT : rectangle de référence ;
    # - prédiction : même rectangle décalé ;
    # - autre : cercle servant de cas visuellement différent.
    img_gt = make_synthetic_scene("rectangle", shift=0)
    img_pred = make_synthetic_scene("rectangle", shift=12)
    img_other = make_synthetic_scene("circle", shift=0)

    # 3) Convertir en niveaux de gris. HOG, SIFT et le seuillage travaillent
    # ici sur l'intensité plutôt que sur les trois canaux couleur.
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    gray_other = cv2.cvtColor(img_other, cv2.COLOR_BGR2GRAY)

    # 4) Extraire automatiquement les boîtes et mesurer la qualité de
    # localisation par IoU.
    box_gt = bbox_from_threshold(gray_gt)
    box_pred = bbox_from_threshold(gray_pred)
    iou_score = iou(box_pred, box_gt)

    # 5) Calculer les descripteurs HOG. La norme L2 entre deux vecteurs HOG
    # mesure ici la proximité de structure globale entre les formes.
    hog_gt = hog_features(gray_gt)
    hog_pred = hog_features(gray_pred)
    hog_other = hog_features(gray_other)

    # 6) Calculer les points clés/descripteurs SIFT pour une comparaison locale.
    kp_gt, desc_gt = sift_features(gray_gt)
    kp_pred, desc_pred = sift_features(gray_pred)
    kp_other, desc_other = sift_features(gray_other)

    # 7) Matcher les descripteurs : on attend plus de bons matches entre deux
    # rectangles proches qu'entre un rectangle et un cercle.
    good_similar = ratio_match_count(desc_gt, desc_pred)
    good_different = ratio_match_count(desc_gt, desc_other)

    # 8) Générer une figure de synthèse pour vérifier visuellement le lab.
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

    # 9) Consolider les métriques dans un dictionnaire sérialisable en JSON.
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

    # 10) Sauvegarder les métriques pour les checkpoints et le rapport étudiant.
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    metrics = run()
    print(json.dumps(metrics, indent=2))
