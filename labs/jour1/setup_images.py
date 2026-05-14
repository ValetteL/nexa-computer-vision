"""Prépare les images utilisées par le lab du Jour 1.

Le script tente d'abord de télécharger une image publique simple. Si le
téléchargement échoue, il génère localement une scène synthétique. Dans les
deux cas, l'objectif est de fournir aux étudiants des images stables pour
tester OpenCV : lecture, histogramme, seuillage, contours et amélioration de
contraste.
"""
from pathlib import Path

import cv2
import numpy as np


# Graine fixe pour que les images générées soient identiques d'une machine à
# l'autre. Cela facilite la correction et la comparaison des résultats.
RNG_SEED = 42


def download_test_image() -> str:
    """Tente de télécharger une image d'exemple ; retourne un chemin ou ""."""
    # Cette image sert uniquement de support visuel optionnel. Le reste du
    # cours ne dépend pas d'elle : en cas d'échec réseau, on génère une scène.
    url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    )
    out = Path(__file__).parent / "assets" / "test_image.png"
    try:
        import urllib.request

        # `urlretrieve` suffit ici car l'image est optionnelle. Contrairement
        # aux modèles lourds, un échec de téléchargement n'arrête pas le lab.
        urllib.request.urlretrieve(url, str(out))
        img = cv2.imread(str(out))
        if img is not None:
            print(f"Downloaded test image: {out}")
            return str(out)
    except Exception:
        # Échec volontairement silencieux : la génération locale prend le relai.
        pass
    return ""


def generate_test_image() -> str:
    """Crée une scène synthétique réaliste avec formes, texte et bruit."""
    rng = np.random.default_rng(RNG_SEED)

    # Image couleur BGR OpenCV de taille 300x400. Le fond noir initial permet
    # de construire progressivement une scène contrôlée.
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Fond en léger dégradé : cela rend l'histogramme moins trivial qu'une
    # image totalement noire, et prépare l'analyse de contraste.
    for i in range(300):
        img[i, :] = [int(20 + 0.1 * i), int(20 + 0.05 * i), int(40 + 0.15 * i)]

    # Grand rectangle blanc : objet facile à segmenter par seuillage.
    cv2.rectangle(img, (30, 40), (180, 200), (255, 255, 255), -1)

    # Cercle coloré : forme alternative utile pour les contours et HOG/SIFT.
    cv2.circle(img, (300, 150), 60, (255, 200, 0), -1)

    # Formes secondaires : elles rendent l'image plus proche d'une scène réelle.
    cv2.rectangle(img, (200, 20), (260, 70), (0, 165, 255), 3)
    cv2.ellipse(img, (100, 250), (50, 30), 0, 0, 360, (128, 255, 128), -1)

    # Texte : ajoute des bords fins et de la texture pour les descripteurs.
    cv2.putText(img, "CV Lab", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Bruit faible : évite une image trop artificielle tout en restant stable.
    noise = rng.integers(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    out = Path(__file__).parent / "assets" / "test_scene.png"
    cv2.imwrite(str(out), img)
    print(f"Generated test image: {out}")
    return str(out)


def generate_low_contrast_image() -> str:
    """Crée une image sombre et peu contrastée pour `equalizeHist`."""
    rng = np.random.default_rng(RNG_SEED + 1)
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # Les rectangles ont des intensités proches du fond. L'objectif est de
    # montrer qu'un prétraitement d'égalisation peut faire ressortir les détails.
    for _ in range(5):
        x1 = int(rng.integers(10, 200))
        y1 = int(rng.integers(10, 150))
        x2 = int(rng.integers(x1 + 20, 280))
        y2 = int(rng.integers(y1 + 20, 180))
        val = int(rng.integers(30, 70))
        cv2.rectangle(img, (x1, y1), (x2, y2), (val, val, val), -1)
    noise = rng.integers(0, 10, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    out = Path(__file__).parent / "assets" / "test_low_contrast.png"
    cv2.imwrite(str(out), img)
    print(f"Generated low-contrast image: {out}")
    return str(out)


if __name__ == "__main__":
    # Point d'entrée manuel : prépare tous les fichiers attendus par le cours.
    (Path(__file__).parent / "assets").mkdir(parents=True, exist_ok=True)
    download_test_image() or generate_test_image()
    generate_low_contrast_image()
    print("All test images ready in labs/jour1/assets/")
