# Jour 1 - S'introduire a la vision par ordinateur et decrire des images

## 1. Objectif du chapitre

- Comprendre les differences entre classification, detection et reconnaissance.
- Maitriser un pipeline de base en vision par ordinateur.
- Manipuler des images avec OpenCV.
- Introduire HOG et SIFT comme premieres representations visuelles.

## 2. Introduction (importance du chapitre)

La vision par ordinateur permet d'extraire de l'information exploitable depuis des images ou des videos. Ce chapitre est fondamental car il donne la base conceptuelle et pratique qui sera reutilisee pour les CNN, Faster R-CNN et YOLO.

## 3. Prerequis

- Python 3 installe.
- Bases sur les tableaux `numpy`.
- Environnement virtuel Python.

## 4. Concepts cles

- **Classification**: predire une classe globale pour une image.
- **Detection**: localiser et classifier des objets avec des boites.
- **Reconnaissance**: identifier un objet/individu precis.
- **Feature**: representation numerique de l'information visuelle.

## 5. Formulation mathematique (quand necessaire)

### 5.1 Formules en format math

Intersection over Union:

$$
IoU = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}
$$

Distance euclidienne entre deux descripteurs (matching SIFT):

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

### 5.2 Signification des symboles

- $B_p$: boite predite.
- $B_{gt}$: boite verite terrain.
- $|\cdot|$: aire d'une region.
- $\mathbf{x}, \mathbf{y}$: vecteurs descripteurs.
- $n$: dimension descripteur.

### 5.3 Decomposition mathematique pas a pas

Pour l'IoU:

$$
\text{Etape 1: calculer l'aire d'intersection } |B_p \cap B_{gt}|
$$

$$
\text{Etape 2: calculer l'aire d'union } |B_p \cup B_{gt}| = |B_p| + |B_{gt}| - |B_p \cap B_{gt}|
$$

$$
\text{Etape 3: diviser intersection par union}
$$

### 5.4 Exemple numerique guide

Si $|B_p|=1200$, $|B_{gt}|=1000$, $|B_p \cap B_{gt}|=800$:

$$
|B_p \cup B_{gt}| = 1200 + 1000 - 800 = 1400
$$

$$
IoU = \frac{800}{1400} \approx 0.571
$$

### 5.5 Resultat attendu et interpretation

- IoU proche de 1: tres bonne superposition.
- IoU proche de 0: mauvaise localisation.
- En detection, un seuil typique d'acceptation commence souvent a $0.5$.

## 6. Exemple Python complet (code commente)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Charger une image en BGR puis convertir en RGB pour affichage
img_bgr = cv2.imread("data/image.jpg")
if img_bgr is None:
    raise FileNotFoundError("Image introuvable: data/image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2) Pretraitements simples
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (128, 64), interpolation=cv2.INTER_AREA)
_, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# 3) HOG
hog = cv2.HOGDescriptor(
    _winSize=(128, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)
hog_features = hog.compute(img_resized)

# 4) SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img_gray, None)
img_kp = cv2.drawKeypoints(
    img_gray,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# 5) Affichage
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].imshow(img_rgb)
axs[0].set_title("Image RGB")
axs[0].axis("off")

axs[1].imshow(img_thresh, cmap="gray")
axs[1].set_title("Seuillage binaire")
axs[1].axis("off")

axs[2].imshow(img_kp, cmap="gray")
axs[2].set_title("Points cles SIFT")
axs[2].axis("off")

plt.tight_layout()
plt.show()

print("Dimension HOG:", hog_features.shape)
print("Nombre de keypoints SIFT:", len(keypoints))
print("Shape descriptors SIFT:", None if descriptors is None else descriptors.shape)
```

## 7. Explication detaillee du code

- Bloc 1: charge l'image et gere explicitement le cas d'erreur de chemin.
- Bloc 2: applique des transformations de base utiles en preprocessing.
- Bloc 3: extrait un vecteur HOG global de taille fixe.
- Bloc 4: detecte des points SIFT et calcule leurs descripteurs locaux.
- Bloc 5: visualise les sorties pour verifier rapidement la qualite.

## 8. Lab pas a pas (tres guide)

### 8.1 Objectif du lab

Comparer HOG et SIFT sur un mini jeu de donnees et interpreter les differences.

### 8.2 Setup environnement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install opencv-python numpy matplotlib
```

### 8.3 Etapes d'execution

1. Creer `data/` et ajouter au moins 10 images.
2. Creer `lab_jour1.py` avec le code de la section 6.
3. Executer `python3 lab_jour1.py`.
4. Verifier les 3 affichages (RGB, seuillage, keypoints).
5. Noter les dimensions HOG et le nombre de keypoints.
6. Refaire l'experience sur plusieurs images et comparer.

### 8.4 Verification (checkpoints)

- Checkpoint A: le script s'execute sans erreur.
- Checkpoint B: `Dimension HOG` est non nulle.
- Checkpoint C: `Nombre de keypoints SIFT` varie selon l'image.

### 8.5 Erreurs frequentes et correction

- Erreur `Image introuvable` -> chemin faux -> corriger `data/image.jpg`.
- Erreur SIFT indisponible -> version OpenCV inadaptee -> mettre a jour `opencv-python`.
- Affichage couleurs incorrect -> oubli conversion BGR->RGB -> utiliser `cv2.cvtColor`.

## 9. Resume et points a retenir

- Classification, detection et reconnaissance sont trois taches differentes.
- L'IoU est une mesure cle pour evaluer une localisation.
- HOG donne une representation globale des gradients.
- SIFT fournit des points clefs et descripteurs locaux robustes.
- Les visualisations intermediaires accelerent le debug.

## 10. Mini exercices

- Exercice 1: modifier le seuil binaire et comparer l'effet.
- Exercice 2: changer la taille d'image HOG et observer la dimension de sortie.
- Exercice 3: comparer deux images avec une distance moyenne entre descripteurs SIFT.

## 11. Livrables attendus

- Script `lab_jour1.py` fonctionnel.
- Dossier de sorties visuelles.
- Court compte-rendu avec interpretation des resultats.
