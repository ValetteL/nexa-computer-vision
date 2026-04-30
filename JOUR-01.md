# Jour 1 - S'introduire a la vision par ordinateur et decrire des images

## 1. Objectif du chapitre

- Comprendre clairement la difference entre classification, detection et reconnaissance.
- Construire un pipeline vision simple et robuste.
- Manipuler les images avec OpenCV: lecture, conversion, seuillage, boites englobantes.
- Calculer et interpreter des metriques de base (IoU, distances entre descripteurs).
- Extraire et comparer des features HOG et SIFT sur des cas concrets.

## 2. Introduction (importance du chapitre)

Cette journee pose les bases pratiques de toute la suite du cours (CNN, Faster R-CNN, YOLO). L'objectif n'est pas seulement de connaitre des definitions, mais de produire un mini pipeline qui tourne, genere des resultats mesurables, et permet de justifier une decision technique.

Cas metier typiques:

- Controle qualite industriel: detecter un defaut et mesurer son recouvrement avec une zone attendue.
- Retail: reperer des objets sur rayon, puis reconnaitre une reference precise.
- Vision embarquee: detecter rapidement un obstacle ou une signalisation.

## 3. Prerequis

- Python 3.
- Bases en `numpy` (tableaux, dimensions).
- Bases Linux terminal.
- OpenCV, NumPy, Matplotlib installes.

## 4. Concepts cles

- **Classification**: une image -> une etiquette globale.
- **Detection**: une image -> plusieurs boites + etiquettes.
- **Reconnaissance**: une image/zone -> identite fine (personne, produit, logo).
- **Feature**: representation numerique d'une image pour comparer, classer, matcher.

Exemple concret:

- Classification: "il y a une voiture".
- Detection: "voiture en `(x1, y1, x2, y2)`".
- Reconnaissance: "cette voiture est le modele X".

## 5. Formulation mathematique (quand necessaire)

### 5.1 Formules en format math

Intersection over Union:

$$
IoU = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}
$$

Distance euclidienne entre descripteurs:

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

### 5.2 Signification des symboles

- $B_p$: boite predite.
- $B_{gt}$: boite verite terrain.
- $|\cdot|$: aire.
- $\mathbf{x}, \mathbf{y}$: vecteurs de features.
- $n$: dimension du vecteur.

### 5.3 Decomposition mathematique pas a pas

Pour l'IoU:

$$
\text{Etape 1: } A_{inter} = |B_p \cap B_{gt}|
$$

$$
\text{Etape 2: } A_{union} = |B_p| + |B_{gt}| - A_{inter}
$$

$$
\text{Etape 3: } IoU = \frac{A_{inter}}{A_{union}}
$$

### 5.4 Exemple numerique guide

Si $|B_p|=1200$, $|B_{gt}|=1000$, $A_{inter}=800$:

$$
A_{union} = 1200 + 1000 - 800 = 1400
$$

$$
IoU = \frac{800}{1400} \approx 0.571
$$

### 5.5 Resultat attendu et interpretation

- $IoU \approx 0.57$: detection correcte mais perfectible.
- Seuil courant en detection classique: $IoU \ge 0.5$.
- Plus le seuil est haut, plus la localisation exigee est precise.

## 6. Exemple Python complet (code commente)

Le script complet est dans `labs/jour1/day1_lab.py`.

```python
import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


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
    union = area_a + area_b - inter
    return inter / union


def bbox_from_threshold(img_gray):
    _, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)


def hog_features(gray_img):
    resized = cv2.resize(gray_img, (128, 64), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor((128, 64), (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(resized)


def sift_features(gray_img):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray_img, None)


img_gt = make_synthetic_scene("rectangle", shift=0)
img_pred = make_synthetic_scene("rectangle", shift=12)
img_other = make_synthetic_scene("circle", shift=0)

gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
gray_other = cv2.cvtColor(img_other, cv2.COLOR_BGR2GRAY)

box_gt = bbox_from_threshold(gray_gt)
box_pred = bbox_from_threshold(gray_pred)
iou_score = iou(box_pred, box_gt)

hog_gt = hog_features(gray_gt)
hog_pred = hog_features(gray_pred)
hog_other = hog_features(gray_other)

kp_gt, desc_gt = sift_features(gray_gt)
kp_pred, desc_pred = sift_features(gray_pred)
kp_other, desc_other = sift_features(gray_other)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches_similar = bf.knnMatch(desc_gt, desc_pred, k=2)
matches_different = bf.knnMatch(desc_gt, desc_other, k=2)


def ratio_count(matches, ratio=0.75):
    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good += 1
    return good


results = {
    "iou_score": float(iou_score),
    "hog_dimension": int(hog_gt.shape[0]),
    "hog_shifted_l2": float(np.linalg.norm(hog_gt - hog_pred)),
    "hog_different_l2": float(np.linalg.norm(hog_gt - hog_other)),
    "sift_kp_gt": len(kp_gt),
    "sift_kp_pred": len(kp_pred),
    "sift_kp_other": len(kp_other),
    "sift_good_matches_similar": ratio_count(matches_similar),
    "sift_good_matches_different": ratio_count(matches_different),
}

Path("outputs/jour1").mkdir(parents=True, exist_ok=True)
Path("outputs/jour1/metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
```

## 7. Explication detaillee du code

- Bloc 1: cree des scenes synthetiques pour eviter toute dependance a un dataset externe.
- Bloc 2: calcule automatiquement des boites par seuillage et en deduit l'IoU.
- Bloc 3: extrait HOG pour comparer deux formes proches vs differentes.
- Bloc 4: extrait SIFT puis compare les matches "similaires" et "differents" via ratio test.
- Bloc 5: sauvegarde les metriques dans `outputs/jour1/metrics.json` pour tracabilite.

## 8. Lab pas a pas (tres guide)

### 8.1 Objectif du lab

Construire un pipeline de comparaison entre scenes, mesurer IoU, HOG, SIFT, puis interpreter les metriques.

### 8.2 Setup environnement

Si `venv` est disponible:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install opencv-python numpy matplotlib
```

Si `venv` n'est pas disponible (cas Debian minimal):

```bash
sudo apt install python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy matplotlib
```

### 8.3 Etapes d'execution

1. Verifier la presence du script `labs/jour1/day1_lab.py`.
2. Lancer `python3 labs/jour1/day1_lab.py`.
3. Verifier que `outputs/jour1/metrics.json` existe.
4. Verifier que `outputs/jour1/figures/jour1_overview.png` existe.
5. Lire les metriques et comparer cas "similaire" vs "different".

### 8.4 Verification (checkpoints)

- Checkpoint A: `iou_score` est dans l'intervalle $(0,1]$.
- Checkpoint B: `hog_different_l2` est generalement plus grand que `hog_shifted_l2`.
- Checkpoint C: `sift_good_matches_similar` est superieur a `sift_good_matches_different`.

### 8.5 Erreurs frequentes et correction

- `ModuleNotFoundError: cv2` -> OpenCV non installe -> installer `opencv-python`.
- `No module named pip/venv` -> systeme minimal -> installer `python3-pip` et `python3-venv`.
- `desc_* is None` (peu probable sur ces scenes) -> verifier que les formes sont bien dessinees en blanc sur fond noir.

## 9. Validation technique du code

- Validation effectuee dans cet environnement: verification syntaxique reussie avec `python3 -m py_compile labs/jour1/day1_lab.py`.
- Execution complete bloquee ici par absence de `pip/venv` systeme, donc dependances OpenCV/NumPy/Matplotlib non installables localement sans paquets OS.
- Commande d'execution a lancer des que dependances installees: `python3 labs/jour1/day1_lab.py`.

## 10. Resume et points a retenir

- La distinction classification/detection/reconnaissance est indispensable avant les modeles deep.
- L'IoU formalise la qualite de localisation.
- HOG est une base solide pour decrire les contours globaux.
- SIFT est utile pour la similarite locale et le matching.
- Un cours exploitable doit produire metriques + artefacts de sortie, pas seulement de la theorie.

## 11. Mini exercices

- Exercice 1: modifier `shift` de 12 a 30 et observer l'impact sur $IoU$.
- Exercice 2: remplacer le rectangle par un ellipse et comparer HOG.
- Exercice 3: faire varier le ratio test SIFT (0.6, 0.75, 0.9) et analyser le compromis precision/rappel de matching.

## 12. Livrables attendus

- Script `labs/jour1/day1_lab.py` execute sans erreur.
- `outputs/jour1/metrics.json`.
- `outputs/jour1/figures/jour1_overview.png`.
- Mini rapport d'interpretation (5 a 10 lignes).
