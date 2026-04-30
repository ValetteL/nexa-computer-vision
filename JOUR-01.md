# Jour 1 - Fondamentaux de vision par ordinateur et descripteurs classiques

## 1. Objectif du chapitre

- Distinguer clairement classification, detection et reconnaissance.
- Comprendre la logique complete d'un pipeline vision, de l'image brute a l'evaluation.
- Manipuler des images avec OpenCV (conversion, seuillage, extraction de region).
- Introduire HOG et SIFT avec une lecture technique et metier.
- Produire un mini lab reproductible avec sorties mesurables.

Alignement syllabus Jour 1:

- Bloc A: "S'introduire a la vision par ordinateur (3h30)".
- Bloc B: "Decrire des images (3h30)".

## 2. Introduction (importance du chapitre)

Ce premier chapitre sert de socle pour tout le module. Avant d'utiliser des modeles profonds, il faut savoir:

- ce qu'on veut exactement predire,
- comment preparer et representer une image,
- comment mesurer si le resultat est bon.

En pratique, cette rigueur evite de "faire tourner un modele" sans savoir interpreter ses sorties. Le Jour 1 te donne donc une methode: **objectif clair -> transformation image -> mesure -> interpretation**.

## 3. Prerequis

- Python 3.
- Bases en tableaux `numpy`.
- Notions de base sur les pixels et l'image en niveaux de gris.
- OpenCV, NumPy et Matplotlib installes.

## 4. Concepts cles

### 4.1 Classification

- Entree: image complete.
- Sortie: une seule classe.
- Question: "Qu'est-ce qu'il y a dans cette image?"

Reference: cours CS231n (vision, classification, detection) [R1][R2].

### 4.2 Detection

- Entree: image complete.
- Sortie: une ou plusieurs boites + classes.
- Question: "Ou sont les objets et quels sont-ils?"

Reference: cours CS231n (object detection) [R1].

### 4.3 Reconnaissance

- Entree: objet ou region detectee.
- Sortie: identite fine ou classe detaillee.
- Question: "Quel objet exact/personne exacte est-ce?"

Reference: cours CS231n (representations visuelles) [R2].

### 4.4 Coherence metier

- En controle qualite: detection d'une zone defectueuse + score de recouvrement.
- En retail: detection de produit + reconnaissance de reference.
- En securite: detection d'un visage + reconnaissance d'identite.

## 5. Formulation mathematique (quand necessaire)

### 5.1 Contexte mathematique

On utilise deux familles de mesures complementaires:

- une mesure de **localisation** (IoU),
- une mesure de **similarite de representation** (distance euclidienne sur descripteurs).

### 5.2 Symboles et notations

- $B_p$: boite predite.
- $B_{gt}$: boite verite terrain (ground truth).
- $|\cdot|$: aire.
- $A_{inter}$: aire d'intersection.
- $A_{union}$: aire d'union.
- $\mathbf{x}, \mathbf{y}$: vecteurs de descripteurs.
- $x_i, y_i$: i-eme composante des vecteurs.
- $n$: dimension du vecteur.

### 5.3 Formules en format math

Intersection over Union:

$$
IoU = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}
$$

Reference: definition Jaccard/IoU [R6].

Distance euclidienne entre descripteurs:

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

Reference: usage standard en matching de descripteurs locaux [R4].

### 5.4 Lecture mathematique

- L'IoU est le quotient entre la zone commune des boites et leur zone totale couverte.
- La distance euclidienne est la racine carree de la somme des ecarts au carre composante par composante.

### 5.5 Lecture textuelle

- Si les boites se superposent bien, l'IoU se rapproche de 1.
- Si deux objets "se ressemblent" dans l'espace des descripteurs, leur distance baisse.

### 5.6 Sens de la formule

- L'IoU controle la qualite geometrique de la detection.
- La distance controle la proximite visuelle des representations.
- Ensemble, elles permettent d'evaluer "ou est l'objet" et "a quoi il ressemble".

### 5.7 Decomposition mathematique pas a pas

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

### 5.8 Exemple numerique guide

Supposons:

- $|B_p| = 1200$
- $|B_{gt}| = 1000$
- $A_{inter} = 800$

Alors:

$$
A_{union} = 1200 + 1000 - 800 = 1400
$$

$$
IoU = \frac{800}{1400} \approx 0.571
$$

### 5.9 Resultat attendu et interpretation

- $IoU \approx 0.57$: detection correcte mais non optimale.
- En pratique, un seuil minimal courant est $IoU \ge 0.5$.
- Pour des usages exigeants, on vise souvent des seuils plus eleves.

## 6. Exemple Python complet (code commente)

Le script complet, commente et structure, est dans `labs/jour1/day1_lab.py`.

Choix techniques relies aux references:

- HOG avec fenetre/blocs/cellules et `nbins` conformes au fonctionnement OpenCV [R3].
- SIFT via `cv2.SIFT_create()` et `detectAndCompute` selon la doc OpenCV [R4].
- Matching local via test de ratio de Lowe [R5].

```python
# Lancer ce script:
# python3 labs/jour1/day1_lab.py

import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def ensure_dirs():
    out_dir = Path("outputs/jour1")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


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
    return inter / (area_a + area_b - inter)


def bbox_from_threshold(gray):
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)


def hog_features(gray):
    resized = cv2.resize(gray, (128, 64), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor((128, 64), (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(resized)


def sift_features(gray):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def ratio_count(matches, ratio=0.75):
    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good += 1
    return good


out_dir, fig_dir = ensure_dirs()
img_gt = make_synthetic_scene("rectangle", 0)
img_pred = make_synthetic_scene("rectangle", 12)
img_other = make_synthetic_scene("circle", 0)

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
plt.savefig(fig_dir / "jour1_overview.png", dpi=130)
plt.close(fig)

(out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
```

## 7. Explication detaillee du code

- On cree des scenes synthetiques controlees pour garantir un comportement reproductible.
- On extrait automatiquement les boites par seuillage pour calculer l'IoU.
- On compare HOG pour verifier qu'un objet proche donne une distance plus faible.
- On compare SIFT via ratio test pour observer une difference de qualite de matching.
- On exporte systematiquement des artefacts (`metrics.json`, image de synthese) pour audit.

Lien direct avec le syllabus:

- Manipulations OpenCV: lecture/transformation/seuillage/visualisation.
- Extraction de descripteurs: HOG et SIFT.
- Comparaison quantitative: IoU + distances + nombre de correspondances.

## 8. Lab pas a pas (tres guide)

### 8.1 Objectif du lab

Executer un mini pipeline vision de bout en bout et interpretrer les mesures de qualite.

### 8.2 Setup environnement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install opencv-python numpy matplotlib
```

Si `venv`/`pip` est absent sur Debian minimal:

```bash
sudo apt install python3-venv python3-pip
```

### 8.3 Etapes d'execution

1. Verifier la presence de `labs/jour1/day1_lab.py`.
2. Lancer `python3 labs/jour1/day1_lab.py`.
3. Verifier `outputs/jour1/metrics.json`.
4. Verifier `outputs/jour1/figures/jour1_overview.png`.
5. Lire et commenter les metriques.

### 8.4 Verification (checkpoints)

- Checkpoint A: `iou_score` est strictement superieur a 0 et inferieur ou egal a 1.
- Checkpoint B: `hog_different_l2` est superieur a `hog_shifted_l2`.
- Checkpoint C: `sift_good_matches_similar` est superieur a `sift_good_matches_different`.

### 8.5 Erreurs frequentes et correction

- `ModuleNotFoundError: cv2` -> installer `opencv-python`.
- `No module named pip/venv` -> installer `python3-pip` et `python3-venv`.
- Valeurs non coherentes -> verifier qu'aucune fonction n'a ete modifiee entre generation et evaluation.

### 8.6 Validation technique du code

- Script principal de lab: `labs/jour1/day1_lab.py`.
- Validation syntaxique: `python3 -m py_compile labs/jour1/day1_lab.py`.
- Validation fonctionnelle attendue: execution avec generation des deux artefacts (`metrics.json` et `jour1_overview.png`).

## 9. Resume et points a retenir

- Une bonne pratique vision commence par une definition nette de la tache.
- L'IoU mesure "ou" est l'objet, la distance de descripteurs mesure "a quoi" il ressemble.
- HOG et SIFT fournissent une base interpretable avant le deep learning.
- Un cours de qualite doit donner des sorties testables et reproductibles.

## 10. Mini exercices

- Exercice 1: augmenter `shift` de 12 a 30 et analyser la baisse d'IoU.
- Exercice 2: remplacer le rectangle par une ellipse et comparer les distances HOG.
- Exercice 3: tester le ratio SIFT a `0.6`, `0.75`, `0.9` et commenter le compromis.

## 11. Livrables attendus

- Script fonctionnel `labs/jour1/day1_lab.py`.
- Fichier `outputs/jour1/metrics.json`.
- Fichier `outputs/jour1/figures/jour1_overview.png`.
- Court commentaire d'interpretation (5 a 10 lignes).

## 12. Cadre version etudiant (obligatoire)

- Ce chapitre est volontairement centre etudiant: objectifs clairs, progression pas a pas, et auto-verification.
- Il ne contient pas de notes formateur ni de corrige detaille "prof" dans le corps du cours.
- L'apprentissage repose sur les checkpoints, les artefacts generes et l'interpretation personnelle des mesures.

## 13. References (sources en ligne)

- [R1] Stanford CS231n (Schedule/lectures, dont detection): `https://cs231n.stanford.edu/2024/schedule.html`
- [R2] CS231n Course Notes (classification, CNN, representations): `https://cs231n.github.io/`
- [R3] OpenCV HOGDescriptor API: `https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html`
- [R4] OpenCV SIFT API: `https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html`
- [R5] D. Lowe, SIFT paper (IJCV 2004): `https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf`
- [R6] Jaccard/IoU definition (set overlap): `https://en.wikipedia.org/wiki/Jaccard_index`
