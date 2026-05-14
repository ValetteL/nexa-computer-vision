# Projet bonus — Reconnaissance faciale avec Keanu Reeves

## 1. Objectif du projet

Ce projet bonus conclut le module de détection et reconnaissance d'objets par une démonstration de reconnaissance faciale.

L'objectif est de construire un pipeline capable de :

- charger une image de référence de **Keanu Reeves** ;
- détecter son visage ;
- extraire un vecteur numérique appelé **embedding facial** ;
- comparer un visage observé sur une image ou via webcam ;
- décider si le visage observé correspond ou non à Keanu Reeves.

Le projet est **non noté**. Il sert de synthèse pratique et de démonstration guidée.

## 2. Contexte pédagogique

La reconnaissance faciale est un cas particulier de vision par ordinateur. Elle combine deux sous-problèmes :

1. **Détection de visage** : trouver où se situe un visage dans une image.
2. **Reconnaissance de visage** : comparer l'identité visuelle d'un visage à une référence.

Dans ce projet, la reconnaissance ne repose pas sur un entraînement complet réalisé par l'étudiant. On utilise un modèle pré-entraîné, exactement comme pour Faster R-CNN ou YOLO dans les jours précédents.

## 3. Lien avec les trois jours du cours

| Notion du cours | Réutilisation dans le projet |
|---|---|
| Jour 1 — OpenCV | Lecture image, webcam, conversion, affichage, dessin de boîtes |
| Jour 1 — Boîtes englobantes | Localisation du visage détecté |
| Jour 1 — Distance entre descripteurs | Comparaison entre embeddings faciaux |
| Jour 2 — CNN | Les embeddings sont produits par un réseau profond pré-entraîné |
| Jour 2 — Classification | Décision finale : Keanu Reeves ou non |
| Jour 2 — Modèle pré-entraîné | SFace est utilisé comme extracteur d'embeddings |
| Jour 3 — Détection d'objet | YuNet détecte le visage comme objet spécialisé |
| Jour 3 — Seuil de confiance | Le score cosinus est comparé à un seuil |
| Jour 3 — Optimisation | Lumière, cadrage, seuil et qualité caméra influencent le résultat |

Ce projet reprend donc le pipeline complet :

```text
Image ou webcam -> detection visage -> alignement -> embedding -> comparaison -> decision
```

## 4. Prérequis

- Avoir installé les dépendances du module avec `requirements.txt`.
- Être dans l'environnement virtuel du projet.
- Disposer d'une connexion internet au premier lancement pour télécharger les modèles OpenCV Zoo.
- Pour le mode webcam : disposer d'une caméra accessible par OpenCV.

```bash
source .venv/bin/activate
```

## 5. Données et modèles utilisés

Le lab utilise :

- **YuNet** : modèle OpenCV Zoo pour détecter les visages.
- **SFace** : modèle OpenCV Zoo pour extraire des embeddings faciaux.
- **Image de référence Keanu Reeves** : image publique Wikimedia Commons.

Les fichiers sont gérés par le script :

```text
labs/projet_face/
├── face_actor_demo.py
├── assets/
│   ├── README.md
│   └── keanu_reference.jpg
└── models/
    ├── face_detection_yunet_2023mar.onnx
    └── face_recognition_sface_2021dec.onnx
```

Si les fichiers sont absents, le script tente de les télécharger automatiquement.

## 6. Enoncé étudiant

Vous devez exécuter et comprendre un système de reconnaissance faciale simple autour de Keanu Reeves.

Votre mission :

1. Lancer le script en mode image.
2. Observer le score obtenu avec l'image de référence.
3. Lancer le script avec une autre image contenant un visage.
4. Comparer le score au seuil.
5. Lancer le mode webcam.
6. Afficher sur votre téléphone une image de Keanu Reeves face à la caméra.
7. Observer si le système affiche `C'est Keanu Reeves`.
8. Tester ensuite avec une autre personne ou une autre image.
9. Expliquer pourquoi le système peut se tromper.

## 7. Étapes demandées

### Étape 1 — Tester l'aide du script

```bash
.venv/bin/python labs/projet_face/face_actor_demo.py --help
```

### Étape 2 — Tester sur l'image de référence

```bash
.venv/bin/python labs/projet_face/face_actor_demo.py --mode image
```

### Étape 3 — Tester sur une image externe

```bash
.venv/bin/python labs/projet_face/face_actor_demo.py --mode image --input chemin/vers/image.jpg
```

### Étape 4 — Tester avec la webcam

```bash
.venv/bin/python labs/projet_face/face_actor_demo.py --mode webcam
```

Afficher ensuite une image de Keanu Reeves sur un téléphone devant la caméra.

### Étape 5 — Modifier le seuil

```bash
.venv/bin/python labs/projet_face/face_actor_demo.py --mode image --threshold 0.40
```

Un seuil plus élevé rend la décision plus stricte. Un seuil plus bas accepte davantage de visages, mais augmente le risque de faux positif.

## 8. Résultats attendus

En mode image avec l'image de référence, le système doit détecter un visage et produire une sortie de ce type :

```text
Acteur de référence : Keanu Reeves
Image test          : labs/projet_face/assets/keanu_reference.jpg
Score cosinus       : 1.000
Seuil               : 0.363
Décision            : C'est Keanu Reeves
Figure sauvegardée  : outputs/projet_face/face_actor_result.png
```

En mode webcam, le résultat dépend de la qualité de l'image affichée, de la lumière, de l'angle et de la netteté.

## 9. Erreurs fréquentes et correction

| Erreur | Cause | Correction |
|---|---|---|
| `Aucun visage détecté` | Visage trop petit, flou ou mal cadré | Rapprocher l'image, améliorer la lumière |
| `Impossible d'ouvrir la caméra` | Caméra absente ou occupée | Vérifier `--camera 0`, fermer les autres applications |
| Score faible avec Keanu Reeves | Photo trop différente de la référence | Tester une image plus proche ou ajuster le seuil |
| Faux positif | Seuil trop bas | Augmenter `--threshold` |
| Téléchargement impossible | Pas d'internet | Placer les modèles et l'image manuellement dans les dossiers indiqués |

## 10. Limites techniques et éthiques

La reconnaissance faciale traite des données biométriques sensibles.

Ce projet est strictement pédagogique. Il ne doit pas être utilisé pour identifier des personnes réelles sans consentement.

Limites techniques :

- sensibilité à la lumière ;
- sensibilité à l'angle du visage ;
- sensibilité à la qualité de la caméra ;
- différence possible entre une photo jeune et une photo récente ;
- risque de faux positifs et de faux négatifs ;
- biais possibles du modèle pré-entraîné.

## 11. Corrigé guidé

### 11.1 Préparer les fichiers

Le script commence par définir les chemins du projet, les modèles utilisés et l'image de référence.

```python
ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "labs" / "projet_face"
ASSETS_DIR = PROJECT_DIR / "assets"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "projet_face"
```

Cette organisation respecte le format du cours : les labs restent dans `labs/` et les résultats sont écrits dans `outputs/`.

### 11.2 Télécharger les modèles

Le script utilise `download_file` pour récupérer les modèles uniquement s'ils sont absents.

```python
def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
```

Cela évite de télécharger les fichiers à chaque exécution.

### 11.3 Détecter le visage

La détection utilise YuNet via OpenCV :

```python
detector = cv2.FaceDetectorYN.create(...)
```

Le détecteur retourne une liste de visages. Chaque visage contient une boîte et des points de repère. On garde le plus grand visage, car il correspond généralement au sujet principal.

### 11.4 Extraire l'embedding

SFace commence par aligner le visage :

```python
aligned = recognizer.alignCrop(image, face)
```

Puis il extrait un vecteur numérique :

```python
embedding = recognizer.feature(aligned)
```

Cet embedding joue le même rôle qu'un descripteur HOG ou SIFT, mais il est appris par un réseau profond.

### 11.5 Comparer deux visages

La comparaison utilise la similarité cosinus :

```python
score = recognizer.match(embedding_a, embedding_b, cv2.FaceRecognizerSF_FR_COSINE)
```

Plus le score est élevé, plus les visages sont considérés proches.

### 11.6 Décider avec un seuil

Le seuil par défaut est :

```python
DEFAULT_THRESHOLD = 0.363
```

Si le score est supérieur ou égal au seuil, le script affiche :

```text
C'est Keanu Reeves
```

Sinon :

```text
Ce n'est pas Keanu Reeves
```

## 12. Corrigé code complet

Le code complet du lab se trouve dans `labs/projet_face/face_actor_demo.py`.

```python
#!/usr/bin/env python3
"""Projet bonus — reconnaissance faciale de Keanu Reeves avec OpenCV.

Le script propose deux modes :
- image : comparer une image fixe à la référence Keanu Reeves ;
- webcam : ouvrir la caméra et comparer le visage visible en temps réel.

Les modèles utilisés viennent d'OpenCV Zoo : YuNet pour la détection de visage
et SFace pour l'extraction d'embeddings faciaux.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "labs" / "projet_face"
ASSETS_DIR = PROJECT_DIR / "assets"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "projet_face"

ACTOR_NAME = "Keanu Reeves"
REFERENCE_IMAGE = ASSETS_DIR / "keanu_reference.jpg"
REFERENCE_URL = "https://upload.wikimedia.org/wikipedia/commons/5/59/Keanu_Reeves_2019_%28cropped%29.jpg"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

DEFAULT_THRESHOLD = 0.363


def download_file(url: str, destination: Path) -> None:
    """Télécharge un fichier si absent, sans réécrire un fichier existant."""
    if destination.exists() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement : {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "nexa-computer-vision-course/1.0"})
    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Téléchargement incomplet : {destination}")


def ensure_assets(download: bool = True) -> None:
    """Prépare les modèles et l'image de référence nécessaires au lab."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not download:
        return

    download_file(YUNET_URL, YUNET_MODEL)
    download_file(SFACE_URL, SFACE_MODEL)
    download_file(REFERENCE_URL, REFERENCE_IMAGE)


def load_image(path: Path) -> np.ndarray:
    """Charge une image OpenCV BGR avec erreur explicite si elle est absente."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image non lisible : {path}")
    return img


def create_face_detector(input_size: tuple[int, int]):
    """Crée le détecteur YuNet pour une taille d'entrée donnée."""
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL),
        "",
        input_size,
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=5000,
    )


def create_face_recognizer():
    """Crée le recognizer SFace."""
    return cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")


def detect_largest_face(detector, image: np.ndarray) -> np.ndarray:
    """Retourne le visage détecté le plus grand au format YuNet."""
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        raise ValueError("Aucun visage détecté dans l'image.")

    faces = np.asarray(faces)
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def extract_embedding(detector, recognizer, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Détecte le visage principal, l'aligne puis extrait son embedding."""
    face = detect_largest_face(detector, image)
    aligned = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned)
    return face, embedding


def cosine_score(recognizer, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Calcule la similarité cosinus SFace entre deux embeddings."""
    return float(recognizer.match(embedding_a, embedding_b, cv2.FaceRecognizerSF_FR_COSINE))


def decision_from_score(score: float, threshold: float) -> str:
    """Transforme un score de similarité en décision lisible."""
    if score >= threshold:
        return f"C'est {ACTOR_NAME}"
    return f"Ce n'est pas {ACTOR_NAME}"


def crop_box_from_face(face: np.ndarray) -> tuple[int, int, int, int]:
    """Convertit la sortie YuNet en boîte entière x1, y1, x2, y2."""
    x, y, w, h = face[:4]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = max(x1 + 1, int(round(x + w)))
    y2 = max(y1 + 1, int(round(y + h)))
    return x1, y1, x2, y2


def draw_result(image: np.ndarray, face: np.ndarray, score: float, threshold: float) -> np.ndarray:
    """Dessine la boîte du visage, le score et la décision sur l'image."""
    output = image.copy()
    x1, y1, x2, y2 = crop_box_from_face(face)
    decision = decision_from_score(score, threshold)
    color = (0, 180, 0) if score >= threshold else (0, 0, 255)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output, decision, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(output, f"score={score:.3f} seuil={threshold:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return output


def prepare_reference(detector, recognizer) -> np.ndarray:
    """Charge la référence Keanu Reeves et extrait son embedding."""
    reference = load_image(REFERENCE_IMAGE)
    _, reference_embedding = extract_embedding(detector, recognizer, reference)
    return reference_embedding


def run_image_mode(input_path: Path, threshold: float, no_download: bool) -> None:
    """Compare une image fixe à la référence Keanu Reeves."""
    ensure_assets(download=not no_download)
    detector = create_face_detector((320, 320))
    recognizer = create_face_recognizer()

    reference_embedding = prepare_reference(detector, recognizer)
    image = load_image(input_path)
    face, embedding = extract_embedding(detector, recognizer, image)
    score = cosine_score(recognizer, reference_embedding, embedding)
    decision = decision_from_score(score, threshold)

    output = draw_result(image, face, score, threshold)
    output_path = OUTPUTS_DIR / "face_actor_result.png"
    cv2.imwrite(str(output_path), output)

    print(f"Acteur de référence : {ACTOR_NAME}")
    print(f"Image test          : {input_path}")
    print(f"Score cosinus       : {score:.3f}")
    print(f"Seuil               : {threshold:.3f}")
    print(f"Décision            : {decision}")
    print(f"Figure sauvegardée  : {output_path}")


def run_webcam_mode(camera_index: int, threshold: float, no_download: bool) -> None:
    """Compare en direct le visage webcam à la référence Keanu Reeves."""
    ensure_assets(download=not no_download)
    detector = create_face_detector((320, 320))
    recognizer = create_face_recognizer()
    reference_embedding = prepare_reference(detector, recognizer)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra index={camera_index}")

    print("Mode webcam lancé. Appuyer sur 'q' pour quitter.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        try:
            face, embedding = extract_embedding(detector, recognizer, frame)
            score = cosine_score(recognizer, reference_embedding, embedding)
            frame = draw_result(frame, face, score, threshold)
        except ValueError:
            cv2.putText(frame, "Aucun visage detecte", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Projet reconnaissance faciale - Keanu Reeves", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconnaissance faciale pédagogique de Keanu Reeves avec OpenCV.")
    parser.add_argument("--mode", choices=["image", "webcam"], default="image", help="Mode d'exécution du lab.")
    parser.add_argument("--input", type=Path, default=REFERENCE_IMAGE, help="Image à comparer en mode image.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Seuil de similarité cosinus SFace.")
    parser.add_argument("--camera", type=int, default=0, help="Index caméra pour le mode webcam.")
    parser.add_argument("--no-download", action="store_true", help="Ne pas télécharger automatiquement les modèles/assets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "image":
            run_image_mode(args.input, args.threshold, args.no_download)
        else:
            run_webcam_mode(args.camera, args.threshold, args.no_download)
    except Exception as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 13. Explication détaillée de l'implémentation

### Pourquoi YuNet ?

YuNet est spécialisé dans la détection de visage. Il joue ici le même rôle que Faster R-CNN ou YOLO : il localise un objet dans une image. La différence est que l'objet recherché est uniquement un visage.

### Pourquoi SFace ?

SFace transforme un visage en vecteur numérique. Deux visages de la même personne doivent produire deux vecteurs proches. Deux personnes différentes doivent produire des vecteurs plus éloignés.

### Pourquoi un seuil ?

Le modèle ne répond pas directement `oui` ou `non`. Il produit un score. Le seuil transforme ce score en décision. C'est exactement la logique vue au Jour 3 avec les seuils de confiance YOLO.

### Pourquoi le résultat peut varier ?

Le score dépend :

- de la qualité de la photo ;
- de l'éclairage ;
- de l'angle du visage ;
- de l'expression ;
- de l'âge de l'image ;
- de la qualité de l'écran du téléphone devant la webcam.

## 14. Validation technique

Commandes de validation :

```bash
.venv/bin/python -m py_compile labs/projet_face/face_actor_demo.py
.venv/bin/python labs/projet_face/face_actor_demo.py --help
.venv/bin/python labs/projet_face/face_actor_demo.py --mode image
```

La sortie visuelle est sauvegardée dans :

```text
outputs/projet_face/face_actor_result.png
```

## 15. Conclusion

Ce projet montre comment les briques étudiées dans le module peuvent être assemblées dans une application réelle :

- acquisition image ou webcam ;
- détection ;
- extraction de représentation ;
- comparaison ;
- décision par seuil ;
- interprétation critique des limites.

Il constitue une synthèse naturelle du module, tout en ouvrant une discussion importante sur la responsabilité et l'éthique des systèmes biométriques.
