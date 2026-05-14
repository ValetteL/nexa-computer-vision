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


# Chemin racine du projet.
# `face_actor_demo.py` est dans `labs/projet_face/`, donc `parents[2]`
# remonte jusqu'au dossier `nexa-computer-vision/`.
ROOT_DIR = Path(__file__).resolve().parents[2]

# Organisation volontairement identique aux autres labs :
# - les entrées pédagogiques restent dans `labs/` ;
# - les modèles téléchargés restent dans un sous-dossier dédié ;
# - les figures produites par l'étudiant sont sauvegardées dans `outputs/`.
PROJECT_DIR = ROOT_DIR / "labs" / "projet_face"
ASSETS_DIR = PROJECT_DIR / "assets"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "projet_face"

# Acteur choisi pour la démonstration. Le script compare tout visage testé
# à cette unique référence : ce n'est pas un système multi-identités.
ACTOR_NAME = "Keanu Reeves"
REFERENCE_IMAGE = ASSETS_DIR / "keanu_reference.jpg"
REFERENCE_URL = "https://upload.wikimedia.org/wikipedia/commons/5/59/Keanu_Reeves_2019_%28cropped%29.jpg"

# Modèles OpenCV Zoo utilisés par le lab :
# - YuNet détecte les visages et retourne une boîte + des landmarks ;
# - SFace transforme un visage aligné en embedding numérique comparable.
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Seuil recommandé par OpenCV Zoo pour la comparaison cosinus SFace.
# Au-dessus du seuil, on considère que les deux embeddings décrivent
# probablement la même personne ; en dessous, on rejette l'identité.
DEFAULT_THRESHOLD = 0.363


def download_file(url: str, destination: Path) -> None:
    """Télécharge un fichier si absent, sans réécrire un fichier existant."""
    # Ne rien faire si le fichier est déjà présent et non vide : cela rend
    # les exécutions suivantes rapides et utilisables hors ligne.
    if destination.exists() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement : {destination}")

    # Wikimedia et GitHub peuvent refuser certaines requêtes sans User-Agent.
    # On précise donc une identité simple de script pédagogique.
    request = urllib.request.Request(url, headers={"User-Agent": "nexa-computer-vision-course/1.0"})
    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())

    # Vérification minimale : un téléchargement qui crée un fichier vide ne
    # doit pas être considéré comme valide.
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Téléchargement incomplet : {destination}")


def ensure_assets(download: bool = True) -> None:
    """Prépare les modèles et l'image de référence nécessaires au lab."""
    # Les dossiers sont créés ici pour que les autres fonctions puissent
    # écrire sans se préoccuper de l'arborescence.
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # `--no-download` permet de tester uniquement avec les fichiers déjà
    # présents, utile en salle sans connexion internet.
    if not download:
        return

    # Les trois fichiers indispensables sont téléchargés séparément afin que
    # le message d'erreur indique clairement lequel manque.
    download_file(YUNET_URL, YUNET_MODEL)
    download_file(SFACE_URL, SFACE_MODEL)
    download_file(REFERENCE_URL, REFERENCE_IMAGE)


def load_image(path: Path) -> np.ndarray:
    """Charge une image OpenCV BGR avec erreur explicite si elle est absente."""
    # OpenCV retourne `None` au lieu de lever une exception si le chemin est
    # invalide. On transforme ce comportement silencieux en erreur claire.
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image non lisible : {path}")
    return img


def create_face_detector(input_size: tuple[int, int]):
    """Crée le détecteur YuNet pour une taille d'entrée donnée."""
    # `input_size` sera mis à jour à chaque image dans `detect_largest_face`.
    # Les valeurs ci-dessous contrôlent le filtrage interne :
    # - score_threshold : confiance minimale de détection ;
    # - nms_threshold : suppression des boîtes redondantes ;
    # - top_k : nombre maximal de candidats analysés.
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
    # SFace attend un visage aligné, pas une image complète. L'alignement est
    # fait plus bas avec `alignCrop`, à partir des landmarks produits par YuNet.
    return cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")


def detect_largest_face(detector, image: np.ndarray) -> np.ndarray:
    """Retourne le visage détecté le plus grand au format YuNet."""
    # YuNet doit connaître la taille exacte de l'image courante. Cela permet
    # d'utiliser la même instance de détecteur sur des images et frames webcam
    # de tailles différentes.
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        raise ValueError("Aucun visage détecté dans l'image.")

    # Si plusieurs visages sont présents, on garde le plus grand. Pour cette
    # démonstration mono-personne, cela correspond généralement au visage que
    # l'utilisateur présente à la caméra.
    faces = np.asarray(faces)
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def crop_box_from_face(face: np.ndarray) -> tuple[int, int, int, int]:
    """Convertit la sortie YuNet en boîte entière x1, y1, x2, y2."""
    # YuNet retourne la boîte sous la forme (x, y, largeur, hauteur), avec des
    # valeurs flottantes. Pour dessiner avec OpenCV, on convertit en entiers
    # et en format (x1, y1, x2, y2), comme dans les labs de détection.
    x, y, w, h = face[:4]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = max(x1 + 1, int(round(x + w)))
    y2 = max(y1 + 1, int(round(y + h)))
    return x1, y1, x2, y2


def extract_embedding(detector, recognizer, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Détecte le visage principal, l'aligne puis extrait son embedding."""
    # Étape 1 : localiser le visage principal.
    face = detect_largest_face(detector, image)

    # Étape 2 : aligner le visage. SFace utilise les landmarks YuNet pour
    # redresser/cadrer le visage, ce qui rend les embeddings plus comparables.
    aligned = recognizer.alignCrop(image, face)

    # Étape 3 : convertir le crop aligné en embedding. Cet embedding joue le
    # rôle de descripteur profond, analogue moderne de HOG/SIFT.
    embedding = recognizer.feature(aligned)
    return face, embedding


def cosine_score(recognizer, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Calcule la similarité cosinus SFace entre deux embeddings."""
    # Le score cosinus augmente quand les deux embeddings pointent dans une
    # direction similaire. Un score élevé indique donc une identité probable.
    return float(recognizer.match(embedding_a, embedding_b, cv2.FaceRecognizerSF_FR_COSINE))


def decision_from_score(score: float, threshold: float) -> str:
    """Transforme un score de similarité en décision lisible."""
    # Cette étape reprend la logique du Jour 3 : un score continu devient une
    # décision métier grâce à un seuil ajustable.
    if score >= threshold:
        return f"C'est {ACTOR_NAME}"
    return f"Ce n'est pas {ACTOR_NAME}"


def draw_result(image: np.ndarray, face: np.ndarray, score: float, threshold: float) -> np.ndarray:
    """Dessine la boîte du visage, le score et la décision sur l'image."""
    # On travaille sur une copie pour ne jamais modifier l'image originale en
    # mémoire. C'est utile si l'on veut réutiliser l'image brute ailleurs.
    output = image.copy()
    x1, y1, x2, y2 = crop_box_from_face(face)
    decision = decision_from_score(score, threshold)

    # Vert : identité acceptée ; rouge : identité rejetée. Ce code couleur
    # permet une lecture rapide en mode webcam.
    color = (0, 180, 0) if score >= threshold else (0, 0, 255)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output, decision, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(output, f"score={score:.3f} seuil={threshold:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return output


def prepare_reference(detector, recognizer) -> np.ndarray:
    """Charge la référence Keanu Reeves et extrait son embedding."""
    # La référence est calculée une seule fois au lancement. En mode webcam,
    # cela évite de recalculer l'embedding de Keanu Reeves à chaque frame.
    reference = load_image(REFERENCE_IMAGE)
    _, reference_embedding = extract_embedding(detector, recognizer, reference)
    return reference_embedding


def run_image_mode(input_path: Path, threshold: float, no_download: bool) -> None:
    """Compare une image fixe à la référence Keanu Reeves."""
    # Préparation des assets et des modèles. Le mode image sert de test
    # reproductible avant de passer à la webcam.
    ensure_assets(download=not no_download)
    detector = create_face_detector((320, 320))
    recognizer = create_face_recognizer()

    # Embedding de référence : ce vecteur représente Keanu Reeves dans
    # l'espace appris par SFace.
    reference_embedding = prepare_reference(detector, recognizer)

    # Embedding de l'image test : même pipeline que la référence pour que la
    # comparaison soit cohérente.
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
    # Le mode webcam reprend exactement le même pipeline que le mode image,
    # mais l'image test est remplacée par chaque frame de la caméra.
    ensure_assets(download=not no_download)
    detector = create_face_detector((320, 320))
    recognizer = create_face_recognizer()
    reference_embedding = prepare_reference(detector, recognizer)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra index={camera_index}")

    print("Mode webcam lancé. Appuyer sur 'q' pour quitter.")
    while True:
        # Lecture d'une frame. Si la caméra cesse de produire des images,
        # on sort proprement de la boucle.
        ok, frame = cap.read()
        if not ok:
            break

        try:
            # Si un visage est détecté, on calcule son embedding et son score.
            face, embedding = extract_embedding(detector, recognizer, frame)
            score = cosine_score(recognizer, reference_embedding, embedding)
            frame = draw_result(frame, face, score, threshold)
        except ValueError:
            # Absence de visage : ce n'est pas une erreur critique en webcam,
            # on affiche simplement un message dans la fenêtre vidéo.
            cv2.putText(frame, "Aucun visage detecte", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Projet reconnaissance faciale - Keanu Reeves", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    # Interface en ligne de commande volontairement simple : elle permet aux
    # étudiants de tester l'image, la webcam, un autre seuil ou une autre image.
    parser = argparse.ArgumentParser(description="Reconnaissance faciale pédagogique de Keanu Reeves avec OpenCV.")
    parser.add_argument("--mode", choices=["image", "webcam"], default="image", help="Mode d'exécution du lab.")
    parser.add_argument("--input", type=Path, default=REFERENCE_IMAGE, help="Image à comparer en mode image.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Seuil de similarité cosinus SFace.")
    parser.add_argument("--camera", type=int, default=0, help="Index caméra pour le mode webcam.")
    parser.add_argument("--no-download", action="store_true", help="Ne pas télécharger automatiquement les modèles/assets.")
    return parser.parse_args()


def main() -> int:
    # Point d'entrée unique : il route vers le mode image ou webcam et convertit
    # les exceptions en message utilisateur lisible.
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
