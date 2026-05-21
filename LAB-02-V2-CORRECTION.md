# LAB — Entraîner un CNN sur un dataset réel personnalisé

## 1. Objectif concret du lab

L’objectif de ce lab est de créer un **modèle CNN capable de reconnaître automatiquement la catégorie d’un objet à partir d’une image réelle**.

Contrairement à un lab basé sur MNIST ou Fashion-MNIST, ici on travaille avec un **dataset personnalisé**, composé d’images réelles rangées par classes.

Exemple d’objectif final :

```text
Image inconnue
↓
CNN entraîné
↓
Prédiction :
clavier : 91 %
souris  : 6 %
ecran   : 3 %
↓
Résultat final : clavier
```

Le modèle devra répondre à cette question :

```text
Quelle est la classe principale de cette image ?
```

Ce lab concerne uniquement la **classification d’image**.

Il ne traite pas encore :

```text
détection d’objets
bounding boxes
segmentation
YOLO
Faster R-CNN
```

## 2. Pipeline général du lab

```text
Images réelles
↓
Organisation par classes
↓
Nettoyage du dataset
↓
Séparation train / validation / test
↓
Chargement avec PyTorch
↓
Prétraitement des images
↓
Data augmentation sur le train
↓
Entraînement du CNN
↓
Évaluation du modèle
↓
Matrice de confusion
↓
Prédiction sur une image inconnue
```

## 3. Résultat attendu à la fin du lab

À la fin du lab, l’étudiant doit avoir produit :

```text
1. un dataset propre et structuré
2. un dossier train
3. un dossier validation
4. un dossier test
5. un CNN entraîné
6. un modèle sauvegardé
7. une accuracy sur validation et test
8. une matrice de confusion
9. un script de prédiction sur image inconnue
```

Fichier modèle attendu :

```text
models/best_cnn.pth
```

Matrice de confusion attendue :

```text
outputs/confusion_matrix.png
```

## 4. Arborescence complète du projet

```text
lab-cnn-dataset-reel/
├── dataset_raw/
│   ├── clavier/
│   ├── souris/
│   └── ecran/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── best_cnn.pth
│
├── outputs/
│   └── confusion_matrix.png
│
├── prepare_dataset.py
├── train_cnn.py
├── predict.py
├── requirements.txt
└── README.md
```

Au départ, seul le dossier suivant est obligatoire :

```text
dataset_raw/
├── clavier/
├── souris/
└── ecran/
```

Le dossier `dataset/` sera généré automatiquement par le script `prepare_dataset.py`.

## 5. Organisation du dataset brut

### Format attendu

Les images doivent être rangées par classe.

Exemple :

```text
dataset_raw/
├── clavier/
│   ├── clavier_001.jpg
│   ├── clavier_002.jpg
│   └── clavier_003.jpg
│
├── souris/
│   ├── souris_001.jpg
│   ├── souris_002.jpg
│   └── souris_003.jpg
│
└── ecran/
    ├── ecran_001.jpg
    ├── ecran_002.jpg
    └── ecran_003.jpg
```

Le nom du dossier correspond au label.

Exemple :

```text
dataset_raw/clavier/clavier_001.jpg
```

signifie :

```text
cette image appartient à la classe clavier
```

## 6. Règles de qualité du dataset

### Images à conserver

```text
images lisibles
images nettes
images correctement classées
images représentatives du vrai cas d’usage
images prises sous différents angles
images avec différents éclairages
images avec différents arrière-plans
```

### Images à supprimer

```text
images cassées
images impossibles à ouvrir
images mal classées
images hors sujet
images trop floues
doublons exacts
images trop petites
```

### Exemple d’erreur dangereuse

Si le fichier suivant contient en réalité une souris :

```text
dataset_raw/clavier/image_018.jpg
```

alors le modèle va apprendre une mauvaise information :

```text
une souris peut être un clavier
```

La qualité des labels est donc essentielle.

## 7. Installation de l’environnement

### Fichier `requirements.txt`

```txt
torch
torchvision
matplotlib
numpy
pillow
```

### Installation sous Linux ou macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Installation sous Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Étape 1 — Préparer automatiquement le dataset

### Objectif

Le script `prepare_dataset.py` transforme ce dossier :

```text
dataset_raw/
├── clavier/
├── souris/
└── ecran/
```

en dataset final :

```text
dataset/
├── train/
│   ├── clavier/
│   ├── souris/
│   └── ecran/
│
├── val/
│   ├── clavier/
│   ├── souris/
│   └── ecran/
│
└── test/
    ├── clavier/
    ├── souris/
    └── ecran/
```

Répartition utilisée :

```text
70 % entraînement
15 % validation
15 % test
```

### Fichier `prepare_dataset.py`

```python
from pathlib import Path
import random
import shutil
from PIL import Image


SOURCE_DIR = Path("dataset_raw")
OUTPUT_DIR = Path("dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(42)


def create_directory(path: Path):
    """
    Crée un dossier s'il n'existe pas déjà.
    """
    path.mkdir(parents=True, exist_ok=True)


def is_valid_image(image_path: Path) -> bool:
    """
    Vérifie qu'une image peut être ouverte correctement.
    """
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def get_valid_images(class_dir: Path):
    """
    Récupère uniquement les images valides d'une classe.
    """
    valid_images = []

    for file_path in class_dir.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        if not is_valid_image(file_path):
            print(f"Image ignorée car invalide : {file_path}")
            continue

        valid_images.append(file_path)

    return valid_images


def split_images(images):
    """
    Sépare les images en train, validation et test.
    """
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    val_end = train_end + val_size

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    return train_images, val_images, test_images


def copy_images(images, split_name: str, class_name: str):
    """
    Copie les images dans le dossier final.
    """
    target_dir = OUTPUT_DIR / split_name / class_name
    create_directory(target_dir)

    for index, image_path in enumerate(images, start=1):
        new_filename = f"{class_name}_{index:04d}{image_path.suffix.lower()}"
        target_path = target_dir / new_filename

        shutil.copy2(image_path, target_path)


def clean_output_directory():
    """
    Supprime l'ancien dataset généré pour repartir proprement.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    create_directory(OUTPUT_DIR)


def main():
    """
    Prépare le dataset final.
    """
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier {SOURCE_DIR} n'existe pas."
        )

    class_dirs = [
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir()
    ]

    if not class_dirs:
        raise ValueError(
            "Aucune classe trouvée dans dataset_raw."
        )

    clean_output_directory()

    print("Préparation du dataset")
    print("----------------------")

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = get_valid_images(class_dir)

        if len(images) < 30:
            print(
                f"Attention : la classe {class_name} contient seulement "
                f"{len(images)} images. Le modèle risque de mal apprendre."
            )

        train_images, val_images, test_images = split_images(images)

        copy_images(train_images, "train", class_name)
        copy_images(val_images, "val", class_name)
        copy_images(test_images, "test", class_name)

        print(f"Classe : {class_name}")
        print(f"  Train : {len(train_images)}")
        print(f"  Val   : {len(val_images)}")
        print(f"  Test  : {len(test_images)}")

    print("----------------------")
    print("Dataset préparé avec succès.")


if __name__ == "__main__":
    main()
```

### Exécution

```bash
python prepare_dataset.py
```

### Résultat attendu

```text
Préparation du dataset
----------------------
Classe : clavier
  Train : 70
  Val   : 15
  Test  : 15

Classe : souris
  Train : 70
  Val   : 15
  Test  : 15

Classe : ecran
  Train : 70
  Val   : 15
  Test  : 15
----------------------
Dataset préparé avec succès.
```

## 9. Explication pédagogique de la séparation des données

### Train

Le dossier `train` sert à entraîner le modèle.

```text
Le CNN apprend sur ces images.
```

### Validation

Le dossier `val` sert à contrôler le modèle pendant l’entraînement.

```text
On vérifie si le CNN généralise correctement.
```

### Test

Le dossier `test` sert à mesurer la performance finale.

```text
Le modèle ne doit jamais apprendre sur ces images.
```

Si le modèle est très bon sur `train` mais mauvais sur `val`, cela indique souvent un **surapprentissage**.

## 10. Étape 2 — Entraîner le CNN

### Objectif

Le script `train_cnn.py` doit :

```text
1. charger le dataset préparé
2. transformer les images
3. créer un CNN
4. entraîner le modèle
5. mesurer la loss
6. mesurer l’accuracy
7. sauvegarder le meilleur modèle
8. générer une matrice de confusion
```

### Fichier `train_cnn.py`

```python
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt


DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_PATH = MODEL_DIR / "best_cnn.pth"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_directories():
    """
    Crée les dossiers nécessaires.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_transforms():
    """
    Définit les transformations appliquées aux images.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    return train_transform, eval_transform


def get_dataloaders():
    """
    Charge les datasets avec ImageFolder.

    ImageFolder utilise les noms des dossiers comme noms de classes.
    """
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=DATASET_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=DATASET_DIR / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=DATASET_DIR / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


class SimpleCNN(nn.Module):
    """
    CNN simple pour classifier des images couleur.
    """

    def __init__(self, number_of_classes: int):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, number_of_classes)
        )

    def forward(self, x):
        """
        Passage avant du modèle.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Entraîne le modèle pendant une époque.
    """
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        total_images += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_images

    return average_loss, accuracy


def evaluate(model, data_loader, criterion):
    """
    Évalue le modèle sur validation ou test.
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            total_images += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_images

    return average_loss, accuracy, all_labels, all_predictions


def save_confusion_matrix(labels, predictions, class_names):
    """
    Génère une matrice de confusion.
    """
    number_of_classes = len(class_names)

    matrix = np.zeros(
        (number_of_classes, number_of_classes),
        dtype=int
    )

    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label][predicted_label] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix)

    plt.title("Matrice de confusion")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")

    plt.xticks(
        ticks=np.arange(number_of_classes),
        labels=class_names,
        rotation=45
    )

    plt.yticks(
        ticks=np.arange(number_of_classes),
        labels=class_names
    )

    for row in range(number_of_classes):
        for col in range(number_of_classes):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    print(f"Matrice de confusion sauvegardée : {CONFUSION_MATRIX_PATH}")


def main():
    create_directories()

    print(f"Device utilisé : {DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print("Classes détectées :")
    for index, class_name in enumerate(class_names):
        print(f"  {index} : {class_name}")

    model = SimpleCNN(
        number_of_classes=len(class_names)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0

    print("Début de l'entraînement")
    print("-----------------------")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": IMAGE_SIZE
            }

            torch.save(checkpoint, MODEL_PATH)

            print(
                f"Meilleur modèle sauvegardé avec "
                f"Val acc: {best_val_accuracy:.2f}%"
            )

    print("-----------------------")
    print("Évaluation finale sur test")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_accuracy, test_labels, test_predictions = evaluate(
        model,
        test_loader,
        criterion
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    save_confusion_matrix(
        labels=test_labels,
        predictions=test_predictions,
        class_names=class_names
    )

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
```

## 11. Explication détaillée de l’architecture CNN

Le modèle reçoit une image couleur redimensionnée en :

```text
128 x 128 x 3
```

Cela signifie :

```text
128 pixels de hauteur
128 pixels de largeur
3 canaux : rouge, vert, bleu
```

Dans PyTorch, l’image est représentée ainsi :

```text
3 x 128 x 128
```

### 11.1 Première convolution

```python
nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)
```

Lecture :

```text
entrée : 3 canaux RGB
sortie : 16 cartes de caractéristiques
filtre : 3 x 3
padding : conservation de la taille spatiale
```

Résultat :

```text
entrée : 3 x 128 x 128
sortie : 16 x 128 x 128
```

Le modèle apprend à détecter des éléments simples :

```text
bords
contrastes
lignes
petites textures
```

### 11.2 ReLU

```python
nn.ReLU()
```

ReLU applique la règle suivante :

```text
si la valeur est négative, elle devient 0
si la valeur est positive, elle est conservée
```

Formule :

```text
ReLU(x) = max(0, x)
```

Rôle :

```text
garder les activations utiles
supprimer les valeurs négatives
ajouter de la non-linéarité au réseau
```

### 11.3 MaxPooling

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

Le MaxPooling réduit la taille spatiale par deux.

Résultat :

```text
avant pooling : 16 x 128 x 128
après pooling : 16 x 64 x 64
```

Rôle :

```text
réduire le coût de calcul
garder les activations fortes
rendre le modèle moins sensible aux petits déplacements
```

### 11.4 Deuxième convolution

Résultat attendu :

```text
entrée : 16 x 64 x 64
sortie après convolution : 32 x 64 x 64
sortie après pooling : 32 x 32 x 32
```

À ce niveau, le CNN apprend des motifs plus riches :

```text
coins
formes locales
textures
motifs caractéristiques
```

### 11.5 Troisième convolution

Résultat attendu :

```text
entrée : 32 x 32 x 32
sortie après convolution : 64 x 32 x 32
sortie après pooling : 64 x 16 x 16
```

À ce stade, le CNN travaille avec des représentations plus abstraites.

Il ne regarde plus seulement les pixels bruts.
Il combine les informations extraites par les couches précédentes.

### 11.6 Flatten

```python
nn.Flatten()
```

Avant Flatten :

```text
64 x 16 x 16
```

Après Flatten :

```text
16384 valeurs
```

Calcul :

```text
64 x 16 x 16 = 16384
```

Le CNN transforme donc ses cartes de caractéristiques en vecteur.

### 11.7 Couche fully connected

```python
nn.Linear(64 * 16 * 16, 256)
```

Cette couche combine les caractéristiques extraites.

Elle apprend à associer certaines combinaisons de motifs à une classe.

Exemple :

```text
forme rectangulaire
bords horizontaux
texture plastique
couleur sombre
↓
classe probable : clavier
```

### 11.8 Couche de sortie

```python
nn.Linear(256, number_of_classes)
```

Si le dataset contient trois classes :

```text
clavier
souris
ecran
```

alors la sortie du modèle contient trois scores.

Exemple :

```text
clavier : 4.8
souris  : 1.2
ecran   : 0.5
```

Le score le plus élevé correspond à la classe prédite.

## 12. Exécution de l’entraînement

Commande :

```bash
python train_cnn.py
```

Sortie possible :

```text
Device utilisé : cpu
Classes détectées :
  0 : clavier
  1 : ecran
  2 : souris

Début de l'entraînement
-----------------------
Epoch 1/10 | Train loss: 1.0821 | Train acc: 45.71% | Val loss: 1.0012 | Val acc: 56.67%
Meilleur modèle sauvegardé avec Val acc: 56.67%

Epoch 2/10 | Train loss: 0.9124 | Train acc: 62.38% | Val loss: 0.8120 | Val acc: 70.00%
Meilleur modèle sauvegardé avec Val acc: 70.00%

Évaluation finale sur test
Test loss: 0.4215
Test accuracy: 86.67%
Matrice de confusion sauvegardée : outputs/confusion_matrix.png
Entraînement terminé.
```

Les résultats dépendent du nombre d’images, de la qualité des labels et de la difficulté des classes.

## 13. Interprétation des résultats

### Situation correcte

```text
Train accuracy : 90 %
Val accuracy   : 86 %
Test accuracy  : 84 %
```

Interprétation :

```text
le modèle apprend correctement
la validation est proche du train
le test est cohérent
```

### Surapprentissage

```text
Train accuracy : 98 %
Val accuracy   : 55 %
Test accuracy  : 52 %
```

Interprétation :

```text
le modèle apprend par cœur les images d’entraînement
il généralise mal sur de nouvelles images
```

Causes possibles :

```text
dataset trop petit
images trop similaires
classes déséquilibrées
modèle trop complexe
pas assez de data augmentation
```

### Sous-apprentissage

```text
Train accuracy : 45 %
Val accuracy   : 42 %
Test accuracy  : 40 %
```

Interprétation :

```text
le modèle n’apprend pas suffisamment
```

Causes possibles :

```text
pas assez d’epochs
learning rate mal choisi
images trop difficiles
labels incorrects
architecture trop simple
dataset trop bruité
```

## 14. Étape 3 — Prédire une image inconnue

### Objectif

Le script `predict.py` permet de tester une image qui n’est pas dans le dataset.

Exemple :

```bash
python predict.py images_test/clavier_test.jpg
```

Résultat attendu :

```text
Classe prédite : clavier
Probabilité : 91.24 %
```

### Fichier `predict.py`

```python
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


MODEL_PATH = Path("models/best_cnn.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleCNN(nn.Module):
    """
    Même architecture que dans train_cnn.py.
    """

    def __init__(self, number_of_classes: int):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, number_of_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model():
    """
    Charge le modèle sauvegardé.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modèle introuvable. Lance d'abord train_cnn.py."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = SimpleCNN(
        number_of_classes=len(class_names)
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, class_names, image_size


def prepare_image(image_path: Path, image_size: int):
    """
    Prépare une image inconnue pour le CNN.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


def predict(image_path: Path):
    """
    Prédit la classe d'une image.
    """
    model, class_names, image_size = load_model()

    image_tensor = prepare_image(
        image_path=image_path,
        image_size=image_size
    )

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][predicted_index].item()

    predicted_class = class_names[predicted_index]

    print(f"Image : {image_path}")
    print(f"Classe prédite : {predicted_class}")
    print(f"Probabilité : {confidence * 100:.2f}%")


def main():
    if len(sys.argv) != 2:
        print("Utilisation : python predict.py chemin/image.jpg")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    predict(image_path)


if __name__ == "__main__":
    main()
```

### Exécution

```bash
python predict.py images_test/clavier_test.jpg
```

### Résultat possible

```text
Image : images_test/clavier_test.jpg
Classe prédite : clavier
Probabilité : 91.24%
```

## 15. Commandes finales à exécuter

Ordre d’exécution :

```bash
python prepare_dataset.py
python train_cnn.py
python predict.py images_test/mon_image.jpg
```

Ordre logique :

```text
1. préparer dataset_raw
2. lancer prepare_dataset.py
3. vérifier dataset/train, dataset/val et dataset/test
4. lancer train_cnn.py
5. vérifier models/best_cnn.pth
6. vérifier outputs/confusion_matrix.png
7. tester une image inconnue avec predict.py
```

## 16. Ce que l’étudiant doit comprendre

À la fin du lab, l’étudiant doit savoir expliquer :

```text
1. pourquoi un dataset doit être propre
2. pourquoi les images doivent être organisées par classes
3. pourquoi il faut séparer train, validation et test
4. pourquoi on redimensionne les images
5. pourquoi on normalise les pixels
6. pourquoi la data augmentation s’applique seulement au train
7. comment une convolution extrait des caractéristiques
8. à quoi sert ReLU
9. à quoi sert MaxPooling
10. pourquoi on utilise Flatten
11. comment le CNN produit une classe finale
12. comment lire la loss et l’accuracy
13. comment interpréter une matrice de confusion
14. comment tester le modèle sur une image inconnue
```

## 17. Questions de vérification

### Question 1

Pourquoi ne doit-on pas entraîner le modèle sur les images de test ?

Réponse attendue :

```text
Parce que le test doit mesurer la capacité du modèle à généraliser sur des images jamais vues.
```

### Question 2

Pourquoi applique-t-on la data augmentation seulement sur le train ?

Réponse attendue :

```text
Parce que le train sert à enrichir l’apprentissage, alors que validation et test doivent rester stables pour mesurer correctement les performances.
```

### Question 3

Pourquoi utilise-t-on `ImageFolder` ?

Réponse attendue :

```text
Parce qu’il permet de charger automatiquement des images organisées par dossiers de classes.
```

### Question 4

Pourquoi redimensionne-t-on les images en `128 x 128` ?

Réponse attendue :

```text
Parce qu’un CNN reçoit des tenseurs de taille fixe dans un batch.
```

### Question 5

Que signifie une accuracy élevée sur train mais faible sur validation ?

Réponse attendue :

```text
Cela indique souvent un surapprentissage.
```

## 18. Améliorations possibles

Après la première version du lab, les étudiants peuvent améliorer le projet en testant :

```text
1. davantage d’images par classe
2. des classes mieux équilibrées
3. davantage d’epochs
4. une taille d’image en 224 x 224
5. une quatrième classe
6. davantage de filtres de convolution
7. une couche de convolution supplémentaire
8. un modèle pré-entraîné comme ResNet18
9. l’affichage des mauvaises prédictions
10. une analyse détaillée par classe
```

## 19. Résumé final

Ce lab montre comment entraîner un CNN sur un **dataset réel personnalisé**, sans MNIST et sans Fashion-MNIST.

Le cœur du lab n’est pas seulement le modèle.
Le cœur du lab est tout le pipeline :

```text
dataset propre
↓
séparation correcte
↓
prétraitement
↓
CNN
↓
entraînement
↓
validation
↓
test
↓
prédiction réelle
```


```text
Un CNN moderne n’
