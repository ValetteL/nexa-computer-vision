Mais il faut bien cadrer l’objectif :

```text
Objectif du lab :
comprendre le pipeline CNN complet sur une tâche simple de classification d’images.
```

Pas besoin d’utiliser HOG ou SIFT, car le CNN apprend lui-même les caractéristiques.

---

# LAB — Classification d’images avec un CNN uniquement

## 1. Objectif pédagogique

À la fin du lab, l’étudiant doit comprendre :

```text
comment une image entre dans un CNN
comment une convolution extrait des caractéristiques
à quoi servent ReLU et MaxPooling
comment le modèle transforme une image en prédiction
comment entraîner un CNN simple
comment lire la loss et l’accuracy
comment tester le modèle sur une image inconnue
```

Le lab reste volontairement centré sur le CNN.

---

# 2. Sujet du lab

## Titre

```text
Créer un CNN simple pour classifier des images
```

## Mission

Vous devez développer un réseau de neurones convolutionnel capable de classifier des images en plusieurs catégories.

Le modèle devra suivre ce pipeline :

```text
Image
↓
Prétraitement
↓
Convolution
↓
ReLU
↓
MaxPooling
↓
Convolution
↓
ReLU
↓
MaxPooling
↓
Flatten
↓
Fully Connected
↓
Sortie
↓
Classe prédite
```

---

# 3. Dataset conseillé

Pour un lab simple, je conseille **MNIST** ou **Fashion-MNIST**.

## Option 1 — MNIST

Images de chiffres manuscrits.

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Avantage :

```text
très simple
rapide à entraîner
bon pour débuter
```

## Option 2 — Fashion-MNIST

Images de vêtements.

```text
t-shirt
pantalon
pull
robe
manteau
sandale
chemise
basket
sac
botte
```

Avantage :

```text
plus proche d’un vrai problème image
toujours simple
rapide à entraîner
```

Je te conseille **Fashion-MNIST**, car c’est plus intéressant pédagogiquement que les chiffres.

---

# 4. Arborescence du lab

```text
lab-cnn/
├── requirements.txt
├── train_cnn.py
├── predict.py
├── models/
│   └── cnn_fashion_mnist.pth
└── README.md
```

---

# 5. requirements.txt

```txt
torch
torchvision
matplotlib
numpy
```

Installation :

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Sous Windows :

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

# 6. Code complet du lab : train_cnn.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 1. Configuration générale
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001


# 2. Prétraitement des images
# Fashion-MNIST contient des images en niveaux de gris de taille 28 x 28.
# ToTensor transforme l'image en tenseur PyTorch.
# Normalize stabilise les valeurs pour faciliter l'apprentissage.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 3. Chargement du dataset d'entraînement
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# 4. Chargement du dataset de test
test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# 5. Création des DataLoaders
# Le DataLoader envoie les images par paquets au modèle.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 6. Noms des classes Fashion-MNIST
class_names = [
    "T-shirt/top",
    "Pantalon",
    "Pull",
    "Robe",
    "Manteau",
    "Sandale",
    "Chemise",
    "Basket",
    "Sac",
    "Botte"
]


# 7. Définition du CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Première couche de convolution
        # Entrée : 1 canal car l'image est en niveaux de gris.
        # Sortie : 16 cartes de caractéristiques.
        # Kernel : filtre de taille 3 x 3.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Deuxième couche de convolution
        # Entrée : 16 cartes.
        # Sortie : 32 cartes.
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # ReLU garde les valeurs positives et remplace les négatives par 0.
        self.relu = nn.ReLU()

        # MaxPooling réduit la taille spatiale de l'image.
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Après deux MaxPooling :
        # image initiale : 28 x 28
        # après pool 1 : 14 x 14
        # après pool 2 : 7 x 7
        # nombre de cartes finales : 32
        # taille du vecteur : 32 x 7 x 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # Couche de sortie : 10 classes.
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Entrée : batch d'images de taille 1 x 28 x 28

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten : transformation des cartes en vecteur.
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


# 8. Création du modèle
model = SimpleCNN().to(DEVICE)


# 9. Fonction de perte
# CrossEntropyLoss est adaptée à la classification multi-classes.
criterion = nn.CrossEntropyLoss()


# 10. Optimiseur
# Adam met à jour les poids du réseau.
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 11. Fonction d'entraînement
def train():
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_images = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Réinitialisation des gradients
            optimizer.zero_grad()

            # Passage avant : prédiction
            outputs = model(images)

            # Calcul de l'erreur
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Mise à jour des poids
            optimizer.step()

            running_loss += loss.item()

            # Récupération de la classe prédite
            _, predicted = torch.max(outputs, 1)

            total_images += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        accuracy = 100 * correct_predictions / total_images
        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )


# 12. Fonction d'évaluation
def evaluate():
    model.eval()

    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_images += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = 100 * correct_predictions / total_images

    print(f"Accuracy sur le jeu de test : {accuracy:.2f}%")


# 13. Affichage de quelques prédictions
def show_predictions():
    model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(10, 6))

    for index in range(8):
        image = images[index].cpu().squeeze()
        prediction = predicted[index].item()
        real_label = labels[index].item()

        plt.subplot(2, 4, index + 1)
        plt.imshow(image, cmap="gray")
        plt.title(
            f"Prédit : {class_names[prediction]}\n"
            f"Réel : {class_names[real_label]}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# 14. Lancement complet
if __name__ == "__main__":
    print(f"Device utilisé : {DEVICE}")

    train()
    evaluate()
    show_predictions()

    torch.save(model.state_dict(), "models/cnn_fashion_mnist.pth")
    print("Modèle sauvegardé dans models/cnn_fashion_mnist.pth")
```

---

# 7. Ce que l’étudiant doit observer

Pendant l’exécution, il doit observer :

```text
la loss diminue progressivement
l'accuracy augmente progressivement
le modèle apprend à reconnaître les classes
certaines classes restent plus difficiles que d’autres
```

Exemple attendu :

```text
Epoch 1/5 | Loss: 0.6200 | Accuracy: 77.50%
Epoch 2/5 | Loss: 0.4100 | Accuracy: 85.20%
Epoch 3/5 | Loss: 0.3500 | Accuracy: 87.30%
Epoch 4/5 | Loss: 0.3200 | Accuracy: 88.50%
Epoch 5/5 | Loss: 0.3000 | Accuracy: 89.10%

Accuracy sur le jeu de test : environ 88% à 91%
```

Les valeurs exactes peuvent varier selon la machine.

---

# 8. Questions pédagogiques à poser

## Question 1

Pourquoi utilise-t-on une convolution dans un CNN ?

Réponse attendue :

```text
La convolution permet d'extraire automatiquement des motifs visuels comme des bords, des textures, des formes ou des parties d'objet.
```

## Question 2

À quoi sert ReLU ?

Réponse attendue :

```text
ReLU remplace les valeurs négatives par 0 et conserve les valeurs positives. Elle ajoute de la non-linéarité au modèle.
```

## Question 3

À quoi sert MaxPooling ?

Réponse attendue :

```text
MaxPooling réduit la taille des cartes de caractéristiques tout en conservant les informations les plus fortes.
```

## Question 4

Pourquoi fait-on un Flatten ?

Réponse attendue :

```text
Flatten transforme les cartes de caractéristiques en vecteur pour les envoyer dans les couches fully connected.
```

## Question 5

Pourquoi utilise-t-on CrossEntropyLoss ?

Réponse attendue :

```text
Elle est adaptée aux problèmes de classification multi-classes.
```

---

# 9. Variante encore plus centrée CNN

Si tu veux vraiment un lab **100% compréhension CNN**, tu peux demander aux étudiants de modifier uniquement l’architecture.

Exercices :

```text
modifier le nombre de filtres de conv1
modifier le nombre de filtres de conv2
ajouter une troisième convolution
changer le nombre d’epochs
observer l’impact sur l’accuracy
observer l’impact sur le temps d’entraînement
```

Exemple :

```text
Architecture A :
Conv2D 16 filtres
Conv2D 32 filtres

Architecture B :
Conv2D 32 filtres
Conv2D 64 filtres

Architecture C :
Conv2D 16 filtres
Conv2D 32 filtres
Conv2D 64 filtres
```

Ils devront comparer :

```text
loss finale
accuracy finale
temps d’entraînement
surapprentissage éventuel
```

---

# 10. Ce que je te conseille pour ton cours

Pour le Jour 1, tu peux faire :

```text
Lab 1 : pipeline image classique avec OpenCV
Lab 2 : CNN simple avec Fashion-MNIST
```

Ou si tu veux éviter HOG et SIFT dans le lab :

```text
Lab unique : CNN uniquement
```

Dans ce cas, le lab doit être très clair :

```text
pas de HOG
pas de SIFT
pas de contours
pas de seuillage
uniquement CNN
```

Phrase de cadrage à mettre dans le sujet :

```text
Dans ce lab, l’objectif n’est pas d’utiliser des descripteurs classiques comme HOG ou SIFT. L’objectif est de comprendre comment un CNN apprend automatiquement ses propres caractéristiques visuelles à partir des images.
```
