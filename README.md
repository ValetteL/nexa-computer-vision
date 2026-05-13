# Syllabus

## Mastère 1

## Détection et reconnaissance d'objets

### Objectifs pédagogiques

- **C3.2** Organiser un jeu de données en sous-ensemble d'entraînement ou de test à l'aide d'un langage de programmation, de manière simple ou aléatoire en vue d'une modélisation d'apprentissage supervisée.
- **C3.3** Exploiter plusieurs modèles d'apprentissage supervisés à l'aide d'un langage de programmation permettant la classification ou la prédiction d'une variable en fonction des données disponibles.
- **C3.4** Comparer les performances des algorithmes d'apprentissages automatiques suivant les métriques choisies afin de sélectionner l'algorithme à utiliser dans le cadre de l'apprentissage supervisé.

### Durée du module

- **3 jour(s)**
- Soit **21 heures**

## Programme

### Jour 1

#### S'introduire à la vision par ordinateur (3H30)

- Comprendre les différences entre classification, détection et reconnaissance
- Identifier les principales d'un pipeline de vision par ordinateur
- Manipuler des images en Python avec OpenCV (lecture, redimensionnement, histogramme, seuillage)

#### Décrire des images (3H30)

- Extraire des caractéristiques visuelles d'une image (feature)
- Implémenter les descripteurs HOG et SIFT avec OpenCV
- Comparer la détection basée sur des caractéristiques locales

### Jour 2

#### Revoir les fondements des réseaux de neurones convolutifs (CNN) (3H30)

- Comprendre la structure et le fonctionnement d'un CNN
- Construire un CNN simple avec PyTorch
- Entraîner et évaluer un modèle de classification d'images

#### Détecter des objets avec Faster R-CNN (3h30)

- Comprendre le fonctionnement des architectures R-CNN, Fast R-CNN et Faster R-CNN
- Utiliser un modèle pré-entraîné Faster R-CNN avec PyTorch
- Évaluer les performances de détection (IoU, précision, rappel)

### Jour 3

#### Détecter en temps réel avec YOLOv3 (3H30)

- Comprendre l'architecture YOLOv3 (grille, anchors, classes)
- Implémenter YOLOv3 avec OpenCV
- Analyser la différence avec Faster R-CNN
- Exécuter des détections en temps réel sur une vidéo

#### Évaluer et optimiser les modèles (3H30)

- Évaluer les performances d'un modèle de détection sur des métriques précises (rappel, IoU, mAP)
- Optimiser les résultats par ajustement et transfert learning (architecture, batch size, learning rate, data augmentation)
- Présenter les résultats du projet filé

## Modalités techniques

- Module assuré en présentiel par un.e intervenant.e
- Dans le cas du campus e-learning ou d'une contrainte spécifique, le module peut être assuré en distanciel par un formateur en synchrone.

## Méthodologie pédagogique

- Travaux pratiques sur PC
- Travail individuel

## Modalités d'évaluation

- Etude de cas
- Partiel S2

## Intervenant.e.s

- Formateurs experts en Data Scientist / Machine Engineer, avec une expérience dans le domaine

## Bloc 3

- Créer une solution d'intelligence artificielle à partir des données collectées et l'intégrer dans une application
- **A3.2** Utilisation d'un algorithme d'apprentissage supervisé machine learning ou deep learning
- **A3.3** Évaluation des performances d'un modèle
- **A3.4** Optimisation des performances d'un algorithme d'apprentissage supervisé

## Horaires de la formation

- Matin : **9h-12h30**
- Après-midi : **13h30-17h**

## Indicateur de satisfaction

- Taux de satisfaction des participants, recueilli via un questionnaire en fin de module.

## État du projet pédagogique

Le projet contient une version complète du module sur 3 jours : cours, labs Python, métriques JSON et figures de validation.

- `JOUR-01.md` : fondamentaux OpenCV, IoU, HOG et SIFT.
- `JOUR-02.md` : CNN avec PyTorch et détection Faster R-CNN.
- `JOUR-03.md` : comparaison Faster R-CNN / YOLOv8n et optimisation du seuil de confiance.
- `labs/` : scripts exécutables par jour.
- `labs/shared/assets/coco_dog.jpg` : image réelle libre utilisée par les labs de détection.
- `outputs/` : métriques et figures générées par les labs.

L'image `coco_dog.jpg` provient de Wikimedia Commons (`File:YellowLabradorLooking_new.jpg`) et est utilisée sous licence CC BY-SA 3.0 / GFDL avec attribution documentée dans `labs/shared/assets/README.md`.

## Installation technique

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Validation rapide

```bash
.venv/bin/python -m py_compile labs/jour1/day1_lab.py labs/jour1/day1_minimal_iou.py labs/jour2/day2_lab.py labs/jour3/day3_lab.py
```

Les labs peuvent ensuite être exécutés individuellement depuis la racine du projet.

## Couverture Du Syllabus

| Exigence officielle | Couverture projet |
|---|---|
| C3.2 Organiser un jeu de données en entraînement/test | `JOUR-02.md` et `labs/jour2/day2_lab.py` avec split aléatoire train/test et métriques séparées |
| C3.3 Exploiter plusieurs modèles supervisés | CNN PyTorch, Faster R-CNN PyTorch, YOLOv8n Ultralytics, extension YOLOv3 OpenCV DNN |
| C3.4 Comparer les performances selon les métriques | IoU, précision, rappel, mAP théorique, vitesse, seuil de confiance, comparaison Faster R-CNN/YOLO |
| Classification, détection, reconnaissance | `JOUR-01.md`, concepts et cas d'usage |
| Pipeline de vision par ordinateur | `JOUR-01.md`, puis repris dans les jours 2 et 3 |
| OpenCV : lecture, redimensionnement, histogramme, seuillage | `JOUR-01.md` et lab associé |
| HOG et SIFT | `JOUR-01.md` et `labs/jour1/day1_lab.py` |
| CNN avec PyTorch | `JOUR-02.md` et `labs/jour2/day2_lab.py` |
| R-CNN, Fast R-CNN, Faster R-CNN | `JOUR-02.md`, architecture et lab Faster R-CNN |
| YOLOv3 : grille, anchors, classes | `JOUR-03.md`, section concepts YOLOv3 |
| Implémentation YOLOv3 avec OpenCV | `JOUR-03.md`, extension OpenCV DNN |
| Détection temps réel vidéo | `JOUR-03.md`, extension vidéo/webcam avec mesure FPS |
| Optimisation : architecture, batch size, learning rate, data augmentation, transfert learning | `JOUR-03.md`, extension optimisation et transfert learning |
| Projet filé et présentation des résultats | Transitions entre jours, synthèse finale, livrables et recommandation modèle |
