# Cahier des charges

# LAB-01 — Pipeline classique de vision par ordinateur avec OpenCV

## 1. Objet du document

Le présent cahier des charges définit les exigences fonctionnelles, techniques, pédagogiques et de validation pour la réalisation d’un lab de vision par ordinateur basé sur les notions abordées au Jour 1 du cours.

Le lab doit permettre de construire un pipeline complet de traitement d’image classique, sans apprentissage profond, afin de manipuler, analyser, transformer et comparer des images à l’aide d’OpenCV et de méthodes traditionnelles de computer vision.

Le projet final devra être directement exploitable dans un contexte de formation, avec un énoncé étudiant, une correction enseignant, des fichiers de sortie, des métriques vérifiables et une documentation d’exécution.

---

## 2. Objectif du lab

Le lab doit permettre de mettre en pratique les notions suivantes :

* représentation numérique d’une image ;
* lecture d’image avec OpenCV ;
* distinction entre image couleur et image en niveaux de gris ;
* analyse de l’histogramme ;
* égalisation d’histogramme ;
* seuillage fixe, Otsu et adaptatif ;
* extraction de contours ;
* génération de boîtes englobantes ;
* comparaison entre boîte prédite et boîte de référence ;
* calcul de l’IoU ;
* extraction de descripteurs HOG ;
* comparaison de descripteurs par distance ;
* détection de points clés SIFT ;
* matching entre images similaires et différentes ;
* interprétation des résultats obtenus.

Le lab doit montrer comment passer d’une image brute à des informations mesurables, sans utiliser de réseau de neurones.

---

## 3. Résultat attendu

Le projet livré doit permettre d’exécuter un pipeline complet produisant automatiquement :

* les images utilisées dans le lab ;
* les images intermédiaires générées par les traitements ;
* les visualisations utiles à l’analyse ;
* les métriques numériques ;
* une synthèse exploitable par l’étudiant ;
* une correction complète ;
* une grille d’évaluation.

Le lab doit être reproductible : une personne qui récupère le projet doit pouvoir l’installer, l’exécuter et obtenir les résultats attendus sans intervention manuelle complexe.

---

## 4. Périmètre fonctionnel

### 4.1 Gestion des images d’entrée

Le projet doit prévoir au minimum trois images exploitables :

| Image              | Rôle                                              |
| ------------------ | ------------------------------------------------- |
| Image de référence | Sert de base au pipeline                          |
| Image similaire    | Sert à comparer une variation légère              |
| Image différente   | Sert à comparer une image visuellement différente |

Les images peuvent être fournies directement ou générées automatiquement par le programme.

Dans tous les cas, le lab doit garantir que les images sont disponibles au moment de l’exécution.

Les images doivent permettre de tester correctement :

* le seuillage ;
* l’extraction de contours ;
* la génération de boîtes ;
* le calcul de l’IoU ;
* l’extraction HOG ;
* la comparaison par distance ;
* la détection SIFT ;
* le matching SIFT.

---

### 4.2 Lecture et inspection d’image

Le programme doit charger une image avec OpenCV et extraire les informations suivantes :

* hauteur ;
* largeur ;
* nombre de canaux ;
* type NumPy ;
* type des pixels ;
* taille mémoire approximative ;
* valeur minimale des pixels ;
* valeur maximale des pixels.

Le programme doit permettre de vérifier explicitement la différence entre :

* une image couleur ;
* une image en niveaux de gris.

Les résultats doivent être visibles dans la sortie console ou dans un fichier de métriques.

---

### 4.3 Conversion en niveaux de gris

Le programme doit convertir l’image couleur en niveaux de gris.

Le résultat doit être sauvegardé comme fichier image.

Le lab doit permettre de comparer :

* l’image d’origine ;
* l’image en niveaux de gris.

Le traitement doit être intégré dans le pipeline principal et ne doit pas être une manipulation isolée.

---

### 4.4 Histogramme

Le programme doit calculer l’histogramme de l’image en niveaux de gris.

Le programme doit sauvegarder une visualisation de cet histogramme.

L’histogramme doit permettre d’observer :

* les zones sombres ;
* les zones claires ;
* les pics de concentration ;
* les zones peu utilisées ;
* la répartition globale des intensités.

Le lab doit produire une interprétation minimale de l’histogramme sous forme textuelle ou dans la correction.

---

### 4.5 Égalisation d’histogramme

Le programme doit appliquer une égalisation d’histogramme sur l’image en niveaux de gris.

Le programme doit sauvegarder :

* l’image avant égalisation ;
* l’image après égalisation ;
* l’histogramme avant égalisation ;
* l’histogramme après égalisation.

Le résultat doit permettre d’observer l’effet de l’égalisation sur le contraste.

Le lab doit indiquer clairement dans quelles situations l’égalisation peut être utile et dans quelles situations elle peut dégrader le résultat.

---

### 4.6 Seuillage

Le programme doit appliquer plusieurs méthodes de seuillage sur l’image en niveaux de gris.

Les méthodes obligatoires sont :

| Méthode             | Exigence                                    |
| ------------------- | ------------------------------------------- |
| Seuillage fixe      | Utilisation d’un seuil choisi explicitement |
| Seuillage Otsu      | Choix automatique du seuil                  |
| Seuillage adaptatif | Seuil calculé localement                    |

Le programme doit sauvegarder une image de résultat pour chaque méthode.

Le programme doit permettre de comparer visuellement les résultats.

Le lab doit expliquer que les méthodes sont comparées afin de sélectionner celle qui est la plus adaptée au cas traité.

Le pipeline principal doit utiliser une méthode de seuillage retenue pour les étapes suivantes.

---

### 4.7 Extraction des contours

Le programme doit extraire les contours à partir d’une image binaire.

Le programme doit sauvegarder une image montrant les contours détectés.

Le programme doit calculer au minimum :

* le nombre total de contours détectés ;
* l’aire de chaque contour principal ;
* le contour retenu comme objet principal ;
* les coordonnées associées au contour principal.

Le programme doit être capable d’ignorer les petits contours parasites à l’aide d’un critère simple, par exemple une aire minimale.

---

### 4.8 Génération de boîte englobante

Le programme doit générer une boîte englobante autour de l’objet principal détecté.

La boîte doit être représentée sous la forme :

```text
x_min, y_min, x_max, y_max
```

Le programme doit sauvegarder une image affichant :

* l’image d’origine ;
* les contours utiles ;
* la boîte englobante prédite.

La boîte prédite doit être enregistrée dans le fichier de métriques.

---

### 4.9 Gestion de la boîte de référence

Le lab doit intégrer une boîte de référence correspondant à la localisation attendue de l’objet.

La boîte de référence peut être :

* définie manuellement ;
* fournie dans un fichier de configuration ;
* générée automatiquement si les images sont synthétiques ;
* indiquée dans un fichier de correction.

La boîte de référence doit être enregistrée dans le fichier de métriques.

Le lab doit clairement distinguer :

* boîte de référence ;
* boîte prédite ;
* erreur de localisation ;
* score d’évaluation.

---

### 4.10 Calcul de l’IoU

Le programme doit calculer l’IoU entre la boîte prédite et la boîte de référence.

L’IoU doit respecter la formule suivante :

```text
IoU = aire de l’intersection / aire de l’union
```

Le score doit être compris entre 0 et 1.

Le programme doit enregistrer le score IoU dans le fichier de métriques.

Le lab doit fournir une interprétation du score :

|             IoU | Interprétation             |
| --------------: | -------------------------- |
|               0 | Aucune superposition       |
|     proche de 0 | Très mauvaise localisation |
|   autour de 0,5 | Localisation moyenne       |
| supérieur à 0,7 | Localisation correcte      |
|     proche de 1 | Localisation excellente    |

Le lab doit inclure une expérimentation permettant de modifier la position de l’objet ou de la boîte afin d’observer l’évolution du score IoU.

---

### 4.11 Expérience sur le décalage

Le lab doit prévoir une expérience où un objet ou une boîte est progressivement décalé.

Le programme doit permettre de mesurer l’impact du décalage sur l’IoU.

Le résultat attendu est :

```text
plus le décalage augmente, plus l’IoU diminue
```

Le lab doit produire un tableau de résultats contenant au minimum :

| Décalage | Boîte prédite |     IoU | Interprétation         |
| -------: | ------------- | ------: | ---------------------- |
|   faible | attendue      |  élevée | bonne localisation     |
|    moyen | attendue      | moyenne | localisation partielle |
|     fort | attendue      |  faible | mauvaise localisation  |

---

### 4.12 Extraction HOG

Le programme doit extraire un descripteur HOG sur au moins deux images.

Le programme doit enregistrer :

* la taille du vecteur HOG ;
* le descripteur ou un résumé exploitable ;
* les paramètres utilisés pour le calcul ;
* la distance entre les images comparées.

Le lab doit montrer que HOG transforme une image en vecteur numérique décrivant les orientations de gradients.

Le programme doit comparer :

* image de référence contre image similaire ;
* image de référence contre image différente.

---

### 4.13 Distance entre descripteurs HOG

Le programme doit calculer une distance entre les vecteurs HOG.

La distance utilisée doit être indiquée explicitement.

La distance attendue doit respecter la logique suivante :

```text
distance image référence / image similaire < distance image référence / image différente
```

Le programme doit enregistrer les distances dans le fichier de métriques.

Le lab doit fournir une interprétation claire :

* distance faible : images proches ;
* distance élevée : images différentes ;
* la comparaison se fait sur des vecteurs, pas directement sur l’image visible.

---

### 4.14 Extraction SIFT

Le programme doit détecter les points clés SIFT sur les images.

Le programme doit enregistrer :

* le nombre de points clés sur l’image de référence ;
* le nombre de points clés sur l’image similaire ;
* le nombre de points clés sur l’image différente.

Le programme doit sauvegarder une visualisation des points clés détectés.

Le lab doit faire apparaître clairement la différence entre :

* HOG comme description globale ;
* SIFT comme description locale.

---

### 4.15 Matching SIFT

Le programme doit effectuer un matching SIFT entre :

* image de référence et image similaire ;
* image de référence et image différente.

Le programme doit enregistrer :

* le nombre total de correspondances ;
* le nombre de bonnes correspondances ;
* le ratio de correspondances retenues ;
* une image de visualisation des matches.

Le résultat attendu est :

```text
nombre de bons matches image similaire > nombre de bons matches image différente
```

Le lab doit fournir une interprétation des résultats obtenus.

---

## 5. Exigences sur les sorties

Le projet doit générer automatiquement les sorties suivantes.

### 5.1 Images de sortie obligatoires

| Sortie                       | Description                              |
| ---------------------------- | ---------------------------------------- |
| Image originale              | Image de départ                          |
| Image niveaux de gris        | Image convertie                          |
| Histogramme original         | Distribution des intensités              |
| Image égalisée               | Image après égalisation                  |
| Histogramme égalisé          | Distribution après égalisation           |
| Résultat seuillage fixe      | Masque binaire                           |
| Résultat seuillage Otsu      | Masque binaire                           |
| Résultat seuillage adaptatif | Masque binaire                           |
| Contours détectés            | Visualisation des contours               |
| Boîte englobante             | Objet localisé                           |
| Points clés SIFT             | Points détectés                          |
| Matching SIFT similaire      | Correspondances entre images proches     |
| Matching SIFT différent      | Correspondances entre images différentes |

---

### 5.2 Fichier de métriques obligatoire

Le projet doit produire un fichier de métriques structuré.

Ce fichier doit contenir au minimum :

| Champ                       | Description                    |
| --------------------------- | ------------------------------ |
| image_width                 | Largeur de l’image             |
| image_height                | Hauteur de l’image             |
| image_channels              | Nombre de canaux               |
| image_dtype                 | Type des pixels                |
| grayscale_min               | Intensité minimale             |
| grayscale_max               | Intensité maximale             |
| contour_count               | Nombre de contours détectés    |
| selected_contour_area       | Aire du contour retenu         |
| predicted_box               | Boîte prédite                  |
| reference_box               | Boîte de référence             |
| iou_score                   | Score IoU                      |
| hog_vector_size             | Taille du vecteur HOG          |
| hog_distance_similar        | Distance avec image similaire  |
| hog_distance_different      | Distance avec image différente |
| sift_keypoints_reference    | Points clés image référence    |
| sift_keypoints_similar      | Points clés image similaire    |
| sift_keypoints_different    | Points clés image différente   |
| sift_good_matches_similar   | Bons matches image similaire   |
| sift_good_matches_different | Bons matches image différente  |

---

### 5.3 Rapport de synthèse

Le projet doit générer ou fournir un rapport court expliquant :

* ce qui a été traité ;
* les résultats obtenus ;
* les valeurs importantes ;
* les écarts observés ;
* les limites du pipeline classique ;
* les points à retenir.

Ce rapport peut être fourni au format Markdown.

---

## 6. Exigences documentaires

Le projet doit contenir une documentation claire permettant d’exécuter le lab.

La documentation doit contenir :

* le titre du lab ;
* l’objectif ;
* les prérequis techniques ;
* les commandes d’installation ;
* les commandes d’exécution ;
* la description des fichiers générés ;
* les résultats attendus ;
* les erreurs fréquentes ;
* la méthode de vérification ;
* les questions à traiter ;
* les consignes de rendu.

La documentation ne doit pas remplacer le code. Elle doit expliquer l’usage du lab et l’interprétation des résultats.


L’agent ne doit pas ajouter de fonctionnalités hors périmètre sans justification explicite.
