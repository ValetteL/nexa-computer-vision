# Assets du projet reconnaissance faciale

Ce dossier contient les fichiers nécessaires au projet bonus de reconnaissance faciale.

## Image de référence

Le script `labs/projet_face/face_actor_demo.py` tente de télécharger automatiquement :

- `keanu_reference.jpg`
- Source : Wikimedia Commons, `File:Keanu Reeves 2019 (cropped).jpg`
- URL : https://commons.wikimedia.org/wiki/File:Keanu_Reeves_2019_(cropped).jpg
- Usage : image publique utilisée comme référence pédagogique pour Keanu Reeves.

Si le téléchargement automatique échoue, placer manuellement une image de Keanu Reeves dans ce dossier sous le nom :

```text
keanu_reference.jpg
```

## Modèles OpenCV Zoo

Le script télécharge automatiquement les modèles OpenCV Zoo dans `labs/projet_face/models/` :

- YuNet : détection de visage.
- SFace : extraction d'embedding et comparaison faciale.

Ces modèles sont utilisés uniquement pour une démonstration pédagogique. La reconnaissance faciale manipule des données biométriques sensibles : ne pas utiliser ce projet pour identifier des personnes sans consentement.
