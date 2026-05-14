# Gabarit Standard de Cours (format hybride)

Ce fichier définit le format obligatoire pour rédiger tout chapitre de cours.

## Règles globales obligatoires

- Produire une version étudiant uniquement (pas de section "notes formateur", pas de corrigé complet dans le corps du cours).
- Toujours rédiger en français clair, pédagogique, et progressif. Adopter un ton narratif et intuitif, pas purement académique.
- Toujours commencer par une introduction qui explique pourquoi le chapitre est important.
- Toujours guider pas à pas, sans sauts logiques.
- Toujours garder une cohérence pédagogique forte : objectif -> concept -> maths -> code -> lab -> vérification.
- Toujours inclure un exemple Python complet et exécutable. Le code peut être divisé en plusieurs blocs, chacun correspondant à un concept précis.
- Toujours commenter le code et expliquer le code en dessous du bloc.
- Toujours fournir un lab avec étapes détaillées, commandes, et points de vérification.
- Toujours tester tout le code avant intégration : le code fourni dans le cours doit s'exécuter et fonctionner.
- Toujours citer les sources utilisées et ajouter une section finale "Références" avec URLs.
- Toujours placer les références uniquement en fin de chapitre (section "Références"), pas de citations dispersées dans le corps.
- Pour chaque lab, inclure une sortie attendue (ordre de grandeur) et une interprétation rapide des métriques produites.
- Toujours soigner le français : accents, ponctuation, orthographe et grammaire doivent être corrects dans tout le cours.
- Interdiction stricte de toute mention d'outil externe de production automatique dans le repo, les cours, les commentaires, les commits et les descriptions GitHub.
- Les commits et pushes doivent être neutres et attribués uniquement à l'auteur du repo, sans signature ou mention d'intervenant secondaire.
- Toute nouvelle règle doit être ajoutée d'abord au gabarit, puis appliquée au chapitre.
- Corriger le gabarit et les chapitres au fur et à mesure, sans attendre la fin du module.
- Toujours aligner le contenu sur le syllabus officiel de l'école. Chaque point du programme doit être couvert explicitement.
- Toujours fournir les fichiers de reproductibilité projet quand du code est livré : `requirements.txt` pour l'environnement Python et `.gitignore` pour exclure environnement virtuel, caches, jetons et artefacts volumineux.

## Style visuel et schéma

- Utiliser des diagrammes Mermaid pour les schémas conceptuels et les pipelines.
- Utiliser des diagrammes ASCII pour les représentations techniques détaillées (axes, grilles, valeurs).
- Les tableaux comparatifs sont encouragés pour opposer des méthodes ou concepts.
- Chaque schéma doit être accompagné d'une courte explication de lecture.

## Règle mathématique obligatoire

Quand une notion mathématique est impliquée, les formules doivent être écrites impérativement en format math LaTeX Markdown :

- Inline: `$a^2 + b^2 = c^2$`
- Bloc:

```text
$$
L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$
```

Interdiction d'écrire les formules importantes en texte brut seulement.

Ordre obligatoire de présentation mathématique (quand il y a des formules) :

1. Contexte mathématique
2. Symboles et notations
3. Formule en format math
4. Lecture mathématique
5. Lecture textuelle
6. Sens de la formule
7. Décomposition pas à pas
8. Calcul numérique guidé
9. Résultat attendu

## Structure standard obligatoire par chapitre

Copier-coller cette structure pour chaque fichier `JOUR-XX.md` ou `CHAPITRE-XX.md`.

---

# [Titre du chapitre]

## 1. Objectif du chapitre

- Rappeler les blocs du syllabus officiel couverts par ce chapitre.
- Compétences visées.
- Résultat concret attendu en fin de chapitre.

## 2. Introduction (importance du chapitre)

- Contexte métier : pourquoi ce sujet compte dans la pratique.
- Problèmes résolus par ce chapitre.
- Lien avec le reste du cours et les chapitres suivants.
- Poser 2-3 questions auxquelles le chapitre va répondre.

## 3. Prérequis

- Connaissances nécessaires.
- Outils logiciels nécessaires avec commandes d'installation.

## 4. Concepts clés

- Définitions simples et intuitives de chaque notion.
- Intuition avant la technique.
- Cas d'usage concrets et réalistes.
- Diagrammes Mermaid et/ou ASCII pour illustrer les relations entre concepts.

## 5. Contenu technique du syllabus

- Couvrir chaque point du syllabus officiel dans l'ordre.
- Pour chaque notion : explication narrative + schéma + exemple code minimal.
- Garder un ton progressif : du plus simple au plus complexe.

## 6. Formulation mathématique (quand nécessaire)

### 6.1 Contexte mathématique

- Quand et pourquoi cette formule est utilisée.
- Type de problème résolu.

### 6.2 Symboles et notations

- $x$ : [définition]
- $y$ : [définition]

### 6.3 Formule(s) en format math

Écrire la ou les formules en LaTeX Markdown (obligatoire) :

$$
[FORMULE_PRINCIPALE]
$$

### 6.4 Lecture mathématique

- Lire la formule avec ses notations mathématiques.

### 6.5 Lecture textuelle

- Reformuler en langage naturel clair, orienté intuition/métier.

### 6.6 Sens de la formule

- Expliquer ce que chaque terme implique.
- Donner l'effet attendu.

### 6.7 Décomposition mathématique pas à pas

Montrer les etapes de calcul dans l'ordre logique:

$$
\text{Étape 1 : ...}
$$

$$
\text{Étape 2 : ...}
$$

### 6.8 Exemple numérique guidé

Donner des valeurs concretes et calculer:

$$
[CALCUL_NUMERIQUE]
$$

### 6.9 Résultat attendu et interprétation

- Valeur obtenue.
- Signification métier/technique.
- Comment juger si le resultat est bon.

## 7. Exemples Python par concept

- Presenter le code en plusieurs blocs, chacun correspondant a un concept.
- Chaque bloc est suivi de son explication detaillee.

```python
# Bloc 1 : notion A
# Bloc 2 : notion B
```

**Explication**
- Bloc 1 : ce que fait le code et pourquoi.
- Bloc 2 : ce que fait le code et pourquoi.

## 8. Lab pas à pas (très guidé)

### 8.1 Objectif du lab

- Ce que l'étudiant doit obtenir à la fin.

### 8.2 Arborescence

- Presenter la structure des fichiers attendue.

### 8.3 Setup environnement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install [DEPENDANCES]
```

### 8.4 Étapes d'exécution

1. Creer l'arborescence demandee.
2. Ajouter les fichiers requis.
3. Lancer la commande 1.
4. Verifier la sortie attendue 1.
5. Lancer la commande 2.
6. Verifier la sortie attendue 2.

### 8.5 Verification (checkpoints)

- Checkpoint A : [attendu].
- Checkpoint B : [attendu].
- Checkpoint C : [attendu].

### 8.6 Sortie attendue

- Donner un exemple JSON ou texte de sortie.
- Donner des ordres de grandeur plausibles pour les métriques.
- Préciser les relations attendues entre métriques (ex. : A > B).
- Interpreter rapidement les resultats.

### 8.7 Erreurs frequentes et correction

- Erreur 1 -> Cause -> Correction.
- Erreur 2 -> Cause -> Correction.

### 8.8 Validation technique du code

- Fournir une commande de vérification rapide (syntaxe + exécution).
- Ajouter une lecture rapide des métriques pour aider le diagnostic.

### 8.9 Parcours progressif recommande

- Niveau 1 : exécution standard.
- Niveau 2 : variation d'un parametre.
- Niveau 3 : analyse de robustesse.

## 9. Resume et points a retenir

- 3 à 7 points essentiels.

## 10. Mini exercices

- Exercice 1 (application directe).
- Exercice 2 (variation).
- Exercice 3 (analyse de resultat).

## 11. Livrables attendus

- Fichier(s) code.
- Captures/sorties.
- Réponse courte d'interprétation.

## 12. Cadre version étudiant (obligatoire)

- Contenu orienté apprentissage et autonomie de l'étudiant.
- Pas de méta-instructions formateur (timing enseignant, script oral, posture de classe).
- Pas de correction exhaustive intégrée directement ; privilégier indices, checkpoints et auto-vérification.
- Conserver des consignes actionnables et des attentes mesurables.

## 13. Références (obligatoire)

- Lister les sources exactes utilisées dans le chapitre (cours universitaires, documentation officielle, papiers).
- Ne pas ajouter de références inline dans les sections du cours ; conserver les sources regroupées en fin de chapitre.
- Fournir des URLs verifiables.

---

## Checkliste qualite avant validation

- Contenu aligne sur le syllabus officiel (chaque point couvert).
- Introduction présente et engageante.
- Définitions et notions clés en début de chapitre.
- Diagrammes Mermaid et/ou ASCII avec explication de lecture.
- Formules en format math LaTeX quand nécessaire.
- Signification de tous les symboles mathématiques.
- Lecture mathématique présente.
- Lecture textuelle présente.
- Sens de la formule explicite.
- Calcul guidé avec résultat attendu.
- Exemples Python par concept, complets et commentes.
- Code exécuté et validé avant intégration.
- Reproductibilite technique assuree (`requirements.txt`, commandes d'installation, fichiers sensibles exclus).
- Explication detaillee du code sous chaque bloc.
- Lab exécutable pas à pas avec arborescence.
- Checkpoints et erreurs frequentes documentes.
- Sortie attendue avec exemple concret et interprétation.
- Parcours progressif recommande (3 niveaux).
- Version étudiant respectée (sans sections formateur).
- Enchaînement pédagogique cohérent sur tout le chapitre.
- Sources explicites et traçables (section Références en fin de chapitre uniquement).
- Nouvelles règles intégrées d'abord dans le gabarit puis appliquées au chapitre.
- Francais correct et soigne (accents, ponctuation, orthographe, grammaire).
