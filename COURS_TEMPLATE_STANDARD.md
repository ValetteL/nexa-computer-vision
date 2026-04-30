# Gabarit Standard de Cours

Ce fichier definit le format obligatoire pour rediger tout chapitre de cours.

## Regles globales obligatoires

- Produire une version etudiant uniquement (pas de section "notes formateur", pas de corrige complet dans le corps du cours).
- Toujours rediger en francais clair, pedagogique, et progressif.
- Toujours commencer par une introduction qui explique pourquoi le chapitre est important.
- Toujours guider pas a pas, sans sauts logiques.
- Toujours garder une coherence pedagogique forte: objectif -> concept -> maths -> code -> lab -> verification.
- Toujours inclure un exemple Python complet et executable.
- Toujours commenter le code et expliquer le code en dessous du bloc.
- Toujours fournir un lab avec etapes detaillees, commandes, et points de verification.
- Toujours tester tout le code avant integration: le code fourni dans le cours doit s'executer et fonctionner.
- Toujours citer les sources utilisees et ajouter une section finale "References" avec URLs.
- Toujours placer les references uniquement en fin de chapitre (section "References"), pas de citations dispersees dans le corps.
- Pour chaque lab, inclure une sortie attendue (ordre de grandeur) et une interpretation rapide des metriques produites.
- Toujours soigner le francais: accents, ponctuation, orthographe et grammaire doivent etre corrects dans tout le cours.
- Interdiction stricte de toute reference a un bot/IA/assistant/outillage de generation dans le repo, les cours, les commentaires, les commits et les descriptions GitHub.
- Les commits et pushes doivent etre neutres et attribues uniquement a l'auteur du repo, sans signature ou mention d'un bot/IA.
- Toute nouvelle regle doit etre ajoutee d'abord au gabarit, puis appliquee au chapitre.
- Corriger le gabarit et les chapitres au fur et a mesure, sans attendre la fin du module.

## Regle mathematique obligatoire

Quand une notion mathematique est impliquee, les formules doivent etre ecrites imperativement en format math LaTeX Markdown:

- Inline: `$a^2 + b^2 = c^2$`
- Bloc:

```text
$$
L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$
```

Interdiction d'ecrire les formules importantes en texte brut seulement.

Ordre obligatoire de presentation mathematique (quand il y a des formules):

1. Contexte mathematique
2. Symboles et notations
3. Formule en format math
4. Lecture mathematique
5. Lecture textuelle
6. Sens de la formule
7. Decomposition pas a pas
8. Calcul numerique guide
9. Resultat attendu

## Structure standard obligatoire par chapitre

Copier-coller cette structure pour chaque fichier `JOUR-XX.md` ou `CHAPITRE-XX.md`.

---

# [Titre du chapitre]

## 1. Objectif du chapitre

- Competences visees
- Resultat concret attendu en fin de chapitre

## 2. Introduction (importance du chapitre)

- Contexte metier
- Problemes resolus par ce chapitre
- Lien avec le reste du cours

## 3. Prerequis

- Connaissances necessaires
- Outils logiciels necessaires

## 4. Concepts cles

- Definitions simples
- Intuition
- Cas d'usage

## 5. Formulation mathematique (quand necessaire)

### 5.1 Contexte mathematique

- Quand et pourquoi cette formule est utilisee
- Type de probleme resolu

### 5.2 Symboles et notations

- $x$: [definition]
- $y$: [definition]
- $\hat{y}$: [definition]
- $N$: [definition]

### 5.3 Formule(s) en format math

Ecrire la ou les formules en LaTeX Markdown (obligatoire):

$$
[FORMULE_PRINCIPALE]
$$

### 5.4 Lecture mathematique

- Lire la formule avec ses notations mathematiques
- Exemple de style: "q_phi de z sachant x,y suit une loi normale parametree par mu et sigma carre"

### 5.5 Lecture textuelle

- Reformuler en langage naturel clair, orientee intuition/metier

### 5.6 Sens de la formule

- Expliquer ce que chaque terme force le modele a apprendre
- Donner l'effet attendu sur le comportement du modele

### 5.7 Decomposition mathematique pas a pas

Montrer les etapes de calcul dans l'ordre logique:

$$
\text{Etape 1: ...}
$$

$$
\text{Etape 2: ...}
$$

### 5.8 Exemple numerique guide

Donner des valeurs concretes et calculer:

$$
[CALCUL_NUMERIQUE]
$$

### 5.9 Resultat attendu et interpretation

- Valeur obtenue
- Signification metier/technique
- Comment juger si le resultat est bon

## 6. Exemple Python complet (code commente)

```python
# 1) Imports
# 2) Chargement/preparation des donnees
# 3) Calcul/modelisation
# 4) Affichage des resultats
```

## 7. Explication detaillee du code

- Bloc 1: ce que fait l'import
- Bloc 2: ce que fait la preparation des donnees
- Bloc 3: ce que fait l'algorithme
- Bloc 4: comment lire la sortie

## 8. Lab pas a pas (tres guide)

### 8.1 Objectif du lab

- Ce que l'etudiant doit obtenir a la fin

### 8.2 Setup environnement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install [DEPENDANCES]
```

### 8.3 Etapes d'execution

1. Creer l'arborescence demandee
2. Ajouter les fichiers requis
3. Lancer la commande 1
4. Verifier la sortie attendue 1
5. Lancer la commande 2
6. Verifier la sortie attendue 2

### 8.4 Verification (checkpoints)

- Checkpoint A: [attendu]
- Checkpoint B: [attendu]
- Checkpoint C: [attendu]

### 8.4.bis Sortie attendue

- Donner des ordres de grandeur plausibles pour les metriques.
- Preciser les relations attendues entre metriques (ex: A > B).

### 8.5 Erreurs frequentes et correction

- Erreur 1 -> Cause -> Correction
- Erreur 2 -> Cause -> Correction

### 8.6 Validation technique du code

- Fournir une commande de verification rapide (syntaxe + execution).
- Ajouter une lecture rapide des metriques pour aider le diagnostic.

## 9. Resume et points a retenir

- 3 a 7 points essentiels

## 10. Mini exercices

- Exercice 1 (application directe)
- Exercice 2 (variation)
- Exercice 3 (analyse de resultat)

## 11. Livrables attendus

- Fichier(s) code
- Captures/sorties
- Reponse courte d'interpretation

## 11.bis References (obligatoire)

- Lister les sources exactes utilisees dans le chapitre (cours universitaires, documentation officielle, papiers).
- Ne pas ajouter de references inline dans les sections du cours; conserver les sources regroupees en fin de chapitre.
- Fournir des URLs verifiables.

## 12. Cadre version etudiant (obligatoire)

- Contenu oriente apprentissage et autonomie de l'etudiant.
- Pas de meta-instructions formateur (timing enseignant, script oral, posture de classe).
- Pas de correction exhaustive integree directement; privilegier indices, checkpoints et auto-verification.
- Conserver des consignes actionnables et des attentes mesurables.

---

## Checkliste qualite avant validation

- Introduction presente
- Formules en format math LaTeX quand necessaire
- Signification de tous les symboles mathematiques
- Lecture mathematique presente
- Lecture textuelle presente
- Sens de la formule explicite
- Calcul guide avec resultat attendu
- Exemple Python complet et commente
- Code execute et valide avant integration
- Explication detaillee du code sous le bloc
- Lab executable pas a pas
- Checkpoints et erreurs frequentes documentes
- Version etudiant respectee (sans sections formateur)
- Enchainement pedagogique coherent sur tout le chapitre
- Sources explicites et tracables (section References en fin de chapitre uniquement)
- Nouvelles regles integrees d'abord dans le gabarit puis appliquees au chapitre
- Sortie attendue et interpretation rapide presentes dans la partie lab
- Francais correct et soigne (accents, ponctuation, orthographe, grammaire)
