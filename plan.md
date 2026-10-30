# Plan étape par étape

## Légende

- **PARALLÈLE** : chacun travaille de son côté, sans attendre l'autre.
- **ATTEND** : l'étape ne peut pas commencer avant qu'un livrable précis de l'autre soit prêt.
- **ENSEMBLE** : point de synchronisation, à faire à deux (10 à 30 min suffisent souvent).
- **Livrable** : ce qui doit exister à la fin de l'étape, pour savoir qu'elle est vraiment terminée.

---

## Étape 0 : Démarrage (ENSEMBLE)

À faire à deux, une seule fois.

- Choisir le langage (R ou Python) et créer le dépôt partagé (Git ou Drive).
- Télécharger `sequencesimu` et `aiguillage` depuis la plateforme du cours.
- Fixer le **contrat d'interface**, pour que chacun puisse coder sans attendre l'autre :
  - `diam(X, a, b)` renvoie un nombre.
  - `clustfisher(X, K)` renvoie les points de rupture `t`, les labels `cluster` et le critère `C`.
  - Les fonctions de comparaison prennent un **vecteur de labels** en entrée.
- Se répartir les fichiers : `fisher.*` (A), `utils.*` et `autres_algos.*` (B), `rapport` (les deux).

**Livrable :** dépôt créé, interfaces écrites dans un fichier README.

---

## Étape 1 : Chacun prépare sa partie (PARALLÈLE)

| Personne A : cœur de Fisher | Personne B : outils et autres algorithmes |
|---|---|
| **1A.** Coder `diam(a, b)` selon la formule (2). | **1B.** Charger `sequencesimu` et `aiguillage`, séparer les 552 variables de la 553ᵉ colonne (vraie partition). |
| **1A'.** Coder `clustfisher` : étapes 0 à 5 (matrice D, matrices M et A, backtracking, labels). | **1B'.** Coder les fonctions utilitaires : tracé de la séquence colorée avec traits de rupture, ARI, matrice de confusion. |
| | **1B''.** Écrire les appels à k-means et CAH (ou GMM) en les testant sur de faux labels. |

Ces deux colonnes sont **indépendantes**. B n'a pas besoin de Fisher pour avancer, car ses outils prennent des labels quelconques.

**Livrable A :** fonctions `diam` et `clustfisher` qui tournent sans erreur.
**Livrable B :** données chargées, fonctions de tracé/ARI/confusion testées, k-means et CAH fonctionnels.

---

## Étape 2 : Validation de Fisher (ATTEND + ENSEMBLE)

**C'est le goulot d'étranglement du projet.** Tout ce qui touche à Fisher en dépend.

- **ATTEND :** B doit avoir fini 1A' (le code de A).
- **Ensemble :** tester `clustfisher` avec :
  - K = 1 (C doit valoir D(1,n)) ;
  - K = 2 (comparer avec un balayage de tous les points de rupture possibles) ;
  - un petit jeu de données (ex. n = 10) comparé à une recherche exhaustive ;
  - une vérification visuelle sur des données 1D évidentes (deux paliers).
- A corrige les bugs d'indices (décalages t−1, intervalles [t, i]). B écrit les tests.

**Livrable :** `clustfisher` validé, tests enregistrés dans le dépôt.

> **Point de synchro n°1 :** on ne passe pas à la suite tant que Fisher n'est pas validé.

---

## Étape 3 : Données simulées

| Personne A | Personne B |
|---|---|
| **3A.** Calculer C pour K = 1…Kmax (un seul appel avec Kmax suffit, car M[n,k] donne C pour chaque k) et tracer la courbe du coude. | **3B.** **PARALLÈLE :** pendant ce temps, lancer k-means et CAH pour plusieurs valeurs de K sur `sequencesimu`, et préparer les graphiques. |
| **3A'.** **ENSEMBLE** : choisir K à deux (lecture du coude). | **3B'.** **ATTEND le K choisi** : calculer les partitions finales k-means et CAH avec ce K. |
| **3A''.** Segmentation finale par Fisher avec ce K, puis figure avec les ruptures. | **3B''.** **ATTEND la partition de Fisher** : comparer (ARI entre partitions, inertie, graphiques côte à côte). |

**Livrable :** courbe du coude, 3 partitions tracées, tableau comparatif (inertie, ARI).

> **Point de synchro n°2 :** K est choisi et la partie simulée est complète.

---

## Étape 4 : Données réelles « aiguillage » (presque entièrement PARALLÈLE)

Ici **K = 4 est imposé**, donc personne n'attend le choix d'un K. Les deux travaillent en même temps.

| Personne A | Personne B |
|---|---|
| **4A.** Fisher avec K = 4 sur les 140 observations. | **4B.** k-means et CAH avec K = 4 sur les 140 observations. |
| **4A'.** Matrice de confusion, ARI et position des ruptures de Fisher face à la vraie partition. | **4B'.** Mêmes mesures pour k-means et CAH. |

- **ATTEND (seulement à la fin) :** assembler un tableau comparatif unique (vraie partition, Fisher, méthode 2, méthode 3). **ENSEMBLE**, 15 min.
- **ENSEMBLE :** discuter des résultats. Fisher est-il meilleur ? Pourquoi ? Rôle de la contrainte d'ordre, dimension 552, temps de calcul.

**Livrable :** tableau comparatif, figures, notes de commentaires.

> **Point de synchro n°3 :** tous les résultats sont produits. Après cette étape, on ne relance plus de calculs, sauf erreur.

---

## Étape 5 : Rédaction du compte-rendu (PARALLÈLE puis ENSEMBLE)

| Personne A | Personne B |
|---|---|
| **5A.** Rédiger la **méthode** (algorithme de Fisher, formules, implémentation). | **5B.** Rédiger la partie **données réelles** (protocole, résultats, figures). |
| **5A'.** Rédiger la partie **données simulées** (coude, comparaison). | **5B'.** Rédiger l'**introduction** (contexte, objectif). |
| **5C.** **ENSEMBLE :** discussion (forces et limites de Fisher, complexité en O(n²K), sensibilité), puis conclusion. | |

**ATTEND :** la discussion ne peut s'écrire qu'une fois toutes les parties précédentes rédigées, car elle les compare.

**Livrable :** un brouillon complet, 10 pages maximum.

---

## Étape 6 : Finalisation (ENSEMBLE, mais avec rôles croisés)

- **Relecture croisée :** A relit le texte de B, B relit celui de A.
- B nettoie et commente le code, A vérifie que les figures du texte correspondent bien au code.
- **Test de bout en bout :** exécuter tout le code sur une machine propre (ou un environnement neuf) pour vérifier que tout tourne.
- Vérifier la limite de 10 pages, exporter le PDF, déposer le rendu.

**Livrable :** PDF final + code, déposés avant le 27/11 (FI) ou le 04/12 (FA).

---

## Récapitulatif des dépendances

| Étape | Mode | Qui attend qui ? |
|---|---|---|
| 0 | Ensemble | Personne |
| 1 | **Parallèle** | Personne |
| 2 | Ensemble | B attend A (code de Fisher) |
| 3 | Mixte | B attend A pour K, puis pour la partition de Fisher |
| 4 | **Parallèle** | Personne, jusqu'à la comparaison finale |
| 5 | Parallèle, puis ensemble | La discussion attend les deux rédactions |
| 6 | Ensemble | Tout est terminé avant |

## Conseils pour ne pas se perdre

- **Cochez les livrables** : à la fin de chaque étape, vérifiez que le livrable existe réellement dans le dépôt avant de passer à la suite.
- **Les 3 points de synchro** (fin des étapes 2, 3 et 4) sont les seuls moments où l'un peut bloquer l'autre. Fixez-les à l'avance dans votre agenda.
- **Si A prend du retard à l'étape 1**, B ne reste pas bloqué : il avance sur les outils, les autres algorithmes, et même le squelette du rapport (introduction).
- **Si B finit tôt**, il peut déjà rédiger l'introduction et la partie « autres algorithmes » du rapport.

Si vous voulez, je peux aussi vous écrire le **squelette du code** (R ou Python) avec les fonctions et les interfaces de l'étape 0, ou une **checklist à cocher** reprenant les livrables.