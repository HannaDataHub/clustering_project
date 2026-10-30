# Projet : clustering sous contrainte d'ordre (Fisher)

## Structure
- `fisher.py`        : diam, clustfisher (Personne A)
- `utils.py`         : chargement, tracés, ARI, matrice de confusion (Personne B)
- `autres_algos.py`  : k-means, CAH (Ward), GMM (Personne B)
- `test_fisher.py`   : tests de validation (Personne B, étape 2)
- `main.py`          : script principal (simulé puis aiguillage)
- `data/`            : sequencesimu et aiguillage (à télécharger sur la plateforme du cours)

## Contrat d'interface (à ne pas changer sans prévenir l'autre)
- Indices `a`, `b`, `t` : convention du sujet, **à partir de 1**, bornes incluses.
  X reste un tableau numpy (n, p) indexé à partir de 0 ; la conversion se fait dans `diam`.
- `diam(X, a, b) -> float`
- `clustfisher(X, K) -> (t, cluster, C, M)`
    - t       : array de K-1 points de rupture (début des segments 2..K, indexés à partir de 1)
    - cluster : array de n labels, valeurs 1..K
    - C       : float, critère d'inertie = M[n, K]
    - M       : matrice (n, K), pour lire C pour tous les k <= K (courbe du coude)
- Toute fonction de comparaison prend un **vecteur de labels** (array de n entiers).

## Installation
    pip install -r requirements.txt