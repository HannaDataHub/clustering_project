"""Algorithme de programmation dynamique de Fisher (Personne A).

Convention : a, b, t indexés à partir de 1 (comme dans le sujet), bornes incluses.
"""
import numpy as np


def diam(X, a, b):
    """Diamètre D(a, b) : somme des carrés des écarts à la moyenne du segment [a ; b].

    Formule (2) du sujet. Valable pour 1 <= a <= b <= n.
    """
    # TODO (A) : segment = X[a-1:b], moyenne par variable, somme des carrés
    raise NotImplementedError


def clustfisher(X, K):
    """Algorithme 1 du sujet.

    Paramètres
    ----------
    X : ndarray (n, p)  séquence ordonnée
    K : int             nombre de segments (K >= 1)

    Retour
    ------
    t       : ndarray (K-1,)  points de rupture (indexés à partir de 1)
    cluster : ndarray (n,)    labels 1..K
    C       : float           critère d'inertie M[n, K]
    M       : ndarray (n, K)  critères optimaux (utile pour le coude)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]

    # Étape 0 : initialisation (matrices et vecteurs nuls)
    D = np.zeros((n, n))
    M = np.zeros((n, K))
    A = np.zeros((n, K), dtype=int)
    t = np.zeros(K - 1, dtype=int)
    cluster = np.zeros(n, dtype=int)

    # Étape 1 : matrice triangulaire supérieure D
    # TODO (A) : D[a-1, b-1] = diam(X, a, b) pour a = 1..n, b = a..n

    # Étape 2 : critères optimaux M et argmin A
    # TODO (A) : M[i,0] = D[0,i] ; puis pour k = 2..K, i = k..n :
    #            M[i,k] = min_t { M[t-1,k-1] + D[t,i] }, t = k..i  (A = argmin)
    #            attention aux décalages d'indice 1 -> 0 en Python

    # Étape 3 : points de rupture par backtracking
    # TODO (A) : k = K-1, m = n ; tant que k >= 1 : t[k] = A[m, k+1] ; m = t[k]-1 ; k -= 1

    # Étape 5 : critère C et labels
    # TODO (A) : C = M[n, K] ; remplir cluster à partir de t

    raise NotImplementedError