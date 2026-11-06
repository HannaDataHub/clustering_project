"""Algorithme de programmation dynamique de Fisher.

a, b, t indexés à partir de 1, bornes incluses.
En interne, D, M et A sont alloués avec une ligne/colonne 0 inutilisée.
Seule la sortie M est recoupée en (n, K).
"""
import numpy as np


def diam(X, a, b):
    """Diamètre D(a, b) : somme des carrés des écarts à la moyenne du segment [a ; b].

    Formule 2 valable pour 1 <= a <= b <= n (indices à partir de 1).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    segment = X[a - 1:b]
    mu = segment.mean(axis=0)
    return float(((segment - mu) ** 2).sum())


def clustfisher(X, K):
    """
    Paramètres
    X : ndarray (n, p)  séquence ordonnée (ou (n,) si p = 1)
    K : int             nombre de segments (K >= 1)

    Retour
    t       : ndarray (K-1,)  points de rupture (début des segments 2..K, à partir de 1)
    cluster : ndarray (n,)    labels 1..K
    C       : float           critère d'inertie M[n, K]
    M       : ndarray (n, K)  M[i-1, k-1] = coût optimal de i premières obs. en k segments
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if not 1 <= K <= n:
        raise ValueError(f"K doit vérifier 1 <= K <= n (K={K}, n={n})")

    # Étape 0 : initialisation
    D = np.zeros((n + 1, n + 1))
    M = np.zeros((n + 1, K + 1))
    A = np.zeros((n + 1, K + 1), dtype=int)
    t = np.zeros(K - 1, dtype=int)
    cluster = np.zeros(n, dtype=int)

    # Étape 1 : matrice triangulaire supérieure D
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            D[a, b] = diam(X, a, b)

    # Étape 2 : critères optimaux M et argmin A
    for i in range(1, n + 1):
        M[i, 1] = D[1, i]

    for k in range(2, K + 1):
        for i in range(k, n + 1):
            ts = np.arange(k, i + 1)
            valeurs = M[ts - 1, k - 1] + D[ts, i]
            j = np.argmin(valeurs)
            M[i, k] = valeurs[j]
            A[i, k] = ts[j]

    # Étape 3 : points de rupture par backtracking
    k = K - 1
    m = n
    while k >= 1:
        t_k = A[m, k + 1]
        t[k - 1] = t_k
        m = t_k - 1
        k -= 1

    # Étape 5 : critère C et labels
    C = float(M[n, K])
    cluster[:] = 1
    for k in range(1, K):
        cluster[t[k - 1] - 1:] = k + 1

    return t, cluster, C, M[1:, 1:]