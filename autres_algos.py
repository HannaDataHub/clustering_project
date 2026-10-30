"""Autres algorithmes de clustering (Personne B).

Tous prennent X (n, p) et K, et renvoient un vecteur de labels (n,).
"""
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def kmeans_labels(X, K, seed=0):
    return KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(X)


def cah_labels(X, K):
    """CAH avec critère de Ward."""
    return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)


def gmm_labels(X, K, seed=0):
    return GaussianMixture(n_components=K, random_state=seed).fit_predict(X)