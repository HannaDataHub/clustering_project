"""Outils : chargement, tracés, mesures de comparaison (Personne B)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, confusion_matrix


# ---------- Chargement ----------
def load_table(path, **kwargs):
    """Charge un fichier tabulaire (csv/txt) en DataFrame.

    Adapter sep / header / decimal selon le format réel des fichiers du cours,
    par ex. load_table("data/aiguillage.csv", sep=";", header=None)
    """
    return pd.read_csv(path, **kwargs)


def split_aiguillage(df):
    """Sépare les 552 variables (X) de la 553e colonne (vraie partition)."""
    X = df.iloc[:, :552].to_numpy(dtype=float)
    y = df.iloc[:, 552].to_numpy()
    return X, y


# ---------- Mesures ----------
def inertie_intra(X, labels):
    """Inertie intra-classe totale d'une partition (comparable au critère C de Fisher)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    total = 0.0
    for k in np.unique(labels):
        Xk = X[labels == k]
        total += ((Xk - Xk.mean(axis=0)) ** 2).sum()
    return total


def ari(labels1, labels2):
    """Adjusted Rand Index entre deux partitions."""
    return adjusted_rand_score(labels1, labels2)


def confusion(vrai, pred):
    """Matrice de confusion (attention : les numéros de classes sont arbitraires)."""
    return confusion_matrix(vrai, pred)


def points_de_rupture(labels):
    """Indices (à partir de 1) où le label change dans la séquence."""
    labels = np.asarray(labels)
    return np.where(labels[1:] != labels[:-1])[0] + 2


# ---------- Tracés ----------
def plot_coude(Cs, title="Méthode du coude"):
    """Cs[k-1] = critère C pour K = k."""
    ks = np.arange(1, len(Cs) + 1)
    plt.figure(figsize=(5, 3.5))
    plt.plot(ks, Cs, "o-")
    plt.xlabel("K")
    plt.ylabel("Critère C")
    plt.title(title)
    plt.tight_layout()


def plot_segmentation(X, labels, ruptures=None, ax=None, title=""):
    """Séquence colorée selon les labels, avec traits de rupture optionnels.

    Pour p >= 2 : on trace la 1re variable (adapter à vos besoins).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(np.arange(1, len(X) + 1), X[:, 0], c=labels, cmap="tab10", s=12)
    if ruptures is not None:
        for r in ruptures:
            ax.axvline(r - 0.5, color="orange", linestyle="--", linewidth=1)
    ax.set_xlabel("ordre des données")
    ax.set_title(title)
    return ax