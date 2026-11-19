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
    assert df.shape[1] == 553, f"Attendu : 553 colonnes, trouvé : {df.shape[1]}"
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
    return pd.crosstab(pd.Series(vrai, name="vraie"),
                       pd.Series(pred, name="estimée"))

def points_de_rupture(labels):
    """Indices (à partir de 1) où le label change dans la séquence."""
    labels = np.asarray(labels)
    return np.where(labels[1:] != labels[:-1])[0] + 2


# ---------- Tracés ----------
def plot_coude(Cs, title="Méthode du coude"):
    """Cs[k-1] = critère C pour K = k."""
    ks = np.arange(1, len(Cs) + 1)
    fig = plt.figure(figsize=(5, 3.5))
    plt.plot(ks, Cs, "o-")
    plt.xlabel("K")
    plt.ylabel("Critère C")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_segmentation(X, labels, ruptures=None, title=""):
    """Une ligne de graphique par variable, points colorés par classe."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    p = X.shape[1]
    fig, axes = plt.subplots(p, 1, sharex=True, figsize=(7, 2.5 * p))
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        ax.scatter(np.arange(1, len(X) + 1), X[:, j], c=labels,
                   cmap="tab10", s=12)
        if ruptures is not None:
            for r in ruptures:
                ax.axvline(r - 0.5, color="orange", linestyle="--", linewidth=1)
        ax.set_ylabel(f"variable {j+1}")
    axes[0].set_title(title)
    axes[-1].set_xlabel("ordre des données")
    return fig


def plot_labels(labels_dict, title=""):
    """Pour aiguillage : un escalier par méthode.
    labels_dict = {"Vraie": vrai, "Fisher": lab_f, "k-means": lab_k}"""
    n = len(labels_dict)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(7, 1.6 * n))
    axes = np.atleast_1d(axes)
    for ax, (nom, lab) in zip(axes, labels_dict.items()):
        ax.step(np.arange(1, len(lab) + 1), lab, where="mid")
        ax.set_ylabel(nom)
    axes[0].set_title(title)
    axes[-1].set_xlabel("ordre des données")
    return fig