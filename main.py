"""Script principal : simulé puis aiguillage."""
import numpy as np
import matplotlib.pyplot as plt

from fisher import clustfisher
from autres_algos import kmeans_labels, cah_labels
from utils import (load_table, split_aiguillage, plot_coude, plot_segmentation,
                   ari, confusion, inertie_intra, points_de_rupture)


def partie_simulee(kmax=8):
    # TODO : adapter le chemin et les paramètres de lecture
    X = load_table("data/sequencesimu.csv").to_numpy(dtype=float)

    # Coude : un seul appel avec Kmax donne C pour tous les k <= Kmax
    _, _, _, M = clustfisher(X, kmax)
    Cs = M[-1, :]                      # M[n, k] pour k = 1..Kmax
    plot_coude(Cs, "Coude : sequencesimu")
    plt.savefig("coude_simule.png", dpi=150)

    K = 3                              # TODO : à choisir à deux après lecture du coude
    t, lab_fisher, C, _ = clustfisher(X, K)
    lab_km = kmeans_labels(X, K)
    lab_cah = cah_labels(X, K)

    for nom, lab in [("Fisher", lab_fisher), ("k-means", lab_km), ("CAH", lab_cah)]:
        print(f"{nom:8s} inertie = {inertie_intra(X, lab):.2f}  ARI/Fisher = {ari(lab_fisher, lab):.3f}")


def partie_reelle(K=4):
    # TODO : adapter le chemin et les paramètres de lecture
    df = load_table("data/aiguillage.csv", header=None)
    X, vrai = split_aiguillage(df)
    X, vrai = X[:140], vrai[:140]      # les 140 premières observations

    t, lab_fisher, C, _ = clustfisher(X, K)
    lab_km = kmeans_labels(X, K)
    lab_cah = cah_labels(X, K)

    for nom, lab in [("Fisher", lab_fisher), ("k-means", lab_km), ("CAH", lab_cah)]:
        print(f"{nom:8s} ARI/vrai = {ari(vrai, lab):.3f}  ruptures = {points_de_rupture(lab)}")
        print(confusion(vrai, lab))
    print("Vraies ruptures :", points_de_rupture(vrai))


if __name__ == "__main__":
    partie_simulee()
    partie_reelle()