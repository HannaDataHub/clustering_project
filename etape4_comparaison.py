"""Étape 4, part B : k-means et CAH sur aiguillage (K = 4), face à la vraie partition."""
import os
import numpy as np
from utils import (load_table, split_aiguillage, ari, confusion, inertie_intra, points_de_rupture, plot_labels)
from autres_algos import kmeans_labels, cah_labels
from fisher import clustfisher

K = 4
os.makedirs("images", exist_ok=True)

df = load_table("data/aiguillage.txt", sep=",", header=None)
X, vrai = split_aiguillage(df)

# Partitions
lab_km = kmeans_labels(X, K)
lab_cah = cah_labels(X, K)
t, lab_fisher, C, _ = clustfisher(X, K)

methodes = {"Fisher": lab_fisher, "k-means": lab_km, "CAH": lab_cah}

# Mesures
print(f"{'Méthode':10s} {'ARI/vrai':>9s} {'Inertie':>14s}  Ruptures")
print(f"{'Vraie':10s} {1:9.3f} {inertie_intra(X, vrai):14.0f}  {points_de_rupture(vrai)}")
for nom, lab in methodes.items():
    print(f"{nom:10s} {ari(vrai, lab):9.3f} {inertie_intra(X, lab):14.0f}  "
          f"{points_de_rupture(lab)}")

for nom, lab in methodes.items():
    print(f"\nConfusion {nom} :")
    print(confusion(vrai, lab))

# Figure : escalier des labels
fig = plot_labels({"Vraie": vrai, **methodes}, title="aiguillage, K = 4")
fig.savefig("images/aiguillage_labels.png", dpi=150)