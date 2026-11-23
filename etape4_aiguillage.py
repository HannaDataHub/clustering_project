"""Étape 4A : Fisher (K = 4) sur les données aiguillage, comparé à la vraie partition."""
import time
import numpy as np
import matplotlib.pyplot as plt

from fisher import clustfisher
from utils import (load_table, split_aiguillage, ari, confusion,
                   inertie_intra, points_de_rupture, plot_labels)

K = 4
N = 140

# ---------- 0. Chargement et vérifications ----------
df = load_table("data/aiguillage.txt", sep=",", header=None)
X, vrai = split_aiguillage(df)
print("Forme du fichier :", df.shape)
print("Classes dans tout le fichier :", {int(k): int(v) for k, v in zip(*np.unique(vrai, return_counts=True))})
X, vrai = X[:N], vrai[:N]
print(f"\nOn garde les {N} premières observations : X = {X.shape}")
print("Effectifs des vraies classes :", {int(k): int(v) for k, v in zip(*np.unique(vrai, return_counts=True))})
print("Vraies ruptures (début des classes 2..) :", points_de_rupture(vrai))
print("Lignes dupliquées dans X :", int(np.sum(
    [(X[i] == X[:i]).all(axis=1).any() for i in range(1, len(X))])))

# ---------- 1. Fisher avec mesure du temps ----------
t0 = time.time()
t, lab_fisher, C, _ = clustfisher(X, K)
temps = time.time() - t0
print(f"\nFisher K = {K} : temps = {temps:.2f} s, critère C = {C:.1f}")
print("Ruptures de Fisher :", t)
print("Effectifs des segments :", np.bincount(lab_fisher)[1:])

# ---------- 2. Comparaison à la vraie partition ----------
print(f"\nARI (Fisher, vraie partition) = {ari(vrai, lab_fisher):.3f}")
print("Inertie intra : Fisher =", round(inertie_intra(X, lab_fisher), 1),
      "| vraie partition =", round(inertie_intra(X, vrai), 1))
print("\nMatrice de confusion (lignes = vraie, colonnes = Fisher) :")
print(confusion(vrai, lab_fisher))

# ---------- 3. Figure et sauvegardes ----------
fig = plot_labels({"Vraie": vrai, "Fisher": lab_fisher}, title="Aiguillage : K = 4")
fig.savefig("fisher_aiguillage.png", dpi=150)
np.save("lab_fisher_aiguillage.npy", lab_fisher)
np.save("t_fisher_aiguillage.npy", t)
with open("temps_fisher_aiguillage.txt", "w") as f:
    f.write(f"{temps:.3f}\n")

plt.show()
