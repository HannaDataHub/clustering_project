"""Étape 3, partie B : k-means et CAH sur sequencesimu, comparaison à Fisher."""
import numpy as np
import matplotlib.pyplot as plt
from utils import (load_table, plot_segmentation, ari, inertie_intra,
                   points_de_rupture)
from autres_algos import kmeans_labels, cah_labels

X = load_table("data/sequencesimu.txt", sep=r"\s+", header=0).to_numpy(dtype=float)

# ---------- 3B : inertie pour plusieurs K ----------
Ks = range(1, 9)
in_km = [inertie_intra(X, kmeans_labels(X, k)) for k in Ks]
in_cah = [inertie_intra(X, cah_labels(X, k)) for k in Ks]

plt.figure(figsize=(5, 3.5))
plt.plot(Ks, in_km, "o-", label="k-means")
plt.plot(Ks, in_cah, "s-", label="CAH")
plt.xlabel("K")
plt.ylabel("Inertie intra-classe")
plt.legend()
plt.tight_layout()
plt.savefig("coude_autres.png", dpi=150)

# ---------- 3B' : K choisi à deux avec A ----------
K = 4   # choisi à deux avec A, après avoir regardé coude_simule.png
lab_km = kmeans_labels(X, K)
lab_cah = cah_labels(X, K)

# ---------- 3B'' : comparaison avec Fisher (fichiers de A) ----------
lab_fisher = np.load("lab_fisher_simu.npy")
t_fisher = np.load("t_fisher_simu.npy")

print(f"K = {K}")
for nom, lab in [("Fisher", lab_fisher), ("k-means", lab_km), ("CAH", lab_cah)]:
    print(f"{nom:8s} inertie = {inertie_intra(X, lab):.2f}   "
          f"ARI/Fisher = {ari(lab_fisher, lab):.3f}")
print("ARI k-means vs CAH :", round(ari(lab_km, lab_cah), 3))

fig = plot_segmentation(X, lab_fisher, ruptures=t_fisher, title="Fisher")
fig.savefig("simu_fisher_B.png", dpi=150)
fig = plot_segmentation(X, lab_km, title="k-means")
fig.savefig("simu_kmeans.png", dpi=150)
fig = plot_segmentation(X, lab_cah, title="CAH")
fig.savefig("simu_cah.png", dpi=150)