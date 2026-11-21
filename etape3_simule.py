"""Étape 3A : courbe du coude puis segmentation par Fisher sur sequencesimu.

Utilisation :
  1) laisser K = None, lancer le script, regarder coude_simule.png
  2) choisir K (avec ta collègue), le mettre ci-dessous, relancer
"""
import numpy as np
import matplotlib.pyplot as plt

from fisher import clustfisher
from utils import load_table, plot_coude, plot_segmentation

KMAX = 8
K = 4

# ---------- 0. Chargement et regard sur les données ----------
df = load_table("data/sequencesimu.txt", sep=r"\s+", header=0)
X = df.to_numpy(dtype=float)
print("Forme de X :", X.shape)          # attendu : (160, 2)
fig = plot_segmentation(X, np.zeros(len(X)), title="Données brutes")
fig.savefig("donnees_simu.png", dpi=150)

# ---------- 1. Courbe du coude ----------
_, _, _, M = clustfisher(X, KMAX)       # un seul appel pour tous les K <= KMAX
Cs = M[-1, :]                           # Cs[k-1] = critère C pour K = k
print("\n K |   critère C | gain par rapport à K-1")
for k, c in enumerate(Cs, start=1):
    if k == 1:
        print(f"{k:2d} | {c:11.1f} |")
    else:
        g = Cs[k - 2] - c
        print(f"{k:2d} | {c:11.1f} | {g:8.1f}  ({100 * g / Cs[k - 2]:4.1f} %)")
fig = plot_coude(Cs, "Coude : sequencesimu")
fig.savefig("coude_simule.png", dpi=150)

# ---------- 2. Segmentation finale (une fois K choisi) ----------
if K is not None:
    t, cluster, C, _ = clustfisher(X, K)
    print(f"\nFisher avec K = {K} : C = {C:.1f}")
    print("Points de rupture (début des segments 2..K) :", t)
    print("Effectifs des segments :", np.bincount(cluster)[1:])
    fig = plot_segmentation(X, cluster, ruptures=t, title=f"Fisher, K = {K}")
    fig.savefig("fisher_simu.png", dpi=150)
    np.save("lab_fisher_simu.npy", cluster)   # à envoyer / pousser pour ta collègue
    np.save("t_fisher_simu.npy", t)

plt.show()
