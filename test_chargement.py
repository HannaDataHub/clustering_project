import numpy as np
from utils import *
from autres_algos import *

# ---------- Chargement aiguillage.txt ----------
df_aigu = load_table("aiguillage.txt", sep=",", header=None)
X, vrai = split_aiguillage(df_aigu)
print("Forme de X :", X.shape)
print("Classes vraies :", np.unique(vrai))

lab = kmeans_labels(X, 4)
print("ARI k-means :", ari(vrai, lab))  
print("Inertie k-means :", inertie_intra(X, lab))
print(confusion(vrai, lab))

lab_cah = cah_labels(X, 4)
print("ARI CAH :", ari(vrai, lab_cah))


# ---------- Chargement sequencesimu.txt ----------
df_simu = load_table("sequencesimu.txt", sep=r"\s+", header=0)
X_simu = df_simu.to_numpy(dtype=float)
print("Forme simu :", X_simu.shape)
print(X_simu[:3])