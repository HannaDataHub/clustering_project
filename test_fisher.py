"""Tests de validation de clustfisher (Personne B, étape 2).

Lancer avec : python test_fisher.py
"""
import itertools
import numpy as np
from fisher import diam, clustfisher


def inertie_partition(X, ruptures):
    """Critère C pour une liste de points de rupture (indexés à partir de 1)."""
    n = len(X)
    bornes = [1] + list(ruptures) + [n + 1]
    return sum(diam(X, bornes[i], bornes[i + 1] - 1) for i in range(len(bornes) - 1))


def test_diam():
    X = np.array([[1.0], [2.0], [3.0]])
    assert abs(diam(X, 1, 3) - 2.0) < 1e-12   # moyenne 2 : 1 + 0 + 1
    assert abs(diam(X, 2, 2)) < 1e-12


def test_K1():
    X = np.random.default_rng(0).normal(size=(20, 2))
    _, cluster, C, _ = clustfisher(X, 1)
    assert abs(C - diam(X, 1, len(X))) < 1e-9
    assert set(cluster) == {1}


def test_exhaustif(n=10, K=3):
    X = np.random.default_rng(1).normal(size=(n, 2))
    best = min(
        inertie_partition(X, r) for r in itertools.combinations(range(2, n + 1), K - 1)
    )
    _, _, C, _ = clustfisher(X, K)
    assert abs(C - best) < 1e-9


def test_paliers():
    X = np.r_[np.zeros(10), 5 * np.ones(10), np.zeros(10)].reshape(-1, 1)
    t, cluster, _, _ = clustfisher(X, 3)
    assert list(t) == [11, 21]


if __name__ == "__main__":
    for f in (test_diam, test_K1, test_exhaustif, test_paliers):
        f()
        print(f"OK  {f.__name__}")