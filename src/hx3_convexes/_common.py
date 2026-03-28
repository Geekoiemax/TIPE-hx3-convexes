"""
_common.py — Imports communs et données d'exemple partagés par tous les sous-modules de TIPE.
"""

# ----------------------------------------------------------------------------
# Imports communs
# ----------------------------------------------------------------------------


import matplotlib.pyplot as plt
import random
from itertools import combinations
import numpy as np
import csv
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import scipy
from mplcursors import cursor
from functools import reduce


# ---------------------------------------------------------------------------
# Fonction utilitaire commune
# ---------------------------------------------------------------------------

def random_cloud(n: int, a: int) -> np.ndarray:
    """
    Génère un nuage de n points aléatoires distincts dans [0,a]².
    Retourne un np.array de forme (n, 2).
    """
    points = np.array([[x, y] for x in range(a + 1) for y in range(a + 1)])
    indices = np.random.choice(len(points), size=n, replace=False)
    return points[indices]


# ---------------------------------------------------------------------------
# Données d'exemple
# ---------------------------------------------------------------------------

# Nuage de base (12 points)
tab1 = np.array([
    [6, -1], [1, 4], [1, 8], [4, 1], [4, 4],
    [5, 9], [5, 6], [0, -1], [7, 2], [8, 5],
    [11, 6], [13, 1]
])

# Approximation d'un cercle (21 points)
circle = np.array([
    [0, 0], [25, 0], [-25, 0], [0, 25], [0, -25],
    [7, 24], [7, -24], [-7, 24], [-7, -24],
    [24, 7], [24, -7], [-24, 7], [-24, -7],
    [15, 20], [15, -20], [-15, 20], [-15, -20],
    [20, 15], [20, -15], [-20, 15], [-20, -15]
])

# Nuage c1 de Droites.py (12 points)
c1 = np.array([
    [6, -1], [1, 4], [1, 8], [4, 1], [4, 4],
    [5, 9], [5, 6], [0, -1], [7, 2], [8, 5],
    [11, 6], [13, 1]
])

# Diagonale (20 points)
diag = np.array([[k, k] for k in range(20)])