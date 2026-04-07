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
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

@dataclass
class profondeurRes:
    rang: int
    corang: int

    queue: np.ndarray
    tete: np.ndarray
    
    pivot: np.ndarray       # best_B
    nuage: np.ndarray            # original cloud

    alignes: np.ndarray | None          # collinear points used

    nuage_sans_etude : np.ndarray
    point_etude : np.ndarray

    def show(self, showhull: bool = True):
        c0 = self.nuage_sans_etude
        A = self.point_etude
        B = self.pivot

        plt.figure(frameon=True)

        # all points
        plt.plot(c0[:, 0], c0[:, 1], 'o', color='blue')

        # best side
        if len(self.queue) > 0:
            plt.plot(self.queue[:, 0], self.queue[:, 1], 'o', color='deeppink')

        # pivot and direction
        plt.plot(A[0], A[1], 'o', color='darkorange')
        plt.plot(B[0], B[1], 'o', color='black')

        # line
        plt.axline(A, B, color='red') # type: ignore

        # convex hull
        if showhull and len(self.queue) >= 3:
            hull = ConvexHull(self.queue)
            hull_pts = np.vstack([
                self.queue[hull.vertices],
                self.queue[hull.vertices[0]]
            ])
            plt.plot(hull_pts[:, 0], hull_pts[:, 1], '-k')

        plt.show()


# ---------------------------------------------------------------------------
# Fonction utilitaire commune
# ---------------------------------------------------------------------------

def random_cloud(n: int, a: int, rng: np.random.Generator | None = None) -> np.ndarray:
    assert n <= (a + 1) ** 2, "Not enough points to generate"
    
    if rng is None:
        rng = np.random.default_rng()
    
    flat = rng.choice((a + 1) ** 2, size=n, replace=False)
    return np.stack([flat // (a + 1), flat % (a + 1)], axis=1)

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