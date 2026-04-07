"""
droites/core.py — Algorithme des droites pour le rang d'un point dans un nuage.

Toutes les fonctions travaillent avec des np.array de forme (n, 2),
chaque ligne étant un point [x, y].
"""

from hx3_convexes._common import *

# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def affiche_nuage(P: np.ndarray, C: list = []) -> None:
    """
    Affiche le nuage de points P (array (n,2)).
    Trace la ligne brisée passant par les points dont les indices sont dans C.
    """
    P = np.asarray(P)
    fig, ax = plt.subplots()
    plt.grid(True)
    xMin, xMax = int(P[:, 0].min()) - 1, int(P[:, 0].max()) + 1
    yMin, yMax = int(P[:, 1].min()) - 1, int(P[:, 1].max()) + 1
    ax.xaxis.set_ticks(range(xMin, xMax + 1))
    ax.yaxis.set_ticks(range(yMin, yMax + 1))
    plt.scatter(P[:, 0], P[:, 1], s=50)
    for i, (x, y) in enumerate(P):
        plt.annotate(str(i), (x + 0.1, y + 0.1), size="large")
    if C:
        pts = P[C]
        plt.plot(pts[:, 0], pts[:, 1], color="red", linewidth=2)


# ---------------------------------------------------------------------------
# Géométrie de base
# ---------------------------------------------------------------------------

def vect(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Retourne le vecteur MN dans R³ (coordonnée z = 0)
    à partir de deux points dans R².
    """
    M, N = np.asarray(M), np.asarray(N)
    return np.array([N[0] - M[0], N[1] - M[1], 0], dtype=float)


def orient(A: np.ndarray, B: np.ndarray, M: np.ndarray) -> float:
    """
    Retourne la composante z de AB × AM.
    Positive ⟹ M est à gauche de AB (sens trigonométrique).
    Négative ⟹ M est à droite.
    Nulle    ⟹ M est sur la droite (AB).
    """
    A, B, M = np.asarray(A), np.asarray(B), np.asarray(M)
    return np.cross(vect(A, B), vect(A, M))[-1]


# ---------------------------------------------------------------------------
# Partition du plan par une droite
# ---------------------------------------------------------------------------

def split_plan(L: np.ndarray, A: np.ndarray, B: np.ndarray) -> list:
    """
    Divise le nuage L en trois parties par rapport à la droite (AB) :
      - E : points strictement au-dessus (épigraphe)
      - H : points strictement en-dessous (hypographe)
      - G : points sur la droite (graphe)
    Retourne [E, H, G], chacun étant un np.array (k,2).
    """
    L = np.asarray(L)
    A, B = np.asarray(A), np.asarray(B)
    orientations = np.array([orient(A, B, M) for M in L])
    E = L[orientations > 0]
    H = L[orientations < 0]
    G = L[orientations == 0]
    return [E, H, G]


def split_line(L: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Parmi les points de L, retourne le sous-ensemble du côté
    le plus peuplé par rapport à l'abscisse de M
    (utilisé pour les points colinéaires avec A).
    """
    L = np.asarray(L)
    M = np.asarray(M)
    if len(L)> 0 and not np.array_equal(L[0][0],M[0]) :
        right = L[L[:, 0] > M[0]]
        left  = L[L[:, 0] < M[0]]
    else :
        right = L[L[:, 1] > M[1]]
        left  = L[L[:, 1] < M[1]]
    return right if len(right) >= len(left) else left


# ---------------------------------------------------------------------------
# Enveloppe convexe (scipy)
# ---------------------------------------------------------------------------

def quickhull(c: np.ndarray) -> list:
    """
    Retourne la liste des indices des sommets de l'enveloppe convexe de c
    via scipy.spatial.ConvexHull.
    """
    c = np.asarray(c)
    return list(scipy.spatial.ConvexHull(c).vertices)


# ---------------------------------------------------------------------------
# Algorithme des droites
# ---------------------------------------------------------------------------

def droites(c: np.ndarray, a: int, show: bool = True, showhull: bool = True):
    """
    Pour le point c[a], trouve la droite passant par c[a] qui maximise
    le nombre de points d'un même côté.
    Retourne la taille du plus grand demi-plan peuplé.

    Paramètres
    ----------
    c       : np.array (n, 2) — nuage de points
    a       : indice du point étudié
    show    : affiche la figure matplotlib si True
    showhull: trace l'enveloppe convexe du demi-plan max si True
    """
    c = np.asarray(c, dtype=float)
    c0 = np.delete(c, a, axis=0)   # nuage sans le point a
    A = c[a]

    best_count   = 0
    best_side    = np.empty((0, 2))
    best_B       = A.copy()
    best_on_line = np.empty((0, 2))

    for B in c0:
        E, H, G = split_plan(c0, A, B)
        for side in (E, H):
            if len(side) >= best_count:
                best_count   = len(side)
                best_side    = side
                best_B       = B
                best_on_line = G
        if best_count == len(c0):
            break   # on ne peut pas faire mieux


    # Ajoute les points colinéaires du bon côté
    if len(best_on_line) > 0:
        extra = split_line(best_on_line, A)
        if len(extra) > 0:
            best_side = np.vstack([best_side, extra])

    if show:
        plt.figure(frameon=True)
        plt.plot(c0[:, 0], c0[:, 1], 'o', color='blue')
        if len(best_side) > 0:
            plt.plot(best_side[:, 0], best_side[:, 1], 'o', color='deeppink')
        plt.plot(best_B[0], best_B[1], 'o', color='black')
        plt.plot(A[0], A[1], 'o', color='darkorange')
        plt.plot([A[0], best_B[0]], [A[1], best_B[1]], color='red')
        plt.axline(A, best_B)
        if showhull and len(best_side) >= 3:
            H_hull = ConvexHull(best_side)
            hull_pts = np.vstack([
                best_side[H_hull.vertices],
                best_side[H_hull.vertices[0]]
            ])
            plt.plot(hull_pts[:, 0], hull_pts[:, 1], '-k')
        cursor(hover=True)
        plt.show()

    return len(c) - len(best_side)


def decapitate(x,L) : 
    return [y for y in L if not np.array_equal(x, y)]

def droites_heuristique(c: np.ndarray, a: int, show: bool = False, showhull: bool = True):
    """
    Pour le point c[a], trouve la droite passant par c[a] qui maximise
    le nombre de points d'un même côté.
    Retourne la taille du plus grand demi-plan peuplé.

    Paramètres
    ----------
    c       : np.array (n, 2) — nuage de points
    a       : indice du point étudié
    show    : affiche la figure matplotlib si True
    showhull: trace l'enveloppe convexe du demi-plan max si True
    """
    c = np.asarray(c, dtype=float)
    c0 = np.delete(c, a, axis=0)   # nuage sans le point a
    iteratives = list(c0.copy())
    A = c[a]

    best_count   = 0
    best_side    = np.empty((0, 2))
    best_B       = A.copy()
    best_on_line = np.empty((0, 2))

    for B in iteratives:
        E,G,H = np.empty((0, 2)),np.empty((0, 2)),np.empty((0, 2))
        for k,M in enumerate(c0) : 
            o = orient(A,B,M)
            if o > 0 : 
                E = np.vstack([E,M])
            elif o == 0 :
                G = np.vstack([G,M])
                iteratives = decapitate(M,iteratives)
            else : 
                H = np.vstack([H,M])
            if max(len(E),len(H)) + len(c0) < best_count + k + 1 : 
                break
        if len(E) >= best_count:
            best_count   = len(E)
            best_side    = E
            best_B       = B
            best_on_line = G
            worst_side = H

        if len(H) >= best_count:
            best_count   = len(H)
            best_side    = H
            best_B       = B
            best_on_line = G
            worst_side = E

        if best_count == len(c0):
            break   # on ne peut pas faire mieux
    # Ajoute les points colinéaires du bon côté

    if len(best_on_line) > 0:
        extra = split_line(best_on_line, A)
        if len(extra) > 0:
            best_side = np.vstack([best_side, extra])

    res = profondeurRes(
            rang=len(c0) - best_count,
            corang=best_count,
            queue=best_side,
            tete=worst_side,
            pivot=best_B,
            nuage=c,
            alignes=best_on_line,
            point_etude= A,
            nuage_sans_etude= c0
        )

    if show:
        res.show(showhull=showhull)
    return res