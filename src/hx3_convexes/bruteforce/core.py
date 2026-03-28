"""
bruteforce/core.py — Algorithmes brute-force pour le rang d'un point dans un nuage.

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


def affiche_quickhull(tab: np.ndarray) -> None:
    """Affiche le nuage tab avec son enveloppe convexe tracée en rouge."""
    hull_idx = quick_hull(tab)
    affiche_nuage(tab, hull_idx + [hull_idx[0]])


# ---------------------------------------------------------------------------
# Enveloppe convexe (QuickHull maison, travaillant sur indices)
# ---------------------------------------------------------------------------

def plus_bas(tab: np.ndarray) -> int:
    """Retourne l'indice du point le plus bas (puis le plus à gauche en cas d'égalité)."""
    tab = np.asarray(tab)
    j = 0
    for i in range(len(tab)):
        if tab[i, 1] < tab[j, 1] or (tab[i, 1] == tab[j, 1] and tab[i, 0] < tab[j, 0]):
            j = i
    return j


def plus_haut(tab: np.ndarray) -> int:
    """Retourne l'indice du point le plus haut (puis le plus à gauche en cas d'égalité)."""
    tab = np.asarray(tab)
    j = 0
    for i in range(len(tab)):
        if tab[i, 1] > tab[j, 1] or (tab[i, 1] == tab[j, 1] and tab[i, 0] < tab[j, 0]):
            j = i
    return j


def orient(tab: np.ndarray, i: int, j: int, k: int) -> int:
    """
    Retourne le signe de l'orientation du triplet (tab[i], tab[j], tab[k]) :
      +1 si sens trigonométrique, -1 si sens horaire, 0 si colinéaires.
    """
    tab = np.asarray(tab)
    cross = ((tab[j, 0] - tab[i, 0]) * (tab[k, 1] - tab[i, 1])
           - (tab[j, 1] - tab[i, 1]) * (tab[k, 0] - tab[i, 0]))
    return int(np.sign(cross))


def find_hull(tab: np.ndarray, i: int, j: int) -> list:
    """
    Retourne récursivement les indices des points de l'enveloppe convexe
    situés à gauche du segment tab[i]→tab[j].
    """
    tab = np.asarray(tab)
    N = [k for k in range(len(tab)) if orient(tab, i, j, k) > 0]
    if not N:
        return []
    # Point le plus éloigné de (i,j)
    def dist_cross(k):
        return abs((tab[j, 0] - tab[i, 0]) * (tab[k, 1] - tab[i, 1])
                 - (tab[j, 1] - tab[i, 1]) * (tab[k, 0] - tab[i, 0]))
    p = max(N, key=dist_cross)
    return find_hull(tab, i, p) + [p] + find_hull(tab, p, j)


def quick_hull(tab: np.ndarray) -> list:
    """Retourne la liste des indices des sommets de l'enveloppe convexe de tab."""
    tab = np.asarray(tab)
    b = plus_bas(tab)
    h = plus_haut(tab)
    return [b] + find_hull(tab, b, h) + [h] + find_hull(tab, h, b)


# ---------------------------------------------------------------------------
# Tests d'appartenance à l'enveloppe convexe
# ---------------------------------------------------------------------------

def inconv(tab: np.ndarray, point: np.ndarray) -> bool:
    """
    Retourne True si l'ajout de `point` ne modifie pas l'enveloppe convexe de tab
    (i.e. point est à l'intérieur ou sur le bord de conv(tab)).
    Utilise la comparaison des enveloppes avant/après ajout.
    """
    tab = np.asarray(tab)
    point = np.asarray(point)
    conv1 = quick_hull(tab)
    tab_aug = np.vstack([tab, point])
    conv2 = quick_hull(tab_aug)
    return conv1 == conv2


def inconv2(tab: np.ndarray, point: np.ndarray) -> bool:
    """
    Retourne True si `point` est strictement à l'intérieur de conv(tab).
    Teste que point est à droite (ou sur) chaque arête orientée de l'enveloppe.
    """
    tab = np.asarray(tab)
    point = np.asarray(point)
    tab_aug = np.vstack([tab, point])
    p = len(tab_aug) - 1          # indice de point dans tab_aug
    conv = quick_hull(tab)
    conv = conv + [conv[0]]       # fermeture du polygone
    for i in range(len(conv) - 1):
        if orient(tab_aug, p, conv[i], conv[i + 1]) >= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Algorithmes brute-force
# ---------------------------------------------------------------------------

def sublist(L: list, n: int) -> list:
    """Retourne toutes les sous-listes de longueur n de L."""
    return [list(k) for k in combinations(L, n)]


def brutforce(tab: np.ndarray, p: int):
    """
    Trouve le plus grand sous-nuage S de tab-{p} tel que tab[p] ∉ conv(S).
    Retourne la taille de S.
    Utilise inconv (comparaison d'enveloppes).
    """
    tab = np.asarray(tab)
    point = tab[p]
    rest = np.delete(tab, p, axis=0)
    indices = [k for k in range(len(tab)) if k != p]
    n = len(rest)

    # Génère toutes les sous-listes d'indices dans l'ordre décroissant de taille
    all_subs = []
    for r in range(1, n + 1):
        all_subs.extend(list(combinations(range(n), r)))
    all_subs = all_subs[::-1]   # plus grandes en premier

    for sub_idx in all_subs:
        sub = rest[list(sub_idx)]
        if not inconv(sub, point):
            conv = quick_hull(sub)
            orig_idx = [indices[k] for k in sub_idx]
            conv_idx = [orig_idx[k] for k in conv]
            affiche_quickhull(sub)
            affiche_nuage(tab, conv_idx + [conv_idx[0]])
            return len(sub_idx)


def brutforce2(tab: np.ndarray, p: int):
    """
    Identique à brutforce mais utilise inconv2 (test géométrique par arêtes).
    """
    tab = np.asarray(tab)
    point = tab[p]
    rest = np.delete(tab, p, axis=0)
    indices = [k for k in range(len(tab)) if k != p]
    n = len(rest)

    all_subs = []
    for r in range(1, n + 1):
        all_subs.extend(list(combinations(range(n), r)))
    all_subs = all_subs[::-1]

    for sub_idx in all_subs:
        sub = rest[list(sub_idx)]
        if not inconv2(sub, point):
            conv = quick_hull(sub)
            orig_idx = [indices[k] for k in sub_idx]
            conv_idx = [orig_idx[k] for k in conv]
            affiche_quickhull(sub)
            affiche_nuage(tab, conv_idx + [conv_idx[0]])
            print("rang =", len(tab) - len(sub_idx))
            return len(sub_idx)


def brut_tranche(tab: np.ndarray, p: int):
    """
    Variante de brutforce2 qui itère par tranches décroissantes de taille,
    ce qui permet de minimiser le précalcul en faisant uniquement les listes de
    taille nécessaire.
    """
    tab = np.asarray(tab)
    point = tab[p]
    rest = np.delete(tab, p, axis=0)
    indices = [k for k in range(len(tab)) if k != p]
    n = len(rest)

    for j in range(n):
        size = n - j
        sub_idx_list = sublist(list(range(n)), size)
        for sub_idx in reversed(sub_idx_list):
            sub = rest[sub_idx]
            if not inconv2(sub, point):
                conv = quick_hull(sub)
                orig_idx = [indices[k] for k in sub_idx]
                conv_idx = [orig_idx[k] for k in conv]
                affiche_quickhull(sub)
                affiche_nuage(tab, conv_idx + [conv_idx[0]])
                print("rang =", len(tab) - len(sub_idx))
                return len(sub_idx)


def dist_to_barycenter(tab: np.ndarray) -> list:
    """
    Pour chaque point tab[p], calcule son rang (via brutforce) et sa distance
    au barycentre du nuage.
    Retourne une liste de [indice, x, y, rang, dist_barycentre].
    """
    tab = np.asarray(tab)
    G = tab.mean(axis=0)
    L = []
    for p in range(len(tab)):
        rang = brutforce(tab, p)
        dist = np.linalg.norm(tab[p] - G)
        L.append([p, tab[p, 0], tab[p, 1], rang, dist])
    return L