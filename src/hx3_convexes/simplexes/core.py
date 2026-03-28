"""
simplexes/core.py — Algorithmes des ensembles intersectants sur les simplexes (Caratheodory) pour le rang d'un point dans un nuage.

Toutes les fonctions travaillent avec des np.array de forme (n, 2),
chaque ligne étant un point [x, y].
"""

from hx3_convexes._common import *
from hx3_convexes.bruteforce.core import inconv2, affiche_nuage, quick_hull, plus_bas, sublist


# ---------------------------------------------------------------------------
# Algorithmes des simplexes
# ---------------------------------------------------------------------------

def _set_minus(A: list, B: list) -> list:
    """Retourne A - B (éléments de A absents de B)."""
    return [a for a in A if a not in B]


def _union(S: list) -> list:
    """Retourne l'union (sans doublons, ordre de première apparition) des listes de S."""
    u = []
    for s in S:
        for x in s:
            if x not in u:
                u.append(x)
    return u

def simplexes(c: np.ndarray, a: int) -> list:
    """
    Pour le point c[a], retourne la liste de tous les simplexes (triplets d'indices
    dans c) dont l'enveloppe convexe contient c[a].

    Paramètres
    ----------
    c : np.array (n, 2) — nuage de points
    a : indice du point étudié

    Sortie
    ------
    S : list de listes d'entiers — chaque élément est un triplet d'indices
        (dans c) formant un simplexe contenant c[a].
    """
    c = np.asarray(c, dtype=float)
    c0 = np.delete(c, a, axis=0)
    A = c[a]

    indeces = list(range(len(c)))
    indeces = indeces[:a] + indeces[a+1:]

    dim = len(c[0])  # = 2 en 2D, taille d'un simplexe = dim+1 = 3

    subpoints  = sublist(list(c0), dim + 1)
    subindeces = sublist(indeces,  dim + 1)

    S = []
    for index in range(len(subpoints)):
        if inconv2(np.array(subpoints[index]), A):
            # simplexe_loc : indices locaux des sommets de l'enveloppe du sous-nuage
            simplexe_loc = quick_hull(np.array([c[i] for i in subindeces[index]]))
            # simplexe : traduit ces indices locaux en indices globaux dans c
            simplexe = [subindeces[index][k] for k in simplexe_loc]
            S.append(simplexe)

    return S


def to_cover(S: list) -> tuple:
    """
    Traduit la liste de simplexes en une structure pour le problème de couverture.

    Paramètres
    ----------
    S : liste de listes d'entiers (simplexes, chacun étant une liste d'indices dans c)

    Sorties
    -------
    Tc : list de listes — Tc[k] est la liste des indices i tels que P[k] ∈ S[i]
         (même structure que l'original)
    P  : list — l'union de tous les indices présents dans S (l'univers)
    """
    P = _union(S)  # union dans l'ordre de première apparition

    n = len(S)
    Tc = [[] for _ in range(len(P))]
    for k, x in enumerate(P):
        for i in range(n):
            if x in S[i]:
                Tc[k].append(i)

    return Tc, P




def greedy_cover(tc: tuple) -> list:
    """
    Solution gloutonne au problème de couverture.
    Prend la sortie de to_cover et retourne un ensemble intersectant minimal.

    Paramètres
    ----------
    tc : (Tc, P) — sortie de to_cover

    Sortie
    ------
    Inter : list d'entiers — indices (dans c) des points à retirer pour que
            c[a] sorte de toute enveloppe convexe de simplexe.
    """
    Tc, P = tc

    C    = []   # simplexes déjà couverts (listes choisies)
    uC   = []   # union courante des simplexes choisis
    S    = _union(Tc)           # tous les indices de simplexes à couvrir
    n    = len(Tc)
    I    = []
    Itot = list(range(n))

    while len(uC) < len(S):
        Itemp = _set_minus(Itot, I)
        i_max = Itemp[0]
        uC = _union(C)
        for i in Itemp:
            if len(_set_minus(Tc[i], uC)) > len(_set_minus(Tc[i_max], uC)):
                i_max = i
        I.append(i_max)
        C.append(Tc[i_max])

    Inter = [P[i] for i in I]
    print("Inter =", Inter)
    return Inter


def greedy_solve(c: np.ndarray, a: int, show: bool = True, Print: bool = False) -> list:
    """
    Résout le problème complet : trouve les points à retirer pour que c[a]
    sorte de l'enveloppe convexe, et affiche le résultat.

    Paramètres
    ----------
    c     : np.array (n, 2) — nuage de points
    a     : indice du point étudié
    show  : affiche la figure matplotlib si True
    Print : imprime la solution si True

    Sortie
    ------
    s : liste d'indices (dans c) des points à retirer
    """
    s = greedy_cover(to_cover(simplexes(c, a)))

    if show:
        c = np.asarray(c)
        k = [idx for idx in range(len(c)) if idx != a]
        # i = indices à GARDER (dans k mais pas dans s)
        # j = indices à RETIRER (dans s)
        i_keep = _set_minus(k, s)
        j_remove = _set_minus(k, i_keep)

        I = c[i_keep]   # points gardés  → affichés mais hors de la solution
        J = c[j_remove] # points retirés → font partie de la solution

        hull_local = quick_hull(I)
        base_local = plus_bas(I)
        affiche_nuage(
            np.concatenate([I, J, np.array([c[a]])]),
            hull_local + [base_local]
        )
        print("A =", c[a])

    if Print:
        print("s, len(s) =", s, len(s))

    return s
