"""
droites/core.py — Algorithme des droites pour le rang d'un point dans un nuage.

Toutes les fonctions travaillent avec des np.array de forme (n, 2),
chaque ligne étant un point [x, y].
"""

from hx3_convexes._common import *
from hx3_convexes.droites import split_line

def angles(c: np.ndarray, a: int, show: bool = True, showhull: bool = True):
    c = np.asarray(c, dtype=float)
    c0 = np.delete(c, a, axis=0)
    A = c[a]
    n = len(c0)

    translate = c0 - A
    angs = np.arctan2(translate[:, 1], translate[:, 0])

    # Tri par angle
    order = np.argsort(angs)
    sorted_pts = c0[order]
    sorted_angs = angs[order]

    # Doublement pour gérer la circularité
    angs_doubled = np.concatenate([sorted_angs, sorted_angs + 2 * np.pi])
    pts_doubled  = np.concatenate([sorted_pts, sorted_pts])

    m          = n
    best_side  = np.empty((0, 2))
    worst_side = np.empty((0, 2))
    best_B     = A.copy()
    best_i     = 0

    edge = 0
    for i in range(n):
        if edge < i:
            edge = i
        while edge < i + n and angs_doubled[edge] < angs_doubled[i] + np.pi:
            edge += 1

        d        = edge - i
        opposite = n - d

        if min(d, opposite) < m:
            m = min(d, opposite)
            if d <= opposite:
                best_side  = pts_doubled[i:edge]
                worst_side = pts_doubled[edge:i + n]
            else:
                best_side  = pts_doubled[edge:i + n]
                worst_side = pts_doubled[i:edge]
            best_B = sorted_pts[i % n]
            best_i = i

        if m == 0:
            break

    # Récupération des colinéaires depuis le tableau trié
    collineaires = []

    # Points dans la même direction (angle identique)
    k = best_i
    while k < edge and np.isclose(angs_doubled[k], angs_doubled[best_i]):
        collineaires.append(pts_doubled[k])
        k += 1

    # Points dans la direction opposée (angle + π)
    j = edge
    while j < best_i + n and np.isclose(angs_doubled[j], angs_doubled[best_i] + np.pi):
        collineaires.append(pts_doubled[j])
        j += 1

    alignes = np.array(collineaires) if collineaires else np.empty((0, 2))
    if len(alignes) > 0:
        alignes = split_line(alignes, A)

    res = profondeurRes(
        rang=m + 1,
        corang=n - m,
        queue=best_side,
        tete=worst_side,
        pivot=best_B,
        nuage=c,
        alignes=alignes,
        point_etude=A,
        nuage_sans_etude=c0
    )

    if show:
        res.show(showhull=showhull)
    return res