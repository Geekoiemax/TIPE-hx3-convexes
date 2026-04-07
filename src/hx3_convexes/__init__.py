"""
TIPE — Module principal.
Expose les sous-modules droites et bruteforce ainsi que les utilitaires communs.
"""

from hx3_convexes._common import random_cloud, tab1, circle, c1, diag,profondeurRes
from hx3_convexes import droites, bruteforce, simplexes,angles

__all__ = ["random_cloud","tab1", "circle", "c1", "diag","droites", "bruteforce", "simplexes","angles"]