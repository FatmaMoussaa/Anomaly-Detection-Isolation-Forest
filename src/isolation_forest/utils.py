"""Fonctions mathématiques utilitaires : nombre harmonique H(i) et facteur c(n).

Rappel (diapos 9 et 15) :
    c(n) = 2 H(n-1) - 2 (n-1) / n      avec c(1) = 0 et c(2) = 1
c(n) est la longueur de chemin moyenne d'une recherche *infructueuse* dans un
arbre binaire de recherche de n éléments. Elle sert à deux choses :
  1. normaliser la longueur de chemin moyenne dans le score s(x, psi) ;
  2. corriger la profondeur d'une feuille qui contient encore plusieurs points
     (branche non terminée, car on a atteint la hauteur limite).
"""
from __future__ import annotations

import math
from functools import lru_cache

# Constante d'Euler-Mascheroni : H(i) ~ ln(i) + gamma pour i grand.
EULER_MASCHERONI = 0.5772156649015329

# En dessous de ce seuil on additionne exactement 1/k ; au-dessus on utilise
# le développement asymptotique (erreur < 1e-12 dès i = 1000).
_SEUIL_EXACT = 1000


def nombre_harmonique(i: int) -> float:
    """Nombre harmonique H(i) = 1 + 1/2 + ... + 1/i  (H(0) = 0)."""
    if i < 0:
        raise ValueError("i doit être >= 0")
    if i == 0:
        return 0.0
    if i <= _SEUIL_EXACT:
        # fsum limite les erreurs d'arrondi lors de la somme
        return math.fsum(1.0 / k for k in range(1, i + 1))
    # développement asymptotique : ln i + gamma + 1/(2i) - 1/(12 i^2)
    return math.log(i) + EULER_MASCHERONI + 1.0 / (2 * i) - 1.0 / (12 * i * i)


@lru_cache(maxsize=None)
def c(n: int) -> float:
    """Facteur de normalisation c(n) (longueur moyenne d'une recherche infructueuse).

    >>> c(1), c(2)
    (0.0, 1.0)
    """
    if n <= 1:          # 0 ou 1 point : rien à ajouter
        return 0.0
    return 2.0 * nombre_harmonique(n - 1) - 2.0 * (n - 1) / n
