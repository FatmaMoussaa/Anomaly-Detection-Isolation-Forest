"""Module 1 : sous-échantillonnage (diapo 12).

Chaque arbre est construit sur un petit sous-échantillon de psi points tirés
*sans remise*. Cela a deux effets :
  - le coût de construction ne dépend plus de n ;
  - on limite le "swamping" (faux positifs dans les zones denses) et le
    "masking" (anomalies groupées qui se cachent mutuellement).
"""
from __future__ import annotations

import numpy as np


def sous_echantillonnage(X: np.ndarray, psi: int, rng: np.random.Generator) -> np.ndarray:
    """Retourne psi lignes de X tirées uniformément sans remise.

    Si X contient au plus psi lignes, on garde tout le jeu de données.

    Note (reverse engineering) : l'indexation avancée ``X[indices]`` crée une
    *copie* des données, alors qu'un slice ``X[a:b]`` crée une *vue*.
    Ici on veut une copie indépendante, ce qui est le cas.
    """
    if psi < 1:
        raise ValueError("psi doit être >= 1")
    n = X.shape[0]
    if n <= psi:                                   # Si |X| <= psi : tout garder
        return X
    indices = rng.choice(n, size=psi, replace=False)   # tirage sans remise
    return X[indices]
