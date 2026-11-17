"""Comparer deux classements de scores (sans étiquettes) : par exemple notre forêt contre scikit-learn.

Comme les jeux du prof n'ont pas d'étiquettes, on ne peut pas calculer d'AUC.
On vérifie à la place que deux implémentations ordonnent les points de la même façon.
"""
from __future__ import annotations

import numpy as np


def rangs(valeurs: np.ndarray) -> np.ndarray:
    """Rang de chaque valeur (0 = plus petite) ; les ex æquo reçoivent le rang moyen."""
    valeurs = np.asarray(valeurs, dtype=float)
    ordre = np.argsort(valeurs, kind="mergesort")
    r = np.empty(len(valeurs), dtype=float)
    r[ordre] = np.arange(len(valeurs))
    for v in np.unique(valeurs):                       # rang moyen pour les ex æquo
        masque = valeurs == v
        if masque.sum() > 1:
            r[masque] = r[masque].mean()
    return r


def correlation_rangs(a: np.ndarray, b: np.ndarray) -> float:
    """Corrélation de Spearman : 1 = même classement, 0 = aucun lien, -1 = classement inversé."""
    ra, rb = rangs(a), rangs(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def recouvrement_top_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Part des k points les plus anormaux (scores les plus élevés) communs aux deux classements."""
    if not 1 <= k <= len(a):
        raise ValueError("k doit être compris entre 1 et le nombre de points")
    top_a = set(np.argsort(-np.asarray(a), kind="mergesort")[:k].tolist())
    top_b = set(np.argsort(-np.asarray(b), kind="mergesort")[:k].tolist())
    return len(top_a & top_b) / k
