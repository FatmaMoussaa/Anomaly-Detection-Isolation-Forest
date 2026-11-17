"""Chargement et génération des jeux de données.

Jeux demandés par le prof (aucune étiquette n'est utilisée : Isolation Forest est NON supervisé) :

    iris  : 150 x 4   -> on garde les 4 premiers attributs, on enlève l'espèce
    glass : 214 x 9   -> on enlève l'Id et la classe (Type)
    wdbc  : 569 x 30  -> on enlève l'identifiant et la classe (diagnosis : M / B)

Les fichiers sont lus dans ``data/`` (format UCI : ``iris.data``, ``glass.data``, ``wdbc.data``).
Pour iris et wdbc, si le fichier est absent, on retombe sur la copie fournie par scikit-learn.

Deux jeux de démonstration restent disponibles (ils servent à la démo en 2D) :
``jeu_jouet`` (le hand-tracing de la présentation) et ``generer_gaussiennes``.
Eux seuls ont des étiquettes, connues *par construction*.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

DOSSIER_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# nom -> (colonnes d'attributs dans le fichier UCI, extensions acceptées)
#   iris.data : sepal_l, sepal_w, petal_l, petal_w, espèce
#   glass.data: Id, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Type
#   wdbc.data : id, diagnosis, 30 mesures
COLONNES = {"iris": range(0, 4), "glass": range(1, 10), "wdbc": range(2, 32)}
EXTENSIONS = (".data", ".csv", ".txt")


# --------------------------------------------------------------------------- #
#  Jeux de démonstration (avec étiquettes connues par construction)
# --------------------------------------------------------------------------- #
def jeu_jouet() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Jeu jouet de la diapo de hand-tracing : 5 points normaux (A à E) et une anomalie F(10, 10)."""
    noms = ["A", "B", "C", "D", "E", "F"]
    X = np.array([[2.0, 3.0], [3.0, 2.0], [3.0, 4.0], [4.0, 3.0], [3.0, 3.0], [10.0, 10.0]])
    y = np.array([0, 0, 0, 0, 0, 1])
    return X, y, noms


def generer_gaussiennes(n: int = 500, d: int = 2, contamination: float = 0.05,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Nuage gaussien normal + anomalies uniformes dans un cube plus large. Renvoie (X, y)."""
    rng = np.random.default_rng(seed)
    n_anom = max(1, int(round(n * contamination)))
    normaux = rng.normal(0.0, 1.0, size=(n - n_anom, d))
    anomalies = rng.uniform(-6.0, 6.0, size=(n_anom, d))
    X = np.vstack([normaux, anomalies])
    y = np.r_[np.zeros(n - n_anom, dtype=int), np.ones(n_anom, dtype=int)]
    return X, y


# --------------------------------------------------------------------------- #
#  Jeux du prof : on ne renvoie QUE les attributs (X), jamais la classe
# --------------------------------------------------------------------------- #
def _trouver_fichier(nom: str, chemin: Optional[str]) -> Optional[str]:
    """Chemin explicite, sinon data/<nom>.data|.csv|.txt ; None si rien n'existe."""
    if chemin:
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"fichier introuvable : {chemin}")
        return chemin
    for ext in EXTENSIONS:
        cand = os.path.join(DOSSIER_DATA, nom + ext)
        if os.path.exists(cand):
            return cand
    return None


def _lire_attributs(fichier: str, colonnes) -> np.ndarray:
    """Lit uniquement les colonnes numériques demandées (les autres, texte ou id, sont ignorées)."""
    brut = np.genfromtxt(fichier, delimiter=",", usecols=list(colonnes), dtype=float)
    brut = np.atleast_2d(brut)
    # une éventuelle ligne d'en-tête donne des NaN : on l'écarte
    return brut[~np.isnan(brut).all(axis=1)]


def charger_jeu(nom: str, chemin: Optional[str] = None) -> np.ndarray:
    """Charge ``iris``, ``glass`` ou ``wdbc`` et renvoie X (n, d) sans aucune étiquette."""
    if nom not in COLONNES:
        raise ValueError(f"jeu inconnu : {nom!r} (choix : {', '.join(COLONNES)})")
    fichier = _trouver_fichier(nom, chemin)
    if fichier is not None:
        X = _lire_attributs(fichier, COLONNES[nom])
    elif nom == "iris":
        from sklearn.datasets import load_iris          # copie de secours
        X = load_iris().data
    elif nom == "wdbc":
        from sklearn.datasets import load_breast_cancer  # copie de secours
        X = load_breast_cancer().data
    else:
        raise FileNotFoundError(
            "glass introuvable : placez le fichier du prof (ou celui de "
            "https://archive.ics.uci.edu/dataset/42/glass+identification) dans data/glass.data")
    if np.isnan(X).any():
        raise ValueError(f"{nom} : valeurs manquantes ou colonnes mal lues, vérifiez le fichier")
    return X


def charger_iris(chemin: Optional[str] = None) -> np.ndarray:
    """Iris : 150 x 4 (longueur/largeur des sépales et des pétales), sans l'espèce."""
    return charger_jeu("iris", chemin)


def charger_glass(chemin: Optional[str] = None) -> np.ndarray:
    """Glass Identification : 214 x 9 (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe), sans Id ni classe."""
    return charger_jeu("glass", chemin)


def charger_wdbc(chemin: Optional[str] = None) -> np.ndarray:
    """Breast Cancer Wisconsin (Diagnostic) : 569 x 30, sans identifiant ni diagnostic."""
    return charger_jeu("wdbc", chemin)


CHARGEURS = {"iris": charger_iris, "glass": charger_glass, "wdbc": charger_wdbc}
