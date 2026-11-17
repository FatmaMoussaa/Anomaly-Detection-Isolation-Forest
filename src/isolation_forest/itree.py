"""Module 2 : l'arbre d'isolation (iTree) et la longueur de chemin (diapos 8, 11, 13, 15).

Structure de données : un arbre binaire dont les nœuds sont
  - ``Noeud(attribut q, seuil p, gauche, droite)`` : nœud interne ;
  - ``Feuille(taille)``                            : feuille, qui mémorise le
    nombre de points restés isolés ensemble.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .utils import c


# --------------------------------------------------------------------------- #
#  Structures de données
# --------------------------------------------------------------------------- #
class NoeudArbre:
    """Classe de base commune aux nœuds internes et aux feuilles."""

    __slots__ = ()

    def est_feuille(self) -> bool:
        """Indique si le nœud est une feuille (à redéfinir dans les sous-classes)."""
        raise NotImplementedError


class Feuille(NoeudArbre):
    """Feuille d'un iTree : mémorise combien de points sont arrivés ici."""

    __slots__ = ("taille",)

    def __init__(self, taille: int) -> None:
        self.taille = int(taille)

    def est_feuille(self) -> bool:
        """Toujours vrai pour une feuille."""
        return True

    def __repr__(self) -> str:
        return f"Feuille(taille={self.taille})"


class Noeud(NoeudArbre):
    """Nœud interne : coupe l'attribut ``attribut`` au seuil ``seuil``.

    Convention : gauche si x[q] < p ; droite si x[q] >= p.
    """

    __slots__ = ("attribut", "seuil", "gauche", "droite")

    def __init__(self, attribut: int, seuil: float, gauche: NoeudArbre, droite: NoeudArbre) -> None:
        self.attribut = int(attribut)
        self.seuil = float(seuil)
        self.gauche = gauche
        self.droite = droite

    def est_feuille(self) -> bool:
        """Toujours faux pour un nœud interne."""
        return False

    def __repr__(self) -> str:
        return f"Noeud(q={self.attribut}, p={self.seuil:.4g})"


# --------------------------------------------------------------------------- #
#  Construction
# --------------------------------------------------------------------------- #
def construire_itree(X_sub: np.ndarray, h: int, limite: int, rng: np.random.Generator) -> NoeudArbre:
    """Construit récursivement un arbre d'isolation (pseudo-code de la diapo 13).

    Paramètres
    ----------
    X_sub  : sous-échantillon (psi lignes, d colonnes)
    h      : hauteur (profondeur) courante, 0 pour la racine
    limite : hauteur limite l = ceil(log2(psi))
    rng    : générateur aléatoire numpy
    """
    n = X_sub.shape[0]
    # Cas d'arrêt 1 : hauteur limite atteinte ou au plus 1 point
    if h >= limite or n <= 1:
        return Feuille(n)

    # 1) attribut aléatoire q
    q = int(rng.integers(X_sub.shape[1]))
    colonne = X_sub[:, q]
    min_val, max_val = colonne.min(), colonne.max()

    # Cas d'arrêt 2 : toutes les valeurs sont égales -> impossible de couper
    if min_val == max_val:
        return Feuille(n)

    # 2) seuil aléatoire uniforme entre min et max
    p = rng.uniform(min_val, max_val)

    # 3) partage : gauche x[q] < p, droite x[q] >= p
    masque_gauche = colonne < p
    gauche = construire_itree(X_sub[masque_gauche], h + 1, limite, rng)
    droite = construire_itree(X_sub[~masque_gauche], h + 1, limite, rng)
    return Noeud(q, p, gauche, droite)


# --------------------------------------------------------------------------- #
#  Longueur de chemin
# --------------------------------------------------------------------------- #
def longueur_chemin_recursive(x: Sequence[float], arbre: NoeudArbre, h: int = 0) -> float:
    """Version récursive, identique au pseudo-code de la diapo 15."""
    if isinstance(arbre, Feuille):
        return h + c(arbre.taille)            # profondeur + correction c(taille)
    if x[arbre.attribut] < arbre.seuil:
        return longueur_chemin_recursive(x, arbre.gauche, h + 1)
    return longueur_chemin_recursive(x, arbre.droite, h + 1)


def longueur_chemin(x: Sequence[float], arbre: NoeudArbre, h: int = 0) -> float:
    """Version itérative (même résultat, pas de pile d'appels)."""
    noeud = arbre
    while isinstance(noeud, Noeud):
        noeud = noeud.gauche if x[noeud.attribut] < noeud.seuil else noeud.droite
        h += 1
    return h + c(noeud.taille)


def trace_chemin(x: Sequence[float], arbre: NoeudArbre, noms: Optional[Sequence[str]] = None) -> List[str]:
    """Décrit pas à pas le chemin de x (utile pour le hand-tracing)."""
    etapes: List[str] = []
    noeud, h = arbre, 0
    while isinstance(noeud, Noeud):
        nom = noms[noeud.attribut] if noms else f"x[{noeud.attribut}]"
        if x[noeud.attribut] < noeud.seuil:
            etapes.append(f"h={h}: {nom}={x[noeud.attribut]:.3g} < {noeud.seuil:.3g} -> gauche")
            noeud = noeud.gauche
        else:
            etapes.append(f"h={h}: {nom}={x[noeud.attribut]:.3g} >= {noeud.seuil:.3g} -> droite")
            noeud = noeud.droite
        h += 1
    etapes.append(f"h={h}: feuille de taille {noeud.taille} -> longueur = {h} + c({noeud.taille}) = {h + c(noeud.taille):.4f}")
    return etapes


# --------------------------------------------------------------------------- #
#  Outils d'inspection
# --------------------------------------------------------------------------- #
def hauteur(arbre: NoeudArbre) -> int:
    """Hauteur de l'arbre (0 si c'est une simple feuille)."""
    if isinstance(arbre, Feuille):
        return 0
    return 1 + max(hauteur(arbre.gauche), hauteur(arbre.droite))


def compter_noeuds(arbre: NoeudArbre) -> Tuple[int, int]:
    """Retourne (nombre de nœuds internes, nombre de feuilles)."""
    if isinstance(arbre, Feuille):
        return 0, 1
    gi, gf = compter_noeuds(arbre.gauche)
    di, df = compter_noeuds(arbre.droite)
    return 1 + gi + di, gf + df


def tailles_feuilles(arbre: NoeudArbre) -> List[int]:
    """Liste des tailles de toutes les feuilles (leur somme vaut psi)."""
    if isinstance(arbre, Feuille):
        return [arbre.taille]
    return tailles_feuilles(arbre.gauche) + tailles_feuilles(arbre.droite)


def _lignes_arbre(arbre: NoeudArbre, noms: Optional[Sequence[str]]) -> List[str]:
    """Lignes de la représentation textuelle (branches dessinées avec ├─ └─ │)."""
    if isinstance(arbre, Feuille):
        return [f"Feuille(taille={arbre.taille})"]
    nom = noms[arbre.attribut] if noms else f"x[{arbre.attribut}]"
    lignes = [f"{nom} < {arbre.seuil:.3g} ?"]
    for etiquette, enfant, dernier in (("oui", arbre.gauche, False), ("non", arbre.droite, True)):
        sous = _lignes_arbre(enfant, noms)
        lignes.append(("└─ " if dernier else "├─ ") + f"{etiquette}: " + sous[0])
        lignes += [("   " if dernier else "│  ") + ligne for ligne in sous[1:]]
    return lignes


def arbre_en_texte(arbre: NoeudArbre, noms: Optional[Sequence[str]] = None) -> str:
    """Représentation textuelle de l'arbre (une ligne par nœud)."""
    return "\n".join(_lignes_arbre(arbre, noms)) + "\n"


def decoupes(arbre: NoeudArbre, bbox: Tuple[float, float, float, float], h: int = 0
             ) -> Iterator[Tuple[int, float, Tuple[float, float, float, float], int]]:
    """Énumère les coupes d'un arbre *en 2D* : (attribut, seuil, boîte englobante, profondeur).

    bbox = (xmin, xmax, ymin, ymax). Sert à dessiner les coupes (module viz).
    """
    if isinstance(arbre, Feuille):
        return
    xmin, xmax, ymin, ymax = bbox
    q, p = arbre.attribut, arbre.seuil
    yield q, p, bbox, h
    if q == 0:
        yield from decoupes(arbre.gauche, (xmin, p, ymin, ymax), h + 1)
        yield from decoupes(arbre.droite, (p, xmax, ymin, ymax), h + 1)
    else:
        yield from decoupes(arbre.gauche, (xmin, xmax, ymin, p), h + 1)
        yield from decoupes(arbre.droite, (xmin, xmax, p, ymax), h + 1)
