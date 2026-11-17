"""Isolation Forest : implémentation pédagogique (TD2, structures de données avancées en Python)."""
from .comparaison import correlation_rangs, recouvrement_top_k
from .iforest import IsolationForest
from .itree import (Feuille, Noeud, NoeudArbre, arbre_en_texte, compter_noeuds, construire_itree,
                    hauteur, longueur_chemin, longueur_chemin_recursive, trace_chemin)
from .sampling import sous_echantillonnage
from .utils import c, nombre_harmonique

__all__ = [
    "IsolationForest", "NoeudArbre", "Noeud", "Feuille", "construire_itree", "longueur_chemin",
    "longueur_chemin_recursive", "trace_chemin", "arbre_en_texte", "hauteur", "compter_noeuds",
    "sous_echantillonnage", "c", "nombre_harmonique", "correlation_rangs", "recouvrement_top_k",
]
__version__ = "1.1.0"
