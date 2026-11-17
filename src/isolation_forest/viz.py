"""Visualisations (matplotlib) : coupes d'un arbre, carte des scores, projection PCA, histogramme, tableau de bord.

Les couleurs reprennent la palette de la présentation.
Toutes les fonctions retournent la figure ; ``sauver(fig, chemin)`` l'enregistre.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .iforest import IsolationForest
from .itree import NoeudArbre, decoupes

MARINE, TEAL, ROUGE, AMBRE, GRIS = "#13213f", "#1f7a8c", "#d62839", "#e8a33d", "#8a94a6"


def sauver(fig: Figure, chemin: str, dpi: int = 150) -> str:
    """Enregistre la figure (crée le dossier si besoin) et la ferme."""
    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)
    fig.savefig(chemin, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chemin


def _marges(X: np.ndarray, marge: float = 0.1):
    mn, mx = X.min(axis=0), X.max(axis=0)
    ecart = (mx - mn) * marge
    ecart[ecart == 0] = 1.0
    return (mn[0] - ecart[0], mx[0] + ecart[0], mn[1] - ecart[1], mx[1] + ecart[1])


def tracer_decoupes(arbre: NoeudArbre, X: np.ndarray, y: Optional[np.ndarray] = None,
                    noms_points: Optional[Sequence[str]] = None, titre: str = "Coupes d'un arbre d'isolation",
                    ax=None) -> Figure:
    """Dessine les coupes aléatoires d'un iTree en 2D (idée clé de la diapo 7)."""
    if X.shape[1] != 2:
        raise ValueError("tracer_decoupes ne gère que les données 2D")
    fig, ax = (plt.subplots(figsize=(6, 5)) if ax is None else (ax.figure, ax))
    bbox = _marges(X)
    for q, p, (x0, x1, y0, y1), h in decoupes(arbre, bbox):
        epaisseur = max(2.2 - 0.5 * h, 0.7)           # les premières coupes sont plus épaisses
        if q == 0:
            ax.plot([p, p], [y0, y1], "--", color=AMBRE, lw=epaisseur)
        else:
            ax.plot([x0, x1], [p, p], "--", color=AMBRE, lw=epaisseur)
    couleurs = np.where(y == 1, ROUGE, TEAL) if y is not None else TEAL
    ax.scatter(X[:, 0], X[:, 1], c=couleurs, s=60, zorder=3, edgecolor="white")
    if noms_points is not None:
        for (a, b), nom in zip(X, noms_points):
            ax.annotate(nom, (a, b), xytext=(5, 5), textcoords="offset points", fontsize=10, weight="bold")
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    ax.set_title(titre, color=MARINE, weight="bold"); ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]")
    return fig


def tracer_carte_scores(modele: IsolationForest, X: np.ndarray, y: Optional[np.ndarray] = None,
                        resolution: int = 120, titre: str = "Carte du score d'anomalie s(x)", ax=None) -> Figure:
    """Surface de décision : score en tout point du plan (modèle entraîné sur des données 2D)."""
    if X.shape[1] != 2:
        raise ValueError("tracer_carte_scores ne gère que les données 2D")
    fig, ax = (plt.subplots(figsize=(6.5, 5)) if ax is None else (ax.figure, ax))
    x0, x1, y0, y1 = _marges(X, 0.25)
    gx, gy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    grille = np.c_[gx.ravel(), gy.ravel()]
    Z = modele.score_samples(grille).reshape(gx.shape)
    cs = ax.contourf(gx, gy, Z, levels=20, cmap="RdYlBu_r", alpha=0.85)
    ax.contour(gx, gy, Z, levels=[modele.seuil_], colors=MARINE, linewidths=1.5)
    fig.colorbar(cs, ax=ax, label="score s(x)")
    couleurs = np.where(y == 1, ROUGE, MARINE) if y is not None else MARINE
    ax.scatter(X[:, 0], X[:, 1], c=couleurs, s=18, edgecolor="white", linewidth=0.4)
    ax.set_title(titre, color=MARINE, weight="bold"); ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]")
    return fig


def projection_pca(X: np.ndarray, k: int = 2) -> np.ndarray:
    """Projection sur les k premières composantes principales (SVD, numpy seul)."""
    centre = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(centre, full_matrices=False)
    return centre @ vt[:k].T


def tracer_projection_scores(X: np.ndarray, scores: np.ndarray, y: Optional[np.ndarray] = None,
                             titre: str = "Projection PCA colorée par le score", ax=None) -> Figure:
    """Nuage de points en 2D (PCA) coloré par score ; les vraies anomalies sont cerclées de rouge."""
    fig, ax = (plt.subplots(figsize=(6, 5)) if ax is None else (ax.figure, ax))
    P = projection_pca(X)
    sc = ax.scatter(P[:, 0], P[:, 1], c=scores, cmap="RdYlBu_r", s=28, edgecolor="none")
    if y is not None:
        ax.scatter(P[y == 1, 0], P[y == 1, 1], facecolors="none", edgecolors=ROUGE, s=90, linewidth=1.3,
                   label="anomalies réelles")
        ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, label="score s(x)")
    ax.set_title(titre, color=MARINE, weight="bold"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    return fig


def tracer_histogramme_scores(scores: np.ndarray, y: Optional[np.ndarray] = None, seuil: Optional[float] = None,
                              titre: str = "Distribution des scores", ax=None) -> Figure:
    fig, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax))
    bins = np.linspace(min(scores.min(), 0.3), max(scores.max(), 0.8), 30)
    if y is None:
        ax.hist(scores, bins=bins, color=TEAL)
    else:
        ax.hist(scores[y == 0], bins=bins, color=TEAL, alpha=0.85, label="normaux")
        ax.hist(scores[y == 1], bins=bins, color=ROUGE, alpha=0.85, label="anomalies")
        ax.legend()
    if seuil is not None:
        ax.axvline(seuil, color=MARINE, ls="--", label=f"seuil = {seuil:.2f}")
        ax.legend()
    ax.set_title(titre, color=MARINE, weight="bold"); ax.set_xlabel("score s(x)"); ax.set_ylabel("effectif")
    return fig


def tracer_scores_tries(scores: np.ndarray, seuil: Optional[float] = None,
                        titre: str = "Scores classés du plus anormal au plus normal", ax=None) -> Figure:
    """Courbe des scores triés : les premiers points (à gauche) sont les plus isolables."""
    fig, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax))
    tries = np.sort(scores)[::-1]
    ax.plot(np.arange(1, len(tries) + 1), tries, color=TEAL, lw=2)
    if seuil is not None:
        ax.axhline(seuil, color=MARINE, ls="--", label=f"seuil = {seuil:.2f}")
        ax.legend()
    ax.axhline(0.5, color=GRIS, ls=":", lw=1)
    ax.set_xlabel("rang du point"); ax.set_ylabel("score s(x)")
    ax.set_title(titre, color=MARINE, weight="bold"); ax.grid(alpha=0.3)
    return fig


def tracer_chemins(modele: IsolationForest, X: np.ndarray, y: Optional[np.ndarray] = None,
                   titre: str = "Longueur de chemin moyenne E(h(x))", ax=None) -> Figure:
    """Histogramme de E(h(x)) : les anomalies sont à gauche (chemin court).

    Si ``y`` est fourni (jeux de démonstration seulement), on distingue normaux et anomalies."""
    fig, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax))
    eh = modele.chemin_moyen(X)
    bins = np.linspace(eh.min(), eh.max(), 30)
    if y is None:
        ax.hist(eh, bins=bins, color=TEAL)
    else:
        ax.hist(eh[y == 0], bins=bins, color=TEAL, alpha=0.85, label="normaux")
        ax.hist(eh[y == 1], bins=bins, color=ROUGE, alpha=0.85, label="anomalies")
        ax.legend()
    ax.set_xlabel("E(h(x))"); ax.set_ylabel("effectif")
    ax.set_title(titre, color=MARINE, weight="bold")
    return fig


def tracer_tableau_de_bord(modele: IsolationForest, X: np.ndarray, y: Optional[np.ndarray] = None,
                           titre: str = "") -> Figure:
    """Figure de synthèse 2x2 : projection PCA, histogramme, scores triés, longueurs de chemin."""
    scores = modele.score_samples(X)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    tracer_projection_scores(X, scores, y, ax=axes[0, 0])
    tracer_histogramme_scores(scores, y, modele.seuil_, ax=axes[0, 1])
    tracer_scores_tries(scores, modele.seuil_, ax=axes[1, 0])
    tracer_chemins(modele, X, y, ax=axes[1, 1])
    fig.suptitle(titre or "Isolation Forest : résultats", color=MARINE, weight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def tracer_courbe_parametre(valeurs: Sequence, moyennes: np.ndarray, ecarts: np.ndarray, etiquette: str,
                            titre: str, ax=None, ylabel: str = "stabilité (corrélation de Spearman)") -> Figure:
    """Valeur moyenne (± écart-type) d'une mesure en fonction d'un hyperparamètre (T ou psi)."""
    fig, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax))
    ax.plot(valeurs, moyennes, "o-", color=TEAL, lw=2)
    ax.fill_between(valeurs, moyennes - ecarts, moyennes + ecarts, color=TEAL, alpha=0.2)
    ax.set_xlabel(etiquette); ax.set_ylabel(ylabel); ax.set_title(titre, color=MARINE, weight="bold")
    ax.grid(alpha=0.3)
    return fig
