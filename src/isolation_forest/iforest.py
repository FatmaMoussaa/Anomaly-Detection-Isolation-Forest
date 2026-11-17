"""Module 3 + 4 : la forêt d'isolation et le score d'anomalie (diapos 9, 10, 14, 16).

Utilisation :
    modele = IsolationForest(n_arbres=100, psi=256, random_state=42).fit(X)
    scores = modele.score_samples(X)      # s(x, psi) dans ]0, 1]
    etiquettes = modele.predict(X)        # 1 = anomalie, 0 = normal
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .itree import NoeudArbre, compter_noeuds, construire_itree, hauteur, longueur_chemin
from .sampling import sous_echantillonnage
from .utils import c


class IsolationForest:
    """Forêt de T arbres d'isolation (Liu, Ting & Zhou, 2008).

    Paramètres
    ----------
    n_arbres : T, nombre d'arbres (100 dans l'article).
    psi : taille du sous-échantillon par arbre (256 dans l'article).
    contamination : proportion attendue d'anomalies dans (0, 0.5]. Si fournie,
        le seuil de décision est calculé comme le quantile (1 - contamination)
        des scores d'entraînement.
    seuil : seuil fixe utilisé si ``contamination`` est None (0,6 d'après la diapo 16).
    random_state : graine pour la reproductibilité.

    Attributs appris (suffixe ``_`` comme en scikit-learn)
    -----------------------------------------------------
    arbres_ : liste des T racines d'iTrees ;
    psi_effectif_ : min(psi, n) ;
    limite_hauteur_ : l = ceil(log2(psi_effectif_)) ;
    facteur_c_ : c(psi_effectif_), dénominateur du score ;
    seuil_ : seuil de décision réellement utilisé ;
    n_features_ : nombre d'attributs d vus à l'entraînement.
    """

    def __init__(self, n_arbres: int = 100, psi: int = 256, contamination: Optional[float] = None,
                 seuil: float = 0.6, random_state: Optional[int] = None) -> None:
        if not isinstance(n_arbres, (int, np.integer)) or n_arbres < 1:
            raise ValueError("n_arbres doit être un entier >= 1")
        if not isinstance(psi, (int, np.integer)) or psi < 2:
            raise ValueError("psi doit être un entier >= 2")
        if contamination is not None and not (0.0 < contamination <= 0.5):
            raise ValueError("contamination doit être dans ]0, 0.5]")
        if not (0.0 < seuil < 1.0):
            raise ValueError("seuil doit être dans ]0, 1[")
        self.n_arbres = int(n_arbres)
        self.psi = int(psi)
        self.contamination = contamination
        self.seuil = seuil
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    #  Entraînement (Construire_iForest, diapo 14)
    # ------------------------------------------------------------------ #
    def fit(self, X) -> "IsolationForest":
        """Construit la forêt. Aucune étiquette n'est utilisée (non supervisé)."""
        X = self._valider(X)
        n, d = X.shape
        if n < 2:
            raise ValueError("Il faut au moins 2 observations pour entraîner la forêt")

        self._rng = np.random.default_rng(self.random_state)
        self.psi_effectif_ = min(self.psi, n)
        self.limite_hauteur_ = math.ceil(math.log2(self.psi_effectif_))   # l = ceil(log2 psi)
        self.facteur_c_ = c(self.psi_effectif_)
        self.n_features_ = d

        self.arbres_: List[NoeudArbre] = []
        for _ in range(self.n_arbres):                                    # Pour i de 1 à T
            X_sub = sous_echantillonnage(X, self.psi_effectif_, self._rng)    # module 1
            arbre = construire_itree(X_sub, 0, self.limite_hauteur_, self._rng)  # module 2
            self.arbres_.append(arbre)

        if self.contamination is None:
            self.seuil_ = self.seuil
        else:
            self.seuil_ = float(np.quantile(self.score_samples(X), 1.0 - self.contamination))
        return self

    # ------------------------------------------------------------------ #
    #  Scoring (Score_Anomalie, diapo 16)
    # ------------------------------------------------------------------ #
    def chemins_par_arbre(self, X) -> np.ndarray:
        """Matrice (n, T) des longueurs de chemin h_i(x) pour chaque arbre."""
        self._verifier_entraine()
        X = self._valider(X, entrainement=False)
        H = np.empty((X.shape[0], self.n_arbres), dtype=float)
        for j, arbre in enumerate(self.arbres_):
            for i in range(X.shape[0]):
                H[i, j] = longueur_chemin(X[i], arbre, 0)
        return H

    def chemin_moyen(self, X) -> np.ndarray:
        """E(h(x)) : moyenne des longueurs de chemin sur les T arbres."""
        return self.chemins_par_arbre(X).mean(axis=1)

    def score_samples(self, X) -> np.ndarray:
        """Score d'anomalie s(x, psi) = 2^(-E(h(x)) / c(psi)), dans ]0, 1].

        s proche de 1 : anomalie probable ; s < 0,5 : plutôt normal.
        """
        return np.power(2.0, -self.chemin_moyen(X) / self.facteur_c_)

    def predict(self, X) -> np.ndarray:
        """1 si score >= seuil_ (anomalie), 0 sinon (normal)."""
        return (self.score_samples(X) >= self.seuil_).astype(int)

    def fit_predict(self, X) -> np.ndarray:
        """fit puis predict sur le même jeu de données."""
        return self.fit(X).predict(X)

    # ------------------------------------------------------------------ #
    #  Inspection
    # ------------------------------------------------------------------ #
    def resume(self) -> Dict[str, float]:
        """Statistiques sur la forêt apprise (hauteurs, tailles des arbres)."""
        self._verifier_entraine()
        hauteurs = [hauteur(a) for a in self.arbres_]
        internes = [compter_noeuds(a)[0] for a in self.arbres_]
        feuilles = [compter_noeuds(a)[1] for a in self.arbres_]
        return {
            "n_arbres": self.n_arbres,
            "psi_effectif": self.psi_effectif_,
            "limite_hauteur": self.limite_hauteur_,
            "c(psi)": round(self.facteur_c_, 4),
            "hauteur_max_observee": max(hauteurs),
            "hauteur_moyenne": round(float(np.mean(hauteurs)), 2),
            "noeuds_internes_moyens": round(float(np.mean(internes)), 1),
            "feuilles_moyennes": round(float(np.mean(feuilles)), 1),
            "seuil": round(self.seuil_, 4),
        }

    def get_params(self) -> Dict[str, object]:
        return {"n_arbres": self.n_arbres, "psi": self.psi, "contamination": self.contamination,
                "seuil": self.seuil, "random_state": self.random_state}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"IsolationForest({args})"

    # ------------------------------------------------------------------ #
    #  Utilitaires internes
    # ------------------------------------------------------------------ #
    def _verifier_entraine(self) -> None:
        if not hasattr(self, "arbres_"):
            raise RuntimeError("Le modèle n'est pas entraîné : appelez fit(X) d'abord")

    def _valider(self, X, entrainement: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X doit être un tableau 2D (n, d), reçu {X.ndim}D")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contient des NaN ou des valeurs infinies")
        if not entrainement and X.shape[1] != self.n_features_:
            raise ValueError(f"X a {X.shape[1]} attributs, le modèle en attend {self.n_features_}")
        return X
