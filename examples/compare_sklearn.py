"""« Implémentation rapide » avec scikit-learn + comparaison avec notre implémentation.

    python examples/compare_sklearn.py

1) Implémentation rapide : 3 lignes avec ``sklearn.ensemble.IsolationForest``.
2) Comme les jeux du prof n'ont pas d'étiquettes (pas d'AUC possible), on vérifie que notre forêt et celle de
   scikit-learn classent les points de la même façon : corrélation de Spearman des scores et recouvrement
   des 10 points les plus anormaux.
"""
import matplotlib
matplotlib.use("Agg")

import os

import numpy as np
from sklearn.ensemble import IsolationForest as IFSklearn

import _chemins
from isolation_forest import IsolationForest, correlation_rangs, recouvrement_top_k
from isolation_forest import viz
from isolation_forest.data import CHARGEURS


def main():
    print("=== Implémentation rapide (scikit-learn) sur iris ===")
    X = CHARGEURS["iris"]()
    rapide = IFSklearn(n_estimators=100, max_samples=min(256, len(X)), random_state=42).fit(X)
    s_rapide = -rapide.score_samples(X)             # scikit-learn renvoie l'OPPOSÉ du score s(x) de l'article
    print("score le plus élevé :", round(float(s_rapide.max()), 3), "| ligne", int(s_rapide.argmax()))
    print("anomalies (predict = -1) :", int((rapide.predict(X) == -1).sum()), "sur", len(X), "(contamination='auto')")

    print("\n=== Notre implémentation contre scikit-learn (T = 100, psi = 256, 5 graines) ===")
    print(f"{'jeu':>6} | {'n x d':>9} | {'Spearman moyen':>15} | {'top-10 communs':>15}")
    for nom, charger in CHARGEURS.items():
        try:
            X = charger()
        except FileNotFoundError as err:
            print(f"{nom:>6} | ignoré : {err}")
            continue
        psi = min(256, len(X))
        rho, top = [], []
        for g in range(5):
            ours = IsolationForest(100, psi, random_state=g).fit(X).score_samples(X)
            ref = -IFSklearn(n_estimators=100, max_samples=psi, random_state=g).fit(X).score_samples(X)
            rho.append(correlation_rangs(ours, ref))
            top.append(recouvrement_top_k(ours, ref, 10))
        print(f"{nom:>6} | {X.shape[0]:>4}x{X.shape[1]:<4}| {np.mean(rho):>15.3f} | {np.mean(top):>14.0%}")
        if nom == "iris":                           # nuage score-contre-score pour la figure du rapport
            fig, ax = viz.plt.subplots(figsize=(5.5, 5))
            ax.scatter(ours, ref, s=18, color=viz.TEAL, edgecolor="white", linewidth=0.3)
            ax.set_xlabel("score : notre implémentation"); ax.set_ylabel("score : scikit-learn")
            ax.set_title(f"iris, graine 4 : Spearman = {rho[-1]:.3f}", color=viz.MARINE, weight="bold")
            ax.grid(alpha=0.3)
            viz.sauver(fig, os.path.join(_chemins.FIGURES, "comparaison_sklearn_iris.png"))


if __name__ == "__main__":
    main()
