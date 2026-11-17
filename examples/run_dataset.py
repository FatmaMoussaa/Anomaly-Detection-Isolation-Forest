"""Lance Isolation Forest sur un jeu de données du prof et produit le tableau de bord.

    python examples/run_dataset.py iris
    python examples/run_dataset.py glass
    python examples/run_dataset.py wdbc --arbres 100 --psi 256 --seed 1 --contamination 0.05
    python examples/run_dataset.py gauss          # démo 2D synthétique (étiquettes connues par construction)

Les jeux du prof n'ont PAS d'étiquettes ici : on classe les points du plus anormal au moins anormal.
Le seuil de décision est choisi par l'utilisateur : par défaut, les 5 % de points les plus isolés (--contamination).
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import time

import numpy as np

import _chemins
from isolation_forest import IsolationForest
from isolation_forest import viz
from isolation_forest.data import CHARGEURS, generer_gaussiennes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["iris", "glass", "wdbc", "gauss"])
    ap.add_argument("--arbres", type=int, default=100, help="T, nombre d'arbres")
    ap.add_argument("--psi", type=int, default=256, help="taille du sous-échantillon")
    ap.add_argument("--contamination", type=float, default=0.05, help="part de points déclarés anomalies")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "gauss":
        X, y = generer_gaussiennes(600, 2, 0.05, args.seed)       # seul jeu avec étiquettes (synthétique)
    else:
        X, y = CHARGEURS[args.dataset](), None
    print(f"=== {args.dataset} : {X.shape[0]} observations, {X.shape[1]} attributs ===")

    modele = IsolationForest(args.arbres, args.psi, contamination=args.contamination, random_state=args.seed)
    t0 = time.perf_counter()
    modele.fit(X)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = modele.score_samples(X)
    t_score = time.perf_counter() - t0
    pred = modele.predict(X)

    print("modèle :", modele)
    print("forêt  :", modele.resume())
    print(f"temps  : fit {t_fit:.2f} s, scoring {t_score:.2f} s")
    print(f"scores : min {scores.min():.3f} | médiane {np.median(scores):.3f} | max {scores.max():.3f}")
    print(f"seuil  : {modele.seuil_:.3f} (quantile {100 * (1 - args.contamination):.0f} %) "
          f"-> {int(pred.sum())} points déclarés anomalies sur {len(X)}")
    print("10 points les plus anormaux (indice de ligne, score, E(h)) :")
    eh = modele.chemin_moyen(X)
    for i in np.argsort(-scores)[:10]:
        print(f"   ligne {i:>4}  s = {scores[i]:.3f}  E(h) = {eh[i]:.2f}")

    sortie = os.path.join(_chemins.FIGURES, f"{args.dataset}_tableau_de_bord.png")
    viz.sauver(viz.tracer_tableau_de_bord(modele, X, y, f"Isolation Forest sur {args.dataset}"), sortie)
    print("figure :", os.path.relpath(sortie, _chemins.RACINE).replace(os.sep, "/"))   # chemin relatif : portable

    if args.dataset == "gauss":                      # données 2D : carte des scores + coupes
        viz.sauver(viz.tracer_carte_scores(modele, X, y), os.path.join(_chemins.FIGURES, "gauss_carte_scores.png"))
        viz.sauver(viz.tracer_decoupes(modele.arbres_[0], X, y, titre="Coupes du premier arbre"),
                   os.path.join(_chemins.FIGURES, "gauss_decoupes.png"))


if __name__ == "__main__":
    main()
