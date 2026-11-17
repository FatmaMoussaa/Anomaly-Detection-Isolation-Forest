"""Expériences sans étiquettes : stabilité en fonction de T et de psi, et complexité.

    python examples/benchmark.py            # complet
    python examples/benchmark.py --rapide   # version allégée

Stabilité = corrélation de Spearman entre les scores de deux forêts construites avec des graines différentes.
Plus elle est proche de 1, plus le classement des points ne dépend pas du hasard.
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import time

import numpy as np

import _chemins
from isolation_forest import IsolationForest, correlation_rangs
from isolation_forest import viz
from isolation_forest.data import CHARGEURS, generer_gaussiennes


def stabilite(X, paires, **params):
    """Moyenne et écart-type de la corrélation de Spearman entre deux forêts de graines différentes."""
    rho = []
    for g in range(paires):
        a = IsolationForest(random_state=2 * g, **params).fit(X).score_samples(X)
        b = IsolationForest(random_state=2 * g + 1, **params).fit(X).score_samples(X)
        rho.append(correlation_rangs(a, b))
    return float(np.mean(rho)), float(np.std(rho))


def courbe(jeux, valeurs, nom_param, fixes, paires, fichier, titre, log_base=None):
    fig, ax = viz.plt.subplots(figsize=(7, 4.5))
    for nom, X in jeux.items():
        res = np.array([stabilite(X, paires, **{nom_param: v}, **fixes) for v in valeurs])
        print(f"{nom:>6}: " + "  ".join(f"{nom_param}={v}:{m:.3f}±{s:.3f}" for v, (m, s) in zip(valeurs, res)))
        ax.errorbar(valeurs, res[:, 0], yerr=res[:, 1], marker="o", capsize=3, label=nom)
    ax.set_xscale("log", **({"base": log_base} if log_base else {}))
    ax.set_xlabel(nom_param); ax.set_ylabel("stabilité (Spearman entre 2 graines)"); ax.grid(alpha=0.3)
    ax.set_title(titre, color=viz.MARINE, weight="bold"); ax.legend()
    viz.sauver(fig, os.path.join(_chemins.FIGURES, fichier))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapide", action="store_true")
    args = ap.parse_args()
    paires = 2 if args.rapide else 5

    jeux = {}
    for nom, charger in CHARGEURS.items():
        try:
            jeux[nom] = charger()
        except FileNotFoundError:
            print(f"({nom} ignoré : fichier absent de data/)")

    print("=== Stabilité du classement en fonction du nombre d'arbres T (psi = 128) ===")
    courbe(jeux, [1, 5, 10, 25, 50, 100, 200], "n_arbres", {"psi": 128}, paires, "benchmark_T.png",
           "Influence de T sur la stabilité")
    print("\n=== Stabilité en fonction de la taille du sous-échantillon psi (T = 100) ===")
    courbe(jeux, [8, 16, 32, 64, 128, 256], "psi", {"n_arbres": 100}, paires, "benchmark_psi.png",
           "Influence de psi sur la stabilité", log_base=2)

    print("\n=== Complexité : temps de fit en fonction de n (psi = 256, T = 100) ===")
    tailles = [500, 2000, 8000, 32000]
    temps = []
    IsolationForest(100, 256, random_state=0).fit(generer_gaussiennes(500, 5, 0.01, 0)[0])   # échauffement
    for n in tailles:
        X, _ = generer_gaussiennes(n, 5, 0.01, 0)
        t0 = time.perf_counter(); IsolationForest(100, 256, random_state=0).fit(X); temps.append(time.perf_counter() - t0)
        print(f"  n = {n:>6} : fit = {temps[-1]:.3f} s")
    fig, ax = viz.plt.subplots(figsize=(6, 4))
    ax.plot(tailles, temps, "o-", color=viz.TEAL); ax.set_xscale("log"); ax.set_ylim(0, max(temps) * 1.5)
    ax.set_xlabel("n (nombre d'observations)"); ax.set_ylabel("temps de construction (s)"); ax.grid(alpha=0.3)
    ax.set_title("Coût de construction ~ indépendant de n", color=viz.MARINE, weight="bold")
    viz.sauver(fig, os.path.join(_chemins.FIGURES, "benchmark_complexite.png"))


if __name__ == "__main__":
    main()
