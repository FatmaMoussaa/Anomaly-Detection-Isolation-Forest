"""Hand-tracing sur le jeu jouet : 5 points normaux (A à E) + une anomalie F(10, 10).

Ce script produit exactement les chiffres à montrer pendant la démonstration :
hyperparamètres, coupes des premiers arbres, chemin de F et d'un point normal,
longueurs moyennes et scores finaux.

    python examples/demo_toy.py
"""
import matplotlib
matplotlib.use("Agg")

import math
import os

import matplotlib.pyplot as plt
import numpy as np

import _chemins
from isolation_forest import (IsolationForest, arbre_en_texte, c, construire_itree, hauteur, longueur_chemin,
                              trace_chemin)
from isolation_forest import viz
from isolation_forest.data import jeu_jouet

SEED, T, PSI = 42, 100, 6

X, y, noms = jeu_jouet()
print("=== 1. Initialisation : données et hyperparamètres ===")
for nom, (a, b) in zip(noms, X):
    print(f"  {nom} = ({a:g}, {b:g})")
limite = math.ceil(math.log2(PSI))
print(f"  T = {T} arbres, psi = {PSI}, limite l = ceil(log2 {PSI}) = {limite}, c(psi) = {c(PSI):.4f}")

print("\n=== 2. Construction : trois premiers arbres ===")
modele = IsolationForest(n_arbres=T, psi=PSI, random_state=SEED).fit(X)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i in range(3):
    arbre = modele.arbres_[i]
    print(f"\n--- Arbre {i + 1} (hauteur {hauteur(arbre)}) ---")
    print(arbre_en_texte(arbre, noms=["x", "y"]))
    viz.tracer_decoupes(arbre, X, y, noms, titre=f"Arbre {i + 1}", ax=axes[i])
viz.sauver(fig, os.path.join(_chemins.FIGURES, "toy_arbres.png"))

print("=== 3. Chemin d'un point dans l'arbre 1 ===")
for nom, point in (("F (anomalie)", X[5]), ("E (centre du nuage)", X[4])):
    print(f"\n{nom}")
    for ligne in trace_chemin(point, modele.arbres_[0], ["x", "y"]):
        print("  ", ligne)

print("\n=== 4. Longueur moyenne et score ===")
H = modele.chemins_par_arbre(X)
scores = modele.score_samples(X)
print(f"  {'pt':<3} {'h arbre1':>9} {'h arbre2':>9} {'h arbre3':>9} {'E(h)':>8} {'s(x)':>7}  verdict")
for i, nom in enumerate(noms):
    verdict = "ANOMALIE" if scores[i] >= modele.seuil_ else "normal"
    print(f"  {nom:<3} {H[i, 0]:>9.3f} {H[i, 1]:>9.3f} {H[i, 2]:>9.3f} {H[i].mean():>8.3f} {scores[i]:>7.3f}  {verdict}")
print(f"  (s = 2^(-E(h)/c(psi)), c(psi) = {c(PSI):.4f}, seuil = {modele.seuil_})")

# diagramme à barres des scores
fig, ax = plt.subplots(figsize=(6, 4))
couleurs = [viz.ROUGE if v else viz.TEAL for v in y]
ax.bar(noms, scores, color=couleurs)
ax.axhline(modele.seuil_, color=viz.MARINE, ls="--", label=f"seuil {modele.seuil_}")
ax.axhline(0.5, color=viz.GRIS, ls=":", label="0,5")
ax.set_ylabel("score s(x)"); ax.set_title("Scores du jeu jouet", color=viz.MARINE, weight="bold"); ax.legend()
viz.sauver(fig, os.path.join(_chemins.FIGURES, "toy_scores.png"))

# carte de scores
viz.sauver(viz.tracer_carte_scores(modele, X, y), os.path.join(_chemins.FIGURES, "toy_carte_scores.png"))
print("\nFigures enregistrées dans", _chemins.FIGURES)
