"""Génère le diagramme de classes UML (PNG) et le même diagramme au format Mermaid.

    python report/diagramme_classes.py

Sorties : report/figures/diagramme_classes.png, report/figures/diagramme_modules.png,
          report/diagramme_classes.mmd (à coller sur https://mermaid.live ou dans un README GitHub).
"""
import matplotlib
matplotlib.use("Agg")

import inspect
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RACINE, "src"))

from isolation_forest import iforest, itree, sampling, utils, viz, data, comparaison  # noqa: E402

MARINE, TEAL, ROUGE, AMBRE, GRIS = "#13213f", "#1f7a8c", "#d62839", "#e8a33d", "#8a94a6"

# Attributs (UML) : on ne peut pas les deviner par introspection pour une instance non créée.
ATTRIBUTS = {
    "IsolationForest": ["+ n_arbres : int  (T)", "+ psi : int", "+ contamination : float | None", "+ seuil : float",
                        "+ random_state : int | None", "+ arbres_ : List[NoeudArbre]", "+ psi_effectif_ : int",
                        "+ limite_hauteur_ : int", "+ facteur_c_ : float", "+ seuil_ : float", "+ n_features_ : int",
                        "- _rng : Generator"],
    "NoeudArbre": [],
    "Noeud": ["+ attribut : int  (q)", "+ seuil : float  (p)", "+ gauche : NoeudArbre", "+ droite : NoeudArbre"],
    "Feuille": ["+ taille : int"],
}
CLASSES = {"IsolationForest": iforest.IsolationForest, "NoeudArbre": itree.NoeudArbre,
           "Noeud": itree.Noeud, "Feuille": itree.Feuille}


def methodes(cls):
    """Méthodes définies dans la classe (introspection), sous forme UML."""
    lignes = []
    for nom, f in cls.__dict__.items():
        if not inspect.isfunction(f) or (nom.startswith("_") and nom != "__init__"):
            continue
        sig = inspect.signature(f)
        params = ", ".join(p.name for p in sig.parameters.values() if p.name != "self")
        retour = "" if sig.return_annotation is inspect.Signature.empty else f" : {sig.return_annotation}".replace("'", "")
        lignes.append(f"+ {nom}({params}){retour}")
    return lignes


def boite(ax, x, y, titre, sections, couleur=TEAL, stereotype=None, largeur=None):
    """Dessine une boîte UML dont le coin haut-gauche est (x, y). Retourne (x, y, largeur, hauteur)."""
    h_ligne = 1.55
    toutes = [titre, stereotype or ""] + [l for s in sections for l in s]
    largeur = largeur or max(len(l) for l in toutes) * 0.62 + 3
    h_titre = 3.6 if stereotype else 2.6
    hauteurs = [max(len(s), 1) * h_ligne + 1.0 for s in sections]
    haut = h_titre + sum(hauteurs)
    ax.add_patch(Rectangle((x, y - haut), largeur, haut, fc="white", ec=MARINE, lw=1.6, zorder=2))
    ax.add_patch(Rectangle((x, y - h_titre), largeur, h_titre, fc=couleur, ec=MARINE, lw=1.6, zorder=3))
    if stereotype:
        ax.text(x + largeur / 2, y - 0.9, stereotype, ha="center", va="center", color="white", fontsize=8, style="italic", zorder=4)
    ax.text(x + largeur / 2, y - h_titre + 1.3, titre, ha="center", va="center", color="white", fontsize=11,
            weight="bold", zorder=4)
    cur = y - h_titre
    for s, hs in zip(sections, hauteurs):
        ax.plot([x, x + largeur], [cur, cur], color=MARINE, lw=1.2, zorder=3)
        for i, ligne in enumerate(s):
            ax.text(x + 1, cur - 1.1 - i * h_ligne, ligne, ha="left", va="center", family="monospace", fontsize=7.6,
                    color=MARINE, zorder=4)
        cur -= hs
    return x, y, largeur, haut


def fleche(ax, p1, p2, style="->", ls="-", couleur=MARINE, texte=None, decal=(0, 0.8), rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, lw=1.4, color=couleur, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))
    if texte:
        ax.text((p1[0] + p2[0]) / 2 + decal[0], (p1[1] + p2[1]) / 2 + decal[1], texte, fontsize=8, ha="center",
                color=couleur, zorder=6, bbox=dict(fc="white", ec="none", pad=0.6))


def diagramme_classes(chemin):
    fig, ax = plt.subplots(figsize=(17, 8.5))
    ax.set_xlim(0, 122); ax.set_ylim(17, 68); ax.axis("off")
    ax.set_title("Diagramme de classes : Isolation Forest", color=MARINE, weight="bold", fontsize=15, pad=8)

    x0, y0, w0, h0 = boite(ax, 2, 66, "IsolationForest", [ATTRIBUTS["IsolationForest"], methodes(CLASSES["IsolationForest"])])
    xb, yb, wb, hb = boite(ax, 64, 66, "NoeudArbre", [ATTRIBUTS["NoeudArbre"], methodes(CLASSES["NoeudArbre"])],
                           couleur=GRIS, stereotype="<<abstraite : nœud d'arbre binaire>>", largeur=40)
    xn, yn, wn, hn = boite(ax, 56, 42, "Noeud", [ATTRIBUTS["Noeud"], methodes(CLASSES["Noeud"])], couleur=ROUGE,
                           stereotype="<<nœud interne>>")
    xf, yf, wf, hf = boite(ax, 97, 42, "Feuille", [ATTRIBUTS["Feuille"], methodes(CLASSES["Feuille"])], couleur=TEAL,
                           stereotype="<<feuille>>")

    # héritage : triangle creux sous NoeudArbre
    for (xc, yc, wc, dx) in ((xn, yn, wn, -8), (xf, yf, wf, 8)):
        p1 = (xc + wc / 2, yc)
        p2 = (xb + wb / 2 + dx, yb - hb - 1.9)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=MARINE, lw=1.4, zorder=1)
        ax.add_patch(Polygon([(p2[0], p2[1] + 1.9), (p2[0] - 1.1, p2[1]), (p2[0] + 1.1, p2[1])],
                             closed=True, fc="white", ec=MARINE, lw=1.4, zorder=6))
    # association IsolationForest -> NoeudArbre (T racines)
    fleche(ax, (x0 + w0, y0 - 6), (xb, yb - 3.5), style="-|>", texte="arbres_  (racines)\n1 ──► T", decal=(0, 1.8))
    # association Noeud -> NoeudArbre (gauche, droite)
    fleche(ax, (xn, yn - 5), (xb + 2, yb - hb), style="-|>", texte="gauche / droite\n(2 enfants)", decal=(-3.5, -2.5), rad=-0.5)

    ax.text(2, 22.5, "Dépendances (fonctions des modules) :  IsolationForest ⟶ sampling.sous_echantillonnage, "
            "itree.construire_itree, itree.longueur_chemin, utils.c", fontsize=9, color=MARINE, style="italic")
    ax.text(2, 19.5, "Légende :  ▷ héritage    ──► association (contient)    + public    - privé    "
            "un attribut suffixé par _ est appris par fit()", fontsize=9, color=GRIS)
    fig.savefig(chemin, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagramme_modules(chemin):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 125); ax.set_ylim(8, 68); ax.axis("off")
    ax.set_title("Modules du projet et leurs fonctions", color=MARINE, weight="bold", fontsize=15, pad=8)
    mods = {"utils": (utils, GRIS, "c(n), H(n)", (2, 64)), "iforest": (iforest, TEAL, "Modules 3 et 4", (38, 64)),
            "sampling": (sampling, TEAL, "Module 1", (72, 64)), "itree": (itree, ROUGE, "Module 2 + chemin", (34, 46)),
            "viz": (viz, AMBRE, "visualisation", (84, 46)),
            "comparaison": (comparaison, GRIS, "comparer deux classements", (2, 42)),
            "data": (data, GRIS, "jeux de données", (2, 28))}
    b = {}
    for nom, (mod, couleur, role, (x, y)) in mods.items():
        def ok(o):
            return inspect.isfunction(o) or hasattr(o, "__wrapped__")
        fonctions = [n for n, o in inspect.getmembers(mod, ok)
                     if getattr(o, "__module__", None) == mod.__name__ and not n.startswith("_")]
        classes = [n for n, o in inspect.getmembers(mod, inspect.isclass) if o.__module__ == mod.__name__]
        sections = ([[f"class {n}" for n in classes]] if classes else []) + ([[f"{n}()" for n in fonctions]] if fonctions else [])
        b[nom] = boite(ax, x, y, f"{nom}.py", sections, couleur=couleur, stereotype=f"<<module>>  {role}")

    def bord(nom, cote, frac=0.5):
        x, y, w, h = b[nom]
        return {"g": (x, y - h * frac), "d": (x + w, y - h * frac), "h": (x + w * frac, y), "b": (x + w * frac, y - h)}[cote]

    fleche(ax, bord("iforest", "g", 0.5), bord("utils", "d", 0.5), ls="--", texte="c(psi)", decal=(0, 1.0))
    fleche(ax, bord("iforest", "d", 0.5), bord("sampling", "g", 0.5), ls="--", texte="sous-échantillon", decal=(0, 1.0))
    fleche(ax, bord("iforest", "b", 0.5), bord("itree", "h", 0.5), ls="--", texte="arbres + chemins", decal=(7, 0))
    fleche(ax, bord("itree", "g", 0.1), bord("utils", "b", 0.6), ls="--", texte="c(taille)", decal=(-3, 1.5))
    fleche(ax, bord("viz", "h", 0.2), bord("iforest", "d", 0.9), ls="--", texte="lit modèle et scores", decal=(8, 1.5))
    ax.text(2, 10, "Flèches pointillées : « importe / utilise ».   comparaison et data sont indépendants de l'algorithme.",
            fontsize=9, color=GRIS)
    fig.savefig(chemin, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


MERMAID = '''classDiagram
    class IsolationForest {
        +int n_arbres
        +int psi
        +float contamination
        +float seuil
        +int random_state
        +List~NoeudArbre~ arbres_
        +int psi_effectif_
        +int limite_hauteur_
        +float facteur_c_
        +float seuil_
        +int n_features_
        +fit(X) IsolationForest
        +chemins_par_arbre(X) ndarray
        +chemin_moyen(X) ndarray
        +score_samples(X) ndarray
        +predict(X) ndarray
        +fit_predict(X) ndarray
        +resume() dict
        +get_params() dict
    }
    class NoeudArbre {
        <<abstract>>
        +est_feuille() bool
    }
    class Noeud {
        +int attribut
        +float seuil
        +NoeudArbre gauche
        +NoeudArbre droite
        +est_feuille() bool
    }
    class Feuille {
        +int taille
        +est_feuille() bool
    }
    NoeudArbre <|-- Noeud
    NoeudArbre <|-- Feuille
    Noeud o-- NoeudArbre : gauche, droite
    IsolationForest "1" o-- "T" NoeudArbre : arbres_ (racines)
'''

if __name__ == "__main__":
    dossier = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(dossier, exist_ok=True)
    diagramme_classes(os.path.join(dossier, "diagramme_classes.png"))
    diagramme_modules(os.path.join(dossier, "diagramme_modules.png"))
    with open(os.path.join(os.path.dirname(__file__), "diagramme_classes.mmd"), "w", encoding="utf-8") as f:
        f.write(MERMAID)
    print("diagrammes écrits dans", dossier)
