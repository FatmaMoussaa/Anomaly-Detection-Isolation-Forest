"""Reverse engineering : comment les structures de données du projet sont réellement codées.

On observe, par introspection et par mesure, les quatre structures de la diapo 11 :
  1. le tableau 2D           -> numpy.ndarray (tampon mémoire, strides, vues / copies) ;
  2. l'arbre binaire         -> objets Noeud / Feuille (__slots__ vs __dict__) ;
  3. la liste (la forêt)     -> list Python (tableau dynamique de références) ;
  4. la pile d'appels        -> frames Python et récursion de construire_itree ;
plus le générateur aléatoire (numpy.random.Generator) et la hiérarchie matplotlib (Figure / Axes).
Enfin (section 7) : comment la bibliothèque scikit-learn code, elle, la même forêt (tableaux parallèles Cython).

    python report/reverse_engineering.py        # écrit aussi report/reverse_engineering_sortie.txt
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

import numpy as np

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RACINE, "src"))

from isolation_forest import IsolationForest, construire_itree, sous_echantillonnage
from isolation_forest.itree import Feuille, Noeud, compter_noeuds, hauteur

LIGNES = []


def sortie(*args):
    texte = " ".join(str(a) for a in args)
    print(texte)
    LIGNES.append(texte)


def titre(texte):
    sortie("\n" + "=" * 78 + f"\n{texte}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
titre("1. LE TABLEAU 2D : numpy.ndarray")
X = np.arange(12, dtype=np.float64).reshape(6, 2).copy()   # .copy() : X possède son tampon (OWNDATA = True)
sortie("type(X)            :", type(X))
sortie("X.dtype / itemsize :", X.dtype, "/", X.itemsize, "octets par élément")
sortie("X.shape / X.ndim   :", X.shape, "/", X.ndim)
sortie("X.strides          :", X.strides, "-> pour passer à la ligne suivante on saute", X.strides[0],
       "octets, à la colonne suivante", X.strides[1], "octets")
sortie("X.nbytes           :", X.nbytes, "octets (= 6 x 2 x 8) dans UN seul tampon contigu")
sortie("flags              : C_CONTIGUOUS =", X.flags["C_CONTIGUOUS"], "| OWNDATA =", X.flags["OWNDATA"])
sortie("adresse du tampon  :", hex(X.__array_interface__["data"][0]))

col = X[:, 1]
sortie("\nColonne X[:, q] (utilisée à chaque coupe) :")
sortie("  strides =", col.strides, "| contiguë =", col.flags["C_CONTIGUOUS"],
       "| est une vue de X =", col.base is X, "| partage la mémoire =", np.shares_memory(col, X))
tranche = X[1:4]
sortie("Tranche X[1:4]       : vue =", tranche.base is X, "| partage la mémoire =", np.shares_memory(tranche, X))
fancy = X[[0, 3, 5]]
sortie("Indexation X[[0,3,5]]: vue =", fancy.base is X, "| partage la mémoire =", np.shares_memory(fancy, X),
       "-> COPIE (indexation avancée)")
masque = X[:, 0] < 6
sortie("Masque booléen X[X[:,0] < 6] : partage la mémoire =", np.shares_memory(X[masque], X), "-> COPIE")

rng = np.random.default_rng(0)
grand = rng.normal(size=(1000, 3))
sub = sous_echantillonnage(grand, 100, rng)
sortie("\nsous_echantillonnage(1000 pts, psi=100) : partage la mémoire avec X =", np.shares_memory(sub, grand),
       "-> copie indépendante, coût O(psi)")
petit = grand[:50]
sortie("sous_echantillonnage(n <= psi) renvoie le MÊME objet que X :", sous_echantillonnage(petit, 100, rng) is petit)

# --------------------------------------------------------------------------- #
titre("2. L'ARBRE BINAIRE : Noeud et Feuille")
n = Noeud(0, 1.5, Feuille(1), Feuille(2))
sortie("MRO de Noeud       :", [c.__name__ for c in Noeud.__mro__])
sortie("MRO de Feuille     :", [c.__name__ for c in Feuille.__mro__])
sortie("Noeud.__slots__    :", Noeud.__slots__, "| a un __dict__ ?", hasattr(n, "__dict__"))
sortie("Feuille.__slots__  :", Feuille.__slots__)
sortie("sys.getsizeof(Noeud)   :", sys.getsizeof(n), "octets (hors enfants)")
sortie("sys.getsizeof(Feuille) :", sys.getsizeof(Feuille(1)), "octets")


class NoeudAvecDict:                       # même contenu, mais sans __slots__
    def __init__(self, a, s, g, d):
        self.attribut, self.seuil, self.gauche, self.droite = a, s, g, d


nd = NoeudAvecDict(0, 1.5, None, None)
taille_dict = sys.getsizeof(nd) + sys.getsizeof(nd.__dict__)
sortie(f"Même nœud SANS __slots__ : objet {sys.getsizeof(nd)} + __dict__ {sys.getsizeof(nd.__dict__)} = "
       f"{taille_dict} octets -> __slots__ économise ~{100 * (1 - sys.getsizeof(n) / taille_dict):.0f} %")

data = np.random.default_rng(1).normal(size=(256, 4))
arbre = construire_itree(data, 0, 8, np.random.default_rng(1))
internes, feuilles = compter_noeuds(arbre)
sortie(f"\nUn arbre construit sur psi=256 points : {internes} nœuds internes + {feuilles} feuilles "
       f"(= {internes + 1} car arbre binaire strict), hauteur {hauteur(arbre)} (limite 8)")
sortie("Empreinte approx. de cet arbre :", internes * sys.getsizeof(n) + feuilles * sys.getsizeof(Feuille(1)),
       "octets (nœuds seulement ; les floats du seuil et les entiers sont des objets Python en plus)")
sortie("L'arbre ne stocke AUCUN point : seulement (q, p) aux nœuds et la taille aux feuilles.")

# --------------------------------------------------------------------------- #
titre("3. LA LISTE (forêt) : list Python = tableau dynamique de références")
L = []
tailles_vues = []
for i in range(40):
    L.append(i)
    t = sys.getsizeof(L)
    if not tailles_vues or t != tailles_vues[-1][1]:
        tailles_vues.append((len(L), t))
sortie("Sur-allocation : (len, getsizeof) à chaque réallocation :", tailles_vues)
sortie("-> la liste réserve de la place d'avance (append en O(1) amorti) ; chaque case = un pointeur de",
       sys.getsizeof([None] * 2) - sys.getsizeof([None]), "octets")
modele = IsolationForest(n_arbres=100, psi=256, random_state=0).fit(data)
sortie("modele.arbres_ :", type(modele.arbres_).__name__, "de", len(modele.arbres_), "racines |",
       "getsizeof =", sys.getsizeof(modele.arbres_), "octets (les arbres eux-mêmes sont ailleurs)")
sortie("Les 100 éléments sont des objets distincts :", len({id(a) for a in modele.arbres_}) == 100)

# --------------------------------------------------------------------------- #
titre("4. LA PILE D'APPELS : récursion de construire_itree")
sortie("sys.getrecursionlimit() =", sys.getrecursionlimit(), "(limite de Python)")
profondeur = {"cour": 0, "max": 0}


def profileur(frame, evenement, arg):
    if frame.f_code.co_name == "construire_itree":
        if evenement == "call":
            profondeur["cour"] += 1
            profondeur["max"] = max(profondeur["max"], profondeur["cour"])
        elif evenement == "return":
            profondeur["cour"] -= 1


for psi in (8, 64, 256, 1024):
    profondeur.update(cour=0, max=0)
    limite = int(np.ceil(np.log2(psi)))
    jeu = np.random.default_rng(2).normal(size=(psi, 3))
    sys.setprofile(profileur)
    construire_itree(jeu, 0, limite, np.random.default_rng(2))
    sys.setprofile(None)
    sortie(f"  psi = {psi:>4} : limite l = ceil(log2 psi) = {limite:>2} -> profondeur max de la pile "
           f"= {profondeur['max']:>2} (= l + 1) <= recursionlimit")
sortie("La limite l = ceil(log2 psi) borne la pile à l + 1 appels : très loin de la limite de récursion de Python, "
       "même pour un grand psi. Sans elle, un arbre déséquilibré pourrait être bien plus profond.")

# --------------------------------------------------------------------------- #
titre("5. LE GÉNÉRATEUR ALÉATOIRE : numpy.random.Generator")
g = np.random.default_rng(42)
sortie("type(rng)                 :", type(g).__name__)
sortie("rng.bit_generator         :", type(g.bit_generator).__name__, "(PCG64 : 128 bits d'état)")
etat = g.bit_generator.state
sortie("état initial (clés)       :", list(etat.keys()), "| état.state =", {k: str(v)[:18] + "..." for k, v in etat["state"].items()})
a = g.integers(1000, size=3)
g.bit_generator.state = etat
b = g.integers(1000, size=3)
sortie("Restaurer l'état rejoue la même séquence :", a.tolist(), "==", b.tolist(), "->", bool((a == b).all()))
sortie("C'est ce qui rend random_state reproductible : un seul rng est partagé par le sous-échantillonnage et la construction.")

# --------------------------------------------------------------------------- #
titre("6. MATPLOTLIB : la hiérarchie Figure -> Axes -> Artists")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sortie("MRO de Figure :", [c.__name__ for c in Figure.__mro__][:5], "...")
sortie("MRO de Axes   :", [c.__name__ for c in Axes.__mro__][:5], "...")
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])
ax.plot([0, 1], [0, 1], "--")
sortie("fig.axes                 :", [type(a).__name__ for a in fig.axes])
sortie("ax.collections / ax.lines:", [type(c).__name__ for c in ax.collections], "/", [type(l).__name__ for l in ax.lines])
sortie("ax.get_children() (types):", sorted({type(c).__name__ for c in ax.get_children()}))
plt.close(fig)

# --------------------------------------------------------------------------- #
titre("7. SCIKIT-LEARN : comment l'implémentation « rapide » code la même forêt")
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest as IFSklearn

Xi = load_iris().data
sk = IFSklearn(n_estimators=100, max_samples=150, random_state=0).fit(Xi)
sortie("MRO de IsolationForest (sklearn) :", [c.__name__ for c in type(sk).__mro__][:5], "...")
sortie("sk.estimators_                   :", type(sk.estimators_).__name__, "de", len(sk.estimators_), "objets",
       type(sk.estimators_[0]).__name__, "(la forêt est aussi une liste Python d'arbres)")
sortie("MRO d'un arbre                   :", [c.__name__ for c in type(sk.estimators_[0]).__mro__][:4], "...")
sortie("hyper-paramètres de chaque arbre : splitter =", repr(sk.estimators_[0].splitter),
       "| max_features =", sk.estimators_[0].max_features, "| max_depth =", sk.estimators_[0].max_depth,
       "(= ceil(log2(150)) = 8)")
sortie("sk.max_samples_ / sk.offset_     :", sk.max_samples_, "/", sk.offset_,
       "(offset_ = -0,5 : seuil du predict quand contamination='auto')")
sortie("sk.estimators_samples_[0]        :", type(sk.estimators_samples_[0]).__name__,
       "d'INDICES de lignes (pas une copie des données) ; 5 premiers =", sk.estimators_samples_[0][:5].tolist())

arbre = sk.estimators_[0].tree_
sortie("\ntype(tree_)                    :", type(arbre), "-> classe écrite en Cython (sklearn/tree/_tree.pyx), pas en Python")
sortie("tree_.node_count / n_leaves / max_depth :", arbre.node_count, "/", arbre.n_leaves, "/", arbre.max_depth,
       "(un arbre binaire strict : feuilles = noeuds internes + 1 ->", arbre.node_count - arbre.n_leaves, "+ 1 =", arbre.n_leaves, ")")
sortie("Les noeuds sont stockés dans des TABLEAUX PARALLELES numpy (pas d'objet Noeud) :")
for nom in ("children_left", "children_right", "feature", "threshold", "n_node_samples"):
    a = getattr(arbre, nom)
    sortie(f"  tree_.{nom:<15}: dtype={a.dtype!s:<8} shape={a.shape} 7 premières valeurs = {np.round(a[:7], 3).tolist()}")
etat = arbre.__getstate__()
sortie("En réalité un SEUL tampon : tree_.__getstate__()['nodes'] est un tableau structuré numpy :")
sortie("  dtype.names =", etat["nodes"].dtype.names)
sortie(f"  itemsize = {etat['nodes'].dtype.itemsize} octets par noeud x {arbre.node_count} noeuds = {etat['nodes'].nbytes} octets"
       f" (les colonnes children_left, threshold... sont des vues de ce tampon)")
sortie("Convention : un noeud i est une feuille quand children_left[i] == -1 (constante TREE_LEAF).")
sortie("Noeuds internes :", int((arbre.children_left != -1).sum()), "| feuilles :", int((arbre.children_left == -1).sum()))
sortie("Le noeud 0 est la racine ; ses enfants sont children_left[0] =", int(arbre.children_left[0]),
       "et children_right[0] =", int(arbre.children_right[0]), "(numérotation en profondeur d'abord)")


def longueur_chemin_sklearn(x, t):
    """Parcours à la main des tableaux parallèles : même résultat que decision_path de scikit-learn."""
    i, h = 0, 0
    while t.children_left[i] != -1:
        i = t.children_left[i] if x[t.feature[i]] <= t.threshold[i] else t.children_right[i]
        h += 1
    return h, int(t.n_node_samples[i])


x0 = Xi[0]
h, taille = longueur_chemin_sklearn(x0, arbre)
chemin_sk = sk.estimators_[0].decision_path(x0.reshape(1, -1))
sortie("\nParcours pour la ligne 0 : profondeur", h, "| taille de la feuille", taille, "(= n_node_samples de la feuille)")
sortie("  decision_path de sklearn : matrice", type(chemin_sk).__name__, "creuse CSR ; nombre de noeuds traversés =",
       chemin_sk.indices.size, "= profondeur + 1 ->", chemin_sk.indices.size == h + 1)
sortie("scikit-learn ajoute ensuite la correction c(n_node_samples) de la feuille (fonction _average_path_length) :",
       "c'est la même formule que notre utils.c.")
sortie("Comparaison des choix de structure : nous = objets Python (Noeud / Feuille, __slots__) lisibles mais lents ;"
       " scikit-learn = tableaux contigus en C, rapides et compacts mais illisibles.")
sortie("  Côté scikit-learn un noeud = 64 octets dans un tampon unique ; côté Python chaque Noeud est un objet de 64 octets"
       " qui pointe en plus vers un float et un int (objets séparés).")

chemin = os.path.join(os.path.dirname(__file__), "reverse_engineering_sortie.txt")
with open(chemin, "w", encoding="utf-8") as f:
    f.write("\n".join(LIGNES) + "\n")
print("\n[sortie enregistrée dans", chemin, "]")
