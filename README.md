# Isolation Forest — TD2 (Structures de données avancées en Python)

Implémentation **from scratch** de l'algorithme *Isolation Forest* (Liu, Ting & Zhou, IEEE ICDM 2008), appliquée aux
trois jeux de données du prof (**iris**, **glass**, **breast cancer wdbc**), avec :

- l'implémentation « rapide » (scikit-learn) et la comparaison avec notre code,
- le diagramme de classes et la liste des méthodes,
- le reverse engineering des structures de données (les nôtres **et** celles de scikit-learn),
- la visualisation des résultats.

Le rapport complet est dans [`report/RAPPORT.md`](report/RAPPORT.md).

## Données

Isolation Forest est **non supervisé** : on enlève donc toute étiquette.

| Jeu | Fichier à placer dans `data/` | Ce qu'on garde |
|---|---|---|
| iris | `iris.data` | les 4 premiers attributs (sans l'espèce) : 150 x 4 |
| glass | `glass.data` | les 9 mesures (sans l'Id ni la classe) : 214 x 9 |
| wdbc | `wdbc.data` | les 30 mesures (sans l'identifiant ni le diagnostic) : 569 x 30 |

Les trois fichiers sont déjà dans `data/`. Sources : <https://archive.ics.uci.edu/dataset/53/iris>, <https://archive.ics.uci.edu/dataset/42/glass+identification>,
<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>.
Si `iris.data` ou `wdbc.data` est absent, le code utilise la copie fournie par scikit-learn ; `glass.data` est obligatoire.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9 ou plus.

## Utilisation rapide

```python
from isolation_forest import IsolationForest
from isolation_forest.data import charger_wdbc

X = charger_wdbc()                                       # (569, 30), sans étiquette
modele = IsolationForest(n_arbres=100, psi=256, contamination=0.05, random_state=42).fit(X)
scores = modele.score_samples(X)                         # s(x, psi) dans ]0, 1] : proche de 1 = anormal
print(scores.argsort()[::-1][:5], modele.predict(X).sum())   # 5 points les plus anormaux, nombre d'anomalies
```

## Structure du projet

```
data/                       fichiers du prof (iris.data, glass.data, wdbc.data)
src/isolation_forest/
  utils.py        nombre harmonique H(i) et facteur c(n)
  sampling.py     MODULE 1 : sous-échantillonnage sans remise
  itree.py        MODULE 2 : Noeud / Feuille, construire_itree, longueur_chemin
  iforest.py      MODULES 3 et 4 : classe IsolationForest (fit, score_samples, predict...)
  comparaison.py  corrélation de Spearman et recouvrement top-k entre deux classements de scores
  data.py         chargement des 3 jeux du prof + jeu jouet + gaussiennes de démonstration
  viz.py          coupes d'un arbre, carte des scores, projection PCA, histogramme, tableau de bord
examples/
  demo_toy.py          hand-tracing sur 5 points normaux + F(10, 10)
  run_dataset.py       iris | glass | wdbc | gauss -> scores, top anomalies, tableau de bord
  compare_sklearn.py   implémentation rapide (scikit-learn) + comparaison avec notre code
  benchmark.py         stabilité selon T et psi, complexité
tests/                 tests unittest (aussi exécutables avec pytest)
report/
  RAPPORT.md                   rapport complet (livrables du TD)
  reverse_engineering.py       observation des structures (ndarray, arbre, liste, pile, rng, matplotlib, scikit-learn)
  diagramme_classes.py         diagrammes UML (PNG) + diagramme_classes.mmd (Mermaid)
  liste_methodes.py            génère LISTE_DES_METHODES.md par introspection
  figures/                     toutes les figures
  resultats_*.txt              sorties des scripts
```

## Tout lancer

- Windows : `run_all.bat` (utilise la commande `py` ; sinon remplacer `PY=py` par `PY=python` dans le fichier)
- Linux / macOS : `./run_all.sh`

Ou séparément :

```bash
python -m unittest discover -s tests -t . -v      # les tests
python examples/demo_toy.py                        # hand-tracing
python examples/run_dataset.py wdbc                # un jeu (iris, glass ou wdbc)
python examples/compare_sklearn.py                 # implémentation rapide + comparaison
python examples/benchmark.py                       # stabilité et complexité (~30 s)
python report/reverse_engineering.py               # reverse engineering
```

Les scores sont reproductibles : tous les tirages dépendent de `random_state` (42 dans les scripts).
