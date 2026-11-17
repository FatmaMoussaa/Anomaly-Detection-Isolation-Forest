# Liste des classes, fonctions et méthodes

_Fichier généré automatiquement par `report/liste_methodes.py`._

## `utils.py` — Mathématiques : nombre harmonique et facteur c(n)

| Fonction | Signature | Rôle |
|---|---|---|
| `c` | `c(n) -> float` | Facteur de normalisation c(n) (longueur moyenne d'une recherche infructueuse). |
| `nombre_harmonique` | `nombre_harmonique(i) -> float` | Nombre harmonique H(i) = 1 + 1/2 + ... + 1/i  (H(0) = 0). |

## `sampling.py` — Module 1 : sous-échantillonnage

| Fonction | Signature | Rôle |
|---|---|---|
| `sous_echantillonnage` | `sous_echantillonnage(X, psi, rng) -> np.ndarray` | Retourne psi lignes de X tirées uniformément sans remise. |

## `itree.py` — Module 2 : arbre d'isolation (structures + construction + chemin)

### Classe `Feuille` (hérite de `NoeudArbre`)

Feuille d'un iTree : mémorise combien de points sont arrivés ici.

| Méthode | Signature | Rôle |
|---|---|---|
| `__init__` | `__init__(taille) -> None` | Constructeur |
| `est_feuille` | `est_feuille() -> bool` | Toujours vrai pour une feuille. |
| `__repr__` | `__repr__() -> str` | Méthode spéciale Python |

### Classe `Noeud` (hérite de `NoeudArbre`)

Nœud interne : coupe l'attribut ``attribut`` au seuil ``seuil``.

| Méthode | Signature | Rôle |
|---|---|---|
| `__init__` | `__init__(attribut, seuil, gauche, droite) -> None` | Constructeur |
| `est_feuille` | `est_feuille() -> bool` | Toujours faux pour un nœud interne. |
| `__repr__` | `__repr__() -> str` | Méthode spéciale Python |

### Classe `NoeudArbre`

Classe de base commune aux nœuds internes et aux feuilles.

| Méthode | Signature | Rôle |
|---|---|---|
| `est_feuille` | `est_feuille() -> bool` | Indique si le nœud est une feuille (à redéfinir dans les sous-classes). |

| Fonction | Signature | Rôle |
|---|---|---|
| `_lignes_arbre` | `_lignes_arbre(arbre, noms) -> List[str]` | Lignes de la représentation textuelle (branches dessinées avec ├─ └─ │). |
| `arbre_en_texte` | `arbre_en_texte(arbre, noms=None) -> str` | Représentation textuelle de l'arbre (une ligne par nœud). |
| `compter_noeuds` | `compter_noeuds(arbre) -> Tuple[int, int]` | Retourne (nombre de nœuds internes, nombre de feuilles). |
| `construire_itree` | `construire_itree(X_sub, h, limite, rng) -> NoeudArbre` | Construit récursivement un arbre d'isolation (pseudo-code de la diapo 13). |
| `decoupes` | `decoupes(arbre, bbox, h=0) -> Iterator[Tuple[int, float, Tuple[float, float, float, float], int]]` | Énumère les coupes d'un arbre *en 2D* : (attribut, seuil, boîte englobante, profondeur). |
| `hauteur` | `hauteur(arbre) -> int` | Hauteur de l'arbre (0 si c'est une simple feuille). |
| `longueur_chemin` | `longueur_chemin(x, arbre, h=0) -> float` | Version itérative (même résultat, pas de pile d'appels). |
| `longueur_chemin_recursive` | `longueur_chemin_recursive(x, arbre, h=0) -> float` | Version récursive, identique au pseudo-code de la diapo 15. |
| `tailles_feuilles` | `tailles_feuilles(arbre) -> List[int]` | Liste des tailles de toutes les feuilles (leur somme vaut psi). |
| `trace_chemin` | `trace_chemin(x, arbre, noms=None) -> List[str]` | Décrit pas à pas le chemin de x (utile pour le hand-tracing). |

## `iforest.py` — Modules 3 et 4 : forêt et score d'anomalie

### Classe `IsolationForest`

Forêt de T arbres d'isolation (Liu, Ting & Zhou, 2008).

| Méthode | Signature | Rôle |
|---|---|---|
| `__init__` | `__init__(n_arbres=100, psi=256, contamination=None, seuil=0.6, random_state=None) -> None` | Constructeur |
| `fit` | `fit(X) -> IsolationForest` | Construit la forêt. Aucune étiquette n'est utilisée (non supervisé). |
| `chemins_par_arbre` | `chemins_par_arbre(X) -> np.ndarray` | Matrice (n, T) des longueurs de chemin h_i(x) pour chaque arbre. |
| `chemin_moyen` | `chemin_moyen(X) -> np.ndarray` | E(h(x)) : moyenne des longueurs de chemin sur les T arbres. |
| `score_samples` | `score_samples(X) -> np.ndarray` | Score d'anomalie s(x, psi) = 2^(-E(h(x)) / c(psi)), dans ]0, 1]. |
| `predict` | `predict(X) -> np.ndarray` | 1 si score >= seuil_ (anomalie), 0 sinon (normal). |
| `fit_predict` | `fit_predict(X) -> np.ndarray` | fit puis predict sur le même jeu de données. |
| `resume` | `resume() -> Dict[str, float]` | Statistiques sur la forêt apprise (hauteurs, tailles des arbres). |
| `get_params` | `get_params() -> Dict[str, object]` |  |
| `__repr__` | `__repr__() -> str` | Méthode spéciale Python |
| `_verifier_entraine` | `_verifier_entraine() -> None` |  |
| `_valider` | `_valider(X, entrainement=True) -> np.ndarray` |  |

## `comparaison.py` — Comparer deux classements de scores (Spearman, top-k)

| Fonction | Signature | Rôle |
|---|---|---|
| `correlation_rangs` | `correlation_rangs(a, b) -> float` | Corrélation de Spearman : 1 = même classement, 0 = aucun lien, -1 = classement inversé. |
| `rangs` | `rangs(valeurs) -> np.ndarray` | Rang de chaque valeur (0 = plus petite) ; les ex æquo reçoivent le rang moyen. |
| `recouvrement_top_k` | `recouvrement_top_k(a, b, k) -> float` | Part des k points les plus anormaux (scores les plus élevés) communs aux deux classements. |

## `data.py` — Jeux de données

| Fonction | Signature | Rôle |
|---|---|---|
| `_lire_attributs` | `_lire_attributs(fichier, colonnes) -> np.ndarray` | Lit uniquement les colonnes numériques demandées (les autres, texte ou id, sont ignorées). |
| `_trouver_fichier` | `_trouver_fichier(nom, chemin) -> Optional[str]` | Chemin explicite, sinon data/<nom>.data\|.csv\|.txt ; None si rien n'existe. |
| `charger_glass` | `charger_glass(chemin=None) -> np.ndarray` | Glass Identification : 214 x 9 (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe), sans Id ni classe. |
| `charger_iris` | `charger_iris(chemin=None) -> np.ndarray` | Iris : 150 x 4 (longueur/largeur des sépales et des pétales), sans l'espèce. |
| `charger_jeu` | `charger_jeu(nom, chemin=None) -> np.ndarray` | Charge ``iris``, ``glass`` ou ``wdbc`` et renvoie X (n, d) sans aucune étiquette. |
| `charger_wdbc` | `charger_wdbc(chemin=None) -> np.ndarray` | Breast Cancer Wisconsin (Diagnostic) : 569 x 30, sans identifiant ni diagnostic. |
| `generer_gaussiennes` | `generer_gaussiennes(n=500, d=2, contamination=0.05, seed=0) -> Tuple[np.ndarray, np.ndarray]` | Nuage gaussien normal + anomalies uniformes dans un cube plus large. Renvoie (X, y). |
| `jeu_jouet` | `jeu_jouet() -> Tuple[np.ndarray, np.ndarray, List[str]]` | Jeu jouet de la diapo de hand-tracing : 5 points normaux (A à E) et une anomalie F(10, 10). |

## `viz.py` — Visualisations

| Fonction | Signature | Rôle |
|---|---|---|
| `_marges` | `_marges(X, marge=0.1)` |  |
| `projection_pca` | `projection_pca(X, k=2) -> np.ndarray` | Projection sur les k premières composantes principales (SVD, numpy seul). |
| `sauver` | `sauver(fig, chemin, dpi=150) -> str` | Enregistre la figure (crée le dossier si besoin) et la ferme. |
| `tracer_carte_scores` | `tracer_carte_scores(modele, X, y=None, resolution=120, titre="Carte du score d'anomalie s(x)", ax=None) -> Figure` | Surface de décision : score en tout point du plan (modèle entraîné sur des données 2D). |
| `tracer_chemins` | `tracer_chemins(modele, X, y=None, titre='Longueur de chemin moyenne E(h(x))', ax=None) -> Figure` | Histogramme de E(h(x)) : les anomalies sont à gauche (chemin court). |
| `tracer_courbe_parametre` | `tracer_courbe_parametre(valeurs, moyennes, ecarts, etiquette, titre, ax=None, ylabel='stabilité (corrélation de Spearman)') -> Figure` | Valeur moyenne (± écart-type) d'une mesure en fonction d'un hyperparamètre (T ou psi). |
| `tracer_decoupes` | `tracer_decoupes(arbre, X, y=None, noms_points=None, titre="Coupes d'un arbre d'isolation", ax=None) -> Figure` | Dessine les coupes aléatoires d'un iTree en 2D (idée clé de la diapo 7). |
| `tracer_histogramme_scores` | `tracer_histogramme_scores(scores, y=None, seuil=None, titre='Distribution des scores', ax=None) -> Figure` |  |
| `tracer_projection_scores` | `tracer_projection_scores(X, scores, y=None, titre='Projection PCA colorée par le score', ax=None) -> Figure` | Nuage de points en 2D (PCA) coloré par score ; les vraies anomalies sont cerclées de rouge. |
| `tracer_scores_tries` | `tracer_scores_tries(scores, seuil=None, titre='Scores classés du plus anormal au plus normal', ax=None) -> Figure` | Courbe des scores triés : les premiers points (à gauche) sont les plus isolables. |
| `tracer_tableau_de_bord` | `tracer_tableau_de_bord(modele, X, y=None, titre='') -> Figure` | Figure de synthèse 2x2 : projection PCA, histogramme, scores triés, longueurs de chemin. |

**Total : 54 fonctions et méthodes.**
