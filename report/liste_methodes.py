"""Génère report/LISTE_DES_METHODES.md par introspection du code (toujours synchronisée avec les sources).

    python report/liste_methodes.py
"""
import importlib
import inspect
import os
import sys

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RACINE, "src"))

MODULES = [
    ("utils", "Mathématiques : nombre harmonique et facteur c(n)"),
    ("sampling", "Module 1 : sous-échantillonnage"),
    ("itree", "Module 2 : arbre d'isolation (structures + construction + chemin)"),
    ("iforest", "Modules 3 et 4 : forêt et score d'anomalie"),
    ("comparaison", "Comparer deux classements de scores (Spearman, top-k)"),
    ("data", "Jeux de données"),
    ("viz", "Visualisations"),
]


def signature(f):
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return "()"
    params = [p for p in sig.parameters.values() if p.name != "self"]
    texte = ", ".join(p.name + ("=" + repr(p.default) if p.default is not inspect.Parameter.empty else "")
                      for p in params)
    retour = "" if sig.return_annotation is inspect.Signature.empty else f" -> {sig.return_annotation}".replace("'", "")
    return f"({texte}){retour}"


def resume(obj):
    """Première ligne de la docstring *propre* à l'objet (pas celle héritée d'object)."""
    doc = inspect.cleandoc(obj.__doc__) if getattr(obj, "__doc__", None) else ""
    if not doc:
        nom = getattr(obj, "__name__", "")
        return "Constructeur" if nom == "__init__" else ("Méthode spéciale Python" if nom.startswith("__") else "")
    return doc.splitlines()[0].replace("|", "\\|")


lignes = ["# Liste des classes, fonctions et méthodes", "",
          "_Fichier généré automatiquement par `report/liste_methodes.py`._", ""]
total = 0
for nom, description in MODULES:
    mod = importlib.import_module(f"isolation_forest.{nom}")
    lignes += [f"## `{nom}.py` — {description}", ""]
    classes = [(n, o) for n, o in inspect.getmembers(mod, inspect.isclass) if o.__module__ == mod.__name__]
    fonctions = [(n, o) for n, o in inspect.getmembers(mod, lambda o: inspect.isfunction(o) or hasattr(o, "__wrapped__"))
                 if getattr(o, "__module__", None) == mod.__name__ and not isinstance(o, type)]
    for n, cls in classes:
        lignes += [f"### Classe `{n}`" + (f" (hérite de `{cls.__bases__[0].__name__}`)" if cls.__bases__[0] is not object else ""),
                   "", resume(cls), "", "| Méthode | Signature | Rôle |", "|---|---|---|"]
        for m, f in cls.__dict__.items():
            if inspect.isfunction(f):
                lignes.append(f"| `{m}` | `{m}{signature(f)}` | {resume(f)} |")
                total += 1
        lignes.append("")
    if fonctions:
        lignes += ["| Fonction | Signature | Rôle |", "|---|---|---|"]
        for n, f in fonctions:
            lignes.append(f"| `{n}` | `{n}{signature(f)}` | {resume(f)} |")
            total += 1
        lignes.append("")
lignes.append(f"**Total : {total} fonctions et méthodes.**")
chemin = os.path.join(os.path.dirname(__file__), "LISTE_DES_METHODES.md")
with open(chemin, "w", encoding="utf-8") as f:
    f.write("\n".join(lignes) + "\n")
print(f"{chemin} écrit ({total} fonctions et méthodes)")
