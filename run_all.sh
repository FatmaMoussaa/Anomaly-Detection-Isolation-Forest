#!/usr/bin/env bash
# Lance les tests, les démos et régénère toutes les figures et les fichiers du rapport.
# Avant : placer iris.data, glass.data et wdbc.data (fichiers du prof) dans data/.
set -e
cd "$(dirname "$0")"
python3 -m unittest discover -s tests -t . -v
python3 examples/demo_toy.py
: > report/resultats_datasets.txt
for jeu in iris glass wdbc gauss; do
  python3 examples/run_dataset.py $jeu | tee -a report/resultats_datasets.txt
done
python3 examples/compare_sklearn.py | tee report/resultats_comparaison_sklearn.txt
python3 examples/benchmark.py > report/resultats_benchmark.txt
python3 report/reverse_engineering.py
python3 report/liste_methodes.py
python3 report/diagramme_classes.py
echo "Terminé : voir report/figures/ et report/RAPPORT.md"
