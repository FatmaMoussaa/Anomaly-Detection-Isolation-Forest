@echo off
rem Lance les tests, les demos et regenere toutes les figures et les fichiers du rapport.
rem Avant : placer iris.data, glass.data et wdbc.data (fichiers du prof) dans data\.
rem Utilise "py" (lanceur Python de Windows) ; si "py" n'existe pas, remplacer PY=py par PY=python.
set PY=py
cd /d "%~dp0"
%PY% -m unittest discover -s tests -t . -v || exit /b 1
%PY% examples\demo_toy.py || exit /b 1
break > report\resultats_datasets.txt
for %%j in (iris glass wdbc gauss) do %PY% examples\run_dataset.py %%j >> report\resultats_datasets.txt || exit /b 1
%PY% examples\compare_sklearn.py > report\resultats_comparaison_sklearn.txt || exit /b 1
%PY% examples\benchmark.py > report\resultats_benchmark.txt || exit /b 1
%PY% report\reverse_engineering.py || exit /b 1
%PY% report\liste_methodes.py || exit /b 1
%PY% report\diagramme_classes.py || exit /b 1
echo Termine : voir report\figures\ et report\RAPPORT.md
