"""Ajoute src/ au chemin Python pour lancer les scripts sans installer le paquet."""
import os
import sys

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RACINE, "src"))
FIGURES = os.path.join(RACINE, "report", "figures")
