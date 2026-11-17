import os
import tempfile
import unittest

import numpy as np

from tests import _chemins  # noqa: F401
from isolation_forest import IsolationForest
from isolation_forest.data import (charger_glass, charger_iris, charger_jeu, charger_wdbc, generer_gaussiennes,
                                   jeu_jouet)


class TestDemo(unittest.TestCase):
    def test_jeu_jouet(self):
        X, y, noms = jeu_jouet()
        self.assertEqual(X.shape, (6, 2))
        self.assertEqual(noms[-1], "F")
        self.assertEqual(tuple(X[-1]), (10.0, 10.0))
        self.assertEqual(y.sum(), 1)

    def test_gaussiennes(self):
        X, y = generer_gaussiennes(200, 3, 0.1, seed=0)
        self.assertEqual(X.shape, (200, 3))
        self.assertEqual(int(y.sum()), 20)


class TestJeuxDuProf(unittest.TestCase):
    """Les jeux du prof sont renvoyés SANS étiquette : seulement la matrice des attributs."""

    def test_iris(self):
        X = charger_iris()
        self.assertEqual(X.shape, (150, 4))

    def test_wdbc(self):
        X = charger_wdbc()
        self.assertEqual(X.shape, (569, 30))

    def test_jeu_inconnu(self):
        with self.assertRaises(ValueError):
            charger_jeu("mnist")

    def _ecrire(self, dossier, nom, lignes):
        chemin = os.path.join(dossier, nom)
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("\n".join(lignes) + "\n")
        return chemin

    def test_format_uci_iris(self):
        with tempfile.TemporaryDirectory() as d:
            f = self._ecrire(d, "iris.data", ["5.1,3.5,1.4,0.2,Iris-setosa", "7.0,3.2,4.7,1.4,Iris-versicolor", ""])
            X = charger_iris(f)
            self.assertEqual(X.shape, (2, 4))                       # l'espèce (texte) est ignorée
            self.assertEqual(X[1].tolist(), [7.0, 3.2, 4.7, 1.4])

    def test_format_uci_glass_sans_id_ni_classe(self):
        with tempfile.TemporaryDirectory() as d:
            f = self._ecrire(d, "glass.data", ["1,1.52101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00,1",
                                               "2,1.51761,13.89,3.60,1.36,72.73,0.48,7.83,0.00,0.00,1"])
            X = charger_glass(f)
            self.assertEqual(X.shape, (2, 9))
            self.assertAlmostEqual(X[0, 0], 1.52101)                # 1re colonne gardée = RI, pas l'Id
            self.assertAlmostEqual(X[0, 8], 0.0)                    # dernière colonne gardée = Fe, pas le Type

    def test_format_uci_wdbc_sans_id_ni_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            ligne = "842302,M," + ",".join(str(float(i)) for i in range(1, 31))
            f = self._ecrire(d, "wdbc.data", [ligne, ligne])
            X = charger_wdbc(f)
            self.assertEqual(X.shape, (2, 30))
            self.assertEqual((X[0, 0], X[0, 29]), (1.0, 30.0))

    def test_fichier_explicite_introuvable(self):
        with self.assertRaises(FileNotFoundError):
            charger_iris("/chemin/qui/n/existe/pas.data")


class TestSurJeuxReels(unittest.TestCase):
    """Tests d'intégration : sans étiquettes, on vérifie la cohérence des scores."""

    def test_scores_valides_sur_iris_et_wdbc(self):
        for X in (charger_iris(), charger_wdbc()):
            s = IsolationForest(n_arbres=50, psi=128, random_state=0).fit(X).score_samples(X)
            self.assertEqual(s.shape, (len(X),))
            self.assertTrue(np.all((s > 0) & (s <= 1)))
            self.assertGreater(s.max(), s.min())                    # les scores ne sont pas tous égaux


if __name__ == "__main__":
    unittest.main()
