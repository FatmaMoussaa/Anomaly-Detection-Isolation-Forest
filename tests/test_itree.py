import math
import unittest

import numpy as np

from tests import _chemins  # noqa: F401
from isolation_forest.itree import (Feuille, Noeud, arbre_en_texte, compter_noeuds, construire_itree, hauteur,
                                    longueur_chemin, longueur_chemin_recursive, tailles_feuilles, trace_chemin)
from isolation_forest.utils import c


class TestStructures(unittest.TestCase):
    def test_feuille_et_noeud(self):
        f = Feuille(3)
        n = Noeud(1, 2.5, Feuille(1), f)
        self.assertTrue(f.est_feuille())
        self.assertFalse(n.est_feuille())
        self.assertEqual((n.attribut, n.seuil), (1, 2.5))
        self.assertIs(n.droite, f)


class TestConstruction(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(size=(64, 3))

    def test_somme_des_feuilles_egale_psi(self):
        arbre = construire_itree(self.X, 0, 6, self.rng)
        self.assertEqual(sum(tailles_feuilles(arbre)), 64)

    def test_hauteur_limitee(self):
        for limite in (0, 1, 3, 6):
            arbre = construire_itree(self.X, 0, limite, np.random.default_rng(1))
            self.assertLessEqual(hauteur(arbre), limite)

    def test_limite_zero_donne_une_feuille(self):
        arbre = construire_itree(self.X, 0, 0, self.rng)
        self.assertIsInstance(arbre, Feuille)
        self.assertEqual(arbre.taille, 64)

    def test_un_seul_point(self):
        arbre = construire_itree(self.X[:1], 0, 5, self.rng)
        self.assertEqual(arbre.taille, 1)

    def test_valeurs_identiques_donnent_une_feuille(self):
        X = np.ones((10, 2))
        arbre = construire_itree(X, 0, 5, self.rng)
        self.assertIsInstance(arbre, Feuille)
        self.assertEqual(arbre.taille, 10)

    def test_seuils_dans_les_bornes(self):
        X = np.array([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0]])
        arbre = construire_itree(X, 0, 4, np.random.default_rng(5))
        pile = [arbre]
        while pile:
            n = pile.pop()
            if isinstance(n, Noeud):
                col = X[:, n.attribut]
                self.assertTrue(col.min() <= n.seuil <= col.max())
                pile += [n.gauche, n.droite]

    def test_compter_noeuds(self):
        arbre = construire_itree(self.X, 0, 6, self.rng)
        internes, feuilles = compter_noeuds(arbre)
        self.assertEqual(feuilles, internes + 1)          # arbre binaire strict

    def test_reproductible(self):
        a = construire_itree(self.X, 0, 6, np.random.default_rng(9))
        b = construire_itree(self.X, 0, 6, np.random.default_rng(9))
        self.assertEqual(arbre_en_texte(a), arbre_en_texte(b))


class TestLongueurChemin(unittest.TestCase):
    def setUp(self):
        # x[0] < 5 ? : gauche = feuille(1) ; droite = feuille(3)
        self.arbre = Noeud(0, 5.0, Feuille(1), Feuille(3))

    def test_chemin_vers_feuille_isolee(self):
        self.assertEqual(longueur_chemin([2.0], self.arbre), 1.0)

    def test_chemin_avec_correction_c(self):
        self.assertAlmostEqual(longueur_chemin([7.0], self.arbre), 1 + c(3))

    def test_seuil_egal_va_a_droite(self):
        self.assertAlmostEqual(longueur_chemin([5.0], self.arbre), 1 + c(3))

    def test_feuille_racine(self):
        self.assertEqual(longueur_chemin([0.0], Feuille(1)), 0.0)
        self.assertAlmostEqual(longueur_chemin([0.0], Feuille(2)), 1.0)

    def test_versions_recursive_et_iterative_identiques(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(128, 4))
        arbre = construire_itree(X, 0, 7, rng)
        for x in X[:40]:
            self.assertAlmostEqual(longueur_chemin(x, arbre), longueur_chemin_recursive(x, arbre))

    def test_trace_chemin(self):
        etapes = trace_chemin([7.0], self.arbre)
        self.assertEqual(len(etapes), 2)
        self.assertIn("droite", etapes[0])
        self.assertIn("feuille de taille 3", etapes[1])


if __name__ == "__main__":
    unittest.main()
