import math
import unittest

from tests import _chemins  # noqa: F401
from isolation_forest.utils import EULER_MASCHERONI, c, nombre_harmonique


class TestUtils(unittest.TestCase):
    def test_harmonique_valeurs_connues(self):
        self.assertEqual(nombre_harmonique(0), 0.0)
        self.assertEqual(nombre_harmonique(1), 1.0)
        self.assertAlmostEqual(nombre_harmonique(4), 1 + 1 / 2 + 1 / 3 + 1 / 4)

    def test_harmonique_negatif(self):
        with self.assertRaises(ValueError):
            nombre_harmonique(-1)

    def test_approximation_asymptotique_proche_de_l_exacte(self):
        exact = math.fsum(1.0 / k for k in range(1, 5001))
        self.assertAlmostEqual(nombre_harmonique(5000), exact, places=10)

    def test_c_cas_particuliers_du_cours(self):
        self.assertEqual(c(0), 0.0)
        self.assertEqual(c(1), 0.0)
        self.assertEqual(c(2), 1.0)

    def test_c_valeur_n3(self):
        self.assertAlmostEqual(c(3), 2 * 1.5 - 2 * 2 / 3)      # 5/3

    def test_c_croissante(self):
        valeurs = [c(n) for n in range(2, 300)]
        self.assertTrue(all(a < b for a, b in zip(valeurs, valeurs[1:])))

    def test_c_256_proche_de_la_valeur_de_l_article(self):
        # c(256) ~ 2(ln 255 + gamma) - 2*255/256 ~ 10.24
        self.assertAlmostEqual(c(256), 2 * (math.log(255) + EULER_MASCHERONI) - 2 * 255 / 256, places=2)


if __name__ == "__main__":
    unittest.main()
