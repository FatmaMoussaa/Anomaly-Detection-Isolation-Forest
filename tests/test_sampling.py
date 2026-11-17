import unittest

import numpy as np

from tests import _chemins  # noqa: F401
from isolation_forest.sampling import sous_echantillonnage


class TestSousEchantillonnage(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(40, dtype=float).reshape(20, 2)

    def test_garde_tout_si_peu_de_points(self):
        res = sous_echantillonnage(self.X, 50, np.random.default_rng(0))
        np.testing.assert_array_equal(res, self.X)

    def test_taille_psi(self):
        res = sous_echantillonnage(self.X, 8, np.random.default_rng(0))
        self.assertEqual(res.shape, (8, 2))

    def test_sans_remise(self):
        res = sous_echantillonnage(self.X, 15, np.random.default_rng(1))
        self.assertEqual(len({tuple(l) for l in res}), 15)

    def test_lignes_appartiennent_a_X(self):
        ens = {tuple(l) for l in self.X}
        res = sous_echantillonnage(self.X, 10, np.random.default_rng(2))
        self.assertTrue(all(tuple(l) in ens for l in res))

    def test_reproductible(self):
        a = sous_echantillonnage(self.X, 10, np.random.default_rng(7))
        b = sous_echantillonnage(self.X, 10, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_ne_modifie_pas_X(self):
        copie = self.X.copy()
        res = sous_echantillonnage(self.X, 5, np.random.default_rng(3))
        res[:] = -1
        np.testing.assert_array_equal(self.X, copie)

    def test_psi_invalide(self):
        with self.assertRaises(ValueError):
            sous_echantillonnage(self.X, 0, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
