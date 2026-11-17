import unittest

import numpy as np

from tests import _chemins  # noqa: F401
from isolation_forest import IsolationForest, correlation_rangs, recouvrement_top_k
from isolation_forest.comparaison import rangs


class TestComparaison(unittest.TestCase):
    def test_rangs_avec_ex_aequo(self):
        self.assertEqual(rangs(np.array([10.0, 30.0, 20.0, 20.0])).tolist(), [0.0, 3.0, 1.5, 1.5])

    def test_correlation_identique_et_inversee(self):
        a = np.array([0.1, 0.5, 0.3, 0.9])
        self.assertAlmostEqual(correlation_rangs(a, a), 1.0)
        self.assertAlmostEqual(correlation_rangs(a, -a), -1.0)
        self.assertAlmostEqual(correlation_rangs(a, a * 3 + 1), 1.0)       # ne dépend que de l'ordre

    def test_recouvrement(self):
        a = np.array([0.9, 0.8, 0.1, 0.2])
        b = np.array([0.7, 0.9, 0.3, 0.1])
        self.assertEqual(recouvrement_top_k(a, b, 2), 1.0)
        self.assertEqual(recouvrement_top_k(a, b[::-1], 2), 0.0)
        with self.assertRaises(ValueError):
            recouvrement_top_k(a, b, 0)

    def test_meme_graine_donne_classement_identique(self):
        X = np.random.default_rng(0).normal(size=(100, 3))
        s1 = IsolationForest(20, 64, random_state=3).fit(X).score_samples(X)
        s2 = IsolationForest(20, 64, random_state=3).fit(X).score_samples(X)
        self.assertAlmostEqual(correlation_rangs(s1, s2), 1.0)


if __name__ == "__main__":
    unittest.main()
