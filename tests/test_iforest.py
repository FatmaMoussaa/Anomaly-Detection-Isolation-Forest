import math
import unittest

import numpy as np

from tests import _chemins  # noqa: F401
from isolation_forest import IsolationForest
from isolation_forest.data import generer_gaussiennes, jeu_jouet
from isolation_forest.itree import hauteur


class TestParametres(unittest.TestCase):
    def test_parametres_invalides(self):
        for kw in ({"n_arbres": 0}, {"psi": 1}, {"contamination": 0.0}, {"contamination": 0.7},
                   {"seuil": 1.5}, {"n_arbres": 2.5}):
            with self.assertRaises(ValueError, msg=str(kw)):
                IsolationForest(**kw)

    def test_predict_avant_fit(self):
        with self.assertRaises(RuntimeError):
            IsolationForest().predict(np.zeros((2, 2)))

    def test_mauvaise_dimension(self):
        m = IsolationForest(n_arbres=5, psi=8, random_state=0).fit(np.random.default_rng(0).normal(size=(20, 3)))
        with self.assertRaises(ValueError):
            m.score_samples(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            m.score_samples(np.zeros(3))

    def test_nan_refuse(self):
        with self.assertRaises(ValueError):
            IsolationForest().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))

    def test_un_seul_point_refuse(self):
        with self.assertRaises(ValueError):
            IsolationForest().fit(np.array([[1.0, 2.0]]))


class TestForet(unittest.TestCase):
    def setUp(self):
        self.X, self.y = generer_gaussiennes(400, 2, 0.05, seed=1)

    def test_attributs_appris(self):
        m = IsolationForest(n_arbres=20, psi=64, random_state=0).fit(self.X)
        self.assertEqual(len(m.arbres_), 20)
        self.assertEqual(m.psi_effectif_, 64)
        self.assertEqual(m.limite_hauteur_, math.ceil(math.log2(64)))
        self.assertEqual(m.n_features_, 2)
        self.assertTrue(all(hauteur(a) <= m.limite_hauteur_ for a in m.arbres_))

    def test_psi_plus_grand_que_n(self):
        X = self.X[:30]
        m = IsolationForest(n_arbres=5, psi=256, random_state=0).fit(X)
        self.assertEqual(m.psi_effectif_, 30)
        self.assertEqual(m.limite_hauteur_, 5)

    def test_scores_dans_0_1(self):
        s = IsolationForest(n_arbres=30, psi=64, random_state=0).fit(self.X).score_samples(self.X)
        self.assertEqual(s.shape, (400,))
        self.assertTrue(np.all(s > 0) and np.all(s <= 1))

    def test_reproductible(self):
        a = IsolationForest(n_arbres=20, psi=64, random_state=3).fit(self.X).score_samples(self.X)
        b = IsolationForest(n_arbres=20, psi=64, random_state=3).fit(self.X).score_samples(self.X)
        np.testing.assert_array_equal(a, b)

    def test_graines_differentes_donnent_des_scores_differents(self):
        a = IsolationForest(n_arbres=20, psi=64, random_state=3).fit(self.X).score_samples(self.X)
        b = IsolationForest(n_arbres=20, psi=64, random_state=4).fit(self.X).score_samples(self.X)
        self.assertFalse(np.array_equal(a, b))

    def test_anomalie_lointaine_mieux_scoree_que_les_normaux(self):
        X = np.vstack([np.random.default_rng(0).normal(size=(200, 2)), [[15.0, 15.0]]])
        m = IsolationForest(n_arbres=100, psi=128, random_state=0).fit(X)
        s = m.score_samples(X)
        self.assertGreater(s[-1], np.percentile(s[:-1], 99))
        self.assertGreater(s[-1], 0.6)

    def test_anomalies_synthetiques_ont_les_scores_les_plus_hauts(self):
        """Sur les gaussiennes, les anomalies injectées ont en moyenne un score bien plus élevé."""
        m = IsolationForest(n_arbres=100, psi=256, random_state=0).fit(self.X)
        s = m.score_samples(self.X)
        self.assertGreater(s[self.y == 1].mean(), s[self.y == 0].mean() + 0.05)

    def test_chemin_anomalie_plus_court(self):
        m = IsolationForest(n_arbres=100, psi=256, random_state=0).fit(self.X)
        eh = m.chemin_moyen(self.X)
        self.assertLess(eh[self.y == 1].mean(), eh[self.y == 0].mean())

    def test_chemins_par_arbre_forme(self):
        m = IsolationForest(n_arbres=7, psi=32, random_state=0).fit(self.X)
        self.assertEqual(m.chemins_par_arbre(self.X[:10]).shape, (10, 7))

    def test_score_formule(self):
        m = IsolationForest(n_arbres=10, psi=32, random_state=0).fit(self.X)
        eh = m.chemin_moyen(self.X[:5])
        np.testing.assert_allclose(m.score_samples(self.X[:5]), 2 ** (-eh / m.facteur_c_))

    def test_contamination(self):
        m = IsolationForest(n_arbres=50, psi=128, contamination=0.05, random_state=0).fit(self.X)
        taux = m.predict(self.X).mean()
        self.assertAlmostEqual(taux, 0.05, delta=0.02)

    def test_fit_predict_et_seuil_fixe(self):
        m = IsolationForest(n_arbres=50, psi=128, seuil=0.55, random_state=0)
        pred = m.fit_predict(self.X)
        self.assertEqual(set(np.unique(pred)) - {0, 1}, set())
        self.assertEqual(m.seuil_, 0.55)

    def test_resume_et_repr(self):
        m = IsolationForest(n_arbres=5, psi=16, random_state=0).fit(self.X)
        r = m.resume()
        self.assertEqual(r["n_arbres"], 5)
        self.assertIn("IsolationForest(", repr(m))

    def test_jeu_jouet(self):
        X, y, _ = jeu_jouet()
        s = IsolationForest(n_arbres=100, psi=6, random_state=42).fit(X).score_samples(X)
        self.assertEqual(int(np.argmax(s)), 5)          # F est la plus anormale


if __name__ == "__main__":
    unittest.main()
