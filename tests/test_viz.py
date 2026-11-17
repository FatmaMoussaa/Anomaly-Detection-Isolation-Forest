import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tests import _chemins  # noqa: F401
from isolation_forest import IsolationForest, construire_itree
from isolation_forest import viz
from isolation_forest.data import generer_gaussiennes, jeu_jouet


class TestViz(unittest.TestCase):
    def setUp(self):
        self.X, self.y = generer_gaussiennes(150, 2, 0.06, seed=0)
        self.m = IsolationForest(n_arbres=20, psi=64, random_state=0).fit(self.X)

    def test_figures_sans_erreur(self):
        rng = np.random.default_rng(0)
        X6, y6, noms = jeu_jouet()
        figs = [
            viz.tracer_decoupes(construire_itree(X6, 0, 3, rng), X6, y6, noms),
            viz.tracer_carte_scores(self.m, self.X, self.y, resolution=30),
            viz.tracer_projection_scores(self.X, self.m.score_samples(self.X), self.y),
            viz.tracer_histogramme_scores(self.m.score_samples(self.X), self.y, self.m.seuil_),
            viz.tracer_scores_tries(self.m.score_samples(self.X), self.m.seuil_),
            viz.tracer_chemins(self.m, self.X, self.y),
            viz.tracer_tableau_de_bord(self.m, self.X, self.y),
            viz.tracer_tableau_de_bord(self.m, self.X),            # sans étiquettes (cas des jeux du prof)
            viz.tracer_courbe_parametre([1, 2, 3], np.array([.5, .6, .7]), np.array([.1, .1, .1]), "T", "test"),
        ]
        self.assertTrue(all(f is not None for f in figs))
        for f in figs:
            viz.plt.close(f)

    def test_sauver(self):
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "sous", "fig.png")
            viz.sauver(viz.tracer_scores_tries(self.m.score_samples(self.X)), chemin)
            self.assertTrue(os.path.getsize(chemin) > 1000)

    def test_donnees_non_2d_refusees(self):
        with self.assertRaises(ValueError):
            viz.tracer_decoupes(construire_itree(np.zeros((4, 3)), 0, 2, np.random.default_rng(0)), np.zeros((4, 3)))

    def test_pca_forme(self):
        self.assertEqual(viz.projection_pca(np.random.default_rng(0).normal(size=(30, 5))).shape, (30, 2))


if __name__ == "__main__":
    unittest.main()
