from unittest import TestCase

import numpy as np

from src.performance.cumulative.storage import FormulaData, MolCandidate
from src.performance.cumulative.investigator import (
    CummulativeInvestigator,
    select_top_n_formulas,
)


# ---------------------------------------------------------------------------
# MolCandidate / FormulaData
# ---------------------------------------------------------------------------


class TestMolCandidate(TestCase):
    def setUp(self):
        self.candidate = MolCandidate(
            num_env_steps=500,
            elements=[6, 1, 1, 1, 1],
            pos=np.zeros((5, 3)),
            reward=0.42,
            energy=-5.3,
        )

    def test_required_fields(self):
        self.assertEqual(self.candidate.num_env_steps, 500)
        self.assertEqual(self.candidate.elements, [6, 1, 1, 1, 1])
        self.assertTrue(np.allclose(self.candidate.pos, np.zeros((5, 3))))
        self.assertAlmostEqual(self.candidate.reward, 0.42)
        self.assertAlmostEqual(self.candidate.energy, -5.3)

    def test_optional_fields_default_to_none(self):
        self.assertIsNone(self.candidate.energy_relaxed)
        self.assertIsNone(self.candidate.rae)

    def test_optional_fields_can_be_set(self):
        c = MolCandidate(
            num_env_steps=1000,
            elements=[6, 8],
            pos=np.zeros((2, 3)),
            reward=0.1,
            energy=-3.0,
            energy_relaxed=-3.5,
            rae=0.05,
        )
        self.assertAlmostEqual(c.energy_relaxed, -3.5)
        self.assertAlmostEqual(c.rae, 0.05)


class TestFormulaData(TestCase):
    def test_empty_on_init(self):
        fd = FormulaData()
        self.assertEqual(fd.molecules, {})

    def test_add_molecule(self):
        fd = FormulaData()
        candidate = MolCandidate(num_env_steps=100, elements=[6], pos=np.zeros((1, 3)), reward=0.5, energy=-2.0)
        fd.molecules["C"] = [candidate]
        self.assertIn("C", fd.molecules)
        self.assertEqual(len(fd.molecules["C"]), 1)

    def test_multiple_candidates_same_smiles(self):
        fd = FormulaData()
        c1 = MolCandidate(100, [6], np.zeros((1, 3)), 0.5, -2.0)
        c2 = MolCandidate(200, [6], np.ones((1, 3)), 0.6, -2.1)
        fd.molecules["C"] = [c1, c2]
        self.assertEqual(len(fd.molecules["C"]), 2)

    def test_independent_default_dicts(self):
        # Each FormulaData instance must have its own dict (not shared via mutable default)
        fd1 = FormulaData()
        fd2 = FormulaData()
        fd1.molecules["C"] = []
        self.assertNotIn("C", fd2.molecules)


# ---------------------------------------------------------------------------
# CummulativeInvestigator._aggregate_discovery_metrics
# ---------------------------------------------------------------------------


class TestAggregateDiscoveryMetrics(TestCase):
    def test_basic_aggregation(self):
        dm = {
            "old_data": [10, 20],
            "rediscovered": [3, 8],
            "novel": [2, 5],
        }
        result = CummulativeInvestigator._aggregate_discovery_metrics(dm)

        self.assertEqual(result["old_data"], 30)
        self.assertEqual(result["rediscovered"], 11)
        self.assertAlmostEqual(result["rediscovery_ratio"], 11 / 30)
        self.assertEqual(result["novel"], 7)
        self.assertAlmostEqual(result["expansion_ratio"], 7 / 30)

    def test_full_rediscovery(self):
        dm = {
            "old_data": [5, 5],
            "rediscovered": [5, 5],
            "novel": [0, 0],
        }
        result = CummulativeInvestigator._aggregate_discovery_metrics(dm)
        self.assertAlmostEqual(result["rediscovery_ratio"], 1.0)
        self.assertAlmostEqual(result["expansion_ratio"], 0.0)

    def test_no_rediscovery(self):
        dm = {
            "old_data": [10],
            "rediscovered": [0],
            "novel": [4],
        }
        result = CummulativeInvestigator._aggregate_discovery_metrics(dm)
        self.assertAlmostEqual(result["rediscovery_ratio"], 0.0)
        self.assertAlmostEqual(result["expansion_ratio"], 4 / 10)

    def test_single_formula(self):
        dm = {
            "old_data": [8],
            "rediscovered": [2],
            "novel": [3],
        }
        result = CummulativeInvestigator._aggregate_discovery_metrics(dm)
        self.assertEqual(result["old_data"], 8)
        self.assertEqual(result["rediscovered"], 2)
        self.assertAlmostEqual(result["rediscovery_ratio"], 2 / 8)


# ---------------------------------------------------------------------------
# select_top_n_formulas
# ---------------------------------------------------------------------------


class TestSelectTopNFormulas(TestCase):
    def setUp(self):
        self.dm = {
            "formulas": ("CH4", "C2H6", "CH3OH"),
            "rediscovered": (3, 8, 1),
            "novel": (2, 5, 0),
            "old_data": (10, 20, 5),
        }

    def test_selects_first_n(self):
        formulas, r, n, o = select_top_n_formulas(n=2, dm=self.dm)
        self.assertEqual(list(formulas), ["CH4", "C2H6"])
        self.assertEqual(list(r), [3, 8])
        self.assertEqual(list(n), [2, 5])
        self.assertEqual(list(o), [10, 20])

    def test_n_larger_than_available(self):
        formulas, r, n, o = select_top_n_formulas(n=10, dm=self.dm)
        self.assertEqual(len(formulas), 3)

    def test_n_equals_one(self):
        formulas, r, n, o = select_top_n_formulas(n=1, dm=self.dm)
        self.assertEqual(list(formulas), ["CH4"])
        self.assertEqual(list(r), [3])

    def test_n_equals_all(self):
        formulas, r, n, o = select_top_n_formulas(n=3, dm=self.dm)
        self.assertEqual(len(formulas), 3)
        self.assertEqual(list(formulas), ["CH4", "C2H6", "CH3OH"])
