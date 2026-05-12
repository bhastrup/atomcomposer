from unittest import TestCase

import numpy as np
import pandas as pd

from src.performance.single_cpkt.stats import (
    get_global_metrics,
    make_serializable,
    single_formula_metrics,
)
from src.performance.single_cpkt.evaluator import get_num_episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_valid=3, n_nan_energy=1, with_rings=False):
    """Return a small synthetic results DataFrame."""
    n = n_valid + n_nan_energy
    abs_energies = list(np.linspace(-5.0, -3.0, n_valid)) + [np.nan] * n_nan_energy
    smiles = ["C", "CCO", "C"] + [None] * n_nan_energy
    valid = [True] * n_valid + [False] * n_nan_energy
    dipole = list(np.linspace(1.0, 3.0, n_valid)) + [0.0] * n_nan_energy
    relax_stable = [True, False, True] + [False] * n_nan_energy
    rmsd = [0.1, np.nan, 0.2] + [np.nan] * n_nan_energy
    rae_relaxed = [-0.1, -0.2, np.nan] + [np.nan] * n_nan_energy
    n_rings = ([0, 0, 1] if not with_rings else [1, 2, 3]) + [0] * n_nan_energy
    ring_max = ([0, 0, 5] if not with_rings else [3, 5, 7]) + [0] * n_nan_energy

    return pd.DataFrame(
        {
            "abs_energy": abs_energies[:n],
            "SMILES": smiles[:n],
            "valid": valid[:n],
            "dipole": dipole[:n],
            "relax_stable": relax_stable[:n],
            "RMSD": rmsd[:n],
            "rae_relaxed": rae_relaxed[:n],
            "n_rings": n_rings[:n],
            "n_atoms_ring_max": ring_max[:n],
        }
    )


# ---------------------------------------------------------------------------
# make_serializable
# ---------------------------------------------------------------------------


class TestMakeSerializable(TestCase):
    def test_converts_np_int64(self):
        d = {"a": np.int64(5), "b": "hello", "c": 3.14}
        result = make_serializable(d)
        self.assertIsInstance(result["a"], int)
        self.assertEqual(result["a"], 5)

    def test_leaves_other_types_unchanged(self):
        d = {"x": 1, "y": 2.0, "z": "text"}
        result = make_serializable(d)
        self.assertIsInstance(result["x"], int)
        self.assertIsInstance(result["y"], float)
        self.assertIsInstance(result["z"], str)


# ---------------------------------------------------------------------------
# single_formula_metrics
# ---------------------------------------------------------------------------


class TestSingleFormulaMetrics(TestCase):
    def setUp(self):
        # 3 rows with valid energy (indices 0-2), 1 NaN energy row (index 3)
        self.df = _make_df()

    def test_empty_df_returns_empty_dict(self):
        result = single_formula_metrics(pd.DataFrame())
        self.assertEqual(result, {})

    def test_total_samples_excludes_nan_energy(self):
        result = single_formula_metrics(self.df)
        self.assertEqual(result["total_samples"], 3)

    def test_validity_ratio(self):
        result = single_formula_metrics(self.df)
        self.assertAlmostEqual(result["valid_per_sample"], 1.0)
        self.assertEqual(result["n_valid"], 3)

    def test_unique_smiles_count(self):
        # SMILES = ["C", "CCO", "C"] → 2 unique
        result = single_formula_metrics(self.df)
        self.assertEqual(result["n_unique"], 2)
        self.assertAlmostEqual(result["unique_per_sample"], 2 / 3)

    def test_energy_average(self):
        result = single_formula_metrics(self.df)
        expected = np.mean([-5.0, -4.0, -3.0])
        self.assertAlmostEqual(result["abs_energy_avg"], expected)

    def test_relax_stable_ratio(self):
        # relax_stable = [True, False, True] → 2/3
        result = single_formula_metrics(self.df)
        self.assertAlmostEqual(result["relax_stable"], 2 / 3)

    def test_ring_ratios(self):
        # n_atoms_ring_max = [0, 0, 5] → only 1 molecule has ring ≥ 3/4/5
        result = single_formula_metrics(self.df)
        self.assertAlmostEqual(result["ring3+_ratio"], 1 / 3)
        self.assertAlmostEqual(result["ring4+_ratio"], 1 / 3)
        self.assertAlmostEqual(result["ring5+_ratio"], 1 / 3)

    def test_rae_relaxed_average_skips_nan(self):
        # rae_relaxed = [-0.1, -0.2, NaN] → mean of 2 values
        result = single_formula_metrics(self.df)
        self.assertAlmostEqual(result["rae_relaxed_avg"], (-0.1 + -0.2) / 2)

    def test_with_smiles_db_rediscovery(self):
        # DB has only "C"; "CCO" is novel
        result = single_formula_metrics(self.df, SMILES_db=["C"])
        self.assertEqual(result["rediscovered"], 1)
        self.assertEqual(result["n_novel"], 1)
        self.assertAlmostEqual(result["rediscovery_ratio"], 1.0)
        self.assertAlmostEqual(result["expansion_ratio"], 1.0)

    def test_with_smiles_db_no_rediscovery(self):
        # DB has only "CCC" — none of the generated SMILES match
        result = single_formula_metrics(self.df, SMILES_db=["CCC"])
        self.assertEqual(result["rediscovered"], 0)
        self.assertEqual(result["n_novel"], 2)
        self.assertAlmostEqual(result["rediscovery_ratio"], 0.0)

    def test_without_smiles_db_fields_are_none(self):
        result = single_formula_metrics(self.df)
        self.assertIsNone(result["rediscovery_ratio"])
        self.assertIsNone(result["expansion_ratio"])
        self.assertIsNone(result["n_novel"])

    def test_all_nan_energy_gives_zero_samples(self):
        # When all abs_energy values are NaN, dropna removes all rows.
        # The function does not early-exit in this case — it returns a dict with total_samples=0.
        df = _make_df(n_valid=0, n_nan_energy=2)
        result = single_formula_metrics(df)
        self.assertEqual(result["total_samples"], 0)
        self.assertEqual(result["n_valid"], 0)
        self.assertAlmostEqual(result["valid_per_sample"], 0.0)


# ---------------------------------------------------------------------------
# get_global_metrics
# ---------------------------------------------------------------------------


class TestGetGlobalMetrics(TestCase):
    def setUp(self):
        # Two formulas with known sizes
        self.formula_metrics = {
            "CH4": {"old_data_size": 2, "valid_per_sample": 0.8, "n_valid": 10},
            "C2H6": {"old_data_size": 4, "valid_per_sample": 0.6, "n_valid": 20},
        }

    def test_size_weighted_average(self):
        result = get_global_metrics(self.formula_metrics, size_weighted=True)
        # weights: CH4=2/6, C2H6=4/6
        expected = 0.8 * (2 / 6) + 0.6 * (4 / 6)
        self.assertAlmostEqual(result["valid_per_sample"], expected, places=6)

    def test_unweighted_average(self):
        result = get_global_metrics(self.formula_metrics, size_weighted=False)
        expected = (0.8 + 0.6) / 2
        self.assertAlmostEqual(result["valid_per_sample"], expected, places=6)

    def test_single_formula_returns_same_values(self):
        metrics = {"CH4": {"old_data_size": 5, "valid_per_sample": 0.75}}
        result = get_global_metrics(metrics, size_weighted=True)
        self.assertAlmostEqual(result["valid_per_sample"], 0.75)

    def test_missing_key_in_one_formula_is_ejected(self):
        # C2H6 is missing "valid_per_sample" → ejected, only CH4 remains
        metrics = {
            "CH4": {"old_data_size": 2, "valid_per_sample": 0.8},
            "C2H6": {"old_data_size": 4},  # missing key
        }
        result = get_global_metrics(metrics, size_weighted=True)
        # Only CH4 in result → its value passes through
        self.assertAlmostEqual(result["valid_per_sample"], 0.8)


# ---------------------------------------------------------------------------
# get_num_episodes
# ---------------------------------------------------------------------------


class TestGetNumEpisodes(TestCase):
    def setUp(self):
        self.smiles = {"CH4": ["C", "CC"], "C2H6": ["CCO", "CCN", "CCC"]}
        self.formulas = ["CH4", "C2H6"]

    def test_const_gives_same_count_for_all(self):
        result = get_num_episodes(self.smiles, num_episodes_const=10, formulas=self.formulas)
        self.assertEqual(result, [10, 10])

    def test_prop_factor_proportional_to_smiles_count(self):
        result = get_num_episodes(self.smiles, prop_factor=3, formulas=self.formulas)
        # CH4: 2 * 3 = 6, C2H6: 3 * 3 = 9
        self.assertEqual(result[0], 6)
        self.assertEqual(result[1], 9)

    def test_both_provided_raises(self):
        with self.assertRaises(AssertionError):
            get_num_episodes(self.smiles, num_episodes_const=5, prop_factor=2, formulas=self.formulas)

    def test_neither_provided_raises(self):
        with self.assertRaises(AssertionError):
            get_num_episodes(self.smiles, formulas=self.formulas)
