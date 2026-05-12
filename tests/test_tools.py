from unittest import TestCase

import numpy as np

from src.tools.util import (
    add_atom_to_formula,
    bag_tuple_to_str_formula,
    discount_cumsum,
    find_count,
    remove_atom_from_formula,
    str_formula_to_bag_tuple,
    string_to_formula,
    zs_to_formula,
)


class TestFormulaConversion(TestCase):
    def test_string_to_formula_water(self):
        formula = string_to_formula("H2O")
        counts = dict(formula)
        self.assertEqual(counts[1], 2)  # H
        self.assertEqual(counts[8], 1)  # O

    def test_string_to_formula_methane(self):
        formula = string_to_formula("CH4")
        counts = dict(formula)
        self.assertEqual(counts[6], 1)  # C
        self.assertEqual(counts[1], 4)  # H

    def test_zs_to_formula(self):
        formula = zs_to_formula([1, 1, 6, 8])
        counts = dict(formula)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[6], 1)
        self.assertEqual(counts[8], 1)
        self.assertEqual(len(formula), 3)

    def test_bag_tuple_to_str_formula(self):
        formula = ((1, 2), (8, 1))
        s = bag_tuple_to_str_formula(formula)
        self.assertIn("H2", s)
        self.assertIn("O", s)

    def test_str_formula_to_bag_tuple(self):
        bag = str_formula_to_bag_tuple("CH4")
        counts = dict(bag)
        self.assertEqual(counts[6], 1)
        self.assertEqual(counts[1], 4)

    def test_round_trip_str_formula(self):
        original = "CH4"
        bag = str_formula_to_bag_tuple(original)
        recovered = bag_tuple_to_str_formula(tuple(bag))
        self.assertIn("C", recovered)
        self.assertIn("H4", recovered)


class TestFormulaManipulation(TestCase):
    def setUp(self):
        self.formula = string_to_formula("CH4")

    def test_remove_atom(self):
        updated = remove_atom_from_formula(self.formula, atomic_number=1)
        counts = dict(updated)
        self.assertEqual(counts[1], 3)
        self.assertEqual(counts[6], 1)

    def test_remove_missing_atom_raises(self):
        with self.assertRaises(RuntimeError):
            remove_atom_from_formula(self.formula, atomic_number=8)

    def test_add_atom(self):
        updated = add_atom_to_formula(self.formula, atomic_number=1)
        counts = dict(updated)
        self.assertEqual(counts[1], 5)

    def test_find_count(self):
        self.assertEqual(find_count(self.formula, atomic_number=1), 4)
        self.assertEqual(find_count(self.formula, atomic_number=6), 1)
        self.assertEqual(find_count(self.formula, atomic_number=8), 0)


class TestDiscountCumsum(TestCase):
    def test_values(self):
        discount = 0.5
        x = np.ones(3, dtype=np.float32)
        y = discount_cumsum(x, discount=discount)
        self.assertAlmostEqual(y[0], 1.0 + 0.5 * 1.0 + 0.25 * 1.0)
        self.assertAlmostEqual(y[1], 1.0 + 0.5 * 1.0)
        self.assertAlmostEqual(y[2], 1.0)

    def test_no_discount(self):
        x = np.array([1.0, 2.0, 3.0])
        y = discount_cumsum(x, discount=1.0)
        self.assertAlmostEqual(y[0], 6.0)
        self.assertAlmostEqual(y[1], 5.0)
        self.assertAlmostEqual(y[2], 3.0)

    def test_zero_discount(self):
        x = np.array([1.0, 2.0, 3.0])
        y = discount_cumsum(x, discount=0.0)
        self.assertAlmostEqual(y[0], 1.0)
        self.assertAlmostEqual(y[1], 2.0)
        self.assertAlmostEqual(y[2], 3.0)
