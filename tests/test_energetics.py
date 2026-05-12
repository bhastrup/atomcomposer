from unittest import TestCase

import numpy as np

from src.performance.energetics import EnergyConverter, EnergyUnit, str_to_EnergyUnit

# Reference values: 1 Hartree = 27.2114 eV = 627.5096 kcal/mol
HARTREE_TO_EV = 27.21140
HARTREE_TO_KCAL = 627.50960


class TestEnergyConverter(TestCase):
    def test_same_unit_returns_unchanged(self):
        for unit in EnergyUnit:
            result = EnergyConverter.convert(1.0, unit, unit)
            self.assertAlmostEqual(result, 1.0)

    def test_hartree_to_ev(self):
        result = EnergyConverter.convert(1.0, EnergyUnit.HARTREE, EnergyUnit.EV)
        self.assertAlmostEqual(result, HARTREE_TO_EV, places=4)

    def test_ev_to_hartree(self):
        result = EnergyConverter.convert(HARTREE_TO_EV, EnergyUnit.EV, EnergyUnit.HARTREE)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_hartree_to_kcal(self):
        result = EnergyConverter.convert(1.0, EnergyUnit.HARTREE, EnergyUnit.KCAL_MOL)
        self.assertAlmostEqual(result, HARTREE_TO_KCAL, places=4)

    def test_kcal_to_hartree(self):
        result = EnergyConverter.convert(HARTREE_TO_KCAL, EnergyUnit.KCAL_MOL, EnergyUnit.HARTREE)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_ev_to_kcal(self):
        result = EnergyConverter.convert(1.0, EnergyUnit.EV, EnergyUnit.KCAL_MOL)
        expected = HARTREE_TO_KCAL / HARTREE_TO_EV
        self.assertAlmostEqual(result, expected, places=4)

    def test_kcal_to_ev(self):
        result = EnergyConverter.convert(1.0, EnergyUnit.KCAL_MOL, EnergyUnit.EV)
        expected = HARTREE_TO_EV / HARTREE_TO_KCAL
        self.assertAlmostEqual(result, expected, places=4)

    def test_round_trip_hartree_ev(self):
        original = 2.5
        converted = EnergyConverter.convert(original, EnergyUnit.HARTREE, EnergyUnit.EV)
        recovered = EnergyConverter.convert(converted, EnergyUnit.EV, EnergyUnit.HARTREE)
        self.assertAlmostEqual(recovered, original, places=5)

    def test_scalar_and_array_equivalent(self):
        arr = np.array([1.0, 2.0, 3.0])
        scalar_results = [EnergyConverter.convert(v, EnergyUnit.HARTREE, EnergyUnit.EV) for v in arr]
        array_result = EnergyConverter.convert(arr, EnergyUnit.HARTREE, EnergyUnit.EV)
        self.assertTrue(np.allclose(array_result, scalar_results))

    def test_unknown_conversion_raises(self):
        with self.assertRaises((ValueError, KeyError)):
            # Force an unsupported conversion by bypassing the enum
            EnergyConverter.conversion_factors[("bad", "unit")]


class TestStrToEnergyUnit(TestCase):
    def test_hartree(self):
        self.assertEqual(str_to_EnergyUnit("hartree"), EnergyUnit.HARTREE)

    def test_ev(self):
        self.assertEqual(str_to_EnergyUnit("eV"), EnergyUnit.EV)

    def test_kcal(self):
        self.assertEqual(str_to_EnergyUnit("kcal/mol"), EnergyUnit.KCAL_MOL)

    def test_case_insensitive(self):
        self.assertEqual(str_to_EnergyUnit("HARTREE"), EnergyUnit.HARTREE)
        self.assertEqual(str_to_EnergyUnit("EV"), EnergyUnit.EV)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            str_to_EnergyUnit("joules")
