from unittest import TestCase

import numpy as np
from ase import Atom, Atoms

from src.rl.spaces import BagSpace, CanvasItemSpace, CanvasSpace, ObservationSpace
from src.tools.util import string_to_formula

ZS = [0, 1, 6, 7, 8]


class TestCanvasItemSpace(TestCase):
    def setUp(self):
        self.space = CanvasItemSpace(zs=ZS)

    def test_round_trip(self):
        atom = Atom(symbol="H", position=(1.0, 2.0, 3.0))
        item = self.space.from_atom(atom)
        recovered = self.space.to_atom(item)
        self.assertEqual(recovered.symbol, "H")
        self.assertTrue(np.allclose(recovered.position, atom.position))

    def test_carbon_index(self):
        atom = Atom(symbol="C", position=(0.0, 0.0, 0.0))
        index, _ = self.space.from_atom(atom)
        self.assertEqual(index, ZS.index(6))

    def test_invalid_label(self):
        with self.assertRaises(RuntimeError):
            self.space.to_atom((-1, (0.0, 0.0, 0.0)))

    def test_element_not_in_zs(self):
        atom = Atom(symbol="He", position=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            self.space.from_atom(atom)


class TestCanvasSpace(TestCase):
    def setUp(self):
        self.space = CanvasSpace(size=5, zs=ZS)

    def test_round_trip(self):
        atoms = Atoms(
            symbols="CH4",
            positions=[
                (0.000, 0.000, 0.000),
                (0.629, 0.629, 0.629),
                (-0.629, -0.629, 0.629),
                (-0.629, 0.629, -0.629),
                (0.629, -0.629, -0.629),
            ],
        )
        canvas = self.space.from_atoms(atoms)
        recovered = self.space.to_atoms(canvas)
        self.assertEqual(list(recovered.symbols), list(atoms.symbols))
        self.assertTrue(np.allclose(recovered.positions, atoms.positions))

    def test_empty_atoms(self):
        canvas = self.space.from_atoms(Atoms())
        recovered = self.space.to_atoms(canvas)
        self.assertEqual(len(recovered), 0)

    def test_padding_to_size(self):
        atoms = Atoms(symbols="CO", positions=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0)])
        canvas = self.space.from_atoms(atoms)
        self.assertEqual(len(canvas), self.space.size)

    def test_too_many_atoms(self):
        atoms = Atoms(
            symbols=["C"] + ["H"] * 5,
            positions=[(float(i), 0.0, 0.0) for i in range(6)],
        )
        with self.assertRaises(RuntimeError):
            self.space.from_atoms(atoms)


class TestBagSpace(TestCase):
    def setUp(self):
        self.space = BagSpace(zs=ZS)

    def test_round_trip(self):
        formula = string_to_formula("CH4")
        bag = self.space.from_formula(formula)
        recovered = self.space.to_formula(bag)
        counts = dict(recovered)
        self.assertEqual(counts[1], 4)
        self.assertEqual(counts[6], 1)

    def test_bag_length(self):
        bag = self.space.from_formula(string_to_formula("H2O"))
        self.assertEqual(len(bag), len(ZS))

    def test_invalid_element(self):
        formula = string_to_formula("He2")
        with self.assertRaises(AssertionError):
            self.space.from_formula(formula)

    def test_empty_formula(self):
        bag = self.space.from_formula(())
        self.assertEqual(sum(bag), 0)


class TestObservationSpace(TestCase):
    def setUp(self):
        self.space = ObservationSpace(canvas_size=5, zs=ZS)

    def test_build_and_parse(self):
        atoms = Atoms(symbols="CO", positions=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0)])
        formula = string_to_formula("CO")
        obs = self.space.build(atoms, formula)
        recovered_atoms, recovered_formula = self.space.parse(obs)
        self.assertEqual(list(recovered_atoms.symbols), list(atoms.symbols))
        counts = dict(recovered_formula)
        self.assertEqual(counts[6], 1)
        self.assertEqual(counts[8], 1)

    def test_canvas_and_bag_sizes(self):
        obs = self.space.build(Atoms(), string_to_formula("H2O"))
        canvas, bag = obs
        self.assertEqual(len(canvas), self.space.canvas_space.size)
        self.assertEqual(len(bag), len(ZS))
